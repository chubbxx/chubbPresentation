import os
import threading
import queue
import tkinter as tk
import logging
from tkinter import scrolledtext
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

class AssistantBot:
    def __init__(self):
        """
        Initialize the AssistantBot with necessary configurations and logging.
        
        Args:
            telegram_token (str): The Telegram bot token.
            assistant_id (str): The ID of the assistant for OpenAI interaction.
        """
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
        )
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("apscheduler").setLevel(logging.WARNING)  # Suppress APScheduler logs
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.API_KEY:
            raise ValueError("API key is missing. Please set the OPENAI_API_KEY environment variable.")

        self.TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        if not self.TELEGRAM_TOKEN:
            raise ValueError("Telegram token is missing. Please set the TELEGRAM_TOKEN environment variable.")

        self.ASSISTANT_ID = os.getenv("ASSISTANT_ID")
        if not self.ASSISTANT_ID:
            raise ValueError("Assistant ID is missing. Please set the ASSISTANT_ID environment variable.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.API_KEY)

        # Initialize variables
        self.is_processing = False
        self.root = None
        self.text_display = None
        self.gui_thread = None
        self.message_queue = queue.Queue()
        self.done_queue = queue.Queue()
        self.bot_app = None

    def stream_events(self, user_message, update):
        """
        Fetch and process events from OpenAI stream. This function will send the user's message to OpenAI 
        and update the GUI with the assistant's response.

        Args:
            user_message (str): The message from the user.
            update (telegram.Update): The update object representing the message.
        """
        self.is_processing = True
        assistant_response = ""
        try:
            # Creating and running the OpenAI stream
            stream = self.client.beta.threads.create_and_run(
                assistant_id=self.ASSISTANT_ID,
                thread={
                    "messages": [
                        {"role": "user", "content": user_message}
                    ]
                },
                stream=True
            )
            # Add a separator for responses in the queue
            self.message_queue.put(f"\n\nAnswers for {user_message}:\n______________________________________\n\n")  
            
            # Process the response from OpenAI stream
            for event in stream:
                if event.event == "thread.message.delta":
                    delta = event.data.delta
                    if delta and delta.content:
                        for block in delta.content:
                            if block.type == 'text':
                                text_value = block.text.value
                                assistant_response += text_value
                                self.message_queue.put(text_value)
                elif event.event == "thread.message.completed":
                    # Signal completion and handle follow-up
                    follow_up_thread = threading.Thread(target=self.stream_follow_up, args=(user_message, assistant_response), daemon=True)
                    follow_up_thread.start()
                    follow_up_thread.join()  # Wait for follow-up to complete
                    self.message_queue.put(None)  # Mark the end of the stream
                    self.done_queue.put(update)  # Signal completion for the current update

        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.message_queue.put(f"An error occurred: {str(e)}")
        finally:
            self.is_processing = False

    def stream_follow_up(self, user_message, assistant_response):
        """
        Generate possible follow-up questions and answers based on the user message and assistant response.

        Args:
            user_message (str): The message from the user.
            assistant_response (str): The response from the assistant.
        """
        try:
            messages = [
                {"role": "system", "content": (
                    "You are an assistant tasked with generating possible follow-up questions based on the input question and answer, along with well-structured responses to aid in addressing examiner questions during a thesis results seminar. "
                    "Focus on commonly asked follow-up questions that frequently arise in thesis defense sessions and are relevant to the input question and answer. "
                    "For each response, generate a structured and concise answer that i can be directly read aloud during the presentation. "
                    "If applicable, highlight supporting data, statistical results, or comparisons with existing literature from the input, and avoid introducing speculative values or additional outcomes not explicitly mentioned. "
                    "Use your general knowledge and other references if the question is a common one or to improve argument "
                    "Always respond in Indonesian, don't use us but me, and don't use language that is too stiff, so that I can read it directly during the presentation."
                )},
                {"role": "user", "content": f"Question: {user_message}\nAnswer: {assistant_response}" }
            ]

            # Notify user about possible follow-up generation
            self.message_queue.put(f"\n\nPossible Follow-up Questions and Answers:\n______________________________________\n\n")

            # Request follow-up from OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=1,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True
            )

            # Stream the follow-up responses
            for chunk in response:
                delta = chunk.choices[0].delta
                content = getattr(delta, 'content', None)
                if content:
                    self.message_queue.put(content)
        except Exception as e:
            self.logger.error(f"An error occurred while generating follow-up: {str(e)}")
            self.message_queue.put(f"\nAn error occurred while generating follow-up: {str(e)}\n")

    def start_streaming(self, user_message, update):
        """
        Start a new thread to handle the streaming process for the given message.

        Args:
            user_message (str): The message from the user.
            update (telegram.Update): The update object representing the message.
        """
        thread = threading.Thread(target=self.stream_events, args=(user_message, update), daemon=True)
        thread.start()

    def update_gui(self):
        """
        Update the GUI to display new messages from the message queue.
        This function is called repeatedly to keep the GUI responsive.
        """
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg is None:
                    self.logger.info("Stream completed")
                    break
                elif msg.startswith("An error occurred:"):
                    self.logger.error(msg)
                    if self.text_display and self.root and self.root.winfo_exists():
                        self.text_display.configure(state='normal')
                        self.text_display.insert(tk.END, f"\nERROR: {msg}\n")
                        self.text_display.configure(state='disabled')
                        self.text_display.see(tk.END)
                    break
                else:
                    if self.text_display and self.root and self.root.winfo_exists():
                        self.text_display.configure(state='normal')
                        if "Answers for" in msg or "Possible Follow-up Questions and Answers" in msg:
                            # Insert separator text with 'separator' tag to apply color
                            self.text_display.insert(tk.END, msg, "separator")
                            self.text_display.see(tk.END)  # Auto-scroll for new responses
                        else:
                            self.text_display.insert(tk.END, msg)
                        self.text_display.configure(state='disabled')
        except queue.Empty:
            pass
        finally:
            if self.root and self.root.winfo_exists():
                self.root.after(100, self.update_gui)

    def create_gui(self):
        """
        Create and run the Tkinter GUI to display the assistant's response.
        It updates the GUI with the messages coming from the message queue.
        """
        if self.root is None or not self.root.winfo_exists():
            self.root = tk.Tk()
            self.root.title("Assistant Output")
            self.root.geometry("600x700")

            self.root.attributes("-topmost", True)

            # Apply Dark Mode Colors
            bg_color = "#000000"
            fg_color = "#ffffff"
            text_bg = "#000000"
            text_fg = "#ffffff"

            self.root.configure(bg=bg_color)

            # Title Label
            title = tk.Label(self.root, text="Assistant Response:", bg=bg_color, fg=fg_color, font=("Helvetica", 14))
            title.pack(pady=10)

            # ScrolledText Widget for Displaying Messages
            self.text_display = scrolledtext.ScrolledText(
                self.root,
                wrap=tk.WORD,
                state='disabled',
                font=("Helvetica", 12),
                bg=text_bg,
                fg=text_fg,
                insertbackground=fg_color
            )
            self.text_display.pack(expand=True, fill='both', padx=10, pady=10)

            self.text_display.tag_configure("separator", foreground="yellow")
            # Start GUI update
            self.update_gui()

            # Run the Tkinter event loop
            self.root.mainloop()

        # If the window was closed, reset the root and text_display
        self.root = None
        self.text_display = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Send a greeting message when the /start command is issued by the user.
        """
        await update.message.reply_text('Hi! Send me a message to process with OpenAI.')

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming messages from the user. It starts the processing of the user's message and 
        triggers GUI creation if necessary.

        Args:
            update (telegram.Update): The update object representing the incoming message.
            context (telegram.ext.CallbackContext): The context for the callback.
        """
        if self.is_processing:
            await update.message.reply_text("I'm still processing the previous request. Please wait.")
            return

        user_message = update.message.text
        self.logger.info(f"Received message: {user_message}")

        # Start new processing
        self.start_streaming(user_message, update)
        
        # Create new GUI if it doesn't exist or was closed
        if self.root is None or not self.root.winfo_exists():
            self.gui_thread = threading.Thread(target=self.create_gui, daemon=True)
            self.gui_thread.start()

        await update.message.reply_text("Processing your request...")

    async def send_done_message(self, update):
        """
        Send a 'Done' message to the user after processing is completed.
        """
        await update.message.reply_text("Done")

    async def check_done_queue(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Periodically check the done_queue to send completion messages to users.

        Args:
            context (telegram.ext.CallbackContext): The context for the callback.
        """
        try:
            while True:
                update = self.done_queue.get_nowait()
                await self.send_done_message(update)
        except queue.Empty:
            pass

    def main(self) -> None:
        """
        Start the Telegram bot. It sets up the application and handlers, 
        then runs the bot with polling.

        Args:
            None
        """
        self.bot_app = Application.builder().token(self.TELEGRAM_TOKEN).build()

        self.bot_app.add_handler(CommandHandler("start", self.start))
        self.bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Add a job to check the done_queue periodically
        self.bot_app.job_queue.run_repeating(self.check_done_queue, interval=1.0, first=0.0)

        self.logger.info("Starting bot...")
        self.bot_app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    bot = AssistantBot()
    bot.main()
