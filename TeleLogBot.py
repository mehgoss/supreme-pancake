import asyncio
import json
import logging
import os
import threading
import time
import telebot
import pytz
from datetime import datetime, timedelta
from queue import Queue
import sys
import bitmex
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import yfinance as yf
import telegram
from telegram import Bot, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes
from io import BytesIO



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)
    
    

class TelegramBot:
    def __init__(self, token, chat_id):
        """
        Initialize the Telegram bot.

        :param token: Telegram bot token
        :param chat_id: Chat ID to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self._bot = None

    async def _async_send_message(self, message=None):
        """
        Async method to send a message.

        :param message: Custom message to send (optional)
        """
        try:
            if not self._bot:
                self._bot = Bot(token=self.token)

            if message is None:
                current_time = get_sast_time()
                message = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')},100 - INFO - This is a test message"

            await self._bot.send_message(chat_id=self.chat_id, text=message)

        except TelegramError as e:
            logger.error(f"Telegram error sending message: {e}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def _async_send_photo(self, photo_buffer, caption=None):
        """
        Async method to send a photo.

        :param photo_buffer: BytesIO buffer containing the image
        :param caption: Optional caption for the photo
        """
        try:
            if not self._bot:
                self._bot = Bot(token=self.token)

            photo_buffer.seek(0)  # Reset buffer position to start
            await self._bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_buffer,
                caption=caption if caption else f"Chart generated at {get_sast_time().strftime('%Y-%m-%d %H:%M:%S')}"
            )

        except TelegramError as e:
            logger.error(f"Telegram error sending photo: {e}")
        except Exception as e:
            logger.error(f"Error sending photo: {e}")

    def send_message(self, message=None):
        """Wrapper for sending text messages synchronously"""
        try:
            if 'get_ipython' in sys.modules['__main__'].__dict__:
                import nest_asyncio
                nest_asyncio.apply()
            asyncio.run(self._async_send_message(message))
        except Exception as e:
            logger.error(f"Error in send_message: {e}")

    def send_photo(self, fig=None, caption=None):
        """
        Send a matplotlib figure or current plot as a photo.

        :param fig: Matplotlib figure object (optional, will use current plot if None)
        :param caption: Optional caption for the photo
        """
        try:
            # Create buffer for image
            buffer = BytesIO()
            
            # Use provided figure or get current plot
            if fig is None:
                plt.savefig(buffer, format='png', bbox_inches='tight')
            else:
                fig.savefig(buffer, format='png', bbox_inches='tight')
            
            buffer.seek(0)
            
            # Handle Jupyter environment
            if 'get_ipython' in sys.modules['__main__'].__dict__:
                import nest_asyncio
                nest_asyncio.apply()
            
            # Send the photo
            asyncio.run(self._async_send_photo(buffer, caption))
            
            # Clean up
            buffer.close()
            if fig is not None:
                plt.close(fig)

        except Exception as e:
            logger.error(f"Error in send_photo: {e}")

class CustomLoggingHandler(logging.Handler):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

    def emit(self, record):
        try:
            log_message = self.format(record)
            self.bot.send_message(log_message)
        except Exception as e:
            print(f"Error in custom logging handler: {e}", file=sys.stderr)

def configure_logging(BOT_TOKEN, CHAT_ID):
    """
    Configure logging with a custom Telegram handler.

    :return: Configured logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create bot instance
    bot = TelegramBot(BOT_TOKEN, CHAT_ID)
    
    # Create and configure custom handler
    custom_handler = CustomLoggingHandler(bot)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    custom_handler.setFormatter(formatter)
    
    # Configure stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Clear and add handlers
    logger.handlers.clear()
    logger.addHandler(custom_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Global logger instance (will be configured later)
logger = logging.getLogger(__name__)

# Example usage:
if __name__ == "__main__":
    # Example configuration - replace with actual token and chat_id
    BOT_TOKEN = "your_bot_token"
    CHAT_ID = "your_chat_id"
    
    # Configure logging
    logger = configure_logging(BOT_TOKEN, CHAT_ID)
    
    # Example bot usage
    bot = TelegramBot(BOT_TOKEN, CHAT_ID)
    
    # Send test message
    bot.send_message("Test message")
    
    # Example matplotlib plot
    plt.plot([1, 2, 3], [4, 5, 6])
    plt.title("Test Plot")
    bot.send_photo(caption="Test chart")