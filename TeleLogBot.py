#TeleLogBot.py 
import asyncio
import logging
import sys
import time
from io import BytesIO
import pytz
from datetime import datetime
from collections import deque
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import TelegramError
from httpx import AsyncClient, Limits

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_sast_time():
    """Get the current time in SAST (South Africa Standard Time)."""
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self._bot = None
        self._client = AsyncClient(
            limits=Limits(max_connections=50, max_keepalive_connections=10),
            timeout=10.0  # Reduced timeout for faster failure detection
        )
        # Message queue for rate limiting
        self.message_queue = deque()
        self.photo_queue = deque()
        self.is_processing = False
        self.rate_limit_delay = 3.0  # 3 seconds between messages (20 messages/minute)

    def _ensure_bot(self):
        """Initialize the bot if not already done."""
        if not self._bot:
            if not self.token or "your_bot_token" in self.token:
                raise ValueError("Invalid bot token.")
            trequest = HTTPXRequest(connection_pool_size=10)
            self._bot = Bot(token=self.token, request=trequest)

    async def _async_send_message(self, message):
        """Send a message asynchronously with flood control handling."""
        try:
            self._ensure_bot()
            await self._bot.send_message(chat_id=self.chat_id, text=message)
        except TelegramError as e:
            if "Flood control exceeded" in str(e):
                retry_after = self._extract_retry_after(e)
                logger.warning(f"Flood control exceeded. Retrying in {retry_after} seconds.")
                await asyncio.sleep(retry_after)
                await self._async_send_message(message)  # Retry after waiting
            else:
                logger.error(f"Telegram error sending message: {e}")
                raise
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def _async_send_photo(self, photo_buffer, caption):
        """Send a photo asynchronously with flood control handling."""
        try:
            self._ensure_bot()
            photo_buffer.seek(0)
            await self._bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_buffer,
                caption=caption
            )
        except TelegramError as e:
            if "Flood control exceeded" in str(e):
                retry_after = self._extract_retry_after(e)
                logger.warning(f"Flood control exceeded. Retrying in {retry_after} seconds.")
                await asyncio.sleep(retry_after)
                await self._async_send_photo(photo_buffer, caption)  # Retry after waiting
            else:
                logger.error(f"Telegram error sending photo: {e}")
                raise
        except Exception as e:
            logger.error(f"Error sending photo: {e}")
            raise

    def _extract_retry_after(self, error):
        """Extract the retry_after value from a Telegram flood control error."""
        try:
            error_str = str(error)
            if "Retry in" in error_str:
                return int(error_str.split("Retry in ")[1].split(" ")[0])
            return 60  # Default to 60 seconds if parsing fails
        except (IndexError, ValueError):
            return 60

    async def _process_queues(self):
        """Process the message and photo queues with rate limiting."""
        if self.is_processing:
            return
        self.is_processing = True
        try:
            while self.message_queue or self.photo_queue:
                # Process messages first
                if self.message_queue:
                    message = self.message_queue.popleft()
                    await self._async_send_message(message)
                    await asyncio.sleep(self.rate_limit_delay)  # Rate limit
                # Then process photos
                if self.photo_queue:
                    photo_data = self.photo_queue.popleft()
                    photo_buffer, caption = photo_data["buffer"], photo_data["caption"]
                    await self._async_send_photo(photo_buffer, caption)
                    photo_buffer.close()  # Close the buffer after sending
                    await asyncio.sleep(self.rate_limit_delay)  # Rate limit
        except Exception as e:
            logger.error(f"Error processing queues: {e}")
        finally:
            self.is_processing = False

    def send_message(self, message=None):
        """Synchronous wrapper for sending text messages."""
        if message is None:
            current_time = get_sast_time()
            message = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - This is a test message"
        self.message_queue.append(message)
        # Run the queue processor in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.create_task(self._process_queues())
            if not loop.is_running():
                loop.run_until_complete(self._process_queues())
        except Exception as e:
            logger.error(f"Error in send_message: {e}")

    def send_photo(self, fig=None, caption=None):
        """Synchronous wrapper for sending a matplotlib figure."""
        try:
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            if fig is None:
                plt.savefig(buffer, format='png', bbox_inches='tight')
            else:
                fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)

            if caption is None:
                caption = f"Chart at {get_sast_time().strftime('%Y-%m-%d %H:%M:%S')}"
            self.photo_queue.append({"buffer": buffer, "caption": caption})

            # Run the queue processor in a new event loop if needed
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.create_task(self._process_queues())
            if not loop.is_running():
                loop.run_until_complete(self._process_queues())

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

def configure_logging(bot_token, chat_id):
    """Configure logging with a Telegram bot handler."""
    logger = logging.getLogger(__name__)
    bot = TelegramBot(bot_token, chat_id)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        custom_handler = CustomLoggingHandler(bot)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        custom_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(custom_handler)
        logger.addHandler(stream_handler)

    return logger, bot

# Example usage (for testing)
if __name__ == "__main__":
    # Replace with your bot token and chat ID
    BOT_TOKEN = "your_bot_token_here"
    CHAT_ID = "your_chat_id_here"

    logger, bot = configure_logging(BOT_TOKEN, CHAT_ID)

    # Test sending messages
    for i in range(5):
        logger.info(f"Test message {i+1}")
        time.sleep(1)  # Simulate some delay between logs
