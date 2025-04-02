import os
import sys
import time
import logging
import pytz
from datetime import datetime

# Ensure the necessary imports are available
try:
    from TeleLogBot import configure_logging
    from MatteGreen import MatteGreen
    from BitMEXApi import BitMEXTestAPI
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Environment Variables (consider using a more secure method in production)
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

def get_sast_time():
    """
    Get current time in South African Standard Time (SAST)
    """
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

def long_running_task(logger,bot):
    """
    Runs the trading strategy for approximately 3 minutes
    """
    start_time = time.time()
    max_runtime = 60 * 60  # 3 minutes in seconds
    
    try:
        api = BitMEXTestAPI(
            api_key=API_KEY,
            api_secret=API_SECRET,
            test=True,
            Log=logger
            
        )
        # Initialize the trader
        trader = MatteGreen(
            api_key=API_KEY,
            api_secret=API_SECRET,
            test=True,
            symbol="XBT-USD",
            timeframe="5m",
            risk_per_trade=0.005,
            log=logger,
            #telegram_bot=bot, 
            #api=api
        )
        
        # Log start time
        current_time = get_sast_time()
        #logger.info(f"Starting MateGreen at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run the trading strategy
        while time.time() - start_time < max_runtime:
            # Call the run method of MateGreen
            signal_found, df = trader.run(2*60)
            
            # Optional: Add additional logging or processing
            if signal_found:
                logger.info("Trading signals detected during this scan")
            
            # Break if no more signals or runtime exceeded
            if not signal_found:
                break
    
    except Exception as e:
        logger.error(f"An error occurred during trading: {e}")
        sys.exit(1)
    
    logger.info("Trading session completed within time limit")

def main():
    """
    Main function to run the script
    """
    # Configure logging
    logger, bot = configure_logging(TOKEN, CHAT_ID)
    
    logger.info("Starting trading script")
    long_running_task(logger,bot)
    logger.info("Script finished")

if __name__ == "__main__":
    main()
