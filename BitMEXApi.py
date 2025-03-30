# -*- coding: utf-8 -*-
import json
import logging
import os
import threading
import time
import pytz
from datetime import datetime, timedelta
import sys
import bitmex
from dotenv import load_dotenv
from TeleLogBot import configure_logging
import pandas as pd

load_dotenv()

# Telegram credentials
TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Initialize logger and Telegram bot (assuming TeleLogBot is configured elsewhere)
logger, telegram_bot = configure_logging(TOKEN, CHAT_ID)

# Set the correct time zone
utc_now = datetime.utcnow()
sast = pytz.timezone('Africa/Johannesburg')
sast_now = utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

import yfinance as yf
from datetime import datetime

def btc_to_usd(btc_amount, fallback_price=80000):
    """
    Convert a BTC amount to USD using the current BTC/USD price from yfinance.

    :param btc_amount: Amount in BTC (e.g., 0.00999386)
    :param fallback_price: Fallback BTC/USD price in case yfinance fails (default: 80000 USD/BTC)
    :return: USD amount (float)
    """
    try:
        # Fetch current BTC/USD price from yfinance
        btc_ticker = yf.Ticker("BTC-USD")
        btc_data = btc_ticker.history(period="1d")  # Get latest daily data
        if not btc_data.empty:
            btc_usd_price = btc_data['Close'].iloc[-1]  # Use the latest closing price
        else:
            btc_usd_price = fallback_price
            logger.warning(f"Failed to fetch BTC/USD price from yfinance. Using fallback price: ${fallback_price}")

        # Calculate USD amount
        usd_amount = btc_amount * btc_usd_price
        logger.info(f"Converted {btc_amount:.8f} BTC to ${usd_amount:.2f} USD at ${btc_usd_price:.2f}/BTC")
        return usd_amount

    except Exception as e:
        logger.error(f"Error fetching BTC/USD price: {str(e)}. Using fallback price: ${fallback_price}")
        usd_amount = btc_amount * fallback_price
        return usd_amount


class BitMEXTestAPI:
    def __init__(self, api_key, api_secret, test=True, symbol='SOL-USD', Log=None):
        """
        Initialize BitMEX API client

        :param api_key: BitMEX API key
        :param api_secret: BitMEX API secret
        :param test: Whether to use testnet (default True)
        :param symbol: Trading symbol (default SOL-USD)
        """
        try:
            self.logger = Log if Log else logger
            self.client = bitmex.bitmex(
                test=test,
                api_key=api_key,
                api_secret=api_secret
            )
            self.symbol = symbol.replace('-', '')  # Symbol parsed here
            self.max_balance_usage = 0.30  # Maximum percentage of account balance to use for a position

            # Log initialization
            network_type = 'testnet' if test else 'mainnet'
            self.logger.info(f"BitMEXTestAPI initialized for {symbol} on {network_type}")

        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def get_profile_info(self):
        """
        Retrieve comprehensive account profile information

        :return: Dictionary with user, balance, and position detail
        """
        try:
            # Get user information
            user_info = self.client.User.User_get().result()[0]

            # Get account balance
            margin = self.client.User.User_getMargin().result()[0]

            # Get open positions
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]

            # Fetch current BTC/USD price for conversion (using BitMEX data)
            btc_price_data = self.client.Trade.Trade_getBucketed(
                symbol="BTCUSD",
                binSize="1m",
                count=1,
                reverse=True
            ).result()[0]
            btc_usd_price = btc_price_data[0]['close'] if btc_price_data else 40000  # Fallback price

            # Convert wallet balance from satoshis (1e8) to BTC and USD
            wallet_balance_btc = margin.get('walletBalance') / 100000000
            wallet_balance_usd = wallet_balance_btc * btc_usd_price

            # Format profile information
            profile_info = {
                "user": {
                    "id": user_info.get('id'),
                    "username": user_info.get('username'),
                    "email": user_info.get('email'),
                    "account": user_info.get('account')
                },
                "balance": {
                    "wallet_balance": margin.get('walletBalance'),
                    "margin_balance": margin.get('marginBalance'),
                    "available_margin": margin.get('availableMargin'),
                    "unrealized_pnl": margin.get('unrealisedPnl'),
                    "realized_pnl": margin.get('realisedPnl'),
                    "usd": btc_to_usd(wallet_balance_btc),
                    "bitmex_usd": wallet_balance_usd # Using BitMEX price for internal consistency
                },
                "positions": [{
                    "symbol": pos.get('symbol'),
                    "current_qty": pos.get('currentQty'),
                    "avg_entry_price": pos.get('avgEntryPrice'),
                    "leverage": pos.get('leverage'),
                    "liquidation_price": pos.get('liquidationPrice'),
                    "unrealized_pnl": pos.get('unrealisedPnl'),
                    "realized_pnl": pos.get('realisedPnl')
                } for pos in positions] if positions else []
            }

            # Logging profile details
            self.logger.info("Profile information retrieved successfully")
            self.logger.info(f"Account: {profile_info['user']['username']}")
            self.logger.info(f"Wallet Balance: {wallet_balance_btc:.8f} BTC ({profile_info['balance']['bitmex_usd']:.2f} USD)")
            self.logger.info(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} BTC")

            if profile_info['positions']:
                for pos in profile_info['positions']:
                    self.logger.info(f"📈🔵🔴Position: {pos['symbol']} | Qty: {pos['current_qty']} | Entry: {pos['avg_entry_price']}")
            else:
                self.logger.info("No open positions")

            return profile_info

        except Exception as e:
            self.logger.error(f"Error getting profile information: {str(e)}")
            return None

    def get_candle(self, timeframe='1m', count=100):
        """
        Retrieve candlestick (OHLCV) data for the specified symbol

        :param timeframe: Candle timeframe (default '1m' for 1 minute)
        :param count: Number of candles to retrieve (default 100)
        :return: DataFrame with candle data or None if error occurs
        """
        try:
            # Extended timeframe mapping
            timeframe_map = {
                '1m': '1m',
                '2m': '1m',
                '5m': '5m',
                '10m': '5m',
                '15m': '5m',
                '30m': '5m',
                '1h': '1h',
                '4h': '1h',
                '1d': '1d'
            }

            if timeframe not in timeframe_map:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(timeframe_map.keys())}")

            base_timeframe = timeframe_map[timeframe]
            multiplier = {
                '2m': 2, '10m': 2, '15m': 3, '30m': 6, '4h': 4
            }.get(timeframe, 1)

            # Adjust count for aggregation
            adjusted_count = count * multiplier if multiplier > 1 else count

            # Retrieve candle data
            candles = self.client.Trade.Trade_getBucketed(
                symbol=self.symbol,
                binSize=base_timeframe,
                count=adjusted_count,
                reverse=True
            ).result()[0]

            # Format initial candle data
            formatted_candles = [{
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            } for candle in candles]

            # Convert to DataFrame
            df = pd.DataFrame(formatted_candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Aggregate if needed
            if multiplier > 1:
                df = df.sort_values('timestamp')
                df = df.resample(f'{timeframe}', on='timestamp').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna().tail(count)

            # Log retrieval success
            self.logger.info(f"Retrieved {len(df)} {timeframe} candles for {self.symbol}")

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

    def open_test_position(self, side="Buy", quantity=100, order_type="Market", take_profit_price=None, stop_loss_price=None):
        """
        Open a test trading position with optional Take Profit and Stop Loss orders.

        :param side: Buy or Sell
        :param quantity: Number of contracts
        :param order_type: Type of order (default Market)
        :param take_profit_price: Price at which to place a Take Profit market order (optional)
        :param stop_loss_price: Price at which to place a Stop Loss market order (optional)
        :return: Order details for the entry order, or None if error
        """
        try:
            self.logger.info(f"Attempting to open {side} position for {quantity} contracts for {self.symbol}")
            normalized_side = "Sell" if str(side).strip().lower() in ["short", "sell"] else "Buy"
            opposite_side = "Buy" if normalized_side == "Sell" else "Sell"

            profile = self.get_profile_info()
            if not profile or not profile['balance']:
                self.logger.error("Could not retrieve profile information to check existing positions or balance.")
                return None

            # Check for existing open position within a price range
            for pos in profile['positions']:
                if pos['symbol'] == self.symbol:
                    existing_entry = float(f"{pos['avg_entry_price']:.2f}")
                    current_market_price = self.get_candle(timeframe='1m', count=1)['close'].iloc[-1] if not self.get_candle(timeframe='1m', count=1).empty else None
                    if current_market_price is not None:
                        current_price_rounded = float(f"{current_market_price:.2f}")
                        if normalized_side == ("Buy" if pos['current_qty'] <= 0 else "Sell"): # Check if trying to open in the same direction
                            if abs(current_price_rounded - existing_entry) <= 0.5:
                                self.logger.warning(f"Existing {pos['current_qty']} contract position at {existing_entry}. Skipping new {side} order near {current_price_rounded}.")
                                return None

            # Check account balance usage (using a rough estimate of current price)
            current_price = self.get_candle(timeframe='1m', count=1)['close'].iloc[-1] if not self.get_candle(timeframe='1m', count=1).empty else None
            available_balance_usd = profile['balance']['bitmex_usd']
            if available_balance_usd is not None and current_price is not None:
                potential_order_value_usd = quantity * current_price / profile['positions'][0].get('leverage', 1) if profile['positions'] else quantity * current_price # Rough estimate
                if available_balance_usd > 0 and (potential_order_value_usd / available_balance_usd) > self.max_balance_usage:
                    self.logger.warning(f"Attempting to use more than {self.max_balance_usage*100}% of account balance. Skipping order.")
                    return None
                elif available_balance_usd <= 0:
                    self.logger.warning("Available balance is zero or negative. Cannot open new position.")
                    return None
            elif current_price is None:
                self.logger.warning("Could not fetch current market price. Skipping order.")
                return None

            self.logger.info(f"🎉🎉🎉🎀Opening {side} position for {quantity} contracts for {self.symbol}🎀🎉🎉🎉")

            # Execute the entry order
            entry_order = self.client.Order.Order_new(
                symbol=self.symbol,
                side=normalized_side,
                orderQty=quantity if quantity > 0 else abs(int(quantity)) + 1 * 5, # Ensure quantity is positive
                ordType="Market"
            ).result()[0]

            # Log entry order details
            self.logger.info(f"Entry order placed: {entry_order['ordStatus']} | OrderID: {entry_order['orderID']}")
            self.logger.info(f"Entry order details: {side} {abs(quantity)} contracts at {entry_order.get('price', 'market price')}")

            # Wait for entry order to potentially fill (adjust time as needed)
            time.sleep(1)

            # Place Take Profit order if a price is provided
            if take_profit_price is not None:
                tp_order = self.client.Order.Order_new(
                    symbol=self.symbol,
                    side=opposite_side,
                    orderQty=abs(quantity),
                    stopPx=take_profit_price,
                    ordType='MarketIfTouched',  # Consider 'LimitIfTouched' for more price control
                    execInst='Close',
                    text='LastPrice'
                ).result()[0]
                self.logger.info(f"Take Profit order placed: {tp_order['orderID']} at trigger price: {take_profit_price}")

            # Place Stop Loss order if a price is provided
            if stop_loss_price is not None:
                sl_order = self.client.Order.Order_new(
                    symbol=self.symbol,
                    side=opposite_side,
                    orderQty=abs(quantity),
                    stopPx=stop_loss_price,
                    ordType='Stop',  # Consider 'StopLimit' for more price control
                    execInst='Close',
                    text='LastPrice'
                ).result()[0]
                self.logger.info(f"Stop Loss order placed: {sl_order['orderID']} at trigger price: {stop_loss_price}")

            # Wait for orders to settle
            time.sleep(2)
            self.get_profile_info()

            return entry_order

        #except bitmex.exceptions.BitMEXAPIError as e:
            #self.logger.error(f"BitMEX API Error opening position: {e}")
            #return None
        except Exception as e:
            self.logger.error(f"Error opening test position: {str(e)}")
            return None

    def _close_position(self, position):
        """
        Close a single position

        :param position: Position dictionary from Position_get
        :return: Order result or None if error
        """
        try:
            symbol = position['symbol']
            current_qty = position['currentQty']

            if current_qty == 0:
                self.logger.info(f"No open position for {symbol}")
                return None

            # Determine closing side
            side = "Sell" if current_qty > 0 else "Buy"
            qty = abs(current_qty)

            self.logger.info(f"Closing position: {symbol} | {current_qty} contracts | Side: {side} | Qty: {qty}")

            # Place closing order
            order = self.client.Order.Order_new(
                symbol=self.symbol,
                side=side,
                orderQty=qty,
                ordType="Market"
            ).result()[0]

            self.logger.info(f"🔴📈⁉️❗Position closed: {order['ordStatus']} |  OrderID: {order['orderID']}")
            print(f"Position closed: {order['ordStatus']} | OrderID: {order['orderID']}")

            return order

        except Exception as e:
            logger.error(f"Error closing position {position['symbol']}: {str(e)}")
            print(f"Error closing position {position['symbol']}: {str(e)}")
            return None

    def close_all_positions(self):
        """
        Close all open positions for the current symbol

        :return: True if successful, None if error
        """
        try:
            # Get current positions
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol if '-' not in self.symbol else self.symbol.replace('-USD', 'USD')})
            ).result()[0]

            if not positions:
                logger.info("No positions to close")
                print("No positions to close")
                return None

            # Close each position
            for position in positions:
                self._close_position(position)

            # Wait for orders to settle
            time.sleep(2)
            self.get_profile_info()

            return True

        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            print(f"Error closing positions: {str(e)}")
            return None

    def run_test_sequence(self):
        """
        Run a comprehensive test sequence of trading operations

        :return: Final profile information or None if error
        """
        try:
            logger.info("Starting test sequence")
            print("Starting test sequence")

            # Initial profile
            logger.info("=== INITIAL PROFILE ===")
            print("=== INITIAL PROFILE ===")
            self.get_profile_info()

            # Open long position
            logger.info("=== OPENING LONG POSITION(BUY)🔵  ===")
            print("=== OPENING LONG POSITION (BUY)🔵 ===")
            self.open_test_position(side="Buy", quantity=1)

            # Wait and check profile
            wait_time = 1
            logger.info(f"Waiting for {wait_time} seconds...")
            print(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            self.get_profile_info()

            # Open short position
            logger.info("=== OPENING SHORT POSITION(SELL)🔴  ===")
            print("=== OPENING SHORT POSITION(SELL)🔴 ===")
            self.open_test_position(side="Sell", quantity=1)

            # Wait and check profile
            logger.info(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            self.get_profile_info()

            # Close all positions
            logger.info("=== CLOSING ALL POSITIONS ===")
            print("=== CLOSING ALL POSITIONS ===")
            self.close_all_positions()

            # Final profile check
            logger.info("=== FINAL PROFILE ===")
            print("=== FINAL PROFILE ===")
            final_profile = self.get_profile_info()

            logger.info("Test sequence completed successfully")
            print("Test sequence completed successfully")
            return final_profile

        except Exception as e:
            logger.error(f"Error in test sequence: {str(e)}")
            print(f"Error in test sequence: {str(e)}")
            return None

# Example usage (optional, for testing):
if __name__ == "__main__":
    api_key = os.getenv("BITMEX_API_KEY")
    api_secret = os.getenv("BITMEX_API_SECRET")
    api = BitMEXTestAPI(api_key, api_secret, test=True, symbol="SOL-USD")
    api.run_test_sequence()
