# -*- coding: utf-8 -*-
import json
import logging
import os
import time
import pytz
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import bitmex

load_dotenv()

def update_order_params(order_params, pre="SL", **updates):
    # Create a new dictionary to avoid modifying the original
    temp = dict(order_params)
    temp["clOrdID"] = f"{pre};{temp.get('clOrdID', '')}"
    temp["ordType"] = "Stop" if pre == "SL" else "Limit"
    temp["execInst"] = "Close"
    temp["side"] = "Sell" if temp["side"] == "Buy" else "Buy"
    temp["contingencyType"] = "OneCancelsTheOther"
    temp["clOrdLinkID"] = temp["clOrdID"]  # Use clOrdID as clOrdLinkID for linking orders
    temp["stopPx"] = None
    temp["price"] = None

    # Update specific parameters based on the updates dictionary
    for key, value in updates.items():
        temp[key] = value

    return temp

class BitMEXTestAPI:
    """BitMEX API client for trading operations."""
    
    def __init__(self, api_key, api_secret, test=True, symbol='SOL-USD', Log=None):
        """
        Initialize BitMEX API client.

        Args:
            api_key (str): BitMEX API key
            api_secret (str): BitMEX API secret
            test (bool): Use testnet if True (default: True)
            symbol (str): Trading symbol (default: 'SOL-USD')
            logger (logging.Logger): Custom logger instance (optional)
        """
        self.logger = Log if Log else logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        try:
            self.client = bitmex.bitmex(
                test=test,
                api_key=api_key,
                api_secret=api_secret
            )
            self.symbol = symbol.replace('-', '')  # Normalize to 'SOLUSD'
            self.max_balance_usage = 0.20  # Max 20% of balance per position
            network_type = 'testnet' if test else 'mainnet'
            self.logger.info(f"BitMEXTestAPI initialized for {self.symbol} on {network_type}")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    # ... (Other methods remain unchanged)

    def open_position(self, side="Buy", quantity=100, order_type="Market", price=None, 
                     exec_inst=None, take_profit_price=None, stop_loss_price=None, 
                     clOrdID=None, text=None):
        """
        Open a trading position with optional Take Profit and Stop Loss orders.

        Args:
            side (str): 'Buy' or 'Sell'
            quantity (int): Number of contracts
            order_type (str): 'Market' or 'Limit'
            price (float): Limit price (required for Limit orders)
            exec_inst (str): Execution instructions
            take_profit_price (float): Price for Take Profit order
            stop_loss_price (float): Price for Stop Loss order
            clOrdID (str): Client Order ID
            text (str): Order text/note

        Returns:
            dict: Entry order details or None if error
        """
        try:
            self.logger.info(f"Attempting to open {side} position for {quantity} contracts on {self.symbol}")
            normalized_side = "Sell" if str(side).strip().lower() in ["short", "sell"] else "Buy"

            profile = self.get_profile_info()
            if not profile or not profile['balance']:
                self.logger.error("Could not retrieve profile information")
                return None

            current_price = self.get_candle(timeframe='1m', count=1)
            current_price = current_price['close'].iloc[-1] if current_price is not None and not current_price.empty else None

            # Check position proximity
            for pos in profile['positions']:
                if pos['symbol'] == self.symbol and pos['current_qty'] != 0:
                    existing_entry = float(f"{pos['avg_entry_price']:.2f}")
                    if current_price:
                        current_price_rounded = float(f"{current_price:.2f}")
                        if normalized_side == ("Buy" if pos['current_qty'] <= 0 else "Sell"):
                            if abs(current_price_rounded - existing_entry) <= 1.5:
                                self.logger.warning(f"Existing position at {existing_entry}. Skipping near {current_price_rounded}")
                                return None

            # Check balance usage
            available_balance_usd = profile['balance']['bitmex_usd']
            if available_balance_usd and current_price:
                leverage = next((pos['leverage'] for pos in profile['positions'] if pos['symbol'] == self.symbol), 1)
                order_value_usd = quantity * current_price / leverage
                if available_balance_usd > 0 and (order_value_usd / available_balance_usd) > self.max_balance_usage:
                    self.logger.warning(f"Order exceeds {self.max_balance_usage*100}% of balance")
                    return None
                elif available_balance_usd <= 0:
                    self.logger.warning("Zero or negative balance")
                    return None

            clOrdID = clOrdID or str(uuid.uuid4())[:6]  # Generate a unique ID if not provided
            
            # Prepare entry order
            order_params = {
                "symbol": self.symbol,
                "side": normalized_side,
                "orderQty": quantity,
                "ordType": order_type,
                "clOrdID": clOrdID,
                "text": text
            }
            
            if order_type == "Limit" and price is not None:
                order_params["price"] = round(price, 2)
            
            if exec_inst:
                order_params["execInst"] = exec_inst

            orders = {"entry": None, "take_profit": None, "stop_loss": None}
            position_qty = sum(pos['current_qty'] for pos in profile['positions'] if pos['symbol'] == self.symbol)
            
            if abs(position_qty) < 20:  # Assuming 20 contracts as a threshold for new positions
                # Entry Order
                orders["entry"] = self.client.Order.Order_new(**order_params).result()[0]
                
                # Stop Loss Order
                if stop_loss_price is not None:
                    sl_params = update_order_params(order_params, pre='SL', stopPx=round(stop_loss_price, 2))
                    orders["stop_loss"] = self.client.Order.Order_new(**sl_params).result()[0]
                    self.logger.info(f"Stop Loss order placed: {orders['stop_loss']['ordStatus']} | OrderID: {orders['stop_loss']['orderID']} | "
                                     f"clOrdID: {orders['stop_loss'].get('clOrdID')} | text: {orders['stop_loss'].get('text')} | "
                                     f"clOrdLinkID: {orders['stop_loss'].get('clOrdLinkID')}")

                # Take Profit Order
                if take_profit_price is not None:
                    tp_params = update_order_params(order_params, pre='TP', price=round(take_profit_price, 2))
                    orders["take_profit"] = self.client.Order.Order_new(**tp_params).result()[0]
                    self.logger.info(f"Take Profit order placed: {orders['take_profit']['ordStatus']} | OrderID: {orders['take_profit']['orderID']} | "
                                     f"clOrdID: {orders['take_profit'].get('clOrdID')} | text: {orders['take_profit'].get('text')} | "
                                     f"clOrdLinkID: {orders['take_profit'].get('clOrdLinkID')}")

                self.logger.info(f"Entry order placed: {orders['entry']['ordStatus']} | OrderID: {orders['entry']['orderID']} | "
                                 f"clOrdID: {orders['entry'].get('clOrdID')} | text: {orders['entry'].get('text')}")
                
            else:
                self.logger.warning(f"Total position quantity {position_qty} >= 20. Skipping order")
                return None

            time.sleep(1)  # Small delay to ensure orders are processed
            self.get_profile_info()  # Refresh profile info after order placement
            return orders
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None

    # ... (Other methods remain unchanged)

if __name__ == "__main__":
    from TeleLogBot import configure_logging  # Assuming this exists
    logger, telegram_bot = configure_logging(os.getenv("TOKEN"), os.getenv("CHAT_ID"))
    api = BitMEXTestAPI(
        api_key=os.getenv("BITMEX_API_KEY"),
        api_secret=os.getenv("BITMEX_API_SECRET"),
        test=True,
        symbol="SOL-USD",
        Log=logger
    )
    api.run_test_sequence()  # Assuming this method exists to test the sequence
