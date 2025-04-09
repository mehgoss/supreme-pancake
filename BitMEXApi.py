# -*- coding: utf-8 -*-
import json
import logging
import os
import time
import pytz
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import bitmex  # Requires 'bitmex' package: pip install bitmex

load_dotenv()

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

    def get_profile_info(self):
        """
        Retrieve account profile information including balance and positions.

        Returns:
            dict: Profile information with user, balance, and position details, or None if error
        """
        try:
            user_info = self.client.User.User_get().result()[0]
            margin = self.client.User.User_getMargin().result()[0]
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]

            btc_price_data = self.client.Trade.Trade_getBucketed(
                symbol="XBTUSD",
                binSize="5m",
                count=1,
                reverse=True
            ).result()[0]
            btc_usd_price = btc_price_data[-1]['close'] if btc_price_data else 90000

            wallet_balance_btc = margin.get('walletBalance', 0) / 100000000  # Satoshis to BTC
            wallet_balance_usd = wallet_balance_btc * btc_usd_price

            profile_info = {
                "user": {
                    "id": user_info.get('id'),
                    "username": user_info.get('username'),
                    "email": user_info.get('email'),
                    "account": user_info.get('account')
                },
                "balance": {
                    "wallet_balance": margin.get('walletBalance', 0),
                    "margin_balance": margin.get('marginBalance', 0),
                    "available_margin": margin.get('availableMargin', 0),
                    "unrealized_pnl": margin.get('unrealisedPnl', 0),
                    "realized_pnl": margin.get('realisedPnl', 0),
                    "bitmex_usd": wallet_balance_usd
                },
                "positions": [{
                    "symbol": pos.get('symbol'),
                    "current_qty": pos.get('currentQty', 0),
                    "avg_entry_price": pos.get('avgEntryPrice'),
                    "leverage": pos.get('leverage', 1),
                    "liquidation_price": pos.get('liquidationPrice'),
                    "unrealized_pnl": pos.get('unrealisedPnl', 0),
                    "realized_pnl": pos.get('realisedPnl', 0)
                } for pos in positions] if positions else []
            }

            self.logger.info("Profile information retrieved successfully")
            self.logger.info(f"Account: {profile_info['user']['username']}")
            self.logger.info(f"Wallet Balance: {wallet_balance_btc:.8f} BTC (${wallet_balance_usd:.2f} USD)")
            self.logger.info(f"Available Margin: {profile_info['balance']['available_margin'] / 100000000:.8f} BTC")
            if profile_info['positions']:
                for pos in profile_info['positions']:
                    self.logger.info(f"Position: {pos['symbol']} | Qty: {pos['current_qty']} | Entry: {pos['avg_entry_price']}")
            else:
                self.logger.info("No open positions")

            return profile_info
        except Exception as e:
            self.logger.error(f"Error getting profile information: {str(e)}")
            return None

    def get_open_orders(self):
        """
        Retrieve open orders for the specified symbol.

        Returns:
            list: List of open orders
        """
        open_orders = self.client.Order.Order_getOrders(
            filter=json.dumps({"symbol": self.symbol})
        ).result()[0]
        return open_orders

    def get_transactions(self):
        """
        Retrieve wallet transaction history.

        Returns:
            list: List of wallet transactions
        """
        wallet_history = self.client.User.User_getWalletHistory().result()[0]
        return wallet_history

    def get_wallet_summary(self):
        """
        Retrieve wallet summary for the specified symbol.

        Returns:
            list: Wallet summary data
        """
        wallet_summary = self.client.User.User_getWalletSummary().result()[0]
        return wallet_summary

    def get_stats_history_usd(self):
        """
        Retrieve historical stats in USD.

        Returns:
            list: Historical statistics
        """
        stats_history = self.client.Stats.Stats_historyUSD().result()[0]
        return stats_history

    def get_positions(self):
        """
        Retrieve all current positions.

        Returns:
            list: List of position data
        """
        positions = self.client.Position.Position_get().result()[0]
        return positions

    def get_candle(self, timeframe='1m', count=100):
        """
        Retrieve candlestick (OHLCV) data for the specified symbol.

        Args:
            timeframe (str): Candle timeframe (e.g., '1m', '5m', '1h')
            count (int): Number of candles to retrieve

        Returns:
            pd.DataFrame: Candle data or None if error
        """
        try:
            valid_timeframes = ['1m', '5m', '1h', '1d']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}")

            candles = self.client.Trade.Trade_getBucketed(
                symbol=self.symbol,
                binSize=timeframe,
                count=count,
                reverse=True
            ).result()[0]

            if not candles:
                raise ValueError("No candle data returned")

            formatted_candles = [{
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            } for candle in candles if all(key in candle for key in ['open', 'high', 'low', 'close'])]

            df = pd.DataFrame(formatted_candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            self.logger.info(f"Retrieved {len(df)} {timeframe} candles for {self.symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

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
            cl_ord_id (str): Client Order ID
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

            cl_ord_id = cl_ord_id or 'No strings attached'
            
            # Prepare entry order
            order_params = {
                "symbol": self.symbol,
                "side": normalized_side,
                "orderQty": quantity,
                "ordType": order_type,
                "clOrdID": clOrdID,
                "text": text
            }
            if order_type == "Limit":
                if price is None:
                    raise ValueError("Price required for Limit order")
                order_params["price"] = price
            if exec_inst:
                order_params["execInst"] = exec_inst

            orders = {"entry": None, "take_profit": None, "stop_loss": None}
            position_qty = sum(pos['current_qty'] for pos in profile['positions'] if pos['symbol'] == self.symbol)
            if abs(position_qty) < 20:
                orders["entry"] = self.client.Order.Order_new(**order_params).result()[0]
                self.logger.info(f"Entry order placed: {orders['entry']['ordStatus']} | OrderID: {orders['entry']['orderID']}")
            else:
                self.logger.warning(f"Total position quantity {position_qty} >= 20. Skipping order")
                return None

            time.sleep(1)
            self.get_profile_info()
            return orders
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, side="Sell", quantity=100, order_type="Market", price=None, 
                      exec_inst="Close", take_profit_price=None, stop_loss_price=None, 
                      clOrdID=None, text=None):
        """
        Close a position.

        Args:
            side (str): 'Buy' or 'Sell'
            quantity (int): Number of contracts
            order_type (str): 'Market' or 'Limit'
            price (float): Limit price
            exec_inst (str): Execution instructions
            take_profit_price (float): Take Profit price
            stop_loss_price (float): Stop Loss price
            cl_ord_id (str): Client Order ID
            text (str): Order text/note

        Returns:
            dict: Order result or None if error
        """
        try:
            order = self.client.Order.Order_new(
                symbol=self.symbol,
                side=side,
                orderQty=quantity,
                ordType=order_type,
                execInst=exec_inst,
                clOrdID=clOrdID,
                text=text
            ).result()[0]
            
            self.logger.info(f"Closed position: {order['ordStatus']} | OrderID: {order['orderID']} | "
                           f"clOrdID: {order.get('clOrdID')} | text: {order.get('text')}")
            return order
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None

    def close_all_positions(self, clOrdID=None, text=None):
        """
        Close all open positions for the current symbol.

        Args:
            cl_ord_id (str): Client Order ID
            text (str): Order text/note

        Returns:
            bool: True if successful, None if error or no positions
        """
        try:
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]
            
            if not positions:
                self.logger.info("No positions to close")
                return None

            for pos in positions:
                if pos['current_qty'] != 0:
                    side = "Sell" if pos['current_qty'] > 0 else "Buy"
                    qty = abs(pos['current_qty'])
                    order = self.client.Order.Order_new(
                        symbol=self.symbol,
                        side=side,
                        orderQty=qty,
                        ordType="Market",
                        execInst='Close',
                        clOrdID=clOrdID,
                        text=text
                    ).result()[0]
                    self.logger.info(f"Closed position: {order['ordStatus']} | OrderID: {order['orderID']} | "
                                   f"clOrdID: {order.get('clOrdID')} | text: {order.get('text')}")

            time.sleep(2)
            return True
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return None

    def set_leverage(self, leverage):
        """
        Set leverage for the specified symbol.

        Args:
            leverage (float): Leverage value (0.01 to 100 for isolated, 0 for cross)

        Returns:
            dict: API response or None if error
        """
        try:
            if not 0 <= leverage <= 100:
                raise ValueError("Leverage must be between 0 and 100")

            response = self.client.Position.Position_updateLeverage(
                symbol=self.symbol,
                leverage=leverage
            ).result()[0]

            margin_type = "isolated" if leverage > 0 else "cross"
            self.logger.info(f"Leverage set to {leverage}x ({margin_type} margin) for {self.symbol}")
            return response
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            return None

    def set_cross_leverage(self, leverage):
        """
        Set cross leverage for the specified symbol.

        Args:
            leverage (float): Leverage value (0.01 to 100)

        Returns:
            dict: API response or None if error
        """
        try:
            if not 0 <= leverage <= 100:
                self.logger.warning("Leverage must be between 0 and 100")

            response = self.client.Position.Position_updateCrossLeverage(
                symbol=self.symbol,
                leverage=leverage
            ).result()[0]

            margin_type = "isolated" if leverage > 0 else "cross"
            self.logger.info(f"Cross leverage set to {leverage}x ({margin_type} margin) for {self.symbol}")
            return response
        except Exception as e:
            self.logger.error(f"Error setting cross leverage: {str(e)}")
            return None

    def run_test_sequence(self):
        """
        Run a test sequence of trading operations.

        Returns:
            dict: Final profile info or None if error
        """
        try:
            self.logger.info("Starting test sequence")
            print("Starting test sequence")

            self.logger.info("=== INITIAL PROFILE ===")
            self.get_profile_info()

            self.logger.info("=== SETTING LEVERAGE ===")
            self.set_leverage(100)

            self.logger.info("=== OPENING LONG POSITION (BUY) ===")
            self.open_position(side="Buy", quantity=1)

            time.sleep(1)
            self.get_profile_info()

            self.logger.info("=== OPENING SHORT POSITION (SELL) ===")
            self.open_position(side="Sell", quantity=1)

            time.sleep(1)
            self.get_profile_info()

            self.logger.info("=== CLOSING ALL POSITIONS ===")
            self.close_all_positions()

            self.logger.info("=== FINAL PROFILE ===")
            final_profile = self.get_profile_info()

            self.logger.info("Test sequence completed successfully")
            return final_profile
        except Exception as e:
            self.logger.error(f"Error in test sequence: {str(e)}")
            return None

if __name__ == "__main__":
    from TeleLogBot import configure_logging  # Assuming this exists
    logger, telegram_bot = configure_logging(os.getenv("TOKEN"), os.getenv("CHAT_ID"))
    api = BitMEXTestAPI(
        api_key=os.getenv("BITMEX_API_KEY"),
        api_secret=os.getenv("BITMEX_API_SECRET"),
        test=True,
        symbol="SOL-USD",
        logger=logger
    )
    api.run_test_sequence()
