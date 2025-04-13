# MatteGreen.py v1
import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from collections import deque
import matplotlib.pyplot as plt
import mplfinance as mpf
import pytz
from datetime import datetime, timedelta
from BitMEXApi import BitMEXTestAPI
from TeleLogBot import configure_logging, TelegramBot
from PerfCalc import get_trading_performance_summary
import logging
import uuid
import yfinance as yf 

load_dotenv()

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

def clOrderID_string(clord_id, text=None):
    try:
        if clord_id is None:
            raise ValueError("clOrdID cannot be None")
        if not isinstance(clord_id, str):
            raise ValueError(f"clOrdID must be a string, got {type(clord_id)}")
        parts = clord_id.split(';')
        if len(parts) < 3:
            raise IndexError(f"clOrdID has fewer than 3 parts: {clord_id}")
        symbol = parts[0].strip('(').strip(')')
        date = parts[1].strip('(').strip(')')
        uid = parts[2].strip('(').strip(')')
        
        if text is not None:
            if not isinstance(text, str):
                raise ValueError(f"text must be a string or None, got {type(text)}")
            text_parts = text.split(';')
            if len(text_parts) < 4:
                raise IndexError(f"text has fewer than 4 parts: {text}")
            status = text_parts[0].strip('(').strip(')')
            action = text_parts[1].strip("'").strip('(').strip(')').replace("'", '').strip()
            entry_data = text_parts[2].strip('(').strip(')').split(', ')
            if len(entry_data) < 4:
                raise IndexError(f"entry_data has fewer than 4 values: {text_parts[2]}")
            entry_price = entry_data[0]
            position_size = entry_data[1]
            direction = entry_data[2]
            current_idx = entry_data[3]
            tp_sl_data = text_parts[3].strip('(').strip(')').split(', ')
            if len(tp_sl_data) < 2:
                raise IndexError(f"tp_sl_data has fewer than 2 values: {text_parts[3]}")
            take_profit = tp_sl_data[0]
            stop_loss = tp_sl_data[1]
        else:
            status = 'open'
            action = 'entry'
            entry_price = position_size = direction = current_idx = take_profit = stop_loss = '0'

        return {
            'action': action,
            'symbol': symbol,
            'date': date,
            'side': direction,
            'price': round(float(entry_price), 2),
            'stop_loss': round(float(stop_loss), 4),
            'take_profit': round(float(take_profit), 4),
            'position_size': int(position_size),
            'entry_idx': current_idx,
            'status': status,
            'uuid': uid
        }
    except Exception as e:
        raise ValueError(f"Failed to parse clOrdID '{clord_id}' or text '{text}': {str(e)}")

def update_clOrderID_string(clord_id, text=None, **updates):
    clOrderID_dict = clOrderID_string(clord_id, text)
    temp = {
        'action': clOrderID_dict['action'],
        'symbol': clOrderID_dict['symbol'],
        'side': clOrderID_dict['side'],
        'price': clOrderID_dict['price'],
        'stop_loss': clOrderID_dict['stop_loss'],
        'take_profit': clOrderID_dict['take_profit'],
        'position_size': clOrderID_dict['position_size'],
        'entry_idx': clOrderID_dict['entry_idx'],
        'status': clOrderID_dict['status'],
        'date': clOrderID_dict['date'],
        'uuid': clOrderID_dict['uuid']
    }
    
    if updates:
        for key, value in updates.items():
            temp[key] = value
    
    new_clord_id = f"({temp['symbol']});({temp['date']});({temp['uuid'][:6]})"
    if len(new_clord_id) > 36:
        excess = len(new_clord_id) - 36
        temp['uuid'] = temp['uuid'][:6 - excess]
        new_clord_id = f"({temp['symbol']});({temp['date']});({temp['uuid']})"
        if len(new_clord_id) > 36:
            raise ValueError(f"clOrdID still exceeds 36 characters after truncation: {new_clord_id}")

    new_text = f"({temp['status']});({temp['action']});({temp['price']}, {temp['position_size']}, {temp['side']}, {temp['entry_idx']});({temp['take_profit']}, {temp['stop_loss']})"
    
    return new_clord_id, new_text

class MatteGreen:
    def __init__(self, api_key, api_secret, test=True, symbol="SOL-USD", timeframe="5m",
                 initial_capital=10000, risk_per_trade=0.02, rr_ratio=1.25, lookback_period=20,
                 fvg_threshold=0.003, telegram_token=None, telegram_chat_id=None, log=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.test = test
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.risk_per_trade = risk_per_trade
        self.rr_ratio = rr_ratio
        self.lookback_period = lookback_period
        self.fvg_threshold = fvg_threshold
        self.logger, self.bot = configure_logging(os.getenv("TOKEN"), os.getenv("CHAT_ID")) 
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        valid_timeframes = ['1m', '5m', '1h', '1d'] 
        self.api = BitMEXTestAPI(api_key, api_secret, test=test, symbol=symbol, timeframe=timeframe if timeframe in valid_timeframes else "5m", Log=self.logger)
        self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id) if telegram_token and telegram_chat_id else None
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.current_trades = []  # (trade_id, entry_idx, entry_price, direction, stop_loss, take_profit, size, clord_id, text, sl_orderID, tp_orderID)
        self.trades = []
        self.equity_curve = [initial_capital]
        self.market_bias = 'neutral'
        self.logger.info(f"MatteGreen initialized for {symbol} on {timeframe}")

    def get_market_data(self):
        try:
            # Fetch 5m candle data from BitMEX instead of Yahoo Finance
            data = self.api.get_candle(timeframe=self.timeframe, count=self.lookback_period + 20)  # Extra candles for safety
            #data = yf.download(self.symbol, period='2d', interval=self.timeframe, auto_adjust=True) 
            #data.columns = [x[0].lower() for x in data.columns]
            if data is None or data.empty:
                self.logger.error("No data from BitMEX API")
                return False
            # Rename columns to match previous structure
            data = data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
            data.index = data['timestamp']  # Set timestamp as index for consistency
            data['higher_high'] = False
            data['lower_low'] = False
            data['bos_up'] = False
            data['bos_down'] = False
            data['choch_up'] = False
            data['choch_down'] = False
            data['bullish_fvg'] = False
            data['bearish_fvg'] = False
            self.df = data
            self.logger.info(f"Fetched {len(self.df)} {self.timeframe} candles from BitMEX for {self.symbol}")
            return True
        except Exception as e:
            self.logger.error(f"Market data fetch failed: {str(e)}")
            return False

    def sync_open_orders(self):
        try:
            open_orders = self.api.get_open_orders() or []
            positions = self.api.get_positions() or []
            self.logger.info(f"🔁Syncing: Found {len(open_orders)} open orders and {len(positions)} positions.")
            
            position_map = {pos['symbol']: pos for pos in positions if pos.get('currentQty', 0) != 0}
            position = position_map.get(self.symbol.replace('-', ''), {})
            has_position = position.get('currentQty', 0) != 0
            position_direction = 'long' if position.get('currentQty', 0) > 0 else 'short' if position.get('currentQty', 0) < 0 else None
            
            exchange_trades = {}
            
            for order in open_orders:
                clord_id = order.get('clOrdID')
                if not clord_id or clord_id in [None, 'No strings attached']:
                    self.logger.debug(f"Skipping order with no clOrdID: {order}")
                    continue
                text = order.get('text', '')
                self.logger.debug(f"Processing order: clOrdID='{clord_id}', text='{text}'")
                try:
                    clOrderID = clOrderID_string(clord_id, text)
                    if clOrderID['symbol'].replace('-', '') != self.symbol.replace('-', ''):
                        self.logger.debug(f"Skipping order for different symbol: {clOrderID['symbol']} vs {self.symbol}")
                        continue
                    if has_position and clOrderID['side'] != position_direction:
                        pass
                    exchange_trades[clord_id] = (
                        order.get('orderID', clord_id),
                        int(clOrderID['entry_idx']),
                        float(clOrderID['price']),
                        clOrderID['side'],
                        float(clOrderID['stop_loss']),
                        float(clOrderID['take_profit']),
                        int(clOrderID['position_size']),
                        clord_id,
                        text
                    )
                except ValueError as e:
                    self.logger.warning(f"Invalid clOrdID format or data: {clord_id} - {str(e)}")
                    continue
            
            local_clord_ids = {trade[7] for trade in self.current_trades if trade[7]}
            exchange_clord_ids = set(exchange_trades.keys())

            for trade in list(self.current_trades):
                clord_id = trade[7]
                sl_orderID = trade[9]
                tp_orderID = trade[10]
                if clord_id and clord_id not in exchange_clord_ids and not has_position:
                    trade_id, entry_idx, entry_price, direction, stop_loss, take_profit, size, _, text = trade[:9]
                    sl_executions = self.api.get_executions(orderID=sl_orderID) if sl_orderID else []
                    tp_executions = self.api.get_executions(orderID=tp_orderID) if tp_orderID else []
                    
                    if sl_executions:
                        exit_price = sum(exec_['price'] * exec_['execQty'] for exec_ in sl_executions) / sum(exec_['execQty'] for exec_ in sl_executions)
                        reason = 'stoploss'
                    elif tp_executions:
                        exit_price = sum(exec_['price'] * exec_['execQty'] for exec_ in tp_executions) / sum(exec_['execQty'] for exec_ in tp_executions)
                        reason = 'takeprofit'
                    else:
                        exit_price = self.df['close'].iloc[-1]  # Fallback if no executions found
                        reason = 'unknown'
                    
                    pl = (exit_price - entry_price) * size if direction == 'long' else (entry_price - exit_price) * size
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': exit_price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': clord_id, 'reason': reason
                    })
                    self.current_balance += pl
                    self.current_trades.remove(trade)
                    self.equity_curve.append(self.current_balance)
                    self.logger.info(f"Closed trade {clord_id} at {exit_price} due to {reason}")
                elif clord_id in exchange_trades and has_position and trade[3] != position_direction:
                    self.logger.warning(f"Correcting direction mismatch for clOrdID {clord_id}: local={trade[3]}, exchange={position_direction}")
                    trade_id, entry_idx, entry_price, _, stop_loss, take_profit, size, clord_id, text = trade
                    self.current_trades.remove(trade)
                    self.current_trades.append((trade_id, entry_idx, entry_price, position_direction, stop_loss, take_profit, size, clord_id, text))

            for clord_id, trade_data in exchange_trades.items():
                if clord_id not in local_clord_ids:
                    self.current_trades.append(trade_data)
            
            self.logger.debug(f"Current trades after sync: {self.current_trades}")
        except Exception as e:
            self.logger.error(f"Failed to sync open orders: {str(e)}")
        self.logger.info('🔄Done Sync....... ')

    def identify_swing_points(self):
        window = min(self.lookback_period // 2, 3)
        self.swing_highs = np.zeros(len(self.df))
        self.swing_lows = np.zeros(len(self.df))
        for i in range(window, len(self.df) - window):
            if all(self.df['high'].iloc[i] >= self.df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['high'].iloc[i] >= self.df['high'].iloc[i+j] for j in range(1, window+1)):
                self.swing_highs[i] = 1
            if all(self.df['low'].iloc[i] <= self.df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['low'].iloc[i] <= self.df['low'].iloc[i+j] for j in range(1, window+1)):
                self.swing_lows[i] = 1

    def detect_market_structure(self):
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        recent_highs = deque(maxlen=self.lookback_period)
        recent_lows = deque(maxlen=self.lookback_period)

        for i in range(self.lookback_period, len(self.df)):
            if self.swing_highs[i]:
                recent_highs.append((i, self.df['high'].iloc[i]))
            if self.swing_lows[i]:
                recent_lows.append((i, self.df['low'].iloc[i]))

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (self.market_bias in ['bullish', 'neutral']) and \
                   recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bearish'))
                    self.market_bias = 'bearish'
                elif (self.market_bias in ['bearish', 'neutral']) and \
                     recent_lows[-1][1] > recent_lows[-2][1] and recent_highs[-1][1] > recent_highs[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bullish'))
                    self.market_bias = 'bullish'

            if self.market_bias == 'bearish' and recent_highs and self.df['high'].iloc[i] > recent_highs[-1][1]:
                self.bos_points.append((i, self.df['high'].iloc[i], 'bullish'))
            elif self.market_bias == 'bullish' and recent_lows and self.df['low'].iloc[i] < recent_lows[-1][1]:
                self.bos_points.append((i, self.df['low'].iloc[i], 'bearish'))

            if i > 1:
                if (self.df['low'].iloc[i] - self.df['high'].iloc[i-2]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i-2], self.df['low'].iloc[i], 'bullish'))
                if (self.df['low'].iloc[i-2] - self.df['high'].iloc[i]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i], self.df['low'].iloc[i-2], 'bearish'))

    def execute_trades(self):
        signals = []
        current_idx = len(self.df) - 1
        current_price = self.df['close'].iloc[current_idx]  # 5m closing price
    
        total_risk_amount = sum(abs(entry_price - stop_loss) * size for _, _, entry_price, _, stop_loss, _, size, _, _ in self.current_trades)
        max_total_risk = self.current_balance * 0.20
    
        for trade in list(self.current_trades):
            trade_id, idx, entry_price, direction, stop_loss, take_profit, size, clord_id, text = trade
            if (direction == 'long' and self.df['low'].iloc[current_idx] <= stop_loss) or \
               (direction == 'short' and self.df['high'].iloc[current_idx] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': round(stop_loss, 4), 'direction': direction, 'pl': pl, 'result': 'loss', 'trade_id': trade_id})
                signals.append({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                self.execute_exit({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                #self.current_trades.remove(trade)
            elif (direction == 'long' and self.df['high'].iloc[current_idx] >= take_profit) or \
                 (direction == 'short' and self.df['low'].iloc[current_idx] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': round(entry_price, 2),
                                    'exit_price': round(take_profit, 4), 'direction': direction, 'pl': pl, 'result': 'win', 'trade_id': trade_id})
                signals.append({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                self.execute_exit({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                #self.current_trades.remove(trade)
    
        if len(self.current_trades) < 3 and current_idx >= self.lookback_period:
            direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
            if direction:
                entry_price = current_price  # Intended 5m entry price
                lookback_start = max(0, current_idx - self.lookback_period)
                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                            max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                take_profit = entry_price + stop_dist * 0.625 if direction == 'long' else entry_price - stop_dist * 0.625
                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                risk_of_new_trade = abs(entry_price - stop_loss) * size
    
                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                    signals.append({'action': 'entry', 'side': direction, 'price': round(entry_price, 2), 'stop_loss': round(stop_loss, 4),
                                    'take_profit': round(take_profit, 4), 'position_size': int(size) if size < 1 else 1, 'entry_idx': current_idx})
                    self.current_trades.append((None, current_idx, entry_price, direction, stop_loss, take_profit, size, None, None))
                    self.logger.info(f"Entry signal: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
    
        self.equity_curve.append(self.current_balance)
        return signals

    def execute_entry(self, signal):
        side = signal['side']
        desired_price = signal['price']  # 5m closing price
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal['position_size']
        entry_idx = signal['entry_idx']
        sast_now = get_sast_time()

        # Fetch the latest 1m candle to check current price
        current_1m_data = self.api.get_candle(timeframe='1m', count=1)
        if current_1m_data is None or current_1m_data.empty:
            self.logger.error("Failed to fetch 1m candle for price validation")
            return
        current_1m_price = current_1m_data['close'].iloc[-1]
        price_threshold = desired_price * 0.005  # 0.5% tolerance

        self.logger.info(f"Desired 5m entry price: {desired_price}, Current 1m price: {current_1m_price}, Threshold: ±{price_threshold}")

        # Check if 1m price is within acceptable range of 5m price
        if abs(current_1m_price - desired_price) > price_threshold:
            self.logger.warning(f"1m price ({current_1m_price}) deviates from 5m desired price ({desired_price}) beyond threshold. Adjusting to limit order.")
            order_type = "Limit"
            entry_price = desired_price  # Use the 5m price as a limit order
        else:
            self.logger.info("1m price within threshold of 5m price. Proceeding with market order.")
            order_type = "Market"
            entry_price = current_1m_price  # Use current 1m price for market order
            stop_loss = stop_loss + price_threshold
            take_profit = take_profit + price_threshold

        date_str = sast_now.strftime("%Y%m%d%H%M")
        uid = str(uuid.uuid4())[:6]
        clord_id = f"({self.symbol});({date_str});({uid})"
        text = f"('open');('entry');({entry_price}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"

        self.logger.info(f"Opening position with clOrdID: '{clord_id}' (length: {len(clord_id)}), text: '{text}', order_type: {order_type}")

        if len(clord_id) > 36:
            self.logger.error(f"clOrdID exceeds 36 characters: {clord_id}")
            raise ValueError(f"clOrdID exceeds 36 characters: {clord_id}")

        pos_side = "Sell" if side.lower() in ['short', 'sell'] else "Buy"
        pos_quantity = max(1, int(position_size))
        profile = self.api.get_profile_info()
        if not profile or 'balance' not in profile:
            self.logger.error("Failed to fetch profile info for margin check")
            return
    
        available_margin = profile['balance']['bitmex_usd']
        required_margin = (position_size * entry_price) / 5  # Assuming 5x leverage

        try:
            orders = self.api.open_position(
                side=pos_side,
                quantity=pos_quantity,
                order_type=order_type,
                price=entry_price if order_type == "Limit" else None,  # Pass price only for limit orders
                take_profit_price=take_profit,
                stop_loss_price=stop_loss,
                clOrdID=clord_id,
                text=text
            )
            if orders and orders.get('entry'):
                trade_id = orders['entry']['orderID']
                sl_orderID = orders['stop_loss']['orderID'] if orders.get('stop_loss') else None
                tp_orderID = orders['take_profit']['orderID'] if orders.get('take_profit') else None
                # Replace the last appended trade with full details
                self.current_trades[-1] = (trade_id, entry_idx, entry_price, side, stop_loss, take_profit, position_size, clord_id, text, sl_orderID, tp_orderID)
                self.logger.info(f"📈🎉Opened {pos_side} at {entry_price}, SL: {stop_loss}, TP: {take_profit}, ID: {trade_id}, clOrdID: {clord_id}")
            else:
                self.logger.warning(f"Order failed, tracking locally with clOrdID: {clord_id}")
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            raise

    def execute_exit(self, signal):
        reason = signal['reason']
        price = signal['price']
        direction = signal['direction']
        entry_idx = signal['entry_idx']
        trade_id = signal.get('trade_id')
        sast_now = get_sast_time()

        for trade in list(self.current_trades):
            stored_trade_id, idx, entry_price, trade_direction, stop_loss, take_profit, size, clord_id, text = trade
            if idx == entry_idx and trade_direction == direction:
                try:
                    positions = self.api.get_positions() or []
                    position_qty = next((pos['currentQty'] for pos in positions if pos['symbol'] == self.symbol.replace('-', '')), 0)
                    if position_qty == 0:
                        pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                        self.current_balance += pl
                        self.equity_curve.append(self.current_balance)
                        self.trades.append({
                            'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                            'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                            'trade_id': trade_id, 'clord_id': clord_id
                        })
                        self.current_trades.remove(trade)
                        continue

                    if trade_id and trade_id == stored_trade_id:
                        if clord_id is None or text is None:
                            self.logger.warning(f"Invalid clOrdID ({clord_id}) or text ({text}) for trade {trade_id}, skipping API close.")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        side = 'Sell' if position_qty > 0 else 'Buy' if position_qty < 0 else None
                        if side is None:
                            self.logger.warning(f"Position quantity is 0 but trade exists locally, skipping API close for {clord_id}")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        if len(new_clord_id) > 36:
                            self.logger.error(f"clOrdID exceeds 36 characters: {new_clord_id}")
                            raise ValueError(f"clOrdID exceeds 36 characters: {new_clord_id}")
                        self.api.close_position(side=side, quantity=size, order_type="Market", 
                                                clOrdID=new_clord_id, text=new_text)
                        self.logger.info(f"Closed position via API: {new_clord_id}")
                    else:
                        if clord_id is None:
                            self.logger.warning(f"No valid clOrdID for trade, skipping manual close.")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        self.logger.warning(f"No valid trade_id, closing manually with clOrdID: {new_clord_id}")

                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': new_clord_id
                    })
                    self.current_trades.remove(trade)
                    self.logger.info(f"Closed {direction} at {price}, Reason: {reason}, PnL: {pl}, clOrdID: {new_clord_id}")
                except Exception as e:
                    self.logger.error(f"Failed to close position {clord_id}: {str(e)}")
                    if clord_id is None:
                        self.logger.warning(f"No valid clOrdID for failed trade, removing locally.")
                    else:
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        self.api.close_all_positions(clOrdID=new_clord_id)
                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': clord_id if clord_id is None else new_clord_id
                    })
                    self.current_trades.remove(trade)
                break

    def visualize_results(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = len(self.df)
        subset = self.df.iloc[start_idx:end_idx]

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        mpf.plot(subset, type='candle', style='charles', ax=ax1, ylabel='Price', show_nontrading=False,
                 datetime_format='%H:%M')
        swing_high_idx = [i - start_idx for i, val in enumerate(self.swing_highs[start_idx:end_idx]) if val]
        swing_low_idx = [i - start_idx for i, val in enumerate(self.swing_lows[start_idx:end_idx]) if val]
      
        for start, end, high, low, fvg_type in self.fvg_areas:
            if start_idx <= end < end_idx:
                color = 'green' if fvg_type == 'bullish' else 'red'
                ax1.fill_between(range(max(0, start - start_idx), min(end - start_idx + 1, len(subset))),
                                 high, low, color=color, alpha=0.2, label=f"{fvg_type.capitalize()} FVG" if start == self.fvg_areas[0][0] else "")
       
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range(len(self.equity_curve)), self.equity_curve, label='Equity', color='blue')
        ax2.set_title("Equity Curve")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def run(self, scan_interval=5*60, max_runtime_minutes=40, sleep_interval_minutes=1, iterations_before_sleep=2):
        start_time = time.time()
        sast_now = get_sast_time()
        Banner = """

        ┏━━━✦❘༻༺❘✦━━━┓
                  Welcome to
                           💵
                 ᙏα𝜏𝜏ҽGɾҽҽɳ
 
        ┗━━━✦❘༻༺❘✦━━━┛
        
        ~'Where the green💸💸 make u happy😊' 
        """
        self.logger.info(Banner) 
        self.logger.info(f"Starting MatteGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting MatteGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

        profile = self.api.get_profile_info()
        if profile:
            self.initial_balance = self.current_balance = profile['balance']['bitmex_usd']
            self.equity_curve = [self.initial_balance]
            self.logger.info(f"Initial balance: ${self.initial_balance:.2f}")
        self.logger.info("=== SETTING LEVERAGE ===")
        self.api.set_cross_leverage(1)
        signal_found = False
        iteration = 0
        while (time.time() - start_time) < max_runtime_minutes * 60:
            sast_now = get_sast_time()
            self.logger.info(f"🤔📈🕵🏽‍♂️🔍🔎Scan {iteration + 1} started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scan {iteration + 1} started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

            if not self.get_market_data() or len(self.df) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(self.df)} candles")
                time.sleep(scan_interval)
                iteration += 1
                continue
            try:
                self.sync_open_orders()
                self.identify_swing_points()
                self.detect_market_structure()
                signals = self.execute_trades()

                for signal in signals:
                    if signal['action'] == 'exit':
                        self.execute_exit(signal)
                    elif signal['action'] == 'entry':
                        self.execute_entry(signal)
                        signal_found = True
                
                wallet_history = self.api.get_transactions() 
                positions = self.api.get_positions()
                performance = get_trading_performance_summary(wallet_history, positions) 
                self.logger.info(f"Performance:\n\n\nOverview: \n{profile['user']} \n\nProfits: \n{profile['balance']}\n\nMetadata: \n{profile['positions']}")

                if self.bot and iteration % 2 == 0:
                    fig = self.visualize_results(start_idx=max(0, len(self.df) - 48))
                    caption = (f"📸Scan {iteration+1}\nTimestamp: {sast_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                               f"Symbol: {self.symbol}\nSignal: {signal_found}\nBalance: ${self.current_balance:.2f}\nPrice @ ${self.df['close'][-1]}")
                    self.bot.send_photo(fig=fig, caption=caption)
                self.logger.info(f"({self.symbol}) Price @ ${self.df['close'][-1]} \n\n 😪😪😪 Sleeping for {scan_interval/60} minutes....") 
            except Exception as e:
                self.logger.error(f"An error occurred during trading: {str(e)}")
            time.sleep(scan_interval)
            iteration += 1

            if iteration % iterations_before_sleep == 0 and iteration > 0:
                self.logger.info(f"Pausing for {sleep_interval_minutes} minutes...")
                print(f"Pausing for {sleep_interval_minutes} minutes...")
                time.sleep(sleep_interval_minutes * 60)
                self.logger.info("🙋🏾‍♂️Resuming...")
                print("Resuming...")
       
        self.logger.info(f"MatteGreen run completed. Final balance: ${self.current_balance:.2f}")

if __name__ == "__main__":
    api_key = os.getenv('API_KEY')
    api_secret = os.getenv('API_SECRET')
    mg = MatteGreen(api_key, api_secret)
    mg.run()
