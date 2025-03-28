import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yfinance as yf

from TeleLogBot import configure_logging
from collections import deque
import matplotlib.pyplot as plt
import pytz
from datetime import datetime, timedelta

load_dotenv()
#TOKEN = os.getenv("TOKEN")
#CHAT_ID = os.getenv("CHAT_ID")
##self.logger , telegram_bot = configure_logging(TOKEN, CHAT_ID)

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class MateGreen:
    def __init__(self, api_key, api_secret, test=True, symbol="BTC-USD", timeframe="5m",
                 initial_capital=10000, risk_per_trade=0.02, rr_ratio=2, lookback_period=20,
                 fvg_threshold=0.003, telegram_bot=None,api=None,log=None):
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
        self.telegram_bot = telegram_bot

        # BitMEX API for trade execution
        self.api = api
        
        # Trading state
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.current_trades = []  # Tracks active trades (index, entry_price, direction, stop_loss, take_profit, size)
        self.trades = []  # Historical trade records
        self.equity_curve = [initial_capital]
        self.market_bias = 'neutral'
        #self.logger  = log
        self.logger .info(f"MateGreen initialized for {symbol} on {timeframe} timeframe")

    def get_market_data(self):
        """
        Fetch market data from BitMEX API or fallback to yfinance.
        """
        try:
            #self.logger .info(f"Fetching {self.symbol} market data from BitMEX")
            data = self.api.get_candle(symbol=self.symbol, timeframe=self.timeframe)
            df = pd.DataFrame(data)
            #self.logger .info(f"Retrieved {len(df)} candles from BitMEX")
            self.df = df
            self.df.columns = [col.lower() for col in self.df.columns]
        except Exception as e:
            #self.logger .warning(f"Failed to get data from BitMEX API: {str(e)}. Falling back to yfinance.")
            crypto_ticker = self.symbol if self.symbol.endswith('USD') else f"{self.symbol}-USD"
            sast_now = get_sast_time()
            end_date = sast_now
            start_date = end_date - timedelta(days=2)
            try:
                data = yf.download(
                    crypto_ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=self.timeframe
                )
                #self.logger .info(f"Retrieved {len(data)} candles from yfinance")
                
                # Ensure the DataFrame is not empty and has required columns
                if data.empty:
                    #self.logger .error("Retrieved empty DataFrame from yfinance")
                    return False
                
                # Standardize column names
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0].lower() if col[1] else col[0].lower() for col in data.columns]
                else:
                    data.columns = [col.lower() for col in data.columns]
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in data.columns for col in required_columns):
                    #self.logger .error(f"Missing required columns. Available: {data.columns}")
                    return False
                
                # Add additional columns for market structure analysis
                data['higher_high'] = False
                data['lower_low'] = False
                data['bos_up'] = False
                data['bos_down'] = False
                data['choch_up'] = False
                data['choch_down'] = False
                data['bullish_fvg'] = False
                data['bearish_fvg'] = False
                
                self.df = data
                return True
            except Exception as e:
                #self.logger .error(f"yfinance fallback failed: {str(e)}")
                return False


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
                if (self.market_bias == 'bullish' or self.market_bias == 'neutral') and \
                   recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bearish'))
                    self.market_bias = 'bearish'
                elif (self.market_bias == 'bearish' or self.market_bias == 'neutral') and \
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
        potential_entries = []
        current_idx = len(self.df) - 1
        current_price = self.df['close'].iloc[current_idx]

        # Check existing trades for exit signals
        for trade in list(self.current_trades):
            idx, entry_price, direction, stop_loss, take_profit, size = trade
            if (direction == 'long' and self.df['low'].iloc[current_idx] <= stop_loss) or \
               (direction == 'short' and self.df['high'].iloc[current_idx] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': stop_loss, 'direction': direction, 'pl': pl, 'result': 'loss'})
                signals.append({
                    'action': 'exit',
                    'price': stop_loss,
                    'reason': 'stoploss',
                    'direction': direction,
                    'entry_idx': idx
                })
                self.current_trades.remove(trade)
                self.logger.info(f"Exit signal: {direction} trade stopped out at {stop_loss}")
            elif (direction == 'long' and self.df['high'].iloc[current_idx] >= take_profit) or \
                 (direction == 'short' and self.df['low'].iloc[current_idx] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': take_profit, 'direction': direction, 'pl': pl, 'result': 'win'})
                signals.append({
                    'action': 'exit',
                    'price': take_profit,
                    'reason': 'takeprofit',
                    'direction': direction,
                    'entry_idx': idx
                })
                self.current_trades.remove(trade)
                self.logger.info(f"Exit signal: {direction} trade took profit at {take_profit}")

        # Check for new entry signals
        for idx, price, choch_type in self.choch_points:
            if idx == current_idx:
                potential_entries.append((idx, price, 'long' if choch_type == 'bullish' else 'short', 'CHoCH'))
        for idx, price, bos_type in self.bos_points:
            if idx == current_idx:
                potential_entries.append((idx, price, 'long' if bos_type == 'bullish' else 'short', 'BOS'))

        if len(self.current_trades) < 3:
            for entry in list(potential_entries):
                entry_idx, entry_price, direction, entry_type = entry
                if current_idx - entry_idx <= 3:
                    fvg_confirmed = any(fvg_type == ('bullish' if direction == 'long' else 'bearish') and
                                        current_idx - fvg_end <= 10 for _, fvg_end, _, _, fvg_type in self.fvg_areas)
                    if fvg_confirmed or entry_type == 'BOS':
                        # Use lookback_period for stop distance, matching SMCBacktester
                        lookback_start = max(0, current_idx - self.lookback_period)
                        stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                                    max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                        stop_loss = entry_price - stop_dist * 1.1 if direction == 'long' else entry_price + stop_dist * 1.1
                        take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                        size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                        signals.append({
                            'action': 'entry',
                            'side': direction,
                            'price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'position_size': int(size),
                            'entry_idx': entry_idx
                        })
                        self.current_trades.append((current_idx, entry_price, direction, stop_loss, take_profit, size))
                        potential_entries.remove(entry)
                        self.logger.info(f"Entry signal: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")

        self.equity_curve.append(self.current_balance)
        return signals

    def execute_signal(self, signal):
        """Execute a trading signal (entry or exit)."""
        if signal is None:
            #self.logger .info("No trading signal detected")
            return
        if signal['action'] == 'entry':
            self.execute_entry(signal)
        elif signal['action'] == 'exit':
            self.execute_exit(signal)

    def execute_entry(self, signal):
        """Open a position using the BitMEX API, supporting multiple trades."""
        side = signal['side']
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal['position_size']
        entry_idx = signal['entry_idx']
        sast_now = get_sast_time()
        try:
            order_side = "Buy" if signal['side'] == "long" else "Sell"
            order = self.api.open_test_position(side=side.capitalize(), quantity=signal['position_size'] if int(signal['position_size'])>1 else int(signal['position_size'])+1*2)
            #order_result = self.api.open_test_position(side=order_side, quantity=position_size)
            trade = {
                'entry_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                'entry_price': price,
                'entry_index': entry_idx,
                'type': side,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'order_id': order['orderID'],
                'risk_amount': self.risk_per_trade * self.current_balance
            }
            self.trades.append(trade)
            self.logger.info(f"Opened {side} position at {price}, SL: {stop_loss}, TP: {take_profit}, Size: {position_size}")
        except Exception as e:
            self.logger.error(f"Error opening {side} position: {e}")

    def execute_exit(self, signal):
        """Close a specific position using the BitMEX API, supporting multiple trades."""
        reason = signal['reason']
        price = signal['price']
        direction = signal['direction']
        entry_idx = signal['entry_idx']
        sast_now = get_sast_time()
        try:
            # Find the trade to close (matching entry_idx and direction)
            for trade in list(self.current_trades):
                idx, entry_price, trade_direction, stop_loss, take_profit, size = trade
                if idx == entry_idx and trade_direction == direction:
                    order_id = next(t['order_id'] for t in self.trades if t['entry_index'] == entry_idx and t['type'] == direction)
                    self.api.close_position(order_id)
                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    for t in self.trades:
                        if t['entry_index'] == entry_idx and t['type'] == direction and 'exit_price' not in t:
                            t.update({
                                'exit_time': sast_now.strftime('%Y-%m-%d %H:%M:%S'),
                                'exit_price': price,
                                'exit_index': len(self.df) - 1,
                                'exit_reason': reason,
                                'pnl': pl
                            })
                            break
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.current_trades.remove(trade)
                    self.logger.info(f"Closed {direction} position at {price} due to {reason}, PnL: {pl}")
                    break
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def visualize_results(self, start_idx=0, end_idx=None):
        """Visualize results for Telegram notifications."""
        if end_idx is None:
            end_idx = len(self.df)
        fig, ax = plt.subplots(figsize=(15, 8))
        subset = self.df.iloc[start_idx:end_idx]
        ax.plot(subset.index, subset['close'], label='Close Price', color='black', linewidth=1)
        for trade in self.trades:
            if start_idx <= trade['entry_index'] < end_idx:
                color = 'green' if trade['type'] == 'long' else 'red'
                marker = '^' if trade['type'] == 'long' else 'v'
                ax.scatter(trade['entry_index'], trade['entry_price'], color=color, marker=marker, s=120, zorder=5)
                if 'exit_index' in trade and trade['exit_index'] < end_idx:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax.scatter(trade['exit_index'], trade['exit_price'], color=color, marker='o', s=120, zorder=5)
                    ax.plot([trade['entry_index'], trade['exit_index']],
                            [trade['entry_price'], trade['exit_price']],
                            color=color, linewidth=1, linestyle='--')
        ax.set_title(f"{self.symbol} - SMC Analysis")
        ax.legend(['Close Price'], loc='best')
        ax.grid(True, alpha=0.3)
        return fig

    def calculate_performance(self):
        """Calculate performance metrics."""
        if not self.trades or sum(t.get('pnl', 0) for t in self.trades) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0}
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(self.trades)
        gross_profit = sum([t['pnl'] for t in self.trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t['pnl'] for t in self.trades if t.get('pnl', 0) <= 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance if self.initial_balance else 0
        peak = self.initial_balance
        max_drawdown = 0
        for balance in self.equity_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        return {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return_pct': total_return * 100,
            'max_drawdown_pct': max_drawdown * 100
        }

    def run(self, scan_interval=120):
        """Live trading loop with 2 scans, matching SMC.run()."""
        sast_now = get_sast_time()
        self.logger.info(f"Starting MateGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting MateGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            profile = self.api.get_profile_info()
            self.initial_balance = float(profile['balance']['usd'])
            self.current_balance = self.initial_balance
            self.equity_curve = [self.initial_balance]
            #self.logger .info(f"Initial balance set to {self.initial_balance:.2f}")
        except Exception as e:
            self.logger .error(f"Failed to initialize balance: {e}")
            return False, pd.DataFrame()

        signal_found = False
        for iteration in range(2):
            sast_now = get_sast_time()
            #self.logger .info(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

            if not self.get_market_data() or len(self.df) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(self.df)} candles retrieved")
                if iteration < 1:
                    time.sleep(scan_interval)
                continue

            self.identify_swing_points()
            self.detect_market_structure()
            signals = self.execute_trades()

            for signal in signals:
                self.execute_signal(signal)
                if signal['action'] == 'entry':
                    signal_found = True
                    self.logger.info("Trading signal detected and trade executed🐻❎🐃")

            try:
                profile = self.api.get_profile_info()
                api_balance = float(profile['balance']['usd'])
                if abs(api_balance - self.current_balance) > 0.01:
                    #self.logger .info(f"Balance updated from API: {self.current_balance:.2f} -> {api_balance:.2f}")
                    self.current_balance = api_balance
                    self.equity_curve.append(self.current_balance)
            except Exception as e:
                self.logger.warning(f"Failed to sync balance with API: {e}")

            performance = self.calculate_performance()
            #self.logger .info(f"Performance snapshot: {performance}")

            if not signal_found and not self.df.empty and self.telegram_bot:
                try:
                    lookback_candles = 16 if self.timeframe == "15m" else 48 if self.timeframe == "5m" else 4
                    fig = self.visualize_results(start_idx=max(0, len(self.df) - lookback_candles))
                    caption = f"📸No signals found - Scan {iteration+1} at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}"
                    self.telegram_bot.send_photo(fig=fig, caption=caption)
                    #self.logger.info(f"📸Sent no-signal analysis plot for scan {iteration+1}")
                    plt.close(fig)
                except Exception as e:
                    self.logger.error(f"Failed to generate or send no-signal analysis: {e}")

            if iteration < 1:
                self.logger .info(f"Waiting {scan_interval} seconds for next scan...")
                print(f"Waiting {scan_interval} seconds for next scan...")
                time.sleep(scan_interval)

        #self.logger .info("Completed 2 scans, stopping MateGreen")
        if self.current_trades:  # Check if any trades are open
            try:
                self.api.close_all_positions()
                self.logger.info("All open positions closed")
            except Exception as e:
                self.logger.error(f"Failed to close positions on exit: {e}")

        final_performance = self.calculate_performance()
        self.logger.info(f"Final performance metrics: {final_performance}")

        return signal_found, self.df




    

        
if __name__ == "__main__":
    trader = MateGreen(
        api_key=os.getenv("BITMEX_API_KEY"),
        api_secret=os.getenv("BITMEX_API_SECRET"),
        test=True,
        symbol="BTC-USD",
        timeframe="5m",
        telegram_bot=telegram_bot
    )
    signal_found, price_data = trader.run()
    print(f"Signal found: {signal_found}, Data length: {len(price_data)}")
