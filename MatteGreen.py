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
from TeleLogBot import configure_logging, TelegramBot  # Import refined TeleLogBot

load_dotenv()

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

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
        self.api = BitMEXTestAPI(api_key, api_secret, test=test, symbol=symbol, Log=self.logger)
        self.telegram_bot = TelegramBot(telegram_token, telegram_chat_id) if telegram_token and telegram_chat_id else None
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.current_trades = []
        self.trades = []
        self.equity_curve = [initial_capital]
        self.market_bias = 'neutral'
        self.logger.info(f"MatteGreen initialized for {symbol} on {timeframe}")

    def get_market_data(self):
        try:
            data = self.api.get_candle(timeframe=self.timeframe, count=self.lookback_period * 2)
            if data is None or data.empty:
                self.logger.error("No data from BitMEX API")
                return False
            data.index = pd.to_datetime(data['timestamp'])
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
            self.logger.error(f"Market data fetch failed: {str(e)}")
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
        current_price = self.df['close'].iloc[current_idx]

        total_risk_amount = sum(abs(entry_price - stop_loss) * size for _, entry_price, _, stop_loss, _, size in self.current_trades)
        max_total_risk = self.current_balance * 0.20

        for trade in list(self.current_trades):
            idx, entry_price, direction, stop_loss, take_profit, size = trade
            if (direction == 'long' and self.df['low'].iloc[current_idx] <= stop_loss) or \
               (direction == 'short' and self.df['high'].iloc[current_idx] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': stop_loss, 'direction': direction, 'pl': pl, 'result': 'loss'})
                signals.append({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'entry_idx': idx})
                self.current_trades.remove(trade)
                self.logger.info(f"Exit: {direction} stopped out at {stop_loss}")
            elif (direction == 'long' and self.df['high'].iloc[current_idx] >= take_profit) or \
                 (direction == 'short' and self.df['low'].iloc[current_idx] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': take_profit, 'direction': direction, 'pl': pl, 'result': 'win'})
                signals.append({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'entry_idx': idx})
                self.current_trades.remove(trade)
                self.logger.info(f"Exit: {direction} took profit at {take_profit}")

        if len(self.current_trades) < 3 and current_idx >= self.lookback_period:
            direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
            if direction:
                entry_price = current_price
                lookback_start = max(0, current_idx - self.lookback_period)
                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                            max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                stop_loss = entry_price - stop_dist * 0.5 if direction == 'long' else entry_price + stop_dist * 0.5
                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                risk_of_new_trade = abs(entry_price - stop_loss) * size

                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                    signals.append({'action': 'entry', 'side': direction, 'price': entry_price, 'stop_loss': stop_loss,
                                    'take_profit': take_profit, 'position_size': int(size), 'entry_idx': current_idx})
                    self.current_trades.append((current_idx, entry_price, direction, stop_loss, take_profit, size))
                    self.logger.info(f"Entry: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")

        self.equity_curve.append(self.current_balance)
        return signals

    def execute_entry(self, signal):
        side = signal['side']
        price = signal['price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal['position_size']
        entry_idx = signal['entry_idx']
        sast_now = get_sast_time()

        pos_side = "Sell" if side.lower() in ['short', 'sell'] else "Buy"
        pos_quantity = max(2, int(position_size))

        orders = self.api.open_test_position(side=pos_side, quantity=pos_quantity, order_type="Market",
                                             take_profit_price=take_profit, stop_loss_price=stop_loss)
        if orders and orders['entry']:
            trade_id = orders['entry']['orderID']
            self.current_trades[-1] = (trade_id, price, side, stop_loss, take_profit, position_size)
            self.logger.info(f"Opened {pos_side} at {price}, SL: {stop_loss}, TP: {take_profit}, ID: {trade_id}")
        else:
            self.logger.warning(f"Order failed, using local ID {entry_idx}")
            self.current_trades[-1] = (entry_idx, price, side, stop_loss, take_profit, position_size)

    def execute_exit(self, signal):
        reason = signal['reason']
        price = signal['price']
        direction = signal['direction']
        entry_idx = signal['entry_idx']
        sast_now = get_sast_time()

        for trade in list(self.current_trades):
            idx, entry_price, trade_direction, stop_loss, take_profit, size = trade
            if idx == entry_idx and trade_direction == direction:
                profile = self.api.get_profile_info()
                position_open = any(p['symbol'] == self.symbol and p['current_qty'] != 0 for p in profile['positions'])
                if not position_open:
                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.current_trades.remove(trade)
                    self.logger.info(f"Closed by BitMEX: {direction} at {price}, Reason: {reason}, PnL: {pl}")
                else:
                    self.api.close_all_positions()
                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.current_trades.remove(trade)
                    self.logger.info(f"Manually closed {direction} at {price}, Reason: {reason}, PnL: {pl}")
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
        ax1.plot(swing_high_idx, subset['high'].iloc[swing_high_idx], 'rv', label='Swing High')
        ax1.plot(swing_low_idx, subset['low'].iloc[swing_low_idx], 'g^', label='Swing Low')

        for idx, price, c_type in self.choch_points:
            if start_idx <= idx < end_idx:
                ax1.plot(idx - start_idx, price, 'mo', label='CHoCH' if idx == self.choch_points[0][0] else "")
        for idx, price, b_type in self.bos_points:
            if start_idx <= idx < end_idx:
                ax1.plot(idx - start_idx, price, 'co', label='BOS' if idx == self.bos_points[0][0] else "")

        for start, end, high, low, fvg_type in self.fvg_areas:
             if start_idx <= end < end_idx:
                 color = 'green' if fvg_type == 'bullish' else 'red'
                 ax1.fill_between(range(max(0, start - start_idx), min(end - start_idx + 1, len(subset))),
                                  high, low, color=color, alpha=0.2, label=f"{fvg_type.capitalize()} FVG" if start == self.fvg_areas[0][0] else "")
 
        # Plot SL and TP for current trades
        for idx, entry_price, direction, stop_loss, take_profit, size in self.current_trades:
            ax1.set_title(f"{self.symbol} - SMC Analysis")
        ax1.legend(loc='upper left')

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range(len(self.equity_curve)), self.equity_curve, label='Equity', color='blue')
        ax2.set_title("Equity Curve")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def calculate_performance(self):
        if not self.trades:
            return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0}
        winning_trades = [t for t in self.trades if t['pl'] > 0]
        win_rate = len(winning_trades) / len(self.trades)
        gross_profit = sum(t['pl'] for t in self.trades if t['pl'] > 0)
        gross_loss = abs(sum(t['pl'] for t in self.trades if t['pl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        total_return = (self.current_balance - self.initial_capital) / self.initial_capital * 100
        max_drawdown = max((max(self.equity_curve[:i+1]) - self.equity_curve[i]) / max(self.equity_curve[:i+1]) * 100
                           for i in range(len(self.equity_curve)))
        return {'total_trades': len(self.trades), 'win_rate': win_rate, 'profit_factor': profit_factor,
                'total_return_pct': total_return, 'max_drawdown_pct': max_drawdown}

    def run(self, scan_interval=300, max_runtime_minutes=45, sleep_interval_minutes=5, iterations_before_sleep=5):
        start_time = time.time()
        sast_now = get_sast_time()
        self.logger.info(f"Starting MatteGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting MatteGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

        profile = self.api.get_profile_info()
        if profile:
            self.initial_balance = self.current_balance = profile['balance']['bitmex_usd']
            self.equity_curve = [self.initial_balance]
            self.logger.info(f"Initial balance: ${self.initial_balance:.2f}")

        signal_found = False
        iteration = 0
        while (time.time() - start_time) < max_runtime_minutes * 60:
            sast_now = get_sast_time()
            self.logger.info(f"Scan {iteration + 1} started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scan {iteration + 1} started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

            if not self.get_market_data() or len(self.df) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(self.df)} candles")
                time.sleep(scan_interval)
                iteration += 1
                continue

            self.identify_swing_points()
            self.detect_market_structure()
            signals = self.execute_trades()

            for signal in signals:
                if signal['action'] == 'entry':
                    self.execute_entry(signal)
                    signal_found = True
                elif signal['action'] == 'exit':
                    self.execute_exit(signal)

            performance = self.calculate_performance()
            self.logger.info(f"Performance: {performance}")

            if self.bot:
                fig = self.visualize_results(start_idx=max(0, len(self.df) - 48))
                caption = (f"📸Scan {iteration+1} at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"Signal: {signal_found}\nBalance: ${self.current_balance:.2f}\nPrice at : {self.df['close'][-1]}")
                self.bot.send_photo(fig=fig, caption=caption)
            self.logger.info(f"Price:{self.df['close'][-1]}\n😪😪😪sleepining for {scan_interval/60} minutes....") 
            time.sleep(scan_interval)
            iteration += 1

            if iteration % iterations_before_sleep == 0 and iteration > 0:
                self.logger.info(f"Pausing for {sleep_interval_minutes} minutes...")
                print(f"Pausing for {sleep_interval_minutes} minutes...")
                time.sleep(sleep_interval_minutes * 60)
                self.logger.info("🙋🏾‍♂️Resuming...")
                print("Resuming...")

        self.logger.info("MatteGreen stopped.")
        final_performance = self.calculate_performance()
        #self.logger.info(f"Final performance: {final_performance}")
        if self.bot:
            fig = self.visualize_results(start_idx=max(0, len(self.df) - 48))
            caption = (f"🏁Final Results\nTotal Trades: {final_performance['total_trades']}\n"
                       f"Win Rate: {final_performance['win_rate']:.2%}\nReturn: {final_performance['total_return_pct']:.2f}%")
            self.bot.send_photo(fig=fig, caption=caption)

        return signal_found, self.df

if __name__ == "__main__":
    logger, bot = configure_logging(os.getenv("TOKEN"), os.getenv("CHAT_ID"))
    trader = MatteGreen(
        api_key=os.getenv("BITMEX_API_KEY"),
        api_secret=os.getenv("BITMEX_API_SECRET"),
        test=True,
        symbol="SOL-USD",
        timeframe="5m",
        telegram_token=os.getenv("TOKEN"),
        telegram_chat_id=os.getenv("CHAT_ID"),
        log=logger
    )
    signal_found, price_data = trader.run()
    print(f"Signal found: {signal_found}, Data length: {len(price_data)}")
