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

def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class MateGreen:
    def __init__(self, api_key, api_secret, test=True, symbol="BTC-USD", timeframe="5m",
                 initial_capital=10000, risk_per_trade=0.02, rr_ratio=2, lookback_period=20,
                 fvg_threshold=0.003, telegram_bot=None, api=None, log=None):
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
        self.api = api
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
        self.logger = log
        self.logger.info(f"MateGreen initialized for {symbol} on {timeframe} timeframe")

    def get_market_data(self):
        # Fallback to yfinance only (removing BitMEX for simplicity and your request)
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
            self.logger.info(f"Retrieved {len(data)} candles from yfinance")
            if data.empty:
                self.logger.error("Retrieved empty DataFrame from yfinance")
                return False
            data.columns = [col.lower() for col in data.columns]
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns. Available: {data.columns}")
                return False
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
            self.logger.error(f"yfinance fetch failed: {str(e)}")
            return False

    # Other methods (identify_swing_points, detect_market_structure, execute_trades, etc.) remain unchanged

    def visualize_results(self, start_idx=0, end_idx=None):
        """Visualize price chart and equity curve."""
        if end_idx is None:
            end_idx = len(self.df)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1], sharex=True)

        # Price chart (top subplot)
        subset = self.df.iloc[start_idx:end_idx]
        ax1.plot(subset.index, subset['close'], label='Close Price', color='black', linewidth=1)
        for trade in self.trades:
            if start_idx <= trade['entry_index'] < end_idx:
                color = 'green' if trade['type'] == 'long' else 'red'
                marker = '^' if trade['type'] == 'long' else 'v'
                ax1.scatter(trade['entry_index'], trade['entry_price'], color=color, marker=marker, s=120, zorder=5)
                if 'exit_index' in trade and trade['exit_index'] < end_idx:
                    color = 'green' if trade['pnl'] > 0 else 'red'
                    ax1.scatter(trade['exit_index'], trade['exit_price'], color=color, marker='o', s=120, zorder=5)
                    ax1.plot([trade['entry_index'], trade['exit_index']],
                             [trade['entry_price'], trade['exit_price']],
                             color=color, linewidth=1, linestyle='--')
        ax1.set_title(f"{self.symbol} - SMC Analysis")
        ax1.legend(['Close Price'], loc='best')
        ax1.grid(True, alpha=0.3)

        # Equity curve (bottom subplot)
        equity_indices = range(len(self.equity_curve))
        ax2.plot(equity_indices, self.equity_curve, label='Equity Curve', color='blue', linewidth=1)
        ax2.set_title("Equity Curve")
        ax2.legend(['Equity'], loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Trade/Scan Index")

        plt.tight_layout()
        return fig

    def run(self, scan_interval=120):
        sast_now = get_sast_time()
        self.logger.info(f"Starting MateGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting MateGreen at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

        self.initial_balance = self.initial_capital  # Use initial_capital since no API balance
        self.current_balance = self.initial_balance
        self.equity_curve = [self.initial_balance]

        signal_found = False
        for iteration in range(2):
            sast_now = get_sast_time()
            self.logger.info(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Scan {iteration + 1}/2 started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")

            if not self.get_market_data() or len(self.df) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(self.df)} candles retrieved")
                if iteration < 1:
                    # Send chart before sleeping if no data
                    if self.telegram_bot and not self.df.empty:
                        lookback_candles = 48 if self.timeframe == "5m" else 16
                        fig = self.visualize_results(start_idx=max(0, len(self.df) - lookback_candles))
                        caption = f"📸Insufficient data - Scan {iteration+1} at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}"
                        self.telegram_bot.send_photo(fig=fig, caption=caption)
                        plt.close(fig)
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

            performance = self.calculate_performance()
            self.logger.info(f"Performance snapshot: {performance}")

            if iteration < 1:  # Before sleeping
                if self.telegram_bot and not self.df.empty:
                    try:
                        lookback_candles = 48 if self.timeframe == "5m" else 16
                        fig = self.visualize_results(start_idx=max(0, len(self.df) - lookback_candles))
                        caption = f"📸Scan {iteration+1} at {sast_now.strftime('%Y-%m-%d %H:%M:%S')} - Signal: {signal_found}"
                        self.telegram_bot.send_photo(fig=fig, caption=caption)
                        time.sleep(5)
                        self.logger.info(f"📸Sent analysis plot for scan {iteration+1}")
                        plt.close(fig)
                    except Exception as e:
                        self.logger.error(f"Failed to send chart: {e}")
                self.logger.info(f"Waiting {scan_interval} seconds for next scan...")
                print(f"Waiting {scan_interval} seconds for next scan...")
                time.sleep(scan_interval)

        self.logger.info("Completed 2 scans, stopping MateGreen")
        final_performance = self.calculate_performance()
        self.logger.info(f"Final performance metrics: {final_performance}")

        return signal_found, self.df

if __name__ == "__main__":
    # Assuming telegram_bot and logger are set up in TeleLogBot
    logger, telegram_bot = configure_logging(os.getenv("TOKEN"), os.getenv("CHAT_ID"))
    trader = MateGreen(
        api_key=os.getenv("BITMEX_API_KEY"),
        api_secret=os.getenv("BITMEX_API_SECRET"),
        test=True,
        symbol="BTC-USD",
        timeframe="5m",
        telegram_bot=telegram_bot,
        log=logger
    )
    signal_found, price_data = trader.run()
    print(f"Signal found: {signal_found}, Data length: {len(price_data)}")
