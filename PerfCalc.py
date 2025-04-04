import pandas as pd
import numpy as np
from datetime import datetime
import json
import argparse
from typing import Dict, List, Union, Any


class TradingPerformanceCalculator:
    def __init__(self, transactions_file=None, positions_file=None):
        """
        Initialize the calculator with transaction and position data.
        
        :param transactions_file: Path to JSON file containing transaction data
        :param positions_file: Path to JSON file containing positions data
        """
        self.transactions = self._load_data(transactions_file) if transactions_file else []
        self.positions = self._load_data(positions_file) if positions_file else []
        
    def _load_data(self, file_path):
        """Load data from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return []
    
    def set_transactions(self, transactions):
        """Set transactions data directly."""
        self.transactions = transactions
    
    def set_positions(self, positions):
        """Set positions data directly."""
        self.positions = positions
    
    def _convert_timestamps(self, data):
        """Convert string timestamps to datetime objects."""
        for item in data:
            if 'transactTime' in item and isinstance(item['transactTime'], str):
                item['transactTime'] = datetime.fromisoformat(item['transactTime'].replace('Z', '+00:00'))
            if 'timestamp' in item and isinstance(item['timestamp'], str):
                item['timestamp'] = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
        return data
    
    def _prepare_transactions_df(self):
        """Convert transactions to DataFrame for analysis."""
        if not self.transactions:
            return pd.DataFrame()
        
        # Handle potential string timestamps
        processed_transactions = self._convert_timestamps(self.transactions)
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_transactions)
        
        # Ensure amounts are numeric
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        if 'walletBalance' in df.columns:
            df['walletBalance'] = pd.to_numeric(df['walletBalance'], errors='coerce')
        if 'fee' in df.columns:
            df['fee'] = pd.to_numeric(df['fee'], errors='coerce')
            
        return df
    
    def _prepare_positions_df(self):
        """Convert positions to DataFrame for analysis."""
        if not self.positions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.positions)
        
        # Ensure values are numeric
        numeric_cols = ['unrealized_pnl', 'realized_pnl', 'size', 'value']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def calculate_performance(self):
        """
        Calculate trading performance metrics.
        
        :return: Dictionary with performance metrics
        """
        # Convert data to DataFrames
        transactions_df = self._prepare_transactions_df()
        positions_df = self._prepare_positions_df()
        
        # Default metrics in case of insufficient data
        metrics = {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "total_return_pct": 0,
            "max_drawdown_pct": 0,
            "margin_utilization": 0,
            "available_margin": 0,
            "realized_pnl": 0,
            "unrealized_pnl": 0
        }
        
        # Return default metrics if no data
        if transactions_df.empty and positions_df.empty:
            return metrics
        
        # Calculate metrics from transactions
        if not transactions_df.empty:
            # Get the most recent wallet balance
            if 'walletBalance' in transactions_df.columns:
                latest_transaction = transactions_df.sort_values('timestamp', ascending=False).iloc[0]
                metrics["wallet_balance"] = latest_transaction['walletBalance'] / 1e8 if 'walletBalance' in latest_transaction else 0
            
            # Calculate realized PnL from transactions
            if 'transactType' in transactions_df.columns and 'amount' in transactions_df.columns:
                pnl_transactions = transactions_df[transactions_df['transactType'] == 'RealisedPNL']
                metrics["realized_pnl"] = pnl_transactions['amount'].sum() / 1e8 if not pnl_transactions.empty else 0
            
            # Calculate total deposits and withdrawals
            if 'transactType' in transactions_df.columns and 'amount' in transactions_df.columns:
                transfers = transactions_df[transactions_df['transactType'] == 'Transfer']
                metrics["total_deposits"] = transfers[transfers['amount'] > 0]['amount'].sum() / 1e8 if not transfers.empty else 0
                metrics["total_withdrawals"] = -1 * transfers[transfers['amount'] < 0]['amount'].sum() / 1e8 if not transfers.empty else 0
            
            # Calculate trading fees
            if 'fee' in transactions_df.columns:
                metrics["total_fees"] = transactions_df['fee'].sum() / 1e8 if 'fee' in transactions_df else 0
        
        # Calculate metrics from positions
        if not positions_df.empty:
            # Calculate total trades
            metrics["total_trades"] = len(positions_df)
            
            # Calculate win rate
            if 'realized_pnl' in positions_df.columns:
                winning_trades = positions_df[positions_df['realized_pnl'] > 0]
                metrics["win_rate"] = round((len(winning_trades) / len(positions_df)) * 100, 2) if len(positions_df) > 0 else 0
            
            # Calculate profit factor
            if 'realized_pnl' in positions_df.columns:
                gross_profit = positions_df[positions_df['realized_pnl'] > 0]['realized_pnl'].sum() / 1e8 if 'realized_pnl' in positions_df else 0
                gross_loss_abs = abs(positions_df[positions_df['realized_pnl'] < 0]['realized_pnl'].sum() / 1e8) if 'realized_pnl' in positions_df else 0
                metrics["profit_factor"] = round(gross_profit / gross_loss_abs, 2) if gross_loss_abs > 0 else float("inf")
            
            # Calculate unrealized PnL
            if 'unrealized_pnl' in positions_df.columns:
                metrics["unrealized_pnl"] = round(positions_df['unrealized_pnl'].sum() / 1e8, 2)
            
            # Calculate total return percentage
            if 'realized_pnl' in positions_df.columns and metrics.get("wallet_balance", 0) > 0:
                initial_balance = metrics["wallet_balance"] - metrics["realized_pnl"] if "realized_pnl" in metrics else metrics["wallet_balance"]
                if initial_balance > 0:
                    metrics["total_return_pct"] = round(((metrics["wallet_balance"] - initial_balance) / initial_balance) * 100, 2)
        
        # Fill in missing metrics with estimates if possible
        if "total_trades" not in metrics and "realized_pnl" in metrics:
            # Estimate trade count from PnL transactions if positions aren't available
            pnl_transactions = transactions_df[transactions_df['transactType'] == 'RealisedPNL'] if not transactions_df.empty else pd.DataFrame()
            metrics["total_trades"] = len(pnl_transactions) if not pnl_transactions.empty else 0
            
        # Round numeric values for readability
        for key in metrics:
            if isinstance(metrics[key], (float, int)) and key != "total_trades" and not np.isinf(metrics[key]):
                metrics[key] = round(metrics[key], 2)
                
        return metrics
    
    def calculate_equity_curve(self):
        """
        Calculate the equity curve from transactions.
        
        :return: DataFrame with timestamps and equity values
        """
        transactions_df = self._prepare_transactions_df()
        if transactions_df.empty or 'walletBalance' not in transactions_df.columns or 'timestamp' not in transactions_df.columns:
            return pd.DataFrame(columns=['timestamp', 'equity'])
        
        # Sort by timestamp
        equity_df = transactions_df.sort_values('timestamp')[['timestamp', 'walletBalance']]
        
        # Convert to BTC
        equity_df['equity'] = equity_df['walletBalance'] / 1e8
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
        
        # Calculate max drawdown
        max_dd = equity_df['drawdown'].max()
        
        return equity_df, max_dd
        
    def calculate_detailed_performance(self):
        """
        Calculate detailed performance metrics including equity curve, drawdowns, etc.
        
        :return: Dictionary with detailed performance metrics
        """
        basic_metrics = self.calculate_performance()
        equity_df, max_drawdown = self.calculate_equity_curve()
        
        # Add max drawdown
        detailed_metrics = {**basic_metrics, "max_drawdown_pct": round(max_drawdown, 2)}
        
        # Add monthly returns if enough data
        if not equity_df.empty and len(equity_df) > 1:
            # Calculate monthly returns
            equity_df['month'] = equity_df['timestamp'].dt.strftime('%Y-%m')
            monthly_equity = equity_df.groupby('month')['equity'].last().reset_index()
            monthly_equity['prev_equity'] = monthly_equity['equity'].shift(1)
            monthly_equity['monthly_return'] = (monthly_equity['equity'] - monthly_equity['prev_equity']) / monthly_equity['prev_equity'] * 100
            
            # Add to metrics
            detailed_metrics["monthly_returns"] = monthly_equity[['month', 'monthly_return']].dropna().to_dict('records')
            
            # Calculate annualized metrics
            if len(monthly_equity) > 1:
                days = (equity_df['timestamp'].max() - equity_df['timestamp'].min()).days
                if days > 30:
                    years = days / 365.25
                    total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0] - 1) * 100
                    detailed_metrics["annualized_return"] = round((((1 + total_return/100) ** (1/years)) - 1) * 100, 2)
        
        return detailed_metrics
    
    def generate_performance_report(self):
        """
        Generate a comprehensive performance report.
        
        :return: Dictionary with all performance metrics and statistics
        """
        detailed_metrics = self.calculate_detailed_performance()
        
        # Format for reporting
        report = {
            "summary": {
                "total_trades": detailed_metrics.get("total_trades", 0),
                "win_rate": f"{detailed_metrics.get('win_rate', 0)}%",
                "profit_factor": detailed_metrics.get("profit_factor", 0),
                "realized_pnl": f"{detailed_metrics.get('realized_pnl', 0)} BTC",
                "unrealized_pnl": f"{detailed_metrics.get('unrealized_pnl', 0)} BTC",
                "total_return": f"{detailed_metrics.get('total_return_pct', 0)}%",
                "max_drawdown": f"{detailed_metrics.get('max_drawdown_pct', 0)}%",
            },
            "detailed_metrics": detailed_metrics
        }
        
        return report
    
    def export_report_to_json(self, output_file):
        """
        Export performance report to JSON file.
        
        :param output_file: Path to save the JSON file
        """
        report = self.generate_performance_report()
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            print(f"Report successfully exported to {output_file}")
        except Exception as e:
            print(f"Error exporting report: {e}")


def parse_transaction_data(raw_data):
    """
    Parse transaction data that might be in various formats.
    
    :param raw_data: String or list containing transaction data
    :return: List of transaction dictionaries
    """
    if isinstance(raw_data, list):
        return raw_data
    
    # Try to parse as JSON
    try:
        data = json.loads(raw_data)
        return data if isinstance(data, list) else [data]
    except:
        pass
    
    # Try to parse as string representation of Python objects
    try:
        # Warning: Using eval is dangerous with untrusted input
        # This is a simplified example - in production use ast.literal_eval
        import ast
        data = ast.literal_eval(raw_data)
        return data if isinstance(data, list) else [data]
    except:
        pass
    
    # Return empty list if parsing fails
    print("Failed to parse transaction data")
    return []


def main():
    parser = argparse.ArgumentParser(description='Calculate trading performance metrics')
    parser.add_argument('--transactions', type=str, help='Path to transactions JSON file or raw transaction data')
    parser.add_argument('--positions', type=str, help='Path to positions JSON file')
    parser.add_argument('--output', type=str, default='performance_report.json', help='Output file path')
    
    args = parser.parse_args()
    
    calculator = TradingPerformanceCalculator()
    
    # Load transactions
    if args.transactions:
        if args.transactions.endswith('.json'):
            # Load from file
            calculator = TradingPerformanceCalculator(args.transactions, args.positions)
        else:
            # Parse raw data
            transactions = parse_transaction_data(args.transactions)
            calculator.set_transactions(transactions)
            
            # Load positions if provided
            if args.positions:
                if args.positions.endswith('.json'):
                    with open(args.positions, 'r') as f:
                        positions = json.load(f)
                else:
                    positions = parse_transaction_data(args.positions)
                calculator.set_positions(positions)
    
    # Generate and export report
    report = calculator.generate_performance_report()
    calculator.export_report_to_json(args.output)
    
    # Print summary
    print("\nTrading Performance Summary:")
    for key, value in report["summary"].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

def get_trading_performance_summary(transactions_data=None, positions_data=None):
    """
    Generate a detailed trading performance summary dictionary.
    
    :param transactions_data: List of transaction dictionaries or JSON file path
    :param positions_data: List of position dictionaries or JSON file path
    :return: Dictionary containing detailed performance summary
    """
    # Initialize calculator
    calculator = TradingPerformanceCalculator()
    
    # Handle transactions data
    if transactions_data:
        if isinstance(transactions_data, str) and transactions_data.endswith('.json'):
            calculator = TradingPerformanceCalculator(transactions_file=transactions_data)
        else:
            calculator.set_transactions(transactions_data)
    
    # Handle positions data
    if positions_data:
        if isinstance(positions_data, str) and positions_data.endswith('.json'):
            calculator = TradingPerformanceCalculator(positions_file=positions_data)
        else:
            calculator.set_positions(positions_data)
    
    # Generate full report
    report = calculator.generate_performance_report()
    
    # Create detailed summary dictionary
    summary = {
        "overview": {
            "total_trades": report["summary"]["total_trades"],
            "win_rate_pct": float(report["summary"]["win_rate"].replace('%', '')),
            "profit_factor": report["summary"]["profit_factor"],
            "total_return_pct": float(report["summary"]["total_return"].replace('%', '')),
            "max_drawdown_pct": float(report["summary"]["max_drawdown"].replace('%', ''))
        },
        "profit_metrics": {
            "realized_pnl_btc": float(report["summary"]["realized_pnl"].replace(' BTC', '')),
            "unrealized_pnl_btc": float(report["summary"]["unrealized_pnl"].replace(' BTC', '')),
            "total_pnl_btc": float(report["summary"]["realized_pnl"].replace(' BTC', '')) + 
                           float(report["summary"]["unrealized_pnl"].replace(' BTC', ''))
        },
        "detailed_metrics": {
            "wallet_balance_btc": report["detailed_metrics"].get("wallet_balance", 0),
            "total_deposits_btc": report["detailed_metrics"].get("total_deposits", 0),
            "total_withdrawals_btc": report["detailed_metrics"].get("total_withdrawals", 0),
            "total_fees_btc": report["detailed_metrics"].get("total_fees", 0),
            "annualized_return_pct": report["detailed_metrics"].get("annualized_return", 0)
        },
        "performance_history": {
            "monthly_returns": report["detailed_metrics"].get("monthly_returns", []),
            "equity_curve": calculator.calculate_equity_curve()[0][['timestamp', 'equity']].to_dict('records')
            if not calculator.calculate_equity_curve()[0].empty else []
        },
        "metadata": {
            "calculation_date": datetime.now().isoformat(),
            "data_points": len(transactions_data) if transactions_data else 0,
            "position_count": len(positions_data) if positions_data else 0
        }
    }
    
    return summary
if __name__ == "__main__":
    main()
