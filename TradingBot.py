from datetime import datetime, timedelta, time as dt_time
import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError
from typing import List, Optional, Dict
import csv
import json
import time
import schedule
import pytz

from Strategy import Strategy, Action
from Market import Market
from Utils import EVALUATION_WINDOW

class SimulationResult:
    """
    Result of trading simulation
    Matches COMP2012 SimulationResult struct exactly
    """
    
    def __init__(self):
        """Constructor matching SimulationResult() initialization"""
        self.best_strategy: Optional[Strategy] = None
        self.total_return: float = float('-inf')  # Matches -std::numeric_limits<double>::max()

class TradingBot:
    """
    Trading bot following COMP2012 design pattern exactly
    Matches TradingBot.h structure with enhanced real trading capabilities
    NO TRY-EXCEPT statements
    """
    
    def __init__(self, market: Market, initial_capacity: int = 10, 
                 api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Constructor matching TradingBot(Market *market, int initialCapacity = 10)
        Enhanced with real trading capabilities
        """
        self.market = market
        self.strategy_count = 0
        self.strategy_capacity = initial_capacity
        
        # Initialize strategies array - equivalent to Strategy **availableStrategies in C++
        self.available_strategies: List[Strategy] = []
        
        # Enhanced functionality: Real trading capabilities
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.trading_client = None
        self.data_client = None
        
        if api_key and secret_key:
            self.trading_client = TradingClient(api_key, secret_key, paper=paper)
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
    
    def add_strategy(self, strategy: Strategy):
        """
        Add new trading strategy to bot
        Matches: void addStrategy(Strategy *strategy)
        
        Following COMP2012 specification:
        - If capacity reached, dynamically increase capacity
        """
        # Check if we need to increase capacity
        if self.strategy_count >= self.strategy_capacity:
            # Double the capacity (dynamic array growth)
            self.strategy_capacity *= 2
        
        # Add strategy to array
        self.available_strategies.append(strategy)
        self.strategy_count += 1
    
    def run_simulation(self) -> SimulationResult:
        """
        Execute trading simulation using available strategies
        Matches: SimulationResult runSimulation()
        
        Following COMP2012 specification:
        - Evaluate strategies for last 101 days (EVALUATION_WINDOW + 1)
        - Return SimulationResult with best strategy and total return
        - Calculate return following specified rules:
          1. Start with zero cumulative profit
          2. BUY only if not holding, SELL only if holding
          3. Record purchase price on BUY, add profit on SELL
          4. On last day, add unrealized profit if still holding
        """
        result = SimulationResult()
        
        if self.strategy_count == 0 or not self.market:
            return result
        
        # Calculate evaluation window (last 101 days including last day)
        total_days = self.market.get_num_trading_days()
        start_day = max(0, total_days - EVALUATION_WINDOW - 1)
        
        for i, strategy in enumerate(self.available_strategies):
            # Initialize trading state
            buy_price = 0.0
            profit = 0.0
            holding = 0.0  # 0 = not holding, 1 = holding
            
            # Simulate trading for evaluation period
            for day in range(start_day, total_days):
                action = strategy.decide_action(self.market, day, holding)
                current_price = self.market.get_price(day)
                
                if action == Action.BUY and holding == 0:
                    # Buy stock - record purchase price
                    buy_price = current_price
                    holding = 1
                elif action == Action.SELL and holding == 1:
                    # Sell stock - calculate and add profit
                    profit += current_price - buy_price
                    holding = 0
                # HOLD - no change in position
            
            # On last trading day, add unrealized profit if still holding
            if holding == 1:
                last_price = self.market.get_last_price()
                profit += last_price - buy_price
            
            # Update best strategy
            if i == 0 or profit > result.total_return:
                result.best_strategy = strategy
                result.total_return = profit
        
        return result
    
    # Enhanced functionality: Real trading methods
    
    def get_buying_power(self) -> float:
        """Get current buying power of the account - NO TRY-EXCEPT"""
        if not self.trading_client:
            return 0.0
        
        account = self.trading_client.get_account()
        return float(account.buying_power)
    
    def get_current_holdings(self) -> Dict[str, Dict]:
        """Get current stock holdings with their details - NO TRY-EXCEPT"""
        if not self.trading_client:
            return {}
        
        positions = self.trading_client.get_all_positions()
        holdings = {}
        
        for position in positions:
            holdings[position.symbol] = {
                'quantity': float(position.qty),
                'market_value': float(position.market_value),
                'avg_entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc)
            }
        
        return holdings
    
    def get_current_stock_price(self, symbol: str) -> float:
        """Get current market price for a stock - NO TRY-EXCEPT"""
        if not self.data_client:
            return 0.0
        
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        latest_quote = self.data_client.get_stock_latest_quote(request)
        return float(latest_quote[symbol].bid_price)
    
    def place_buy_order(self, symbol: str, quantity: int = 1) -> bool:
        """Place a market buy order for the specified stock - NO TRY-EXCEPT"""
        if not self.trading_client:
            print(f"❌ No trading client configured for {symbol}")
            return False
        
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(order_data=market_order_data)
        print(f"✅ Buy order placed for {quantity} share(s) of {symbol}, Order ID: {order.id}")
        return True
    
    def place_sell_order(self, symbol: str, quantity: int) -> bool:
        """Place a market sell order for the specified stock - NO TRY-EXCEPT"""
        if not self.trading_client:
            print(f"❌ No trading client configured for {symbol}")
            return False
        
        market_order_data = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        order = self.trading_client.submit_order(order_data=market_order_data)
        print(f"✅ Sell order placed for {quantity} share(s) of {symbol}, Order ID: {order.id}")
        return True
    
    def display_all_holdings(self):
        """Display comprehensive holdings information - NO TRY-EXCEPT"""
        if not self.trading_client:
            print("❌ No trading client available")
            return
        
        holdings = self.get_current_holdings()
        
        if not holdings:
            print("📊 No current holdings in account")
            return
        
        print("\n" + "="*80)
        print("🏢 COMPLETE ACCOUNT HOLDINGS")
        print("="*80)
        
        # Get account info
        account = self.trading_client.get_account()
        print(f"💰 Total Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"💵 Buying Power: ${float(account.buying_power):,.2f}")
        print(f"📈 Total Equity: ${float(account.equity):,.2f}")
        print(f"📊 Day Trade Buying Power: ${float(account.daytrading_buying_power):,.2f}")
        
        print(f"\n📋 ALL HOLDINGS ({len(holdings)} positions):")
        print(f"{'Symbol':<8} {'Qty':<6} {'Entry $':<10} {'Current $':<10} {'Market Val':<12} {'P&L':<10} {'P&L %':<8}")
        print("-" * 70)
        
        total_market_value = 0
        total_pl = 0
        
        for symbol, holding in holdings.items():
            qty = holding['quantity']
            entry_price = holding['avg_entry_price']
            current_price = holding['current_price']
            market_value = holding['market_value']
            unrealized_pl = holding['unrealized_pl']
            unrealized_plpc = holding['unrealized_plpc']
            
            total_market_value += market_value
            total_pl += unrealized_pl
            
            print(f"{symbol:<8} {qty:<6.0f} ${entry_price:<9.2f} ${current_price:<9.2f} "
                  f"${market_value:<11.2f} ${unrealized_pl:<9.2f} {unrealized_plpc:<7.1%}")
        
        print("-" * 70)
        print(f"{'TOTALS':<8} {'':<6} {'':<10} {'':<10} ${total_market_value:<11.2f} ${total_pl:<9.2f}")
        print("\n" + "="*80)
        
        # Add this method to the TradingBot class in TradingBot.py
    def get_account_info(self) -> Dict:
        """Get comprehensive account information"""
        if not self.trading_client:
            return {'status': 'No trading client available'}
        
        account = self.trading_client.get_account()
        return {
            'status': account.status,
            'account_blocked': account.account_blocked,
            'trading_blocked': account.trading_blocked,
            'transfers_blocked': account.transfers_blocked,
            'account_number': account.account_number,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'last_equity': float(account.last_equity),
            'multiplier': float(account.multiplier),
            'currency': account.currency,
            'created_at': str(account.created_at)
        }


class AutomatedTradingSystemLoop:
    """Automated Trading System with position tracking - NO TRY-EXCEPT"""
    
    def __init__(self, api_key: str, secret_key: str, paper_trading: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper_trading = paper_trading
        
        # Trading bot for real trading
        self.trading_bot = TradingBot(None, api_key=api_key, secret_key=secret_key, paper=paper_trading)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Load pre-analyzed stock data from CSV
        self.stock_database = self._load_stock_database()
        self.sp500_symbols = self.get_sp500_symbols()  # ✅ Now this method exists
        
        # Trading state
        self.active_positions = {}
        self.is_running = False
        self.min_cash_reserve = 1000.0
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        print(f"🚀 Automated Trading System Initialized")
        print(f"📊 Loaded {len(self.stock_database)} stocks from database")
        print(f"📈 Tracking {len(self.sp500_symbols)} S&P 500 symbols")
        print(f"💰 Paper Trading: {paper_trading}")
        
    
    def _load_stock_database(self) -> Dict[str, Dict]:
        """Load stock analysis data from CSV - NO TRY-EXCEPT"""
        try:
            results = ResultsManager.load_from_csv()
            stock_db = {}
            
            for stock in results:
                symbol = stock.get('symbol', '')
                if symbol:
                    winner = stock.get('winner', 'GBM')
                    best_strategy_name = stock.get(f'best_strategy_{winner.lower()}', 'Unknown')
                    
                    stock_db[symbol] = {
                        'grade': stock.get('grade', 'F'),
                        'score': stock.get('score', 0.0),
                        'rank': stock.get('rank', 999),
                        'best_strategy_name': best_strategy_name,
                        'winner': winner,
                        'percentage_return': max(
                            stock.get('percentage_return_gbm', 0.0),
                            stock.get('percentage_return_ml', 0.0)
                        )
                    }
            
            return stock_db
            
        except Exception as e:
            print(f"⚠️  Error loading stock database: {e}")
            print("📊 Using empty database")
            return {}
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.eastern_tz)
        current_time = now.time()
        current_weekday = now.weekday()
        
        # Market is closed on weekends
        if current_weekday >= 5:
            return False
        
        return self.market_open <= current_time <= self.market_close
    
    def start_automated_trading(self):
        """Start the automated trading system - NO TRY-EXCEPT"""
        print("\n🚀 STARTING AUTOMATED TRADING SYSTEM")
        print("="*60)
        
        # Display initial holdings
        print("\n📋 INITIAL ACCOUNT STATUS:")
        self.trading_bot.display_all_holdings()
        
        if not self._is_market_open():
            print("⏰ Market is currently closed. System will wait for market open.")
        
        # Schedule monitoring
        schedule.every(1).minutes.do(self._minute_task)
        self.is_running = True
        
        print("✅ Automated trading system is now running...")
        print("🛑 Press Ctrl+C to stop")
        
        while self.is_running:
            if self._is_market_open():
                schedule.run_pending()
            else:
                now = datetime.now(self.eastern_tz)
                if now.time() > self.market_close:
                    print(f"🌙 Market closed for today.")
                    time.sleep(3600)
                else:
                    print(f"⏰ Waiting for market open...")
                    time.sleep(300)
            time.sleep(1)
    
    def _minute_task(self):
        """Task that runs every minute during market hours"""
        if self._is_market_open():
            print(f"🔍 Market monitoring cycle at {datetime.now().strftime('%H:%M:%S')}")
    
    def stop_trading(self):
        """Stop the automated trading system"""
        self.is_running = False
        print("🛑 Automated trading system stopped")
        
        # Display final holdings
        print("\n📋 FINAL ACCOUNT STATUS:")
        self.trading_bot.display_all_holdings()
        
        
    def get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols from Wikipedia"""
        try:
            import pandas as pd
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (replace . with -)
            cleaned_symbols = []
            for symbol in symbols:
                cleaned_symbol = str(symbol).replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)
            
            print(f"📊 Loaded {len(cleaned_symbols)} S&P 500 symbols")
            return cleaned_symbols
            
        except Exception as e:
            print(f"❌ Error fetching S&P 500 symbols: {e}")
            print("📊 Using fallback symbol list")
            # Fallback list of major S&P 500 stocks
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'TSM', 'UNH',
                'XOM', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX', 'MA', 'BAC', 'ABBV',
                'PFE', 'AVGO', 'COST', 'DIS', 'KO', 'MRK', 'PEP', 'TMO', 'WMT', 'ABT'
            ]


class ResultsManager:
    """Manages saving and loading of analysis results - NO TRY-EXCEPT"""
    
    @staticmethod
    def save_to_csv(results_data: List[Dict], filename: str = "sp500_analysis_results.csv"):
        """Save results data to CSV file - NO TRY-EXCEPT"""
        
        if not results_data:
            print("No data to save")
            return
        
        # Enhanced CSV columns
        fieldnames = [
            'symbol', 'rank', 'score', 'grade', 'grade_category', 
            'quartile', 'scoring_method',
            'best_return_gbm', 'best_strategy_gbm', 
            'best_return_ml', 'best_strategy_ml',
            'percentage_return_gbm', 'percentage_return_ml',
            'auc_gbm', 'slope_gbm', 'sharpe_ratio_gbm',
            'auc_ml', 'slope_ml', 'sharpe_ratio_ml',
            'total_return_gbm', 'total_return_ml',
            'winner', 'advantage', 'assessment'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for result in results_data:
                csv_row = {}
                for field in fieldnames:
                    csv_row[field] = result.get(field, 'N/A')
                writer.writerow(csv_row)
        
        print(f"\nResults saved to {filename}")
        print(f"Total records saved: {len(results_data)}")
    
    @staticmethod
    def save_summary_to_json(summary_data: Dict, filename: str = "sp500_analysis_summary.json"):
        """Save summary statistics to JSON file - NO TRY-EXCEPT"""
        
        with open(filename, 'w') as jsonfile:
            json.dump(summary_data, jsonfile, indent=2, default=str)
        
        print(f"Summary saved to {filename}")
    
    @staticmethod
    def load_from_csv(filename: str = "sp500_enhanced_yfinance_results.csv") -> List[Dict]:
        """Load analysis results from CSV file - NO TRY-EXCEPT"""
        
        results = []
        with open(filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields
                numeric_fields = ['rank', 'score', 'raw_score', 'percentile', 
                                'best_return_gbm', 'best_return_ml',
                                'percentage_return_gbm', 'percentage_return_ml',
                                'auc_gbm', 'slope_gbm', 'sharpe_ratio_gbm',
                                'auc_ml', 'slope_ml', 'sharpe_ratio_ml',
                                'total_return_gbm', 'total_return_ml', 'advantage']
                
                for field in numeric_fields:
                    if field in row and row[field] != 'N/A':
                        row[field] = float(row[field])
                    else:
                        row[field] = 0.0
                
                results.append(row)
        
        print(f"Loaded {len(results)} stock analysis results from {filename}")
        return results