"""
StabilityAnalysis.py
Core stability analysis functionality for stock performance metrics
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

# Import from Market module
from Market import Market
from Utils import TRADING_DAYS_PER_YEAR, round_to_decimals

class StabilityAnalyzer(Market):
    """Enhanced stability analyzer inheriting from Market class for comprehensive performance analysis"""
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def calculate_area_under_curve(prices, start_price):
        """
        Method 1: Area under curve relative to starting price
        Shows how stable and long the stock price growth is
        Larger area = more stable/longer growth
        """
        normalized_prices = np.array(prices) - start_price
        area = np.trapz(normalized_prices)
        return area
    
    @staticmethod
    def calculate_overall_slope(prices):
        """
        Method 2: Overall slope from start to end
        Shows how much the stock can grow overall
        Higher slope = better overall growth
        """
        if len(prices) < 2:
            return 0
        days = np.arange(len(prices))
        slope = np.polyfit(days, prices, 1)[0]
        return slope
    
    @staticmethod  
    def calculate_sharpe_ratio(prices, risk_free_rate=0.02):
        """
        Method 3: Sharpe ratio (risk-adjusted returns)
        Shows return per unit of risk
        Higher Sharpe ratio = better risk-adjusted performance
        """
        if len(prices) < 2:
            return 0
        
        # Calculate daily returns
        daily_returns = np.diff(prices) / np.array(prices[:-1])
        
        if len(daily_returns) == 0:
            return 0
        
        # Annualize returns and volatility
        mean_return = np.mean(daily_returns) * TRADING_DAYS_PER_YEAR
        std_return = np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        if std_return == 0:
            return 0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        return sharpe_ratio
    
    @staticmethod
    def calculate_max_drawdown(prices):
        """Calculate maximum drawdown (additional risk metric)"""
        if len(prices) < 2:
            return 0
        
        prices_array = np.array(prices)
        cumulative_max = np.maximum.accumulate(prices_array)
        drawdowns = (prices_array - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        return max_drawdown
    
    @staticmethod
    def calculate_volatility(prices):
        """Calculate annualized volatility"""
        if len(prices) < 2:
            return 0
        
        daily_returns = np.diff(prices) / np.array(prices[:-1])
        volatility = np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        return volatility
    
    @staticmethod
    def get_enhanced_stability_metrics(prices):
        """Get comprehensive stability metrics with detailed explanations"""
        start_price = prices[0]
        end_price = prices[-1]
        
        # Calculate all metrics
        auc = StabilityAnalyzer.calculate_area_under_curve(prices, start_price)
        slope = StabilityAnalyzer.calculate_overall_slope(prices)
        sharpe_ratio = StabilityAnalyzer.calculate_sharpe_ratio(prices)
        total_return = (end_price - start_price) / start_price
        max_drawdown = StabilityAnalyzer.calculate_max_drawdown(prices)
        volatility = StabilityAnalyzer.calculate_volatility(prices)
        
        return {
            'auc': auc,
            'slope': slope, 
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'start_price': start_price,
            'end_price': end_price
        }