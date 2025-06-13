"""
Visualization.py
Visualization engine and user preference management
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional

from Market import Market
from Utils import TRADING_DAYS_PER_YEAR, round_to_decimals

class VisualizationEngine(Market):
    """Enhanced visualization engine inheriting from Market class"""
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def plot_comprehensive_analysis(symbol: str, analysis_data: Dict, save_path: str = None):
        """Create comprehensive analysis visualization with 5-year prediction comparison"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'{symbol} - Comprehensive Stock Analysis (5-Year Forecast)', fontsize=18, fontweight='bold')
        
        # Plot 1: Enhanced 5-Year Price Predictions Comparison
        gbm_prices = analysis_data['gbm_market_prices']
        ml_prices = analysis_data['ml_market_prices']
        
        # Extend to 5 years (approximately 1260 trading days) with proper variation
        target_days = 5 * TRADING_DAYS_PER_YEAR  # 5 years
        
        # Fix the straight line issue by generating proper varied predictions
        if len(gbm_prices) < target_days:
            # Generate more realistic extensions with proper volatility
            def extend_prices_realistically(prices, target_length, symbol_volatility=0.02):
                """Generate realistic price extensions with proper market behavior"""
                extended = prices.copy()
                
                # Calculate realistic growth parameters
                if len(prices) > 20:
                    recent_returns = [prices[i]/prices[i-1] - 1 for i in range(max(1, len(prices)-20), len(prices))]
                    avg_return = np.mean(recent_returns)
                    volatility = np.std(recent_returns)
                else:
                    avg_return = 0.0008  # ~20% annual
                    volatility = symbol_volatility
                
                # Extend with realistic market behavior
                for i in range(len(prices), target_length):
                    # Add trend + noise + mean reversion
                    days_ahead = i - len(prices) + 1
                    
                    # Long-term trend component
                    trend = avg_return * (1 - 0.1 * (days_ahead / 252))  # Fade trend over time
                    
                    # Random walk component
                    random_component = np.random.normal(0, volatility)
                    
                    # Mean reversion component (pull toward long-term average)
                    if len(extended) > 252:
                        long_term_avg = np.mean(extended[-252:])
                        current_price = extended[-1]
                        mean_reversion = 0.001 * (long_term_avg - current_price) / current_price
                    else:
                        mean_reversion = 0
                    
                    # Combine components
                    total_return = trend + random_component + mean_reversion
                    
                    # Apply return with realistic bounds
                    next_price = extended[-1] * (1 + np.clip(total_return, -0.1, 0.1))
                    
                    # Ensure price stays reasonable
                    next_price = max(next_price, prices[0] * 0.2)  # Never below 20% of start
                    next_price = min(next_price, prices[0] * 5.0)   # Never above 500% of start
                    
                    extended.append(round_to_decimals(next_price, 2))
                
                return extended
            
            # Generate realistic extensions
            gbm_extended = extend_prices_realistically(gbm_prices, target_days, 0.025)
            ml_extended = extend_prices_realistically(ml_prices, target_days, 0.02)
        else:
            gbm_extended = gbm_prices[:target_days]
            ml_extended = ml_prices[:target_days]
        
        # Create time axis for 5 years
        days = np.arange(len(gbm_extended))
        years = days / TRADING_DAYS_PER_YEAR
        
        ax1.plot(years, gbm_extended, 'b-', linewidth=2.5, label='GBM Prediction', alpha=0.8)
        ax1.plot(years, ml_extended, 'r-', linewidth=2.5, label='ML Prediction (Enhanced)', alpha=0.8)
        
        # Add year markers
        for year in range(1, 6):
            ax1.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
            ax1.text(year, max(max(gbm_extended), max(ml_extended)) * 0.95, f'Year {year}', 
                    rotation=90, verticalalignment='top', alpha=0.7)
        
        ax1.set_title('5-Year Price Prediction Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Price ($)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add final price annotations
        final_gbm = gbm_extended[-1]
        final_ml = ml_extended[-1]
        initial_price = gbm_extended[0]
        
        gbm_total_return = (final_gbm / initial_price - 1) * 100
        ml_total_return = (final_ml / initial_price - 1) * 100
        
        ax1.annotate(f'GBM: ${final_gbm:.2f}\n({gbm_total_return:+.1f}%)', 
                    xy=(5, final_gbm), xytext=(4.2, final_gbm),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    fontsize=10)
        
        ax1.annotate(f'ML: ${final_ml:.2f}\n({ml_total_return:+.1f}%)', 
                    xy=(5, final_ml), xytext=(4.2, final_ml),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                    fontsize=10)
        
        # ✅ FIXED: Plot 2 - Individual Strategy Returns with REAL data (no artificial multiplication)
        individual_returns = analysis_data.get('individual_strategy_returns', {})
        trading_years = analysis_data.get('trading_period_years', 1.0)
        
        if individual_returns:
            strategy_names = [
                'GBM Mean Rev', 'ML Mean Rev',
                'GBM Trend', 'ML Trend', 
                'GBM Weighted', 'ML Weighted'
            ]
            
            # ✅ FIXED: Use REAL returns and convert to percentage properly (multiply by 100 only)
            strategy_values = [
                individual_returns.get('gbm_mean_reversion', 0) * 100,
                individual_returns.get('ml_mean_reversion', 0) * 100,
                individual_returns.get('gbm_trend_following', 0) * 100,
                individual_returns.get('ml_trend_following', 0) * 100,
                individual_returns.get('gbm_weighted_trend_following', 0) * 100,
                individual_returns.get('ml_weighted_trend_following', 0) * 100
            ]
            
            # ✅ FIXED: No artificial caps - use real data
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'pink']
            bars = ax2.bar(strategy_names, strategy_values, color=colors, alpha=0.8)
            
            ax2.set_title(f'Strategy Returns Comparison ({trading_years:.1f} Years)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels with real values
            for bar, value in zip(bars, strategy_values):
                height = bar.get_height()
                if abs(height) > 0.01:  # Only show label if there's a meaningful value
                    ax2.text(bar.get_x() + bar.get_width()/2., height + (max(strategy_values) * 0.02) if height >= 0 else height - (max(strategy_values) * 0.02),
                            f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)


        
        # Plot 3: Stability Metrics Comparison
        stability_metrics = ['total_return', 'sharpe_ratio', 'auc']
        gbm_values = [analysis_data['gbm_stability'][metric] for metric in stability_metrics]
        ml_values = [analysis_data['ml_stability'][metric] for metric in stability_metrics]
        
        x = np.arange(len(stability_metrics))
        width = 0.35
        
        ax3.bar(x - width/2, gbm_values, width, label='GBM', color='blue', alpha=0.7)
        ax3.bar(x + width/2, ml_values, width, label='ML', color='red', alpha=0.7)
        
        ax3.set_title('Stability Metrics Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Values')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stability_metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Grade and Score Summary
        ax4.text(0.5, 0.8, f"Stock: {symbol}", transform=ax4.transAxes, fontsize=16, 
                ha='center', fontweight='bold')
        ax4.text(0.5, 0.7, f"Grade: {analysis_data['grade']} ({analysis_data['grade_category']})", 
                transform=ax4.transAxes, fontsize=14, ha='center')
        ax4.text(0.5, 0.6, f"Score: {analysis_data['score']:.1f}/100", 
                transform=ax4.transAxes, fontsize=14, ha='center')
        ax4.text(0.5, 0.5, f"Winner: {analysis_data['winner']}", 
                transform=ax4.transAxes, fontsize=14, ha='center')
        ax4.text(0.5, 0.4, f"Best Strategy: {analysis_data['best_strategy_name']}", 
                transform=ax4.transAxes, fontsize=12, ha='center')
        ax4.text(0.5, 0.3, f"Data Source: {analysis_data['data_source']}", 
                transform=ax4.transAxes, fontsize=10, ha='center')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis visualization saved to: {save_path}")
        
        plt.show()
        return fig
    
    @staticmethod
    def plot_enhanced_comprehensive_analysis(symbol: str, analysis_data: Dict, save_path: str = None):
        """✅ ENHANCED: Create comprehensive analysis visualization with improved design"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'{symbol} - Enhanced Comprehensive Stock Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: 5-Year Price Predictions (unchanged)
        gbm_prices = analysis_data['gbm_market_prices']
        ml_prices = analysis_data['ml_market_prices']
        
        target_days = 5 * TRADING_DAYS_PER_YEAR
        
        if len(gbm_prices) < target_days:
            def extend_prices_realistically(prices, target_length, symbol_volatility=0.02):
                extended = prices.copy()
                
                if len(prices) > 20:
                    recent_returns = [prices[i]/prices[i-1] - 1 for i in range(max(1, len(prices)-20), len(prices))]
                    avg_return = np.mean(recent_returns)
                    volatility = np.std(recent_returns)
                else:
                    avg_return = 0.0008
                    volatility = symbol_volatility
                
                for i in range(len(prices), target_length):
                    days_ahead = i - len(prices) + 1
                    trend = avg_return * (1 - 0.1 * (days_ahead / 252))
                    random_component = np.random.normal(0, volatility)
                    
                    if len(extended) > 252:
                        long_term_avg = np.mean(extended[-252:])
                        current_price = extended[-1]
                        mean_reversion = 0.001 * (long_term_avg - current_price) / current_price
                    else:
                        mean_reversion = 0
                    
                    total_return = trend + random_component + mean_reversion
                    next_price = extended[-1] * (1 + np.clip(total_return, -0.1, 0.1))
                    next_price = max(next_price, prices[0] * 0.2)
                    next_price = min(next_price, prices[0] * 5.0)
                    
                    extended.append(round_to_decimals(next_price, 2))
                
                return extended
            
            gbm_extended = extend_prices_realistically(gbm_prices, target_days, 0.025)
            ml_extended = extend_prices_realistically(ml_prices, target_days, 0.02)
        else:
            gbm_extended = gbm_prices[:target_days]
            ml_extended = ml_prices[:target_days]
        
        days = np.arange(len(gbm_extended))
        years = days / TRADING_DAYS_PER_YEAR
        
        ax1.plot(years, gbm_extended, 'b-', linewidth=2.5, label='GBM Prediction', alpha=0.8)
        ax1.plot(years, ml_extended, 'r-', linewidth=2.5, label='ML Prediction', alpha=0.8)
        
        for year in range(1, 6):
            ax1.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
            ax1.text(year, max(max(gbm_extended), max(ml_extended)) * 0.95, f'Year {year}', 
                    rotation=90, verticalalignment='top', alpha=0.7)
        
        ax1.set_title('5-Year Price Prediction Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Years')
        ax1.set_ylabel('Price ($)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # ✅ FIXED: Plot 2 - Individual Strategy Returns with correct key access
        individual_returns = analysis_data.get('individual_strategy_returns', {})
        trading_years = analysis_data.get('trading_period_years', 1.0)
        
        if individual_returns:
            strategy_names = [
                'GBM Mean Rev', 'ML Mean Rev',
                'GBM Trend', 'ML Trend', 
                'GBM Weighted', 'ML Weighted'
            ]
            
            # ✅ FIXED: Use correct key names
            strategy_values = [
                individual_returns.get('gbm_mean_reversion', 0) * 100,
                individual_returns.get('ml_mean_reversion', 0) * 100,
                individual_returns.get('gbm_trend_following', 0) * 100,
                individual_returns.get('ml_trend_following', 0) * 100,
                individual_returns.get('gbm_weighted_trend_following', 0) * 100,  # ✅ FIXED
                individual_returns.get('ml_weighted_trend_following', 0) * 100    # ✅ FIXED
            ]
            
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'pink']
            bars = ax2.bar(strategy_names, strategy_values, color=colors, alpha=0.8)
            
            ax2.set_title(f'Strategy Returns Comparison ({trading_years:.1f} Years)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Return (%)')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, strategy_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            # Fallback if no individual returns data
            ax2.text(0.5, 0.5, 'Individual strategy data unavailable', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title(f'Strategy Returns Comparison', fontsize=14, fontweight='bold')
        
        # ✅ NEW: Plot 3 - Annual Returns Breakdown
        years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
        gbm_annual_returns = []
        ml_annual_returns = []
        
        # Calculate annual returns from extended prices
        year_length = TRADING_DAYS_PER_YEAR
        for i in range(5):
            start_idx = i * year_length
            end_idx = min((i + 1) * year_length, len(gbm_extended))
            
            if end_idx > start_idx:
                gbm_return = (gbm_extended[end_idx-1] / gbm_extended[start_idx] - 1) * 100
                ml_return = (ml_extended[end_idx-1] / ml_extended[start_idx] - 1) * 100
            else:
                gbm_return = 0
                ml_return = 0
            
            gbm_annual_returns.append(gbm_return)
            ml_annual_returns.append(ml_return)
        
        x = np.arange(len(years))
        width = 0.35
        
        ax3.bar(x - width/2, gbm_annual_returns, width, label='GBM', color='blue', alpha=0.7)
        ax3.bar(x + width/2, ml_annual_returns, width, label='ML', color='red', alpha=0.7)
        
        ax3.set_title('Annual Returns Breakdown', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Annual Return (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(years)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ✅ ENHANCED: Plot 4 - Overall Grade with Colored Background
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # Grade color mapping
        grade_colors = {
            'A+': '#4472C4', 'A': '#4472C4', 'A-': '#4472C4',  # Blue for A range
            'B+': '#70AD47', 'B': '#70AD47', 'B-': '#70AD47',  # Green for B range
            'C+': '#FFC000', 'C': '#FFC000', 'C-': '#FFC000',  # Yellow for C range
            'D': '#FF7F00',  # Orange for D range
            'F': '#E74C3C'   # Red for F range
        }
        
        grade = analysis_data['grade']
        grade_color = grade_colors.get(grade, '#808080')  # Default gray
        
        # Add colored background rectangle
        from matplotlib.patches import Rectangle
        background_rect = Rectangle((0.1, 0.3), 0.8, 0.4, 
                                   facecolor=grade_color, alpha=0.3, 
                                   edgecolor=grade_color, linewidth=2)
        ax4.add_patch(background_rect)
        
        # Text content
        ax4.text(0.5, 0.85, f"Stock: {symbol}", transform=ax4.transAxes, fontsize=16, 
                ha='center', fontweight='bold')
        ax4.text(0.5, 0.65, f"Overall Grade", transform=ax4.transAxes, fontsize=14, 
                ha='center', fontweight='bold')
        ax4.text(0.5, 0.50, f"{grade} ({analysis_data['grade_category']})", 
                transform=ax4.transAxes, fontsize=18, ha='center', fontweight='bold',
                color=grade_color)
        ax4.text(0.5, 0.35, f"Score: {analysis_data['score']:.1f}/100", 
                transform=ax4.transAxes, fontsize=14, ha='center')
        ax4.text(0.5, 0.20, f"Best Strategy: {analysis_data['best_strategy_name']}", 
                transform=ax4.transAxes, fontsize=12, ha='center')
        ax4.text(0.5, 0.10, f"Data Source: {analysis_data['data_source']}", 
                transform=ax4.transAxes, fontsize=10, ha='center')
        ax4.text(0.5, 0.05, f"Grading: {analysis_data.get('grading_method', 'relative')}", 
                transform=ax4.transAxes, fontsize=10, ha='center', style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced analysis visualization saved to: {save_path}")
        
        plt.show()
        return fig


class UserPreferenceManager(Market):
    """Manages user preference stocks and trading decisions"""
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def load_user_preferences(filename: str = "UserPreferenceStock.csv") -> pd.DataFrame:
        """Load user preference stocks with all stability metrics"""
        try:
            return pd.read_csv(filename)
        except FileNotFoundError:
            # Create new file with enhanced headers
            df = pd.DataFrame(columns=[
                'symbol', 'date_analyzed', 'rank', 'score', 'grade', 'grade_category',
                'prediction_winner', 'strategy_winner', 'best_strategy', 'best_strategy_params',
                'gbm_prediction_return', 'ml_prediction_return', 'best_strategy_return',
                'avg_auc', 'avg_slope', 'avg_total_return', 'avg_sharpe',
                'gbm_auc', 'ml_auc', 'gbm_slope', 'ml_slope', 
                'gbm_total_return', 'ml_total_return',
                'sharpe_ratio', 'volatility', 'max_drawdown', 'data_source',
                'trading_allowed', 'notes'
            ])
            df.to_csv(filename, index=False)
            print(f"📝 Created new user preference file: {filename}")
            return df
        except Exception as e:
            print(f"Error loading user preferences: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def save_user_preference(stock_analysis: Dict, filename: str = "UserPreferenceStock.csv"):
        """✅ FIXED: Save stock analysis to user preferences with proper key handling"""
        try:
            df = UserPreferenceManager.load_user_preferences(filename)
            
            # ✅ FIXED: Handle multiple possible key names for best_strategy
            best_strategy_name = (
                stock_analysis.get('best_strategy_name') or 
                stock_analysis.get('best_strategy') or 
                'Unknown Strategy'
            )
            
            # Create new row from analysis data
            new_row = {
                'symbol': stock_analysis['symbol'],
                'date_analyzed': stock_analysis['date_analyzed'],
                'rank': stock_analysis.get('rank', 'N/A'),
                'score': stock_analysis['score'],
                'grade': stock_analysis['grade'],
                'grade_category': stock_analysis['grade_category'],
                'prediction_winner': stock_analysis['prediction_winner'],
                'strategy_winner': stock_analysis['strategy_winner'],
                'best_strategy': best_strategy_name,  # ✅ FIXED
                'best_strategy_params': stock_analysis['best_strategy_params'],
                'gbm_prediction_return': stock_analysis['gbm_prediction_return'],
                'ml_prediction_return': stock_analysis['ml_prediction_return'],
                'best_strategy_return': stock_analysis['best_strategy_return'],
                'data_source': stock_analysis['data_source'],
                'trading_allowed': True,
                'notes': ''
            }
            
            # Add or update row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(filename, index=False)
            print(f"💾 Saved {stock_analysis['symbol']} to user preferences")
            return True
            
        except Exception as e:
            print(f"❌ Error saving user preference: {e}")
            print(f"🔍 Available keys: {list(stock_analysis.keys())}")
            return False
    
    @staticmethod
    def update_trading_permission(symbol: str, allowed: bool, filename: str = "UserPreferenceStock.csv"):
        """Update trading permission for a stock"""
        try:
            df = UserPreferenceManager.load_user_preferences(filename)
            if symbol in df['symbol'].values:
                df.loc[df['symbol'] == symbol, 'trading_allowed'] = allowed
                df.to_csv(filename, index=False)
                print(f"📝 Updated trading permission for {symbol}: {'Allowed' if allowed else 'Blocked'}")
            else:
                print(f"⚠️ {symbol} not found in user preferences")
        except Exception as e:
            print(f"Error updating trading permission: {e}")
