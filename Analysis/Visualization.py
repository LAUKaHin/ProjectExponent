"""
Visualization.py
Modern ML-only visualization engine with consistent data handling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from Market import Market
from Utils import TRADING_DAYS_PER_YEAR, round_to_decimals


class VisualizationEngine(Market):
    """Modern ML-only visualization engine with consistent data handling"""
    
    def __init__(self):
        super().__init__()
        # Set modern matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
    
    @staticmethod
    def _validate_analysis_data(analysis_data: Dict) -> Dict:
        """Validate and clean analysis data for consistent visualization"""
        required_keys = ['symbol', 'ml_market_prices', 'ml_stability', 'grade', 'score']
        
        for key in required_keys:
            if key not in analysis_data:
                raise ValueError(f"Missing required key: {key}")
        
        # Ensure ml_market_prices is a list of numbers
        prices = analysis_data.get('ml_market_prices', [])
        if not prices or not isinstance(prices, list):
            raise ValueError("ml_market_prices must be a non-empty list")
        
        # Clean and validate prices
        clean_prices = []
        for price in prices:
            try:
                clean_prices.append(float(price))
            except (ValueError, TypeError):
                continue
        
        if len(clean_prices) < 2:
            raise ValueError("Need at least 2 valid price points")
        
        analysis_data['ml_market_prices'] = clean_prices
        return analysis_data
    
    @staticmethod
    def _create_time_axis(num_days: int, years: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Create consistent time axis for all plots"""
        target_days = years * TRADING_DAYS_PER_YEAR
        
        if num_days < target_days:
            # Extend to target length
            days = np.arange(target_days)
        else:
            # Use actual length
            days = np.arange(num_days)
            
        years_axis = days / TRADING_DAYS_PER_YEAR
        return days, years_axis
    
    @staticmethod
    def _extend_prices_realistically(prices: List[float], target_length: int, 
                                   volatility: float = 0.02) -> List[float]:
        """Extend price series with realistic market behavior"""
        if len(prices) >= target_length:
            return prices[:target_length]
        
        extended = prices.copy()
        
        # Calculate realistic parameters from existing data
        if len(prices) > 20:
            recent_returns = [prices[i]/prices[i-1] - 1 for i in range(max(1, len(prices)-20), len(prices))]
            avg_return = np.mean(recent_returns)
            vol = np.std(recent_returns)
        else:
            avg_return = 0.0008  # ~20% annual
            vol = volatility
        
        # Extend with realistic market behavior
        np.random.seed(42)  # For reproducible results
        for i in range(len(prices), target_length):
            days_ahead = i - len(prices) + 1
            
            # Trend component (fades over time)
            trend = avg_return * (1 - 0.1 * (days_ahead / 252))
            
            # Random component
            random_component = np.random.normal(0, vol)
            
            # Mean reversion component
            if len(extended) > 252:
                long_term_avg = np.mean(extended[-252:])
                current_price = extended[-1]
                mean_reversion = 0.001 * (long_term_avg - current_price) / current_price
            else:
                mean_reversion = 0
            
            # Combine components
            total_return = trend + random_component + mean_reversion
            next_price = extended[-1] * (1 + np.clip(total_return, -0.1, 0.1))
            
            # Keep price reasonable
            next_price = max(next_price, prices[0] * 0.2)
            next_price = min(next_price, prices[0] * 5.0)
            
            extended.append(round_to_decimals(next_price, 2))
        
        return extended
    
    @staticmethod
    def plot_price_prediction(ax, analysis_data: Dict) -> None:
        """Plot ML price prediction with proper scaling"""
        prices = analysis_data['ml_market_prices']
        symbol = analysis_data['symbol']
        
        # Extend prices to 5 years
        target_days = 5 * TRADING_DAYS_PER_YEAR
        extended_prices = VisualizationEngine._extend_prices_realistically(prices, target_days)
        
        # Create time axis
        days, years = VisualizationEngine._create_time_axis(len(extended_prices))
        
        # Plot price line
        ax.plot(years, extended_prices, 'b-', linewidth=2.5, label='ML Prediction', alpha=0.8)
        
        # Add year markers
        for year in range(1, 6):
            if year <= max(years):
                ax.axvline(x=year, color='gray', linestyle='--', alpha=0.3)
                ax.text(year, max(extended_prices) * 0.95, f'Y{year}', 
                       rotation=90, verticalalignment='top', alpha=0.6, fontsize=8)
        
        # Final price annotation
        final_price = extended_prices[-1]
        initial_price = extended_prices[0]
        total_return = (final_price / initial_price - 1) * 100
        
        ax.annotate(f'${final_price:.2f}\n({total_return:+.1f}%)', 
                   xy=(years[-1], final_price), xytext=(years[-1] - 0.5, final_price),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                   fontsize=9, ha='center')
        
        ax.set_title(f'{symbol} - 5-Year ML Price Prediction', fontweight='bold')
        ax.set_xlabel('Years')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def plot_strategy_comparison(ax, analysis_data: Dict) -> None:
        """Plot ML strategy returns with consistent data"""
        individual_returns = analysis_data.get('individual_strategy_returns', {})
        trading_years = analysis_data.get('trading_period_years', 1.0)
        
        if not individual_returns:
            ax.text(0.5, 0.5, 'Strategy data unavailable', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('ML Strategy Returns Comparison', fontweight='bold')
            return
        
        # Extract ML strategy returns
        strategy_names = ['Mean Reversion', 'Trend Following', 'Weighted Trend']
        strategy_values = [
            individual_returns.get('ml_mean_reversion', 0) * 100,
            individual_returns.get('ml_trend_following', 0) * 100,
            individual_returns.get('ml_weighted_trend_following', 0) * 100
        ]
        
        # Use consistent colors
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Create bars
        bars = ax.bar(strategy_names, strategy_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, strategy_values):
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (max(strategy_values) * 0.02) if height >= 0 else height - (abs(min(strategy_values)) * 0.02),
                       f'{value:.1f}%', ha='center', 
                       va='bottom' if height >= 0 else 'top', fontsize=9, fontweight='bold')
        
        # Set dynamic y-axis limits
        if strategy_values:
            max_val = max(strategy_values)
            min_val = min(strategy_values)
            range_padding = max(10, (max_val - min_val) * 0.2)
            ax.set_ylim(min_val - range_padding, max_val + range_padding)
        
        ax.set_title(f'ML Strategy Returns ({trading_years:.1f} Years)', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels to prevent overlap
        ax.tick_params(axis='x', rotation=15)
    
    @staticmethod
    def plot_performance_metrics(ax, analysis_data: Dict) -> None:
        """Plot ML performance metrics with Slope instead of Total Return"""
        ml_stability = analysis_data.get('ml_stability', {})
        
        # Define metrics to display - replaced Total Return with Slope
        metrics = ['Slope', 'Sharpe Ratio', 'AUC Score']
        values = [
            ml_stability.get('slope', 0),  # Use slope instead of total_return
            ml_stability.get('sharpe_ratio', 0),
            ml_stability.get('auc', 0) / 1000  # Scale AUC for better visualization
        ]
        
        # Create horizontal bar chart for better readability
        colors = ['#4CAF50', '#FF9800', '#9C27B0']
        bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels
        for bar, value, metric in zip(bars, values, metrics):
            width = bar.get_width()
            if metric == 'Slope':
                label = f'{value:.3f}'  # Show slope with 3 decimal places
            elif metric == 'AUC Score':
                label = f'{value:.1f}k'
            else:
                label = f'{value:.2f}'
            
            ax.text(width + max(abs(v) for v in values) * 0.05, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_title('ML Performance Metrics', fontweight='bold')
        ax.set_xlabel('Values')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout to handle potentially negative slope values
        max_abs_value = max(abs(v) for v in values) if values else 1
        ax.set_xlim(-max_abs_value * 0.1, max_abs_value * 1.3)
    
    @staticmethod
    def plot_grade_summary(ax, analysis_data: Dict) -> None:
        """Plot grade summary with colored background"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Grade color mapping
        grade_colors = {
            'A+': '#1B5E20', 'A': '#2E7D32', 'A-': '#388E3C',  # Dark green for A range
            'B+': '#1565C0', 'B': '#1976D2', 'B-': '#1E88E5',  # Blue for B range
            'C+': '#EF6C00', 'C': '#F57C00', 'C-': '#FF8F00',  # Orange for C range
            'D': '#D84315',  # Red-orange for D
            'F': '#C62828'   # Red for F
        }
        
        symbol = analysis_data['symbol']
        grade = analysis_data['grade']
        grade_category = analysis_data.get('grade_category', 'Unknown')
        score = analysis_data['score']
        data_source = analysis_data.get('data_source', 'Unknown')
        grading_method = analysis_data.get('grading_method', 'simplified_absolute')
        
        grade_color = grade_colors.get(grade, '#757575')  # Default gray
        
        # Add colored background
        background_rect = patches.Rectangle((0.1, 0.2), 0.8, 0.6, 
                                          facecolor=grade_color, alpha=0.2, 
                                          edgecolor=grade_color, linewidth=2)
        ax.add_patch(background_rect)
        
        # Add text content with proper spacing
        ax.text(0.5, 0.85, f'{symbol}', transform=ax.transAxes, fontsize=16, 
               ha='center', fontweight='bold')
        
        ax.text(0.5, 0.70, 'Final Grade', transform=ax.transAxes, fontsize=12, 
               ha='center', fontweight='bold', alpha=0.8)
        
        ax.text(0.5, 0.50, f'{grade}', transform=ax.transAxes, fontsize=24, 
               ha='center', fontweight='bold', color=grade_color)
        
        ax.text(0.5, 0.35, f'{grade_category}', transform=ax.transAxes, fontsize=11, 
               ha='center', style='italic')
        
        ax.text(0.5, 0.25, f'Score: {score:.1f}/100', transform=ax.transAxes, fontsize=11, 
               ha='center', fontweight='bold')
        
        ax.text(0.5, 0.10, f'Data: {data_source}', transform=ax.transAxes, fontsize=9, 
               ha='center', alpha=0.7)
        
        ax.text(0.5, 0.05, f'Method: {grading_method}', transform=ax.transAxes, fontsize=8, 
               ha='center', alpha=0.6, style='italic')
    
    @staticmethod
    def plot_comprehensive_analysis(symbol: str, analysis_data: Dict, save_path: str = None) -> plt.Figure:
        """Create comprehensive ML-only analysis visualization"""
        
        # Validate data first
        try:
            analysis_data = VisualizationEngine._validate_analysis_data(analysis_data)
        except ValueError as e:
            print(f"❌ Data validation error: {e}")
            return None
        
        # Create figure with proper layout
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'{symbol} - Comprehensive ML Stock Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Create subplots with custom layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                             left=0.08, right=0.95, top=0.90, bottom=0.08)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Price prediction
        ax2 = fig.add_subplot(gs[0, 1])  # Strategy comparison
        ax3 = fig.add_subplot(gs[1, 0])  # Performance metrics
        ax4 = fig.add_subplot(gs[1, 1])  # Grade summary
        
        try:
            # Plot each component
            VisualizationEngine.plot_price_prediction(ax1, analysis_data)
            VisualizationEngine.plot_strategy_comparison(ax2, analysis_data)
            VisualizationEngine.plot_performance_metrics(ax3, analysis_data)
            VisualizationEngine.plot_grade_summary(ax4, analysis_data)
            
            # Add timestamp
            fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.6)
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"📊 Comprehensive analysis saved to: {save_path}")
            
            plt.show()
            return fig
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            plt.close(fig)
            return None
    
    @staticmethod
    def plot_enhanced_comprehensive_analysis(symbol: str, analysis_data: Dict, save_path: str = None) -> plt.Figure:
        """Enhanced version with additional features - delegates to main function"""
        return VisualizationEngine.plot_comprehensive_analysis(symbol, analysis_data, save_path)


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
                'ml_prediction_return', 'best_strategy_return',
                'avg_auc', 'avg_slope', 'avg_total_return', 'avg_sharpe',
                'ml_auc', 'ml_slope', 'ml_total_return',
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
    def save_user_preference(stock_analysis: Dict, filename: str = "UserPreferenceStock.csv") -> bool:
        """Save stock analysis to user preferences with ML-only data"""
        try:
            df = UserPreferenceManager.load_user_preferences(filename)
            
            # Safe key extraction with fallbacks
            def safe_get(key, fallback='N/A'):
                return stock_analysis.get(key, fallback)
            
            # Handle multiple possible key names for best_strategy
            best_strategy_name = (
                safe_get('best_strategy_name') or 
                safe_get('best_strategy') or 
                safe_get('strategy_winner') or
                'Unknown Strategy'
            )
            
            # Create new row with ML-only data
            new_row = {
                'symbol': safe_get('symbol', 'UNKNOWN'),
                'date_analyzed': safe_get('date_analyzed', datetime.now().strftime('%Y-%m-%d')),
                'rank': safe_get('rank', 'N/A'),
                'score': safe_get('score', 0.0),
                'grade': safe_get('grade', 'F'),
                'grade_category': safe_get('grade_category', 'Unknown'),
                'prediction_winner': safe_get('prediction_winner', safe_get('winner', 'ML')),
                'strategy_winner': safe_get('strategy_winner', 'ML'),
                'best_strategy': best_strategy_name,
                'best_strategy_params': safe_get('best_strategy_params', 'N/A'),
                'ml_prediction_return': safe_get('ml_prediction_return', 0.0),
                'best_strategy_return': safe_get('best_strategy_return', 0.0),
                'data_source': safe_get('data_source', 'Unknown'),
                'trading_allowed': True,
                'notes': ''
            }
            
            # Add optional ML-only stability metrics if available
            stability_keys = ['avg_auc', 'avg_slope', 'avg_total_return', 'avg_sharpe',
                            'ml_auc', 'ml_slope', 'ml_total_return',
                            'sharpe_ratio', 'volatility', 'max_drawdown']
            
            for key in stability_keys:
                if key in stock_analysis:
                    new_row[key] = stock_analysis[key]
                elif key in df.columns:
                    new_row[key] = 0.0  # Default value for missing metrics
            
            # Remove duplicates if symbol already exists
            df = df[df['symbol'] != new_row['symbol']]
            
            # Add new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(filename, index=False)
            print(f"💾 Saved {new_row['symbol']} to user preferences")
            return True
            
        except Exception as e:
            print(f"❌ Error saving user preference: {e}")
            print(f"🔍 Available keys: {list(stock_analysis.keys()) if stock_analysis else 'None'}")
            return False
    
    @staticmethod
    def update_trading_permission(symbol: str, allowed: bool, filename: str = "UserPreferenceStock.csv") -> None:
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
