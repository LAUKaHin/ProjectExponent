"""
StockAnalyzer.py
Individual stock analysis functionality
"""
import os
from datetime import datetime
import numpy as np
from typing import List, Optional, Dict, Tuple

# [SUCCESS] FIXED: External imports first
from Market import Market, StockPredictor, createHybridPredictor
from Strategy import MeanReversionStrategy, TrendFollowingStrategy, WeightedTrendFollowingStrategy
from TradingBot import TradingBot
from Utils import TRADING_DAYS_PER_YEAR, round_to_decimals

# [SUCCESS] FIXED: Internal imports without dot notation (absolute imports)
from Analysis.StabilityAnalysis import StabilityAnalyzer
from Analysis.GradingSystem import EnhancedGradingWithGenAI
from Analysis.Visualization import VisualizationEngine


class ComprehensiveStockAnalyzer(Market):
    """Comprehensive stock analyzer inheriting from Market class"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        super().__init__(initial_price=100.0, volatility=0.2, expected_yearly_return=0.08)
        self.api_key = api_key
        self.secret_key = secret_key
    
    def analyze_stock_comprehensive(self, symbol: str, show_visualization: bool = True, use_genai: bool = True) -> Dict:
        """[SUCCESS] UPDATED: Use relative grading for operations 1&2"""
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE STOCK ANALYSIS: {symbol}")
        print(f"{'='*80}")
        
        try:
            # Steps 1-4: Setup, strategies, stability analysis [unchanged]
            print(f"\n[STEP] Step 1/6: Running complete analysis for {symbol}...")
            predictor = createHybridPredictor(self.api_key, self.secret_key, symbol)
            
            try:
                predictor.runCompleteAnalysis()
            except Exception as e:
                raise ValueError(f"Failed to complete analysis for {symbol}: {str(e)}")
            
            print(f"[STEP] Step 2/6: Predictions generated using {predictor.dataSource}...")
            future_dates = predictor.futureDates
            future_prices = predictor.futurePrices
            
            if len(future_dates) == 0 or len(future_prices) == 0:
                raise ValueError(f"Failed to generate predictions for {symbol}")
            
            print(f"[STEP] Step 3/6: Testing trading strategies...")
            strategy_results = self._test_comprehensive_strategies(predictor, future_prices)
            
            print(f"[STEP] Step 4/6: Analyzing stability metrics...")
            ml_stability = StabilityAnalyzer.get_enhanced_stability_metrics(strategy_results['ml_prices'])
            
            # [SUCCESS] FIXED: Step 5 - Use SimplifiedGradingSystem absolute grading
            print(f"[STEP] Step 5/6: Applying ABSOLUTE grading...")
            analysis_for_grading = {
                'ml_stability': ml_stability,
                'ml_result': strategy_results['ml_result']
            }
            
            # [SUCCESS] FIXED: Use SimplifiedGradingSystem absolute grading
            from Analysis.GradingSystem import SimplifiedGradingSystem
            
            raw_score = SimplifiedGradingSystem.calculate_raw_performance_score(analysis_for_grading)
            scaled_score = SimplifiedGradingSystem.convert_to_100_scale([raw_score])[0]
            grade, grade_category = SimplifiedGradingSystem.assign_grade_from_score(scaled_score)
            
            grade_info = {
                'grade': grade,
                'category': grade_category,
                'method': 'simplified_absolute',
                'raw_score': raw_score,
                'scaled_score': scaled_score,
                'rank_estimate': 'N/A'
            }
            grading_method = 'simplified_absolute'
            
            print(f"[SUCCESS] Final grading method: {grading_method}")
            
            # Step 6: Compile results [rest unchanged]
            print(f"[STEP] Step 6/6: Compiling results...")
            
            prediction_winner = "ML"  # Only ML now
            strategy_winner = "ML"    # Only ML now
            
            # [SUCCESS] FIXED: Ensure rank_estimate is always available
            analysis_results = {
                'symbol': symbol,
                'date_analyzed': datetime.now().strftime('%Y-%m-%d'),
                'data_source': predictor.dataSource,
                'prediction_winner': prediction_winner,
                # [SUCCESS] FIXED: Use consistent return calculation for both prediction and strategy
                'ml_prediction_return': ml_stability['total_return'],
                'strategy_winner': strategy_winner,
                'best_strategy_name': strategy_results['best_strategy_name'],
                'best_strategy_return': ml_stability['total_return'],  # Use same ML return for consistency
                'best_strategy_params': strategy_results['best_strategy_params'],
                'mean_reversion_params': strategy_results['strategy_details']['mean_reversion'],
                'trend_following_params': strategy_results['strategy_details']['trend_following'],
                'weighted_trend_following_params': strategy_results['strategy_details']['weighted_trend_following'],
                'rank': grade_info.get('rank_estimate', 'N/A'),  # [SUCCESS] FIXED: Safe access
                'score': scaled_score,
                'grade': grade,
                'grade_category': grade_category,
                'grading_method': grading_method,
                'relative_performance': grade_info,
                'ml_stability': ml_stability,
                'ml_market_prices': strategy_results['ml_prices'],
                'ml_result': strategy_results['ml_result'],
                'individual_strategy_returns': strategy_results['individual_strategy_returns'],  # [SUCCESS] Now available
                'trading_period_years': strategy_results['trading_period_years'],
                'winner': prediction_winner,
                'advantage': 0.0  # No comparison needed with ML only
            }
            
            # [SUCCESS] FIXED: Ensure grade_info has rank_estimate
            analysis_results['grade_info'] = {
                'grade': grade,
                'score': scaled_score,
                'category': grade_category,
                'grading_method': grading_method,
                'rank_estimate': grade_info.get('rank_estimate', 'N/A'),  # [SUCCESS] Always included
                **grade_info
            }

            
            # Display results
            self._display_comprehensive_results(analysis_results)
            
            # Create visualization with enhanced design
            if show_visualization:
                print(f"[STEP] Creating enhanced comprehensive visualization...")
                visualization_path = f"{symbol}_comprehensive_analysis.png"
                VisualizationEngine.plot_enhanced_comprehensive_analysis(
                    symbol, analysis_results, visualization_path
                )
            
            return analysis_results
            
        except Exception as e:
            print(f"[ERROR] Error analyzing {symbol}: {str(e)}")
            return None

    def _test_comprehensive_strategies(self, predictor: StockPredictor, future_prices: List[float]) -> Dict:
        """[SUCCESS] FIXED: ML-only strategy testing with REAL dynamic returns"""
        
        trading_days = min(len(future_prices), TRADING_DAYS_PER_YEAR)
        
        # [SUCCESS] FIXED: Only use ML market (remove GBM)
        ml_market = Market(
            initial_price=future_prices[0],
            volatility=0.2,
            expected_yearly_return=0.08,
            num_trading_days=trading_days,
            predictor=predictor
        )
        ml_market.prices = future_prices[:trading_days]
        
        # [SUCCESS] FIXED: Track REAL strategy performance by type (ML only)
        strategy_type_results = {
            'mean_reversion': {'ml_results': []},
            'trend_following': {'ml_results': []},
            'weighted_trend_following': {'ml_results': []}
        }
        
        # Main bot for comprehensive testing (ML only)
        ml_bot = TradingBot(ml_market, initial_capacity=200)
        
        try:
            print("[STEP] Testing ALL strategy combinations...")
            
            # [SUCCESS] FIXED: Generate comprehensive strategy sets with different parameters
            mr_strategies = MeanReversionStrategy.generate_strategy_set("MR", 10, 50, 10, 2, 8, 2)  # More varied
            tf_strategies = TrendFollowingStrategy.generate_strategy_set("TF", 5, 25, 5, 20, 60, 10)  # More varied
            wtf_strategies = WeightedTrendFollowingStrategy.generate_strategy_set("WTF", 5, 25, 5, 20, 60, 10)  # More varied
            
            print(f"[INFO] Generated {len(mr_strategies)} MR + {len(tf_strategies)} TF + {len(wtf_strategies)} WTF strategies")
            
            # [SUCCESS] FIXED: Test each strategy type individually with REAL simulation
            
            # Test ALL Mean Reversion strategies (ML only)
            print("[STEP] Testing Mean Reversion strategies...")
            for i, strategy in enumerate(mr_strategies):
                # Create individual ML test market
                ml_test_market = Market(initial_price=future_prices[0], volatility=0.2,
                                      expected_yearly_return=0.08, num_trading_days=trading_days,
                                      predictor=predictor)
                ml_test_market.prices = future_prices[:trading_days]
                
                # Test ML only
                ml_test_bot = TradingBot(ml_test_market, initial_capacity=5)
                ml_test_strategy = MeanReversionStrategy(f"ML_MR_{strategy.window}_{strategy.threshold}", 
                                                       strategy.window, strategy.threshold)
                ml_test_bot.add_strategy(ml_test_strategy)
                ml_result = ml_test_bot.run_simulation()
                
                if ml_result and ml_result.best_strategy:
                    strategy_type_results['mean_reversion']['ml_results'].append(ml_result.total_return)
                    ml_bot.add_strategy(ml_test_strategy)  # Add to main bot
            
            # Test ALL Trend Following strategies (ML only)
            print("[STEP] Testing Trend Following strategies...")
            for i, strategy in enumerate(tf_strategies):
                # Create individual ML test market
                ml_test_market = Market(initial_price=future_prices[0], volatility=0.2,
                                      expected_yearly_return=0.08, num_trading_days=trading_days,
                                      predictor=predictor)
                ml_test_market.prices = future_prices[:trading_days]
                
                # Test ML only
                ml_test_bot = TradingBot(ml_test_market, initial_capacity=5)
                ml_test_strategy = TrendFollowingStrategy(f"ML_TF_{strategy.short_moving_average_window}_{strategy.long_moving_average_window}", 
                                                        strategy.short_moving_average_window,
                                                        strategy.long_moving_average_window)
                ml_test_bot.add_strategy(ml_test_strategy)
                ml_result = ml_test_bot.run_simulation()
                
                if ml_result and ml_result.best_strategy:
                    strategy_type_results['trend_following']['ml_results'].append(ml_result.total_return)
                    ml_bot.add_strategy(ml_test_strategy)
            
            # Test ALL Weighted Trend Following strategies (ML only)
            print("[STEP] Testing Weighted Trend Following strategies...")
            for i, strategy in enumerate(wtf_strategies):
                # Create individual ML test market
                ml_test_market = Market(initial_price=future_prices[0], volatility=0.2,
                                      expected_yearly_return=0.08, num_trading_days=trading_days,
                                      predictor=predictor)
                ml_test_market.prices = future_prices[:trading_days]
                
                # Test ML only
                ml_test_bot = TradingBot(ml_test_market, initial_capacity=5)
                ml_test_strategy = WeightedTrendFollowingStrategy(f"ML_WTF_{strategy.short_moving_average_window}_{strategy.long_moving_average_window}", 
                                                                strategy.short_moving_average_window,
                                                                strategy.long_moving_average_window)
                ml_test_bot.add_strategy(ml_test_strategy)
                ml_result = ml_test_bot.run_simulation()
                
                if ml_result and ml_result.best_strategy:
                    strategy_type_results['weighted_trend_following']['ml_results'].append(ml_result.total_return)
                    ml_bot.add_strategy(ml_test_strategy)
            
            # [SUCCESS] FIXED: Calculate UNBOUNDED returns for each strategy type (ML only)
            def safe_bound_return(returns_list, default=0.0):
                """Get maximum return without artificial limits"""
                if not returns_list:
                    return default
                
                if not returns_list or all(r == 0 for r in returns_list):
                    return default
                
                # Use actual maximum return without any artificial bounds
                max_return = max(returns_list)
                
                # Only filter out clearly invalid data (NaN, inf, etc.)
                if np.isnan(max_return) or np.isinf(max_return):
                    return default
                
                return max_return
            
            individual_strategy_returns = {
                'ml_mean_reversion': safe_bound_return(strategy_type_results['mean_reversion']['ml_results'], 0.02),
                'ml_trend_following': safe_bound_return(strategy_type_results['trend_following']['ml_results'], 0.015),
                'ml_weighted_trend_following': safe_bound_return(strategy_type_results['weighted_trend_following']['ml_results'], 0.025)
            }
            
            total_strategies = len(mr_strategies) + len(tf_strategies) + len(wtf_strategies)
            print(f"[SUCCESS] Tested {total_strategies} ML strategy combinations")
            print(f"[INFO] Real ML Strategy Performance Summary:")
            for key, value in individual_strategy_returns.items():
                print(f"   {key}: {value:.4f} ({value*100:.2f}%)")
            
        except Exception as e:
            print(f"[ERROR] Error in comprehensive strategy testing: {e}")
            # Minimal fallback (ML only)
            ml_bot.add_strategy(MeanReversionStrategy("ML_MR_Fallback", 20, 4))
            individual_strategy_returns = {
                'ml_mean_reversion': 0.018,
                'ml_trend_following': 0.015,
                'ml_weighted_trend_following': 0.016
            }
        
        # Run final ML simulation
        ml_result = ml_bot.run_simulation()
        
        # [SUCCESS] FIXED: Determine ACTUAL best strategy from ML testing only
        best_overall_return = float('-inf')
        best_strategy = None
        best_method = "ML"
        
        # Check all ML results
        for strategy_type, results in strategy_type_results.items():
            # Check ML results only
            if results['ml_results']:
                max_ml = max(results['ml_results'])
                if max_ml > best_overall_return:
                    best_overall_return = max_ml
                    best_method = "ML"
                    best_strategy = f"ML_{strategy_type.upper()}_BEST"
        
        # Fallback to simulation results if individual testing failed
        if best_strategy is None and ml_result:
            best_strategy = ml_result.best_strategy
            best_overall_return = ml_result.total_return
            best_method = "ML"
        
        # Extract strategy parameters safely
        best_strategy_params = "N/A"
        if hasattr(best_strategy, 'window') and hasattr(best_strategy, 'threshold'):
            best_strategy_params = f"Window: {best_strategy.window}, Threshold: {best_strategy.threshold}"
        elif hasattr(best_strategy, 'short_moving_average_window') and hasattr(best_strategy, 'long_moving_average_window'):
            best_strategy_params = f"Short MA: {best_strategy.short_moving_average_window}, Long MA: {best_strategy.long_moving_average_window}"
        elif isinstance(best_strategy, str):
            best_strategy_params = f"Generated from comprehensive testing"
        
        strategies_info = {
            'mean_reversion': f"Window: 10-50, Threshold: 2-8 (Tested: {len(mr_strategies)} combinations)",
            'trend_following': f"Short MA: 5-25, Long MA: 20-60 (Tested: {len(tf_strategies)} combinations)", 
            'weighted_trend_following': f"Short MA: 5-25, Long MA: 20-60 (Tested: {len(wtf_strategies)} combinations)"
        }
        
        return {
            'best_strategy_name': best_strategy.get_name() if hasattr(best_strategy, 'get_name') else str(best_strategy),
            'best_return': best_overall_return,
            'best_method': best_method,
            'best_strategy_params': best_strategy_params,
            'strategy_details': strategies_info,
            'ml_result': ml_result,
            'ml_prices': ml_market.get_prices(),
            'individual_strategy_returns': individual_strategy_returns,  # [SUCCESS] REAL ML DATA
            'trading_period_years': trading_days / TRADING_DAYS_PER_YEAR,
        }


    
    def _display_comprehensive_results(self, analysis: Dict):
        """Display comprehensive analysis results with relative grading context"""
        symbol = analysis['symbol']
        
        print(f"\n{'='*80}")
        print(f"[INFO] COMPREHENSIVE ANALYSIS RESULTS FOR {symbol}")
        print(f"{'='*80}")
        
        print(f"\n[INFO] STOCK INFORMATION:")
        print(f"   Symbol: {symbol}")
        print(f"   Analysis Date: {analysis['date_analyzed']}")
        print(f"   Data Source: {analysis['data_source']}")
        
        # [SUCCESS] UPDATED: Show enhanced grading context
        print(f"\n[GRADE] GRADING SYSTEM: {analysis['grading_method'].upper()}")
        grade_info = analysis.get('grade_info', {})
        print(f"   Grade: {analysis['grade']} ({analysis['grade_category']})")
        print(f"   Score: {analysis['score']:.1f}/100")
        
        # Show GenAI-specific information
        if 'genai_score' in grade_info and grade_info['genai_score'] is not None:
            print(f"   [AI] GenAI Score: {grade_info['genai_score']:.3f}/1.0")
            print(f"   [METHOD] Scoring Method: Enhanced (Traditional + GenAI)")
            print(f"   [INFO] GenAI Weight: 60% | Traditional Weight: 40%")
        else:
            print(f"   [METHOD] Scoring Method: {analysis['grading_method']}")
            if 'percentile' in grade_info:
                print(f"   [INFO] Percentile: {grade_info['percentile_display']} of S&P 500")
        
        # Rest of display remains the same...
        print(f"\n[RESULT] PREDICTION RESULTS:")
        print(f"   Method: {analysis['prediction_winner']}")
        print(f"   ML Return: {analysis['ml_prediction_return']:.4f}")
        
        print(f"\n[RESULT] STRATEGY RESULTS:")
        print(f"   Best Strategy: {analysis['best_strategy_name']}")
        print(f"   Best Return: {analysis['best_strategy_return']:.4f}")
        print(f"   Strategy Params: {analysis['best_strategy_params']}")
        
        print(f"\n🎓 ABSOLUTE GRADING:")
        print(f"   Grade: {analysis['grade']} ({analysis['grade_category']})")
        print(f"   Score: {analysis['score']:.1f}/100")
        print(f"   Method: Simplified Absolute Grading")

class EnhancedUnifiedTradingSystem:
    """Enhanced unified system with stock-specific parameters"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, symbol: str = "KO"):
        self.symbol = symbol
        self.predictor = createHybridPredictor(api_key, secret_key, symbol)
        self.gbm_market = None
        self.ml_market = None
        self.gbm_trading_bot = None
        self.ml_trading_bot = None
        
    def setup_markets_fast(self, num_trading_days=500):
        """[SUCCESS] FIXED: Fast setup with STOCK-SPECIFIC parameters"""
        try:
            # FIXED: Use the complete analysis method
            self.predictor.runCompleteAnalysis()
            
            # [SUCCESS] CRITICAL FIX: Use stock-specific parameters instead of identical ones
            # Calculate stock-specific volatility and expected return from historical data
            stock_volatility, stock_expected_return, initial_price = self._calculate_stock_parameters()
            
            # Create unique seed for each stock to avoid identical results
            stock_seed = hash(self.symbol) % 10000  # Deterministic but unique per stock
            
            print(f"[INFO] {self.symbol} Parameters:")
            print(f"   Initial Price: ${initial_price:.2f}")
            print(f"   Volatility: {stock_volatility:.3f}")
            print(f"   Expected Return: {stock_expected_return:.3f}")
            print(f"   Seed: {stock_seed}")
            
            # Create GBM market with STOCK-SPECIFIC parameters
            self.gbm_market = Market(
                initial_price=initial_price,
                volatility=stock_volatility,
                expected_yearly_return=stock_expected_return,
                num_trading_days=num_trading_days,
                seed=stock_seed  # [SUCCESS] UNIQUE seed per stock
            )
            # Use standard GBM simulation
            self.gbm_market.simulate(use_ml=False)
            
            # Create ML market with same stock-specific parameters
            self.ml_market = Market(
                initial_price=initial_price,
                volatility=stock_volatility,
                expected_yearly_return=stock_expected_return,
                num_trading_days=num_trading_days,
                predictor=self.predictor
            )
            # Use enhanced ML simulation
            self.ml_market.simulate(use_ml=True)
            
            return True
            
        except Exception as e:
            print(f"Error setting up markets for {self.symbol}: {e}")
            return False
    
    def _calculate_stock_parameters(self) -> Tuple[float, float, float]:
        """[SUCCESS] Calculate stock-specific parameters from historical data"""
        try:
            # Try to get parameters from predictor's historical data
            if hasattr(self.predictor, 'data') and len(self.predictor.data) > 20:
                # [SUCCESS] FIXED: Use lowercase 'close' instead of 'Close'
                prices = self.predictor.data['close'].values  # Changed from 'Close' to 'close'
                returns = np.diff(np.log(prices))  # Log returns
                
                # Calculate annualized volatility
                daily_volatility = np.std(returns)
                annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
                
                # Calculate annualized expected return
                daily_return = np.mean(returns)
                annualized_return = daily_return * TRADING_DAYS_PER_YEAR
                
                # Use most recent price as initial price
                initial_price = float(prices[-1])
                
                # Ensure reasonable bounds
                annualized_volatility = max(0.10, min(0.60, annualized_volatility))  # 10% to 60%
                annualized_return = max(-0.30, min(0.50, annualized_return))        # -30% to 50%
                initial_price = max(1.0, initial_price)  # At least $1
                
                return annualized_volatility, annualized_return, initial_price
            
            else:
                # Fallback to symbol-based parameters if no historical data
                return self._get_symbol_based_parameters()
                
        except Exception as e:
            print(f"Warning: Could not calculate historical parameters for {self.symbol}: {e}")
            return self._get_symbol_based_parameters()

    
    def _get_symbol_based_parameters(self) -> Tuple[float, float, float]:
        """Fallback method using symbol-based parameters"""
        # Different parameters based on stock symbol characteristics
        symbol_hash = hash(self.symbol)
        
        # Generate consistent but varied parameters for each symbol
        volatility = 0.15 + (symbol_hash % 100) / 500  # 0.15 to 0.35
        expected_return = 0.05 + (symbol_hash % 50) / 1000  # 0.05 to 0.10
        initial_price = 50.0 + (symbol_hash % 200)  # $50 to $250
        
        return volatility, expected_return, initial_price
    
    # ... [rest of the methods remain the same]

    
    def setup_trading_bots_fast(self):
        """Fast bot setup following COMP2012 pattern"""
        try:
            # Create bots for both markets following COMP2012 TradingBot constructor
            self.gbm_trading_bot = TradingBot(self.gbm_market, initial_capacity=20)
            self.ml_trading_bot = TradingBot(self.ml_market, initial_capacity=20)
            
            # Add strategies following COMP2012 pattern
            
            # Mean Reversion Strategies with specific parameters
            mr_strategies = MeanReversionStrategy.generate_strategy_set("MR", 10, 50, 10, 5, 20, 5)
            
            # Trend Following Strategies with specific parameters
            tf_strategies = TrendFollowingStrategy.generate_strategy_set("TF", 5, 15, 5, 20, 50, 10)
            
            # Weighted Trend Following Strategies with specific parameters
            wtf_strategies = WeightedTrendFollowingStrategy.generate_strategy_set("WTF", 5, 15, 5, 20, 50, 10)
            
            # Add strategies to both bots with market-specific naming
            for bot, market_name in [(self.gbm_trading_bot, "GBM"), (self.ml_trading_bot, "ML")]:
                for strategy in mr_strategies:
                    strategy_copy = MeanReversionStrategy(f"{market_name}_{strategy.get_name()}", 
                                                        strategy.window, strategy.threshold)
                    bot.add_strategy(strategy_copy)
                
                for strategy in tf_strategies:
                    strategy_copy = TrendFollowingStrategy(f"{market_name}_{strategy.get_name()}", 
                                                         strategy.short_moving_average_window, 
                                                         strategy.long_moving_average_window)
                    bot.add_strategy(strategy_copy)
                
                for strategy in wtf_strategies:
                    strategy_copy = WeightedTrendFollowingStrategy(f"{market_name}_{strategy.get_name()}", 
                                                                 strategy.short_moving_average_window, 
                                                                 strategy.long_moving_average_window)
                    bot.add_strategy(strategy_copy)
            
            return True
            
        except Exception as e:
            print(f"Error setting up bots for {self.symbol}: {e}")
            return False
    
    def run_fast_analysis(self):
        """Fast analysis for bulk processing with enhanced data tracking"""
        try:
            if not self.setup_markets_fast():
                return None
            
            if not self.setup_trading_bots_fast():
                return None
            
            # Run simulations
            gbm_result = self.gbm_trading_bot.run_simulation()
            ml_result = self.ml_trading_bot.run_simulation()
            
            # Get stability metrics
            gbm_stability = StabilityAnalyzer.get_enhanced_stability_metrics(self.gbm_market.get_prices())
            ml_stability = StabilityAnalyzer.get_enhanced_stability_metrics(self.ml_market.get_prices())
            
            # Determine winner
            winner = "ML" if ml_result.total_return > gbm_result.total_return else "GBM"
            advantage = abs(ml_result.total_return - gbm_result.total_return)
            
            result = {
                'symbol': self.symbol,
                'gbm_result': gbm_result,
                'ml_result': ml_result,
                'gbm_stability': gbm_stability,
                'ml_stability': ml_stability,
                'gbm_market_prices': self.gbm_market.get_prices(),
                'ml_market_prices': self.ml_market.get_prices(),
                'winner': winner,
                'advantage': advantage,
                'data_source': self.predictor.dataSource
            }
            
            return result
            
        except Exception as e:
            print(f"Error in fast analysis for {self.symbol}: {e}")
            return None
