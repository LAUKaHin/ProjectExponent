"""
SP500Analysis.py
S&P 500 bulk analysis and data management
"""
import pandas as pd
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ✅ CLEAN: External imports first
from Market import Market

# ✅ CLEAN: Internal imports second
from .StockAnalyzer import EnhancedUnifiedTradingSystem
from .GradingSystem import HKUSTStrictPercentileSystem, EnhancedGradingWithGenAI

class SP500ComprehensiveAnalyzer(Market):
    """Two-pass analyzer that calculates global benchmarks first with GenAI grading"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.results = []
        self.global_benchmarks = {}
        
        # ✅ NEW: Initialize GenAI grader
        self.enhanced_grader = None
        self.use_genai = False
        
        # Check if GenAI is available
        try:
            from .GradingSystem import EnhancedGradingWithGenAI
            import os
            
            openrouter_key = os.getenv('OPENROUTER_API_KEY')
            if openrouter_key:
                self.enhanced_grader = EnhancedGradingWithGenAI(openrouter_key)
                self.use_genai = True
                print("🤖 GenAI grading ENABLED for S&P 500 analysis")
            else:
                print("⚠️  OPENROUTER_API_KEY not found - using traditional grading")
                
        except Exception as e:
            print(f"⚠️  GenAI grading unavailable: {e}")
            print("📊 Falling back to traditional HKUST grading")
    
    def run_comprehensive_analysis(self, max_workers: int = 3, sample_size: Optional[int] = None):
        """Two-pass analysis with global benchmarks and ENHANCED GenAI grading"""
        
        print("="*80)
        print("S&P 500 COMPREHENSIVE ANALYSIS WITH OPTIMIZED GENAI GRADING")
        print("="*80)
        print(f"🚀 OPTIMIZATION: Single-batch processing with 10k token limit")
        print(f"🤖 GenAI Enhanced Grading: {'ENABLED (OPTIMIZED)' if self.use_genai else 'DISABLED'}")
        
        # Get symbols
        symbols = self._get_sp500_symbols()
        if sample_size:
            symbols = symbols[:sample_size]
        
        # Pass 1: Generate predictions
        all_raw_results = self._generate_all_predictions(symbols, max_workers)
        
        if not all_raw_results:
            print("❌ No successful predictions generated")
            return
        
        # Pass 2: Calculate global benchmarks
        self.global_benchmarks = self._calculate_global_benchmarks(all_raw_results)
        
        # Pass 3: Score and grade with GenAI enhancement
        processed_results = self._score_against_global_benchmarks(all_raw_results)
        
        # Display and save
        self._display_enhanced_results(processed_results)
        self._save_results(processed_results, "sp500_enhanced_yfinance_results.csv")
        
        return processed_results
    
    def _assign_final_grades(self, raw_scores: List[float], symbols: List[str], 
                   grading_method: str = "enhanced") -> List[Dict]:
        """✅ UPDATED: Apply STRICT percentile grading system with safety checks"""
        
        if self.use_genai and grading_method == "enhanced":
            print(f"\n🤖 Applying ENHANCED GRADING with GenAI for {len(symbols)} stocks...")
            
            # Get GenAI evaluations for all symbols
            genai_scores = self.enhanced_grader.genai_evaluator.evaluate_multiple_stocks(symbols)
            
            # Calculate enhanced scores for each stock
            enhanced_scores = []
            for i, (symbol, raw_score) in enumerate(zip(symbols, raw_scores)):
                genai_score = genai_scores.get(symbol, None)
                
                if genai_score is not None:
                    enhanced_score = 0.08 * raw_score + 0.6 * (genai_score * 100)
                    enhanced_score = max(0, min(100, enhanced_score))
                else:
                    enhanced_score = raw_score
                
                enhanced_scores.append(enhanced_score)
            
            final_scores = enhanced_scores
            scoring_method = "enhanced_with_genai"
            
        else:
            print(f"\n🎓 Applying Traditional HKUST Grading System...")
            final_scores = raw_scores
            scoring_method = "traditional_hkust"
        
        # ✅ APPLY STRICT percentile grading to ensure proper distribution
        grades = EnhancedGradingWithGenAI.apply_strict_percentile_grading_to_scores(final_scores)
        
        # ✅ FIXED: Add safety checks for each grade record
        for i, grade_info in enumerate(grades):
            if grade_info is None:  # ✅ SAFETY CHECK
                print(f"⚠️  Warning: Grade info is None for index {i}")
                continue
                
            symbol = symbols[i]
            grade_info['scoring_method'] = scoring_method
            grade_info['original_score'] = raw_scores[i]
            
            if self.use_genai and symbol in genai_scores:
                grade_info['genai_score'] = genai_scores[symbol]
                grade_info['genai_available'] = True
            else:
                grade_info['genai_score'] = None
                grade_info['genai_available'] = False
        
        return grades

    def _score_against_global_benchmarks(self, all_results: List[Dict]) -> List[Dict]:
        """✅ ENHANCED: Score ALL stocks with GenAI integration"""
        
        print("📊 Scoring all stocks against fixed global benchmarks...")
        
        # Calculate raw scores for all stocks
        raw_scores = []
        symbols = []
        detailed_scores = []
        
        for result in all_results:
            symbols.append(result['symbol'])
            
            # Calculate averages
            avg_return = (result['gbm_stability']['total_return'] + 
                         result['ml_stability']['total_return']) / 2
            avg_auc = (result['gbm_stability']['auc'] + 
                      result['ml_stability']['auc']) / 2
            avg_slope = (result['gbm_stability']['slope'] + 
                        result['ml_stability']['slope']) / 2
            avg_sharpe = (result['gbm_stability']['sharpe_ratio'] + 
                         result['ml_stability']['sharpe_ratio']) / 2
            
            # Score against FIXED global benchmarks
            return_score = (avg_return / self.global_benchmarks['best_return']) * 100
            auc_score = (avg_auc / self.global_benchmarks['best_auc']) * 100
            slope_score = (avg_slope / self.global_benchmarks['best_slope']) * 100
            sharpe_score = (avg_sharpe / self.global_benchmarks['best_sharpe']) * 100
            
            # Ensure no score exceeds 100
            return_score = min(100, max(0, return_score))
            auc_score = min(100, max(0, auc_score))
            slope_score = min(100, max(0, slope_score))
            sharpe_score = min(100, max(0, sharpe_score))
            
            # Traditional weighted final score
            raw_score = (0.40 * return_score + 0.20 * auc_score + 
                        0.20 * slope_score + 0.20 * sharpe_score)
            
            raw_scores.append(raw_score)
            detailed_scores.append({
                'symbol': result['symbol'],
                'return_score': return_score,
                'auc_score': auc_score,
                'slope_score': slope_score,
                'sharpe_score': sharpe_score,
                'raw_score': raw_score,
                'avg_return': avg_return,
                'avg_auc': avg_auc,
                'avg_slope': avg_slope,
                'avg_sharpe': avg_sharpe
            })
        
        print(f"✅ Traditional Score Summary:")
        print(f"   Range: {min(raw_scores):.1f} - {max(raw_scores):.1f}")
        print(f"   Best score: {max(raw_scores):.1f}")
        
        # ✅ ENHANCED: Apply GenAI-enhanced grading
        grades = self._assign_final_grades(raw_scores, symbols, "enhanced" if self.use_genai else "traditional")
        
        # Create final processed results
        processed_results = []
        for i, result in enumerate(all_results):
            grade_info = grades[i]
            
            # Extract strategy info
            best_strategy = (result['gbm_result'].best_strategy if result['winner'] == 'GBM' 
                           else result['ml_result'].best_strategy)
            best_return = (result['gbm_result'].total_return if result['winner'] == 'GBM'
                          else result['ml_result'].total_return)
            
            strategy_params = "N/A"
            if hasattr(best_strategy, 'window') and hasattr(best_strategy, 'threshold'):
                strategy_params = f"Window: {best_strategy.window}, Threshold: {best_strategy.threshold}"
            elif hasattr(best_strategy, 'short_moving_average_window') and hasattr(best_strategy, 'long_moving_average_window'):
                strategy_params = f"Short MA: {best_strategy.short_moving_average_window}, Long MA: {best_strategy.long_moving_average_window}"
            
            processed_result = {
                'symbol': result['symbol'],
                'rank': grade_info['rank'],
                'score': grade_info['scaled_score'],
                'grade': grade_info['grade'],
                'grade_category': grade_info['grade_category'],
                
                # ✅ NEW: GenAI grading information
                'scoring_method': grade_info['scoring_method'],
                'original_score': grade_info['original_score'],
                'genai_score': grade_info.get('genai_score', None),
                'genai_available': grade_info.get('genai_available', False),
                
                # Component scores for verification
                'return_score': detailed_scores[i]['return_score'],
                'auc_score': detailed_scores[i]['auc_score'],
                'slope_score': detailed_scores[i]['slope_score'],
                'sharpe_score': detailed_scores[i]['sharpe_score'],
                
                # Raw metric values
                'avg_return': detailed_scores[i]['avg_return'],
                'avg_auc': detailed_scores[i]['avg_auc'],
                'avg_slope': detailed_scores[i]['avg_slope'],
                'avg_sharpe': detailed_scores[i]['avg_sharpe'],
                
                # Individual GBM and ML values
                'gbm_auc': result['gbm_stability']['auc'],
                'ml_auc': result['ml_stability']['auc'],
                'gbm_slope': result['gbm_stability']['slope'],
                'ml_slope': result['ml_stability']['slope'],
                
                # Original data
                'prediction_winner': result['winner'],
                'gbm_prediction_return': result['gbm_stability']['total_return'],
                'ml_prediction_return': result['ml_stability']['total_return'],
                'best_strategy': best_strategy.get_name(),
                'best_strategy_params': strategy_params,
                'best_strategy_return': best_return,
                'strategy_winner': result['winner'],
                'sharpe_ratio': max(result['gbm_stability']['sharpe_ratio'], 
                                  result['ml_stability']['sharpe_ratio']),
                'max_drawdown': min(result['gbm_stability']['max_drawdown'], 
                                  result['ml_stability']['max_drawdown']),
                'volatility': min(result['gbm_stability']['volatility'], 
                                result['ml_stability']['volatility']),
                'data_source': result.get('data_source', 'unknown')
            }
            
            processed_results.append(processed_result)
        
        # Sort by rank
        processed_results.sort(key=lambda x: x['rank'])
        return processed_results
    
    def _display_enhanced_results(self, all_results: List[Dict]):
        """✅ ENHANCED: Display results with GenAI grading information and safe error handling"""
        
        print(f"\n{'='*80}")
        print("🤖 S&P 500 ANALYSIS WITH ENHANCED GENAI GRADING")
        print(f"{'='*80}")
        
        print(f"Total Stocks Analyzed: {len(all_results)}")
        
        # ✅ ENHANCED: Safe GenAI statistics calculation
        genai_available_results = [r for r in all_results if r.get('genai_available', False)]
        valid_genai_scores = [r['genai_score'] for r in genai_available_results 
                             if r.get('genai_score') is not None]
        
        if len(genai_available_results) > 0:
            print(f"🤖 GenAI Enhanced: {len(genai_available_results)}/{len(all_results)} stocks ({len(genai_available_results)/len(all_results)*100:.1f}%)")
            
            if valid_genai_scores:
                print(f"🤖 GenAI Score Range: {min(valid_genai_scores):.3f} - {max(valid_genai_scores):.3f}")
                print(f"🤖 Valid GenAI Scores: {len(valid_genai_scores)}/{len(genai_available_results)}")
            else:
                print("⚠️  GenAI Enhancement: Available but no valid scores returned")
        else:
            print("⚠️  GenAI Enhancement: Not available")
        
        # Group by grades using HKUST categories
        excellent_stocks = [r for r in all_results if r['grade'] in ['A+', 'A', 'A-']]
        good_stocks = [r for r in all_results if r['grade'] in ['B+', 'B', 'B-']]
        satisfactory_stocks = [r for r in all_results if r['grade'] in ['C+', 'C', 'C-']]
        poor_stocks = [r for r in all_results if r['grade'] in ['D', 'F']]
        
        print(f"\n🎓 ENHANCED GRADE DISTRIBUTION:")
        print(f"   Excellent (A-grades): {len(excellent_stocks)} stocks ({len(excellent_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Good (B-grades): {len(good_stocks)} stocks ({len(good_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Satisfactory (C-grades): {len(satisfactory_stocks)} stocks ({len(satisfactory_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Poor/Fail (D/F-grades): {len(poor_stocks)} stocks ({len(poor_stocks)/len(all_results)*100:.1f}%)")
        
        # Display top performers with GenAI information
        if excellent_stocks:
            print(f"\n{'='*80}")
            print("🌟 EXCELLENT PERFORMERS (A-GRADES) WITH GENAI ENHANCEMENT")
            print(f"{'='*80}")
            print(f"{'Rank':<4} {'Symbol':<8} {'Grade':<3} {'Score':<7} {'GenAI':<8} {'Method':<15}")
            print(f"{'-'*55}")
            
            for stock in excellent_stocks[:15]:  # Show top 15
                genai_score = stock.get('genai_score', None)
                genai_display = f"{genai_score:.3f}" if genai_score is not None else "N/A"
                method = "Enhanced" if stock.get('genai_available', False) and genai_score is not None else "Traditional"
                
                print(f"{stock['rank']:<4} {stock['symbol']:<8} {stock['grade']:<3} {stock['score']:<7.1f} "
                      f"{genai_display:<8} {method:<15}")
        
        # ✅ ENHANCED: Investment recommendations with GenAI context
        print(f"\n{'='*70}")
        print("🤖 GENAI-ENHANCED INVESTMENT RECOMMENDATIONS")
        print(f"{'='*70}")
        
        # Show GenAI-enhanced vs traditional breakdown
        genai_enhanced_excellent = [r for r in excellent_stocks if r.get('genai_available', False) and r.get('genai_score') is not None]
        traditional_excellent = [r for r in excellent_stocks if not (r.get('genai_available', False) and r.get('genai_score') is not None)]
        
        print(f"EXCELLENT (A-grades): {len(excellent_stocks)} stocks")
        if genai_enhanced_excellent:
            print(f"  • GenAI Enhanced: {len(genai_enhanced_excellent)} stocks")
        if traditional_excellent:
            print(f"  • Traditional Grading: {len(traditional_excellent)} stocks")
        
        print(f"GOOD (B-grades): {len(good_stocks)} stocks - Above average performers")
        print(f"SATISFACTORY (C-grades): {len(satisfactory_stocks)} stocks - Average performers")
        print(f"MARGINAL/FAIL (D/F-grades): {len(poor_stocks)} stocks - Below average performers")
        
        if excellent_stocks:
            print(f"\n🌟 Top 5 Enhanced Performers:")
            for i, stock in enumerate(excellent_stocks[:5], 1):
                genai_score = stock.get('genai_score', None)
                if genai_score is not None:
                    genai_info = f" (GenAI: {genai_score:.3f})"
                    enhancement = " 🤖"
                else:
                    genai_info = " (Traditional)"
                    enhancement = ""
                
                print(f"  {i}. {stock['symbol']} (Grade: {stock['grade']}, Score: {stock['score']:.1f}){genai_info}{enhancement}")
        
        # ✅ NEW: Show GenAI statistics summary
        if valid_genai_scores:
            print(f"\n📊 GENAI ENHANCEMENT SUMMARY:")
            print(f"   Successfully Enhanced: {len(valid_genai_scores)} stocks")
            print(f"   Average GenAI Score: {sum(valid_genai_scores)/len(valid_genai_scores):.3f}")
            print(f"   GenAI Score Distribution:")
            
            # Score distribution
            high_genai = len([s for s in valid_genai_scores if s >= 0.8])
            medium_genai = len([s for s in valid_genai_scores if 0.6 <= s < 0.8])
            low_genai = len([s for s in valid_genai_scores if s < 0.6])
            
            print(f"     High (≥0.8): {high_genai} stocks ({high_genai/len(valid_genai_scores)*100:.1f}%)")
            print(f"     Medium (0.6-0.8): {medium_genai} stocks ({medium_genai/len(valid_genai_scores)*100:.1f}%)")
            print(f"     Low (<0.6): {low_genai} stocks ({low_genai/len(valid_genai_scores)*100:.1f}%)")
  
    def _get_sp500_symbols(self):
        """Get S&P 500 symbols from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            cleaned_symbols = []
            for symbol in symbols:
                cleaned_symbol = str(symbol).replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)
            
            return cleaned_symbols
        except Exception as e:
            print(f"Error fetching S&P 500 symbols: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    def analyze_single_stock(self, symbol: str) -> Dict:
        """Analyze a single stock using enhanced system with yfinance"""
        try:
            print(f"Analyzing {symbol}...")
            
            system = EnhancedUnifiedTradingSystem(self.api_key, self.secret_key, symbol)
            result = system.run_fast_analysis()
            
            if result:
                print(f"✓ {symbol} completed successfully using {system.predictor.dataSource}")
                return result
            else:
                print(f"✗ {symbol} failed")
                return None
                
        except Exception as e:
            print(f"✗ {symbol} error: {e}")
            return None
    
    def _generate_all_predictions(self, symbols: List[str], max_workers: int) -> List[Dict]:
        """✅ PASS 1: Generate predictions for all stocks without scoring"""
        
        successful_results = []
        failed_symbols = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_single_stock, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=240)
                    if result:
                        successful_results.append(result)
                        print(f"✓ {symbol}: predictions generated")
                    else:
                        failed_symbols.append(symbol)
                        print(f"✗ {symbol}: failed")
                except Exception as e:
                    print(f"✗ {symbol}: error - {e}")
                    failed_symbols.append(symbol)
        
        print(f"\n✅ Prediction Summary:")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_symbols)}")
        
        return successful_results
    
    def _calculate_global_benchmarks(self, all_results: List[Dict]) -> Dict:
        """✅ PASS 2: Calculate GLOBAL benchmarks from ALL results"""
        
        print("📊 Calculating global benchmarks from all results...")
        
        # Extract all performance metrics
        all_avg_returns = []
        all_avg_aucs = []
        all_avg_slopes = []
        all_avg_sharpes = []
        
        for result in all_results:
            # Calculate averages for each stock
            avg_return = (result['gbm_stability']['total_return'] + 
                         result['ml_stability']['total_return']) / 2
            avg_auc = (result['gbm_stability']['auc'] + 
                      result['ml_stability']['auc']) / 2  
            avg_slope = (result['gbm_stability']['slope'] + 
                        result['ml_stability']['slope']) / 2
            avg_sharpe = (result['gbm_stability']['sharpe_ratio'] + 
                         result['ml_stability']['sharpe_ratio']) / 2
            
            all_avg_returns.append(avg_return)
            all_avg_aucs.append(avg_auc)
            all_avg_slopes.append(avg_slope)
            all_avg_sharpes.append(avg_sharpe)
        
        # ✅ Calculate GLOBAL maximums (these are the true benchmarks)
        global_benchmarks = {
            'best_return': max(all_avg_returns),
            'best_auc': max(all_avg_aucs),
            'best_slope': max(all_avg_slopes),
            'best_sharpe': max(all_avg_sharpes)
        }
        
        # Ensure no benchmark is zero
        for key, value in global_benchmarks.items():
            if value <= 0:
                global_benchmarks[key] = 0.0001  # Small positive value
        
        print(f"✅ GLOBAL BENCHMARKS (FIXED):")
        print(f"   Best Return: {global_benchmarks['best_return']:.6f}")
        print(f"   Best AUC: {global_benchmarks['best_auc']:.2f}")
        print(f"   Best Slope: {global_benchmarks['best_slope']:.6f}")
        print(f"   Best Sharpe: {global_benchmarks['best_sharpe']:.4f}")
        
        return global_benchmarks
    
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            print(f"✅ Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")