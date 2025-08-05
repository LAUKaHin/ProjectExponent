"""
SP500Analysis.py
S&P 500 bulk analysis and data management - ML-only system with absolute grading
"""
import pandas as pd
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# [SUCCESS] CLEAN: External imports first
from Market import Market

# [SUCCESS] CLEAN: Internal imports second - ML-only system
from .StockAnalyzer import ComprehensiveStockAnalyzer
from .GradingSystem import SimplifiedGradingSystem

class SP500ComprehensiveAnalyzer(Market):
    """[SUCCESS] FIXED: ML-only analyzer with absolute grading for S&P 500 stocks"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.results = []
        
        # [SUCCESS] FIXED: Use SimplifiedGradingSystem for absolute grading
        self.grader = SimplifiedGradingSystem()
        print("[INFO] Using ML-only system with absolute grading")
    
    def run_comprehensive_analysis(self, max_workers: int = 3, sample_size: Optional[int] = None):
        """[SUCCESS] FIXED: Single-pass ML-only analysis with absolute grading"""
        
        print("="*80)
        print("S&P 500 COMPREHENSIVE ANALYSIS - ML-ONLY WITH ABSOLUTE GRADING")
        print("="*80)
        print(f"[START] ML-only system with simplified absolute grading")
        
        # Get symbols
        symbols = self._get_sp500_symbols()
        if sample_size:
            symbols = symbols[:sample_size]
        
        # Single pass: Generate ML predictions and grade them
        all_results = self._analyze_all_stocks(symbols, max_workers)
        
        if not all_results:
            print("[ERROR] No successful analyses completed")
            return
        
        # Display and save
        self._display_results(all_results)
        self._save_results(all_results, "sp500_ml_only_results.csv")
        
        return all_results
    
    def _analyze_all_stocks(self, symbols: List[str], max_workers: int) -> List[Dict]:
        """[SUCCESS] FIXED: Analyze all stocks using ML-only system"""
        
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
                        print(f"[SUCCESS] {symbol}: Grade {result['grade']} (Score: {result['score']:.1f})")
                    else:
                        failed_symbols.append(symbol)
                        print(f"[ERROR] {symbol}: failed")
                except Exception as e:
                    print(f"[ERROR] {symbol}: error - {e}")
                    failed_symbols.append(symbol)
        
        print(f"\n[SUCCESS] Analysis Summary:")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_symbols)}")
        
        # Sort by score (highest first)
        successful_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add ranks
        for i, result in enumerate(successful_results, 1):
            result['rank'] = i
        
        return successful_results
    
    def analyze_single_stock(self, symbol: str) -> Dict:
        """[SUCCESS] FIXED: Analyze a single stock using ML-only system with absolute grading"""
        try:
            print(f"Analyzing {symbol}...")
            
            # [SUCCESS] FIXED: Use ComprehensiveStockAnalyzer (ML-only)
            analyzer = ComprehensiveStockAnalyzer(self.api_key, self.secret_key)
            result = analyzer.analyze_stock_comprehensive(symbol, show_visualization=False)
            
            if result:
                print(f"[SUCCESS] {symbol} completed successfully using {result.get('data_source', 'ML')}")
                return result
            else:
                print(f"[ERROR] {symbol} failed")
                return None
                
        except Exception as e:
            print(f"[ERROR] {symbol} error: {e}")
            return None
    
    def _display_results(self, all_results: List[Dict]):
        """[SUCCESS] FIXED: Display ML-only results with absolute grading"""
        
        print(f"\n{'='*80}")
        print("[INFO] S&P 500 ML-ONLY ANALYSIS WITH ABSOLUTE GRADING")
        print(f"{'='*80}")
        
        print(f"Total Stocks Analyzed: {len(all_results)}")
        
        # Group by grades
        excellent_stocks = [r for r in all_results if r['grade'] in ['A+', 'A', 'A-']]
        good_stocks = [r for r in all_results if r['grade'] in ['B+', 'B', 'B-']]
        satisfactory_stocks = [r for r in all_results if r['grade'] in ['C+', 'C', 'C-']]
        poor_stocks = [r for r in all_results if r['grade'] in ['D', 'F']]
        
        print(f"\n[GRADUATION] ABSOLUTE GRADE DISTRIBUTION:")
        print(f"   Excellent (A-grades): {len(excellent_stocks)} stocks ({len(excellent_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Good (B-grades): {len(good_stocks)} stocks ({len(good_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Satisfactory (C-grades): {len(satisfactory_stocks)} stocks ({len(satisfactory_stocks)/len(all_results)*100:.1f}%)")
        print(f"   Poor/Fail (D/F-grades): {len(poor_stocks)} stocks ({len(poor_stocks)/len(all_results)*100:.1f}%)")
        
        # Score statistics
        scores = [r['score'] for r in all_results]
        print(f"\n[INFO] SCORE STATISTICS:")
        print(f"   Range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"   Average: {sum(scores)/len(scores):.1f}")
        print(f"   Median: {sorted(scores)[len(scores)//2]:.1f}")
        
        # Display top performers
        if excellent_stocks:
            print(f"\n{'='*80}")
            print("[STAR] EXCELLENT PERFORMERS (A-GRADES)")
            print(f"{'='*80}")
            print(f"{'Rank':<4} {'Symbol':<8} {'Grade':<3} {'Score':<7} {'Strategy':<20} {'Return':<8}")
            print(f"{'-'*60}")
            
            for stock in excellent_stocks[:15]:  # Show top 15
                strategy_name = stock.get('best_strategy_name', 'Unknown')[:18]
                best_return = stock.get('best_strategy_return', 0) * 100
                
                print(f"{stock['rank']:<4} {stock['symbol']:<8} {stock['grade']:<3} {stock['score']:<7.1f} "
                      f"{strategy_name:<20} {best_return:<7.1f}%")
        
        # Investment recommendations
        print(f"\n{'='*70}")
        print("[TIP] INVESTMENT RECOMMENDATIONS (ABSOLUTE GRADING)")
        print(f"{'='*70}")
        
        print(f"EXCELLENT (A-grades): {len(excellent_stocks)} stocks - Strong buy candidates")
        print(f"GOOD (B-grades): {len(good_stocks)} stocks - Good investment options")
        print(f"SATISFACTORY (C-grades): {len(satisfactory_stocks)} stocks - Average performers")
        print(f"MARGINAL/FAIL (D/F-grades): {len(poor_stocks)} stocks - Avoid or short candidates")
        
        if excellent_stocks:
            print(f"\n[STAR] Top 5 ML-Only Performers:")
            for i, stock in enumerate(excellent_stocks[:5], 1):
                strategy_name = stock.get('best_strategy_name', 'Unknown')
                best_return = stock.get('best_strategy_return', 0) * 100
                
                print(f"  {i}. {stock['symbol']} (Grade: {stock['grade']}, Score: {stock['score']:.1f}, "
                      f"Strategy: {strategy_name}, Return: {best_return:.1f}%)")
        
        # ML-specific statistics
        print(f"\n[INFO] ML SYSTEM SUMMARY:")
        print(f"   Grading Method: Absolute (SimplifiedGradingSystem)")
        print(f"   Data Source: ML predictions only")
        print(f"   Best Score: {max(scores):.1f}/100")
        print(f"   Stocks with Score ≥80: {len([s for s in scores if s >= 80])}")
        print(f"   Stocks with Score ≥60: {len([s for s in scores if s >= 60])}")
  
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
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save results to CSV file"""
        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False)
            print(f"[SUCCESS] Results saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Error saving results: {e}")

    def analyze_top_performers(self, grade_threshold: str = 'B', max_stocks: int = 20):
        """[SUCCESS] NEW: Analyze only top-performing stocks for detailed review"""
        
        print(f"\n[TARGET] ANALYZING TOP PERFORMERS (Grade {grade_threshold}+ only)")
        print("="*60)
        
        # Define grade hierarchy
        grade_hierarchy = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']
        threshold_index = grade_hierarchy.index(grade_threshold)
        target_grades = grade_hierarchy[:threshold_index + 1]
        
        symbols = self._get_sp500_symbols()[:50]  # Start with top 50 S&P stocks
        
        top_performers = []
        
        for symbol in symbols:
            result = self.analyze_single_stock(symbol)
            if result and result['grade'] in target_grades:
                top_performers.append(result)
                print(f"[SUCCESS] {symbol}: {result['grade']} (Score: {result['score']:.1f}) - QUALIFIED")
                
                if len(top_performers) >= max_stocks:
                    break
            elif result:
                print(f"[INFO] {symbol}: {result['grade']} (Score: {result['score']:.1f}) - Below threshold")
        
        # Sort and rank
        top_performers.sort(key=lambda x: x['score'], reverse=True)
        for i, result in enumerate(top_performers, 1):
            result['rank'] = i
        
        print(f"\n[SUCCESS] FOUND {len(top_performers)} TOP PERFORMERS:")
        self._display_results(top_performers)
        
        return top_performers
