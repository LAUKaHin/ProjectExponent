"""
GradingSystem.py
Essential grading systems: simplified, HKUST percentile, and GenAI-enhanced grading
"""
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

from Market import Market

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Genesis import Genesis
except ImportError:
    print("⚠️  Genesis.py not found. GenAI scoring will be disabled.")
    Genesis = None

class SimplifiedGradingSystem(Market):
    """Simplified grading system inheriting from Market class"""
    
    # Grade configuration following academic grading
    GRADE_CONFIG = [
        ('A+', 95, 'Excellent Performance'),
        ('A',  90, 'Excellent Performance'),
        ('A-', 85, 'Excellent Performance'),
        ('B+', 80, 'Good Performance'),
        ('B',  75, 'Good Performance'),
        ('B-', 70, 'Good Performance'),
        ('C+', 60, 'Satisfactory Performance'),
        ('C',  55, 'Satisfactory Performance'),
        ('C-', 50, 'Satisfactory Performance'),
        ('D',  40, 'Marginal Pass'),
        ('F',   0, 'Fail')
    ]
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def load_sp500_benchmarks(filename: str = "sp500_enhanced_yfinance_results.csv") -> Optional[Dict]:
        """Load benchmarks using new CSV structure with AUC and slope"""
        try:
            df = pd.read_csv(filename)
            if df.empty:
                print("⚠️  S&P 500 results file is empty")
                return None
            
            print(f"📊 Available columns in CSV: {list(df.columns)}")
            
            benchmarks = {}
            
            # Use average return column
            if 'avg_return' in df.columns:
                benchmarks['best_return'] = df['avg_return'].max()
            elif 'gbm_prediction_return' in df.columns and 'ml_prediction_return' in df.columns:
                avg_returns = (df['gbm_prediction_return'] + df['ml_prediction_return']) / 2
                benchmarks['best_return'] = avg_returns.max()
            else:
                print("⚠️  Return columns not found, using fallback")
                return None
            
            # Use actual AUC and slope values
            if 'avg_auc' in df.columns:
                benchmarks['best_auc'] = df['avg_auc'].max()
            else:
                if 'gbm_auc' in df.columns and 'ml_auc' in df.columns:
                    avg_aucs = (df['gbm_auc'] + df['ml_auc']) / 2
                    benchmarks['best_auc'] = avg_aucs.max()
                else:
                    benchmarks['best_auc'] = benchmarks['best_return'] * 1000
            
            if 'avg_slope' in df.columns:
                benchmarks['best_slope'] = df['avg_slope'].max()
            else:
                if 'gbm_slope' in df.columns and 'ml_slope' in df.columns:
                    avg_slopes = (df['gbm_slope'] + df['ml_slope']) / 2
                    benchmarks['best_slope'] = avg_slopes.max()
                else:
                    benchmarks['best_slope'] = benchmarks['best_return'] * 0.01
            
            if 'avg_sharpe' in df.columns:
                benchmarks['best_sharpe'] = df['avg_sharpe'].max()
            elif 'sharpe_ratio' in df.columns:
                benchmarks['best_sharpe'] = df['sharpe_ratio'].max()
            else:
                print("⚠️  Sharpe ratio column not found, using fallback")
                return None
            
            # Ensure no benchmark is zero or negative
            for key, value in benchmarks.items():
                if value <= 0:
                    benchmarks[key] = 0.01
            
            print(f"📊 Benchmarks loaded:")
            for key, value in benchmarks.items():
                print(f"   {key}: {value:.6f}")
            
            return benchmarks
            
        except Exception as e:
            print(f"❌ Error loading S&P 500 benchmarks: {e}")
            return None

    @staticmethod
    def calculate_raw_performance_score(result: Dict) -> float:
        """Calculate raw performance score using benchmarks"""
        try:
            # Calculate averages
            gbm_return = result['gbm_stability']['total_return']
            ml_return = result['ml_stability']['total_return']
            currAvg_return = (gbm_return + ml_return) / 2
            
            gbm_auc = result['gbm_stability']['auc']
            ml_auc = result['ml_stability']['auc']
            currAvg_auc = (gbm_auc + ml_auc) / 2
            
            gbm_slope = result['gbm_stability']['slope']
            ml_slope = result['ml_stability']['slope']
            currAvg_slope = (gbm_slope + ml_slope) / 2
            
            gbm_sharpe = result['gbm_stability']['sharpe_ratio']
            ml_sharpe = result['ml_stability']['sharpe_ratio']
            currAvg_sharpe = (gbm_sharpe + ml_sharpe) / 2
            
            # Try to load benchmarks
            benchmarks = SimplifiedGradingSystem.load_sp500_benchmarks()
            
            if benchmarks is not None:
                # Direct ratio scoring
                return_score = (currAvg_return / benchmarks['best_return']) * 100
                auc_score = (currAvg_auc / benchmarks['best_auc']) * 100  
                slope_score = (currAvg_slope / benchmarks['best_slope']) * 100
                sharpe_score = (currAvg_sharpe / benchmarks['best_sharpe']) * 100
                
                # Weighted combination
                raw_score = (0.40 * return_score + 0.20 * auc_score + 0.20 * slope_score + 0.20 * sharpe_score)
                
                if raw_score > 100:
                    raw_score = 100.0
                
                return raw_score
            else:
                # Fallback method
                return_score = max(0, min(100, currAvg_return * 500))
                auc_score = max(0, min(100, abs(currAvg_auc) / 100))
                slope_score = max(0, min(100, currAvg_slope * 5000))
                sharpe_score = max(0, min(100, currAvg_sharpe * 50))
                
                raw_score = (0.40 * return_score + 0.20 * auc_score + 0.20 * slope_score + 0.20 * sharpe_score)
                return raw_score
            
        except Exception as e:
            print(f"❌ Error calculating raw score: {e}")
            return 20.0

    @staticmethod
    def convert_to_100_scale(raw_scores: List[float]) -> List[float]:
        """Convert raw scores to 100-point scale"""
        final_scores = []
        for score in raw_scores:
            final_score = max(0, min(100, score))
            final_scores.append(final_score)
        return final_scores
    
    @staticmethod
    def assign_grade_from_score(score: float) -> Tuple[str, str]:
        """Assign grade from score"""
        for grade, min_score, category in SimplifiedGradingSystem.GRADE_CONFIG:
            if score >= min_score:
                return grade, category
        return 'F', 'Fail'

class HKUSTStrictPercentileSystem(Market):
    """Strict percentile-based grading with exact distributions"""
    
    STRICT_DISTRIBUTION = [
        ('A+', 0.05), ('A',  0.10), ('A-', 0.10), ('B+', 0.10), ('B',  0.15),
        ('B-', 0.15), ('C+', 0.10), ('C',  0.10), ('C-', 0.05), ('D',  0.05), ('F',  0.05)
    ]
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def assign_strict_percentile_grades(scores: List[float]) -> List[Dict]:
        """
        Assign strict percentile grades to a list of scores
        
        Args:
            scores: List of numerical scores
            
        Returns:
            List of dictionaries with grade information
        """
        if not scores:
            return []
        
        # Sort scores in descending order with original indices
        indexed_scores = [(score, i) for i, score in enumerate(scores)]
        indexed_scores.sort(reverse=True, key=lambda x: x[0])
        
        total_count = len(scores)
        results = [None] * total_count
        
        # HKUST strict percentile boundaries
        grade_boundaries = {
            'A+': 0.05,   # Top 5%
            'A':  0.15,   # Next 10% (5-15%)
            'A-': 0.25,   # Next 10% (15-25%)
            'B+': 0.40,   # Next 15% (25-40%)
            'B':  0.55,   # Next 15% (40-55%)
            'B-': 0.70,   # Next 15% (55-70%)
            'C+': 0.80,   # Next 10% (70-80%)
            'C':  0.90,   # Next 10% (80-90%)
            'C-': 0.95,   # Next 5% (90-95%)
            'D':  1.00    # Bottom 5% (95-100%)
        }
        
        # Grade categories
        grade_categories = {
            'A+': 'Excellent', 'A': 'Excellent', 'A-': 'Very Good',
            'B+': 'Good', 'B': 'Good', 'B-': 'Satisfactory',
            'C+': 'Satisfactory', 'C': 'Pass', 'C-': 'Pass',
            'D': 'Marginal Pass'
        }
        
        # Assign grades based on percentile position
        for rank, (score, original_index) in enumerate(indexed_scores):
            percentile = (rank + 1) / total_count
            
            # Find appropriate grade
            grade = 'F'  # Default
            for grade_level, boundary in grade_boundaries.items():
                if percentile <= boundary:
                    grade = grade_level
                    break
            
            category = grade_categories.get(grade, 'Fail')
            
            results[original_index] = {
                'rank': rank + 1,
                'scaled_score': score,
                'grade': grade,
                'grade_category': category,
                'percentile': percentile,
                'z_score': 0.0  # Will be calculated later if needed
            }
        
        return results

class GenAIStockEvaluator(Market):
    """GenAI-powered stock evaluation system using Genesis"""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.genesis_instance = None
        
        if not self.api_key:
            print("⚠️  No OpenRouter API key provided. GenAI evaluation disabled.")
        elif Genesis is None:
            print("⚠️  Genesis module not available. GenAI evaluation disabled.")
        else:
            self._initialize_genesis()
    
    def _initialize_genesis(self):
        """Initialize Genesis instance with stock evaluation system prompt"""
        try:
            self.genesis_instance = Genesis(
                key=self.api_key,
                httpRef="",
                projTitle="ProjectExponent"
            )
            
            # Set comprehensive system prompt
            system_prompt = self._get_stock_evaluation_system_prompt()
            self.genesis_instance.PushMsgToSystem(system_prompt)
            
        except Exception as e:
            print(f"❌ Failed to initialize Genesis: {e}")
            self.genesis_instance = None
    
    def _get_stock_evaluation_system_prompt(self) -> str:
        """System prompt optimized for large batch processing"""
        return """You are a professional stock analysis AI that provides numerical investment scores for large batches.
    
    **CRITICAL INSTRUCTIONS FOR LARGE BATCHES:**
    1. You MUST respond ONLY with a single JSON object containing ALL stock symbols and their scores
    2. Each score MUST be between 0.0 and 1.0 (where 1.0 = excellent investment, 0.0 = avoid)
    3. NO explanations, reasoning, or additional text - ONLY the complete JSON response
    4. Format: {"SYMBOL1": 0.xx, "SYMBOL2": 0.yy, "SYMBOL3": 0.zz, ...}
    5. INCLUDE ALL SYMBOLS from the input - do not truncate or omit any symbols
    6. Use proper stock ticker symbols exactly as provided in the input
    
    **LARGE BATCH PROCESSING:**
    - Process ALL symbols in the input list
    - Maintain consistent JSON formatting throughout
    - Ensure the JSON is complete and properly closed
    - Each symbol should appear exactly once in the response
    
    **STOCK SYMBOL RECOGNITION:**
    - Use symbols exactly as provided (e.g., "BRK-B", not "BRK.B")
    - Maintain case sensitivity as requested
    - Handle special characters in symbols correctly
    
    **EVALUATION CRITERIA (Apply consistently to all symbols):**
    - Financial health and stability (revenue growth, profit margins, debt levels)
    - Market position and competitive advantages
    - Innovation and future growth prospects  
    - Dividend history and yield
    - ESG factors and sustainability
    - Current market valuation vs intrinsic value
    - Economic and industry trends impact
    - Management quality and corporate governance
    - Recent news and events affecting the company
    - Technical analysis indicators
    
    **SCORING GUIDELINES (0.0 to 1.0):**
    - 0.9-1.0: Exceptional investment opportunity (strong buy)
    - 0.8-0.89: Very good investment (buy)
    - 0.7-0.79: Good investment (buy/hold)
    - 0.6-0.69: Decent investment (hold)
    - 0.5-0.59: Average investment (hold/neutral)
    - 0.4-0.49: Below average (cautious/sell)
    - 0.3-0.39: Poor investment (sell)
    - 0.2-0.29: Very poor investment (strong sell)
    - 0.1-0.19: Terrible investment (avoid)
    - 0.0-0.09: Extremely poor/distressed (avoid completely)
    
    **EXAMPLE FOR LARGE BATCH:**
    Input: "AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, ..."
    Response: {"AAPL": 0.85, "MSFT": 0.82, "GOOGL": 0.78, "AMZN": 0.88, "TSLA": 0.68, "META": 0.75, "NVDA": 0.89, ...}
    
    **CRITICAL:** Ensure the JSON response includes ALL symbols and is properly formatted and complete."""

    def evaluate_single_stock(self, symbol: str) -> Optional[float]:
        """Evaluate a single stock using GenAI"""
        if not self.genesis_instance:
            print("⚠️  GenAI evaluation not available")
            return None
        
        try:
            # Clear previous user content
            self.genesis_instance.userContents.clear()
            
            # Send stock symbol for evaluation
            self.genesis_instance.PushMsgToUser("text", symbol.upper())
            
            # OPTIMIZED: 10k token limit for Gemini 2.5 Flash
            response = self.genesis_instance.TXRX(
                LLM="google/gemini-2.5-flash-preview-05-20",
                provider=["google-ai-studio", "google-vertex"],
                max_tokens=10000,
                temperature=0.3
            )
            
            if not response:
                print(f"❌ No response from GenAI for {symbol}")
                return None
            
            # Parse JSON response
            score = self._parse_single_stock_response(response, symbol)
            
            if score is not None:
                print(f"🤖 GenAI Score for {symbol}: {score:.3f}")
            
            return score
            
        except Exception as e:
            print(f"❌ Error evaluating {symbol}: {e}")
            return None
    
    def evaluate_multiple_stocks(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Evaluate ALL stocks in single batch with 10k token limit"""
        if not self.genesis_instance:
            print("⚠️  GenAI evaluation not available")
            return {symbol: None for symbol in symbols}
        
        try:
            print(f"🚀 SINGLE BATCH PROCESSING: {len(symbols)} stocks with Gemini 2.5 Flash")
            print(f"📊 Using 10k token limit for complete S&P 500 coverage")
            
            # Clear previous user content
            self.genesis_instance.userContents.clear()
            
            # Send ALL symbols for single batch evaluation
            symbols_text = ", ".join(symbols)
            self.genesis_instance.PushMsgToUser("text", symbols_text)
            
            print(f"🔄 Sending request to GenAI for {len(symbols)} symbols...")
            print(f"📝 Input length: {len(symbols_text)} characters")
            
            # OPTIMIZED: 10k token limit for Gemini 2.5 Flash
            response = self.genesis_instance.TXRX(
                LLM="google/gemini-2.5-flash-preview-05-20",
                provider=["google-ai-studio", "google-vertex"],
                max_tokens=10000,
                temperature=0.3
            )
            
            if not response:
                print(f"❌ No response from GenAI")
                return {symbol: None for symbol in symbols}
            
            print(f"✅ Received response from GenAI")
            print(f"📏 Response length: {len(response)} characters")
            
            # Parse JSON response
            scores = self._parse_multiple_stocks_response(response, symbols)
            
            successful_scores = len([s for s in scores.values() if s is not None])
            print(f"🎯 SINGLE BATCH RESULTS: {successful_scores}/{len(symbols)} symbols evaluated successfully")
            print(f"📊 Success rate: {successful_scores/len(symbols)*100:.1f}%")
            
            return scores
            
        except Exception as e:
            print(f"❌ Error in single batch evaluation: {e}")
            return {symbol: None for symbol in symbols}
    
    def _parse_single_stock_response(self, response: str, symbol: str) -> Optional[float]:
        """Parse AI response for single stock evaluation"""
        try:
            # Clean the response
            response = response.strip()
            
            # Try to extract JSON
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                # Look for the symbol in various formats
                symbol_upper = symbol.upper()
                
                if symbol_upper in data:
                    score = float(data[symbol_upper])
                    return max(0.0, min(1.0, score))  # Ensure 0.0-1.0 range
                
                # If exact symbol not found, try first value
                if data:
                    first_score = next(iter(data.values()))
                    score = float(first_score)
                    return max(0.0, min(1.0, score))
            
            # Try to extract just a number
            number_match = re.search(r'(\d+\.?\d*)', response)
            if number_match:
                score = float(number_match.group(1))
                # If it's > 1, assume it's a percentage and convert
                if score > 1:
                    score = score / 100
                return max(0.0, min(1.0, score))
            
            print(f"⚠️  Could not parse GenAI response for {symbol}: {response}")
            return None
            
        except Exception as e:
            print(f"❌ Error parsing response for {symbol}: {e}")
            return None
    
    def _parse_multiple_stocks_response(self, response: str, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Parse AI response for multiple stocks evaluation with better JSON handling"""
        result = {}
        
        try:
            # Clean the response
            response = response.strip()
            
            # Better JSON extraction for large responses
            # Find the start and end of JSON object
            start_idx = response.find('{')
            if start_idx == -1:
                print(f"⚠️  No JSON object found in response")
                return {symbol: None for symbol in symbols}
            
            # Find the matching closing brace
            brace_count = 0
            end_idx = -1
            
            for i in range(start_idx, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx == -1:
                # JSON might be truncated, try to find the last complete entry
                print(f"⚠️  JSON appears to be truncated, attempting partial parsing...")
                
                # Find the last complete key-value pair
                last_comma = response.rfind(',')
                if last_comma > start_idx:
                    # Try to complete the JSON by adding closing brace after last comma
                    truncated_json = response[start_idx:last_comma] + '}'
                    try:
                        data = json.loads(truncated_json)
                        print(f"✅ Successfully parsed truncated JSON with {len(data)} entries")
                    except:
                        # If that fails, try to find last complete entry before comma
                        entries = response[start_idx+1:last_comma].split(',')
                        valid_entries = []
                        
                        for entry in entries:
                            if ':' in entry and entry.count('"') >= 2:
                                valid_entries.append(entry.strip())
                        
                        if valid_entries:
                            reconstructed = '{' + ','.join(valid_entries) + '}'
                            try:
                                data = json.loads(reconstructed)
                                print(f"✅ Reconstructed partial JSON with {len(data)} entries")
                            except Exception as e:
                                print(f"❌ Failed to reconstruct JSON: {e}")
                                return {symbol: None for symbol in symbols}
                        else:
                            return {symbol: None for symbol in symbols}
                else:
                    return {symbol: None for symbol in symbols}
            else:
                # Complete JSON found
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                print(f"✅ Successfully parsed complete JSON with {len(data)} entries")
            
            # Match symbols with scores more flexibly
            matched_count = 0
            for symbol in symbols:
                symbol_upper = symbol.upper()
                symbol_clean = symbol.replace('-', '').replace('.', '')  # Handle special characters
                
                # Try multiple variations
                possible_keys = [
                    symbol_upper,
                    symbol,
                    symbol.replace('-', '.'),  # BRK-B -> BRK.B
                    symbol_clean
                ]
                
                found_score = None
                for key in possible_keys:
                    if key in data:
                        try:
                            score = float(data[key])
                            found_score = max(0.0, min(1.0, score))  # Clamp to 0-1 range
                            matched_count += 1
                            break
                        except (ValueError, TypeError):
                            continue
                
                result[symbol] = found_score
            
            print(f"📊 GenAI Matching Summary: {matched_count}/{len(symbols)} symbols matched")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"🔍 Response preview: {response[:200]}...")
            return {symbol: None for symbol in symbols}
        except Exception as e:
            print(f"❌ Error parsing multiple stocks response: {e}")
            return {symbol: None for symbol in symbols}

class EnhancedGradingWithGenAI(SimplifiedGradingSystem):
    """Enhanced grading system with STRICT percentile grading"""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.genai_evaluator = GenAIStockEvaluator(api_key)
    
    @staticmethod
    def apply_strict_percentile_grading_to_scores(final_scores: List[float]) -> List[Dict]:
        """Apply strict HKUST percentile grading with proper null handling"""
        
        if not final_scores:
            return []
        
        # Sort scores with indices to maintain original order
        indexed_scores = [(score, i) for i, score in enumerate(final_scores)]
        indexed_scores.sort(reverse=True, key=lambda x: x[0])
        
        total_count = len(final_scores)
        results = [None] * total_count  # Initialize with None
        
        # STRICT HKUST percentile boundaries (ENFORCED)
        grade_thresholds = [
            ('A+', 0.09),   
            ('A',  0.09),
            ('A-', 0.09),
            ('B+', 0.09),
            ('B',  0.09),
            ('B-', 0.09),
            ('C+', 0.09),
            ('C',  0.09),
            ('C-', 0.09),
            ('D',  0.09),
            ('F',  0.10)
        ]
        
        # Grade categories
        grade_categories = {
            'A+': 'Excellent', 'A': 'Excellent', 'A-': 'Very Good',
            'B+': 'Good', 'B': 'Good', 'B-': 'Satisfactory',
            'C+': 'Satisfactory', 'C': 'Pass', 'C-': 'Pass',
            'D': 'Marginal Pass', 'F': 'Fail'
        }
        
        # ENFORCE strict percentile distribution
        current_position = 0
        
        for grade, percentage in grade_thresholds:
            count_for_grade = max(1, int(total_count * percentage))  # Ensure at least 1 for each grade
            
            # Assign grades to students in this percentile range
            for i in range(current_position, min(current_position + count_for_grade, total_count)):
                if i < len(indexed_scores):  # SAFETY CHECK
                    score, original_index = indexed_scores[i]
                    
                    # Ensure original_index is valid
                    if 0 <= original_index < total_count:
                        results[original_index] = {
                            'rank': i + 1,
                            'scaled_score': score,
                            'grade': grade,
                            'grade_category': grade_categories[grade],
                            'percentile': (i + 1) / total_count,
                            'z_score': 0.0
                        }
            
            current_position += count_for_grade
            
            # Stop if we've assigned all students
            if current_position >= total_count:
                break
        
        # Handle any remaining None values
        for i in range(total_count):
            if results[i] is None:
                # Assign lowest grade to any unassigned positions
                score = final_scores[i] if i < len(final_scores) else 0.0
                results[i] = {
                    'rank': total_count,
                    'scaled_score': score,
                    'grade': 'F',
                    'grade_category': 'Fail',
                    'percentile': 1.0,
                    'z_score': 0.0
                }
        
        # Display grades in proper order
        print(f"✅ STRICT PERCENTILE GRADING APPLIED:")
        
        # Count grades in proper order
        grade_order = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']
        grade_counts = {}
        
        for result in results:
            if result:  # Safety check
                grade = result['grade']
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        # Display in proper order
        for grade in grade_order:
            if grade in grade_counts:
                count = grade_counts[grade]
                percentage = count / total_count * 100
                print(f"   {grade}: {count} stocks ({percentage:.1f}%)")
        
        return results

    @staticmethod
    def get_relative_grade_from_csv(result: Dict, symbol: str) -> Tuple[str, str, float, Dict]:
        """Get grade by finding closest score match in S&P 500 CSV""" 
        try:
            # Check for actual S&P 500 CSV files
            csv_files = [
                "sp500_enhanced_yfinance_results.csv",
                "sp500_fixed_grading_results.csv", 
                "sp500_analysis_results.csv"
            ]
            
            csv_data = None
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    try:
                        test_df = pd.read_csv(csv_file)
                        if not test_df.empty and 'score' in test_df.columns and not test_df['score'].isna().all():
                            csv_data = test_df
                            print(f"✅ Using S&P 500 data from: {csv_file}")
                            break
                    except Exception:
                        continue
            
            if csv_data is None:
                print("❌ No S&P 500 benchmark data found!")
                print("💡 Please run Operation 3 (S&P 500 comprehensive analysis) first.")
                
                raw_score = SimplifiedGradingSystem.calculate_raw_performance_score(result)
                scaled_score = max(0, min(100, raw_score))
                grade, category = SimplifiedGradingSystem.assign_grade_from_score(scaled_score)
                
                grade_info = {
                    'grade': grade,
                    'category': category,
                    'method': 'absolute_no_benchmark',
                    'rank_estimate': 'N/A',
                    'note': 'Run Operation 3 to enable relative grading'
                }
                
                return grade, category, scaled_score, grade_info
            
            # Calculate raw score for current stock
            raw_score = SimplifiedGradingSystem.calculate_raw_performance_score(result)
            
            # Find closest score in CSV and use its grade
            csv_data_clean = csv_data.dropna(subset=['score', 'grade'])
            
            if len(csv_data_clean) == 0:
                print("❌ No valid score/grade data in CSV")
                return EnhancedGradingWithGenAI._fallback_absolute_grading(result)
            
            # Find the row with the closest score
            score_differences = abs(csv_data_clean['score'] - raw_score)
            closest_idx = score_differences.idxmin()
            closest_row = csv_data_clean.loc[closest_idx]
            
            # Use the grade and category from the closest match
            grade = closest_row['grade']
            category = closest_row.get('grade_category', 'Unknown')
            
            # Calculate percentile for information
            all_csv_scores = csv_data_clean['score'].values
            better_than_count = len([s for s in all_csv_scores if s < raw_score])
            percentile = better_than_count / len(all_csv_scores)
            estimated_rank = int(len(all_csv_scores) * (1 - percentile)) + 1
            
            grade_info = {
                'grade': grade,
                'category': category,
                'method': 'closest_match_csv',
                'percentile': percentile,
                'percentile_display': f"{percentile*100:.1f}%",
                'rank_estimate': estimated_rank,
                'total_sp500_stocks': len(all_csv_scores),
                'better_than_count': better_than_count,
                'closest_score': float(closest_row['score']),
                'score_difference': abs(raw_score - float(closest_row['score'])),
                'closest_symbol': closest_row.get('symbol', 'Unknown')
            }
            
            print(f"✅ RELATIVE GRADING - Closest Match:")
            print(f"   Current Score: {raw_score:.1f}")
            print(f"   Closest S&P 500: {closest_row.get('symbol', 'Unknown')} (Score: {closest_row['score']:.1f})")
            print(f"   Score Difference: {abs(raw_score - closest_row['score']):.1f}")
            print(f"   Assigned Grade: {grade} ({category})")
            print(f"   Percentile: {percentile*100:.1f}%")
            
            return grade, category, raw_score, grade_info
            
        except Exception as e:
            print(f"❌ Error in relative grading: {e}")
            return EnhancedGradingWithGenAI._fallback_absolute_grading(result)

    @staticmethod
    def _fallback_absolute_grading(result: Dict) -> Tuple[str, str, float, Dict]:
        """Fallback to absolute grading if relative grading fails"""
        raw_score = SimplifiedGradingSystem.calculate_raw_performance_score(result)
        scaled_score = SimplifiedGradingSystem.convert_to_100_scale([raw_score])[0]
        grade, category = SimplifiedGradingSystem.assign_grade_from_score(scaled_score)
        
        grade_info = {
            'grade': grade,
            'category': category,
            'method': 'absolute_fallback',
            'note': 'Relative grading unavailable'
        }
        
        return grade, category, scaled_score, grade_info

    def get_enhanced_grade_with_genai(self, analysis_for_grading: Dict, symbol: str) -> Tuple[str, str, float, Dict]:
        """Enhanced grading method that combines traditional and GenAI scoring"""
        
        print(f"\n🤖 APPLYING ENHANCED GRADING WITH GENAI for {symbol}")
        
        if self.genai_evaluator and self.genai_evaluator.genesis_instance:
            # Get GenAI evaluation
            genai_score = self.genai_evaluator.evaluate_single_stock(symbol)
            
            if genai_score is not None:
                # Calculate traditional score
                traditional_score = self.calculate_raw_performance_score(analysis_for_grading)
                
                # Combine scores: 40% traditional + 60% GenAI
                enhanced_score = 0.4 * traditional_score + 0.6 * (genai_score * 100)
                enhanced_score = max(0, min(100, enhanced_score))
                
                # Apply grading
                grade, category = self.assign_grade_from_score(enhanced_score)
                
                grade_info = {
                    'grade': grade,
                    'category': category,
                    'method': 'enhanced_with_genai',
                    'traditional_score': traditional_score,
                    'genai_score': genai_score,
                    'enhanced_score': enhanced_score,
                    'genai_weight': 0.6,
                    'traditional_weight': 0.4,
                    'rank_estimate': 'N/A'  # Add this for compatibility
                }
                
                print(f"✅ Enhanced Grading Results:")
                print(f"   Traditional Score: {traditional_score:.1f}")
                print(f"   GenAI Score: {genai_score:.3f}")
                print(f"   Enhanced Score: {enhanced_score:.1f}")
                print(f"   Final Grade: {grade} ({category})")
                
                return grade, category, enhanced_score, grade_info
        
        # Fallback to relative grading if GenAI unavailable
        print("⚠️  GenAI unavailable, using relative grading fallback")
        return self.get_relative_grade_from_csv(analysis_for_grading, symbol)
