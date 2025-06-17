# 📊 ProjectExponent (0.0.4 Delta)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)

An advanced stock trading and analysis system that combines traditional financial modeling with modern AI/ML techniques. The system provides comprehensive stock analysis, S&P 500 bulk analysis, automated trading, and GenAI-enhanced grading.

## 🎯 Project Overview

This project implements a comprehensive stock analysis and trading system following Object-Oriented Programming principles. It integrates multiple data sources, advanced prediction models, trading strategies, and AI-enhanced grading systems.

### Key Features

- **Individual Stock Analysis** - Comprehensive 5-year forecasting with multiple strategies
- **S&P 500 Bulk Analysis** - Parallel processing of 500+ stocks with percentile grading
- **GenAI Enhancement** - OpenRouter/Gemini integration for intelligent stock scoring
- **Automated Trading** - Paper and live trading with user preferences
- **Discord Bot** - Daily account status reporting
- **Enhanced Visualizations** - Color-coded grade backgrounds and comprehensive charts

## 🏗 System Architecture

### Core Pattern: Inheritance Structure

```
Market (Base Class)
├── StockPredictor
├── ComprehensiveStockAnalyzer
├── SP500ComprehensiveAnalyzer
├── SP500DataProvider
├── VisualizationEngine
└── UserPreferenceManager
```

### Data Flow

```
Raw Stock Data → Prediction Models → Strategy Testing → Stability Analysis → Grading → Visualization
                                                                         ↓
                                                               GenAI Enhancement
```

## 📁 File Structure

```
📦 ProjectExponent
├── 📄 PA2.py                     # Main entry point with 8 operations
├── 📄 Market.py                  # Base Market class, StockPredictor, data fetching
├── 📄 Strategy.py                # Trading strategies (MR, TF, WTF)
├── 📄 TradingBot.py               # Trading simulation and real trading
├── 📄 GradingSystem.py           # Grading systems with GenAI integration
├── 📄 Genesis.py                 # Universal GenAI interface
├── 📄 Utils.py                   # Constants and utility functions
├── 📄 Visualization.py           # Visualization engine and user preferences
├── 📂 Analysis/
│   ├── 📄 __init__.py
│   ├── 📄 StockAnalyzer.py       # Individual stock analysis
│   ├── 📄 SP500Analysis.py       # S&P 500 bulk analysis
│   ├── 📄 StabilityAnalysis.py   # Risk and performance metrics
│   ├── 📄 GradingSystem.py       # Multiple grading systems
│   └── 📄 Visualization.py       # Chart generation and user management
├── 📄 .env                       # Environment variables (create this)
└── 📄 requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install yfinance alpaca-trade-api xgboost scikit-learn pandas numpy matplotlib requests python-dotenv schedule pytz markitdown
```

### Environment Setup

1. **Create `.env` file** with your API keys:

```env
# Alpaca Trading Configuration
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# OpenRouter Configuration (for GenAI features)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Discord Bot Configuration (optional)
DISCORD_BOT_TOKEN=your_discord_bot_token_here
DISCORD_CHANNEL_ID=your_discord_channel_id_here

# Environment Setting
ENVIRONMENT=development
```

2. **Run the main program**:

```bash
python PA2.py
```

## 📚 Core Classes and Usage

### 1. Market Class (Market.py)

**Purpose**: Base class for market simulation and data management

```python
from Market import Market, createHybridPredictor

# Create a basic market
market = Market(
    initial_price=100.0,
    volatility=0.2,
    expected_yearly_return=0.08,
    num_trading_days=252
)

# Simulate GBM prices
market.simulate_gbm()
prices = market.get_prices()

# Create stock predictor with hybrid data fetching
predictor = createHybridPredictor(
    apiKey="your_alpaca_key",
    secretKey="your_alpaca_secret", 
    symbol="AAPL"
)

# Run complete analysis
predictor.runCompleteAnalysis()
```

### 2. Strategy Classes (Strategy.py)

**Purpose**: Implement different trading strategies

```python
from Strategy import MeanReversionStrategy, TrendFollowingStrategy, WeightedTrendFollowingStrategy

# Mean Reversion Strategy
mr_strategy = MeanReversionStrategy("MR_20_5", window=20, threshold=5)

# Generate multiple strategies
mr_strategies = MeanReversionStrategy.generate_strategy_set(
    "MR", 
    min_window=10, max_window=50, window_step=10,
    min_threshold=2, max_threshold=8, threshold_step=2
)

# Trend Following Strategy
tf_strategy = TrendFollowingStrategy("TF_10_30", short_window=10, long_window=30)

# Weighted Trend Following Strategy
wtf_strategy = WeightedTrendFollowingStrategy("WTF_5_20", short_window=5, long_window=20)
```

### 3. TradingBot Class (TradingBot.py)

**Purpose**: Execute trading strategies and manage positions

```python
from TradingBot import TradingBot
from Market import Market
from Strategy import MeanReversionStrategy

# Create market and bot
market = Market(initial_price=100, num_trading_days=252)
market.simulate_gbm()

bot = TradingBot(market, initial_capacity=10)

# Add strategies
strategy1 = MeanReversionStrategy("MR_Test", 20, 3)
strategy2 = TrendFollowingStrategy("TF_Test", 10, 30)

bot.add_strategy(strategy1)
bot.add_strategy(strategy2)

# Run simulation
result = bot.run_simulation()
print(f"Best strategy: {result.best_strategy.get_name()}")
print(f"Total return: {result.total_return}")

# Real trading (with API keys)
real_bot = TradingBot(None, api_key="key", secret_key="secret", paper=True)
real_bot.display_all_holdings()
```

### 4. ComprehensiveStockAnalyzer (Analysis/StockAnalyzer.py)

**Purpose**: Perform comprehensive analysis of individual stocks

```python
from Analysis import ComprehensiveStockAnalyzer

# Create analyzer
analyzer = ComprehensiveStockAnalyzer(
    api_key="your_alpaca_key",
    secret_key="your_alpaca_secret"
)

# Analyze a stock
result = analyzer.analyze_stock_comprehensive(
    symbol="AAPL",
    show_visualization=True
)

if result:
    print(f"Grade: {result['grade']}")
    print(f"Score: {result['score']}")
    print(f"Best Strategy: {result['best_strategy_name']}")
```

### 5. SP500ComprehensiveAnalyzer (Analysis/SP500Analysis.py)

**Purpose**: Bulk analysis of S&P 500 stocks with GenAI grading

```python
from Analysis import SP500ComprehensiveAnalyzer

# Create S&P 500 analyzer
analyzer = SP500ComprehensiveAnalyzer(
    api_key="your_alpaca_key",
    secret_key="your_alpaca_secret"
)

# Run comprehensive analysis
# Option 1: Quick test (10 stocks)
analyzer.run_comprehensive_analysis(max_workers=5, sample_size=10)

# Option 2: Full analysis (all ~500 stocks)
analyzer.run_comprehensive_analysis(max_workers=25)
```

### 6. Grading Systems (GradingSystem.py)

**Purpose**: Grade stocks using various methods including GenAI

```python
from GradingSystem import EnhancedGradingWithGenAI, SimplifiedGradingSystem

# Enhanced grading with GenAI
grader = EnhancedGradingWithGenAI(api_key="your_openrouter_key")

# Grade a stock
analysis_data = {
    'gbm_stability': {'total_return': 0.15, 'auc': 1000, 'slope': 0.002, 'sharpe_ratio': 1.5},
    'ml_stability': {'total_return': 0.18, 'auc': 1200, 'slope': 0.003, 'sharpe_ratio': 1.8}
}

grade, category, score, grade_info = grader.get_enhanced_grade_with_genai(
    analysis_data, "AAPL"
)

print(f"Grade: {grade} ({category})")
print(f"Score: {score}")
```

### 7. Genesis GenAI Interface (Genesis.py)

**Purpose**: Universal GenAI interface for stock evaluation

```python
from Genesis import Genesis

# Initialize Genesis
ai = Genesis(
    key="your_openrouter_key",
    httpRef="https://your-website.com",
    projTitle="Stock Analysis"
)

# Set system prompt
ai.PushMsgToSystem("You are a stock analysis expert...")

# Evaluate single stock
ai.PushMsgToUser("text", "AAPL")
response = ai.TXRX(
    LLM="openai/gpt-4o-2024-11-20",
    max_tokens=100,
    temperature=0.3
)

print(f"AI Response: {response}")
```

### 8. VisualizationEngine (Visualization.py)

**Purpose**: Generate comprehensive analysis charts

```python
from Visualization import VisualizationEngine

# Create comprehensive visualization
VisualizationEngine.plot_enhanced_comprehensive_analysis(
    symbol="AAPL",
    analysis_data=result,  # From ComprehensiveStockAnalyzer
    save_path="AAPL_analysis.png"
)
```

## 🎮 Operations Guide

The main program (`PA2.py`) provides 8 operations:

1. **📈 Individual Stock Analysis** - Analyze single stock with comprehensive metrics
2. **📊 Multiple Stocks Testing** - Test several stocks with comparative analysis  
3. **🏆 S&P 500 Comprehensive Analysis** - Bulk analysis with GenAI grading
4. **🤖 Automated Trading Loop** - Paper/live trading with user preferences
5. **📋 View Account Holdings** - Display current positions and P&L
6. **💾 User Preference Management** - Manage saved stock analyses
7. **🔧 Configuration Management** - Setup API keys and environment
8. **🤖 Discord Bot** - Start daily account reporting bot

## 📊 Example Analysis Output

```
✅ COMPREHENSIVE ANALYSIS RESULTS FOR AAPL
================================================================================

📈 STOCK INFORMATION:
   Symbol: AAPL
   Analysis Date: 2024-01-15
   Data Source: yfinance

🎓 GRADING SYSTEM: ENHANCED_WITH_GENAI
   Grade: A (Excellent Performance)
   Score: 87.3/100
   🤖 GenAI Score: 0.850/1.0
   🔬 Scoring Method: Enhanced (Traditional + GenAI)

🏆 PREDICTION RESULTS:
   Winning Method: ML
   GBM Return: 0.1547
   ML Return: 0.1823

⚔️ STRATEGY RESULTS:
   Best Strategy: ML_WTF_15_40
   Best Return: 0.2156
   Strategy Params: Short MA: 15, Long MA: 40
```

## ⚠️ Known Issues and Limitations

### Critical Issues (Unfixed)

1. **Relative Grading System**
   - **Issue**: Grading system still uses hardcoded percentile thresholds instead of finding closest match in S&P 500 CSV data
   - **Impact**: Grades may not accurately reflect relative performance vs S&P 500
   - **Workaround**: Run Operation 3 first to generate S&P 500 benchmark data

2. **Strategy Visualization Inconsistency**
   - **Issue**: Strategy Returns Comparison chart shows unrealistic percentages (e.g., 1078%) that don't match Annual Returns breakdown
   - **Impact**: Misleading visualization data
   - **Workaround**: Focus on the numerical strategy results in text output rather than visualization

3. **Deterministic Strategy Selection**
   - **Issue**: System consistently returns same strategies (e.g., GBM_WTF_5_20, ML_MR_20_2) instead of comprehensively testing all parameter combinations
   - **Impact**: May miss better-performing strategy configurations
   - **Root Cause**: Fixed random seeds in Market class simulation methods

### Minor Issues

4. **GenAI Rate Limiting**
   - Large batch requests (500+ stocks) may hit API rate limits
   - Partial responses may occur with incomplete JSON parsing

5. **Data Source Dependencies**
   - yfinance may occasionally fail for certain symbols
   - Alpaca fallback requires valid API credentials

6. **Memory Usage**
   - Full S&P 500 analysis with all strategies can consume significant memory
   - Recommend running in batches for resource-constrained environments

## 🛠 Development Notes

### Architecture Patterns

- **Inheritance**: All analysis classes inherit from base `Market` class
- **Dependency Injection**: Data fetchers are injected into predictors
- **Factory Pattern**: Creator functions for different market types
- **Strategy Pattern**: Pluggable trading strategies

### Performance Considerations

- **Parallel Processing**: ThreadPoolExecutor for S&P 500 bulk analysis
- **Data Caching**: Predictor results cached to avoid re-computation
- **Memory Management**: Large datasets cleaned after use

### Security Features

- **Environment Variables**: Secure credential storage
- **API Key Validation**: Checks before making external calls
- **Paper Trading Default**: Prevents accidental live trading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **yfinance** for reliable stock data
- **Alpaca Markets** for trading API
- **OpenRouter** for GenAI integration
- **XGBoost** for machine learning predictions

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check the known issues section above
- Review the configuration management in PA2.py Operation 7

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Not financial advice. Use at your own risk.
