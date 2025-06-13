"""
OOP Optimized Market.py with yfinance primary and Alpaca fallback
Enhanced with maximum historical data collection
"""
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from abc import ABC, abstractmethod
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import math
from typing import List, Optional

# Add yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

from Utils import TRADING_DAYS_PER_YEAR, round_to_decimals, generate_z

warnings.filterwarnings('ignore')

# ============================================================================
# Abstract Base Classes
# ============================================================================

class DataFetcher(ABC):
    """Abstract Base Class for data fetching from different sources"""
    
    @abstractmethod
    def fetchData(self, symbol: str, startDate: datetime.datetime, endDate: datetime.datetime) -> pd.DataFrame:
        """Fetch stock data for given symbol and date range"""
        pass
    
    @abstractmethod
    def isAvailable(self) -> bool:
        """Check if data source is available and properly configured"""
        pass
    
    @abstractmethod
    def getDataSource(self) -> str:
        """Get the name of the data source"""
        pass

class BaseMarket(ABC):
    """Abstract Base Class for different market simulation strategies"""
    
    def __init__(self, initialPrice: float = 0, volatility: float = 0, 
                 expectedYearlyReturn: float = 0, numTradingDays: int = TRADING_DAYS_PER_YEAR, 
                 seed: int = -1):
        """Initialize base market with common parameters"""
        self.initialPrice = initialPrice
        self.volatility = volatility
        self.expectedYearlyReturn = expectedYearlyReturn
        self.numTradingDays = numTradingDays
        self.seed = seed
        
        # Initialize prices array
        self.prices = [0.0] * numTradingDays
        
        if seed != -1:
            np.random.seed(seed)
    
    @abstractmethod
    def simulate(self):
        """Abstract method for price simulation - must be implemented by subclasses"""
        pass
    
    # Getter methods using camelCase
    def getVolatility(self) -> float:
        return self.volatility
    
    def getExpectedYearlyReturn(self) -> float:
        return self.expectedYearlyReturn
    
    def getPrices(self) -> List[float]:
        return self.prices
    
    def getPrice(self, index: int) -> float:
        if 0 <= index < len(self.prices):
            return self.prices[index]
        return 0.0
    
    def getLastPrice(self) -> float:
        if len(self.prices) > 0:
            return self.prices[-1]
        return 0.0
    
    def getNumTradingDays(self) -> int:
        return self.numTradingDays

# ============================================================================
# Concrete Data Fetcher Implementations
# ============================================================================

class YahooDataFetcher(DataFetcher):
    """PRIMARY: Yahoo Finance data fetcher with maximum historical data - FIXED"""
    
    def __init__(self):
        self.available = YFINANCE_AVAILABLE
    
    def fetchData(self, symbol: str, startDate: datetime.datetime, endDate: datetime.datetime) -> pd.DataFrame:
        """Fetch maximum historical data from Yahoo Finance - FIXED for compatibility"""
        if not self.isAvailable():
            raise ValueError("yfinance is not available or not installed")
        
        try:
            # Get maximum historical data available (start from very early date)
            max_start_date = datetime.datetime(1960, 1, 1)  # Go back to 2000 for maximum data
            actual_start = min(startDate, max_start_date)
            
            print(f"📊 Fetching MAXIMUM historical data for {symbol}")
            print(f"   Requesting from: {actual_start.date()} to {endDate.date()}")
            print(f"   Data source: Yahoo Finance (yfinance)")
            
            # FIXED: Use simplified parameters compatible with all yfinance versions
            ticker = yf.Ticker(symbol)
            
            # Try different parameter combinations for maximum compatibility
            try:
                # First try: Modern yfinance parameters
                data = ticker.history(
                    start=actual_start,
                    end=endDate,
                    interval="1d",
                    auto_adjust=True
                )
            except Exception as e1:
                print(f"⚠️  Modern parameters failed, trying basic parameters...")
                try:
                    # Fallback: Basic parameters only
                    data = ticker.history(
                        start=actual_start,
                        end=endDate
                    )
                except Exception as e2:
                    print(f"⚠️  Basic parameters failed, trying period method...")
                    # Last resort: Use period instead of start/end
                    data = ticker.history(period="max")
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names to match Alpaca format
            data = data.reset_index()
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Handle different possible column names
            column_mapping = {}
            for col in data.columns:
                if 'date' in col.lower():
                    column_mapping[col] = 'timestamp'
                elif 'adj' in col.lower() and 'close' in col.lower():
                    column_mapping[col] = 'close'  # Prefer adjusted close
            
            data = data.rename(columns=column_mapping)
            
            # Ensure we have a timestamp column
            if 'timestamp' not in data.columns:
                if data.index.name and 'date' in data.index.name.lower():
                    data['timestamp'] = data.index
                else:
                    data['timestamp'] = data.index
            
            # Add required columns if missing
            if 'vwap' not in data.columns:
                # Calculate VWAP approximation: (high + low + close) / 3
                data['vwap'] = ((data['high'] + data['low'] + data['close']) / 3)
            
            if 'trade_count' not in data.columns:
                # Estimate trade count based on volume (rough approximation)
                data['trade_count'] = (data['volume'] / 100).fillna(100).astype(int)
            
            # Ensure timestamp is datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Sort by date and remove duplicates
            data = data.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # Remove rows with missing essential data
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            available_essential = [col for col in essential_cols if col in data.columns]
            data = data.dropna(subset=available_essential)
            
            if data.empty:
                raise ValueError(f"No valid data after cleaning for {symbol}")
            
            print(f"✅ Yahoo Finance: Fetched {len(data)} data points")
            print(f"   Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
            print(f"   Years of data: {(data['timestamp'].max() - data['timestamp'].min()).days / 365.25:.1f}")
            print(f"   Columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            print(f"❌ Yahoo Finance error for {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch data from Yahoo Finance: {str(e)}")
    
    def isAvailable(self) -> bool:
        """Check if yfinance is available"""
        return self.available
    
    def getDataSource(self) -> str:
        return "yfinance"


class AlpacaDataFetcher(DataFetcher):
    """FALLBACK: Alpaca data fetcher"""
    
    def __init__(self, apiKey: str, secretKey: str):
        self.apiKey = apiKey
        self.secretKey = secretKey
        self.client = StockHistoricalDataClient(apiKey, secretKey) if self.isAvailable() else None
    
    def fetchData(self, symbol: str, startDate: datetime.datetime, endDate: datetime.datetime) -> pd.DataFrame:
        """Fetch data from Alpaca API with maximum duration"""
        if not self.isAvailable():
            raise ValueError("Alpaca data fetcher is not properly configured")
        
        try:
            # For Alpaca, also try to get maximum data (5+ years)
            max_start_date = datetime.datetime(2016, 1, 1)  # Alpaca has data from ~2016
            actual_start = min(startDate, max_start_date)
            
            print(f"📊 Fetching data for {symbol} (Alpaca fallback)")
            print(f"   Requesting from: {actual_start.date()} to {endDate.date()}")
            print(f"   Data source: Alpaca Markets")
            
            data = self.client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=actual_start,
                end=endDate
            )).df
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Reset index for easier manipulation
            data = data.reset_index()
            
            print(f"✅ Alpaca: Fetched {len(data)} data points")
            print(f"   Date range: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
            
            return data
            
        except Exception as e:
            print(f"❌ Alpaca error for {symbol}: {str(e)}")
            raise ValueError(f"Failed to fetch data from Alpaca: {str(e)}")
    
    def isAvailable(self) -> bool:
        """Check if API keys are provided"""
        return self.apiKey is not None and self.secretKey is not None
    
    def getDataSource(self) -> str:
        return "alpaca"

class HybridDataFetcher(DataFetcher):
    """HYBRID: Try yfinance first, fallback to Alpaca with maximum historical data"""
    
    def __init__(self, apiKey: str = None, secretKey: str = None):
        self.yahooFetcher = YahooDataFetcher()
        self.alpacaFetcher = AlpacaDataFetcher(apiKey, secretKey) if apiKey and secretKey else None
        self.lastUsedSource = None
    
    def fetchData(self, symbol: str, startDate: datetime.datetime, endDate: datetime.datetime) -> pd.DataFrame:
        """Try yfinance first, then Alpaca fallback"""
        
        # First try Yahoo Finance (primary)
        if self.yahooFetcher.isAvailable():
            try:
                print("🔄 Attempting Yahoo Finance (primary source)...")
                data = self.yahooFetcher.fetchData(symbol, startDate, endDate)
                self.lastUsedSource = "yfinance"
                print("✅ Successfully used Yahoo Finance")
                return data
            except Exception as e:
                print(f"⚠️  Yahoo Finance failed: {str(e)}")
                print("🔄 Falling back to Alpaca...")
        
        # Fallback to Alpaca
        if self.alpacaFetcher and self.alpacaFetcher.isAvailable():
            try:
                data = self.alpacaFetcher.fetchData(symbol, startDate, endDate)
                self.lastUsedSource = "alpaca"
                print("✅ Successfully used Alpaca (fallback)")
                return data
            except Exception as e:
                print(f"❌ Alpaca also failed: {str(e)}")
                raise ValueError(f"Both data sources failed for {symbol}")
        
        # No data sources available
        raise ValueError("No data sources available. Install yfinance or provide Alpaca credentials.")
    
    def isAvailable(self) -> bool:
        """Check if any data source is available"""
        return (self.yahooFetcher.isAvailable() or 
                (self.alpacaFetcher and self.alpacaFetcher.isAvailable()))
    
    def getDataSource(self) -> str:
        """Return the data source that was actually used"""
        return self.lastUsedSource if self.lastUsedSource else "none"

# ============================================================================
# Enhanced StockPredictor with Hybrid Data Fetching
# ============================================================================

class StockPredictor:
    """Enhanced StockPredictor with yfinance primary + Alpaca fallback"""
    
    def __init__(self, dataFetcher: DataFetcher, symbol: str = "KO"):
        """Initialize with injected data fetcher dependency"""
        self.dataFetcher = dataFetcher
        self.symbol = symbol
        self.dataSource = None  # Will be set after fetching
        
        # Data storage
        self.data = None
        self.dataClean = None
        self.model = None
        
        # Features configuration
        self.priceFeatures = ["open", "high", "low", "close", "vwap"]
        self.normFeatures = []
        self.rateFeatures = []
        self.allFeatures = []
        
        # Training data
        self.trainData = None
        self.testData = None
        self.splitRatio = 0.85
        
        # Results storage
        self.predictedPricesClean = np.array([])
        self.actualPricesClean = np.array([])
        self.testTimestampsClean = np.array([])
        self.futurePrices = []
        self.futureDates = []
    
    def fetchData(self, startDate: datetime.datetime = None, endDate: datetime.datetime = None):
        """Fetch maximum historical data using hybrid approach"""
        if startDate is None:
            # Start from 2000 to get maximum historical data
            startDate = datetime.datetime(2000, 1, 1)
        if endDate is None:
            endDate = datetime.datetime.now()
        
        print(f"🚀 Fetching MAXIMUM historical data for {self.symbol}")
        self.data = self.dataFetcher.fetchData(self.symbol, startDate, endDate)
        self.dataSource = self.dataFetcher.getDataSource()
        
        if self.data is not None and not self.data.empty:
            print(f"📈 Data summary for {self.symbol}:")
            print(f"   Source: {self.dataSource}")
            print(f"   Records: {len(self.data)}")
            print(f"   Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            print(f"   Years of data: {(self.data['timestamp'].max() - self.data['timestamp'].min()).days / 365.25:.1f}")
    
    @staticmethod
    def rollingNormalizeSafe(series: pd.Series, window: int = 252) -> pd.Series:
        """Normalize using rolling statistics with NaN handling"""
        rollingMean = series.rolling(window=window, min_periods=30).mean()
        rollingStd = series.rolling(window=window, min_periods=30).std()
        
        # Handle division by zero and NaN values
        rollingStd = rollingStd.fillna(1.0)
        rollingStd = rollingStd.replace(0, 1.0)
        
        normalized = (series - rollingMean) / rollingStd
        return normalized.fillna(0)
    
    def prepareFeatures(self):
        """Prepare and engineer features for the model"""
        print("Preparing features...")
        
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Create a copy for feature engineering
        dataNorm = self.data.copy()
        
        # Apply rolling normalization to price features
        for feature in self.priceFeatures:
            if feature in dataNorm.columns:
                dataNorm[f"{feature}_norm"] = self.rollingNormalizeSafe(self.data[feature])
        
        # Set up normalized features (exclude close from features)
        available_features = ["open", "high", "low", "vwap"]
        self.normFeatures = [f"{feature}_norm" for feature in available_features if f"{feature}_norm" in dataNorm.columns]
        
        # Add rate of change features
        self.rateFeatures = []
        for feature in self.normFeatures:
            rateCol = f"{feature}_diff"
            dataNorm[rateCol] = dataNorm[feature].diff().fillna(0)
            self.rateFeatures.append(rateCol)
        
        # Combine all features (check availability)
        basic_features = ["volume", "trade_count"]
        available_basic = [f for f in basic_features if f in dataNorm.columns]
        self.allFeatures = self.normFeatures + self.rateFeatures + available_basic
        
        # Clean data
        self.dataClean = dataNorm.copy()
        required_cols = self.allFeatures + ["close", "close_norm"]
        available_cols = [col for col in required_cols if col in self.dataClean.columns]
        
        for col in available_cols:
            self.dataClean[col] = self.dataClean[col].ffill().fillna(0)
        
        # Remove problematic rows
        self.dataClean = self.dataClean.dropna(subset=available_cols)
        
        print(f"Data shape after cleaning: {self.dataClean.shape}")
        print(f"Features: {len(self.allFeatures)} - {self.allFeatures}")
    
    def splitData(self, splitRatio: float = None):
        """Split data into training and testing sets"""
        if splitRatio is not None:
            self.splitRatio = splitRatio
        
        if self.dataClean is None:
            raise ValueError("No clean data available. Please prepare features first.")
        
        splitIndex = int(self.splitRatio * len(self.dataClean))
        self.trainData = self.dataClean.iloc[:splitIndex, :]
        self.testData = self.dataClean.iloc[splitIndex:, :]
        
        print(f"Training data: {len(self.trainData)} days")
        print(f"Test data: {len(self.testData)} days")
    
    def trainModel(self, **xgbParams):
        """Train the XGBoost model with consistent target"""
        if self.trainData is None:
            raise ValueError("No training data available. Please split data first.")
        
        # Default XGBoost parameters
        defaultParams = {
            'random_state': 13,
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Update with user-provided parameters
        defaultParams.update(xgbParams)
        
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(**defaultParams)
        
        # Always use close_norm as target
        self.model.fit(self.trainData[self.allFeatures], self.trainData["close_norm"])
        print("Model training completed")
    
    def predictFuture(self, daysAhead: int = 780):
        """Predict future stock prices with consistent random seed"""
        if self.model is None or self.dataClean is None:
            raise ValueError("Model not trained or data not available")
        
        # Set consistent random seed for reproducible results
        np.random.seed(13)
        
        print(f"Starting prediction for {daysAhead} days...")
        
        # Get base statistics
        baseMean = self.dataClean["close"].rolling(window=252, min_periods=30).mean().iloc[-1]
        baseStd = self.dataClean["close"].rolling(window=252, min_periods=30).std().iloc[-1]
        if pd.isna(baseMean):
            baseMean = self.dataClean["close"].mean()
        if pd.isna(baseStd) or baseStd == 0:
            baseStd = self.dataClean["close"].std() * 0.1
        
        # Starting point
        lastPrice = self.dataClean["close"].iloc[-1]
        lastDate = pd.to_datetime(self.dataClean["timestamp"].iloc[-1])
        
        print(f"Starting from: {lastDate.date()}, Price: ${lastPrice:.2f}")
        print(f"Base stats - Mean: ${baseMean:.2f}, Std: ${baseStd:.2f}")
        
        # Generate future business dates
        self.futureDates = pd.bdate_range(start=lastDate + pd.Timedelta(days=1), periods=daysAhead)
        
        # Initialize prediction arrays
        self.futurePrices = []
        currentPrice = lastPrice
        
        # Get the last known features as template
        lastFeatures = self.dataClean[self.allFeatures].iloc[-1].copy()
        
        # Track basic statistics for realistic movement
        dailyReturns = self.dataClean["close"].pct_change().dropna()
        avgReturn = dailyReturns.mean()
        returnStd = dailyReturns.std()
        
        print(f"Historical daily return: {avgReturn:.6f} ± {returnStd:.6f}")
        
        for i, futureDate in enumerate(self.futureDates):
            try:
                # Create feature vector for prediction
                currentFeatures = lastFeatures.copy()
                
                # Add realistic noise and trend to features
                if i > 0:
                    # Update normalized features
                    for feature in self.normFeatures:
                        currentFeatures[feature] += np.random.normal(0, 0.01)
                        currentFeatures[feature] = np.clip(currentFeatures[feature], -3, 3)
                    
                    # Update rate features
                    for feature in self.rateFeatures:
                        currentFeatures[feature] = np.random.normal(0, 0.005)
                        currentFeatures[feature] = np.clip(currentFeatures[feature], -0.1, 0.1)
                
                # Make prediction
                featuresArray = currentFeatures.values.reshape(1, -1)
                predNorm = self.model.predict(featuresArray)[0]
                
                # Convert normalized prediction to actual price
                predPrice = predNorm * baseStd + baseMean
                
                # Apply realistic constraints
                maxDailyChange = 0.15  # 15% max daily change
                minPrice = currentPrice * (1 - maxDailyChange)
                maxPrice = currentPrice * (1 + maxDailyChange)
                predPrice = np.clip(predPrice, minPrice, maxPrice)
                
                # Add trend component
                if i > 0:
                    trendFactor = 1 + avgReturn + np.random.normal(0, returnStd * 0.5)
                    predPrice = currentPrice * trendFactor
                    predPrice = np.clip(predPrice, minPrice, maxPrice)
                
                self.futurePrices.append(predPrice)
                currentPrice = predPrice
                
                # Progress updates
                if i % 260 == 0 and i > 0:  # Every ~year
                    years = i // 260 + 1
                    print(f"Year {years}: ${predPrice:.2f}")
                
            except Exception as e:
                if i < 5:  # Only show first few errors
                    print(f"Error at day {i}: {e}")
                
                # Fallback prediction
                if self.futurePrices:
                    growth = np.random.normal(avgReturn, returnStd)
                    predPrice = self.futurePrices[-1] * (1 + growth)
                else:
                    predPrice = currentPrice * (1 + avgReturn)
                
                self.futurePrices.append(predPrice)
                currentPrice = predPrice
    
    def runCompleteAnalysis(self, daysAhead: int = 780):
        """Run the complete analysis pipeline using maximum historical data"""
        print("=== Starting Complete Stock Prediction Analysis (Enhanced) ===\n")
        
        # Execute full pipeline with maximum data
        self.fetchData()  # This now fetches maximum historical data
        self.prepareFeatures()
        self.splitData()
        self.trainModel()
        self.predictFuture(daysAhead)
        
        print(f"\n=== Analysis Complete - Used {self.dataSource} ===")

# ============================================================================
# Keep existing Market classes unchanged
# ============================================================================

class GBMMarket(BaseMarket):
    """Geometric Brownian Motion Market implementation"""
    
    def simulate(self):
        """Simulate prices using GBM"""
        if len(self.prices) == 0:
            return
        
        self.prices[0] = round_to_decimals(self.initialPrice, 3)
        
        for i in range(1, self.numTradingDays):
            dt = 1.0 / TRADING_DAYS_PER_YEAR
            z = generate_z()
            
            drift = (self.expectedYearlyReturn - 0.5 * self.volatility ** 2) * dt
            diffusion = self.volatility * math.sqrt(dt) * z
            
            nextPrice = self.prices[i-1] * math.exp(drift + diffusion)
            self.prices[i] = round_to_decimals(nextPrice, 3)

class MLMarket(BaseMarket):
    """Machine Learning based Market implementation"""
    
    def __init__(self, initialPrice: float = 0, volatility: float = 0, 
                 expectedYearlyReturn: float = 0, numTradingDays: int = TRADING_DAYS_PER_YEAR, 
                 seed: int = -1, predictor: StockPredictor = None):
        """Initialize ML Market with predictor dependency"""
        super().__init__(initialPrice, volatility, expectedYearlyReturn, numTradingDays, seed)
        self.predictor = predictor
    
    def simulate(self):
        """Simulate prices using ML predictions"""
        if self.predictor is None:
            raise ValueError("No predictor provided for ML simulation")
        
        try:
            # Check if predictor has been run
            if len(self.predictor.futurePrices) == 0:
                print("No existing predictions found, running complete analysis...")
                self.predictor.runCompleteAnalysis()
            
            futurePrices = self.predictor.futurePrices
            
            if len(futurePrices) == 0:
                raise ValueError("Predictor generated no predictions")
            
            # Use predictions as market prices
            self.prices = [round_to_decimals(price, 3) for price in futurePrices[:self.numTradingDays]]
            
            # Pad with last price if needed
            while len(self.prices) < self.numTradingDays:
                if len(self.prices) > 0:
                    self.prices.append(self.prices[-1])
                else:
                    self.prices.append(self.initialPrice)
            
            print(f"✓ ML simulation completed")
            print(f"✓ Price range: ${min(self.prices):.2f} - ${max(self.prices):.2f}")
            
        except Exception as e:
            print(f"Error in ML simulation: {str(e)}")
            print("Falling back to GBM simulation")
            self._fallbackToGBM()
    
    def _fallbackToGBM(self):
        """Private method for GBM fallback"""
        if len(self.prices) == 0:
            return
        
        self.prices[0] = round_to_decimals(self.initialPrice, 3)
        
        for i in range(1, self.numTradingDays):
            dt = 1.0 / TRADING_DAYS_PER_YEAR
            z = generate_z()
            
            drift = (self.expectedYearlyReturn - 0.5 * self.volatility ** 2) * dt
            diffusion = self.volatility * math.sqrt(dt) * z
            
            nextPrice = self.prices[i-1] * math.exp(drift + diffusion)
            self.prices[i] = round_to_decimals(nextPrice, 3)

class Market(BaseMarket):
    """Backward compatible Market class"""
    
    def __init__(self, initial_price: float = 0, volatility: float = 0, 
                 expected_yearly_return: float = 0, num_trading_days: int = TRADING_DAYS_PER_YEAR, 
                 seed: int = -1, predictor: StockPredictor = None):
        """Original Market constructor for backward compatibility"""
        super().__init__(initial_price, volatility, expected_yearly_return, num_trading_days, seed)
        self.predictor = predictor
    
    def simulate(self, use_ml=False):
        """Main simulation method with original interface"""
        if use_ml and self.predictor is not None:
            self.simulate_with_ml()
        else:
            self.simulate_gbm()
    
    def simulate_gbm(self):
        """GBM simulation using original method names"""
        if len(self.prices) == 0:
            return
        
        self.prices[0] = round_to_decimals(self.initialPrice, 3)
        
        for i in range(1, self.numTradingDays):
            dt = 1.0 / TRADING_DAYS_PER_YEAR
            z = generate_z()
            
            drift = (self.expectedYearlyReturn - 0.5 * self.volatility ** 2) * dt
            diffusion = self.volatility * math.sqrt(dt) * z
            
            nextPrice = self.prices[i-1] * math.exp(drift + diffusion)
            self.prices[i] = round_to_decimals(nextPrice, 3)
    
    def simulate_with_ml(self):
        """ML simulation using original method names"""
        if self.predictor is None:
            raise ValueError("No predictor provided for ML simulation")
        
        try:
            # Check if predictor has been run
            if len(self.predictor.futurePrices) == 0:
                print("No existing predictions found, running complete analysis...")
                self.predictor.runCompleteAnalysis()
            
            futurePrices = self.predictor.futurePrices
            
            if len(futurePrices) == 0:
                raise ValueError("Predictor generated no predictions")
            
            # Use predictions as market prices
            self.prices = [round_to_decimals(price, 3) for price in futurePrices[:self.numTradingDays]]
            
            # Pad with last price if needed
            while len(self.prices) < self.numTradingDays:
                if len(self.prices) > 0:
                    self.prices.append(self.prices[-1])
                else:
                    self.prices.append(self.initialPrice)
            
            print(f"✓ ML simulation completed")
            print(f"✓ Price range: ${min(self.prices):.2f} - ${max(self.prices):.2f}")
            
        except Exception as e:
            print(f"Error in ML simulation: {str(e)}")
            print("Falling back to GBM simulation")
            self.simulate_gbm()
    
    # Original getter methods for backward compatibility
    def get_volatility(self) -> float:
        return self.getVolatility()
    
    def get_expected_yearly_return(self) -> float:
        return self.getExpectedYearlyReturn()
    
    def get_prices(self) -> List[float]:
        return self.getPrices()
    
    def get_price(self, index: int) -> float:
        return self.getPrice(index)
    
    def get_last_price(self) -> float:
        return self.getLastPrice()
    
    def get_num_trading_days(self) -> int:
        return self.getNumTradingDays()

# ============================================================================
# UPDATED Factory Functions with yfinance Priority
# ============================================================================

def createHybridPredictor(apiKey: str = None, secretKey: str = None, symbol: str = "KO") -> StockPredictor:
    """NEW: Create StockPredictor with yfinance primary + Alpaca fallback"""
    hybridFetcher = HybridDataFetcher(apiKey, secretKey)
    return StockPredictor(hybridFetcher, symbol)

def createAlpacaPredictor(apiKey: str, secretKey: str, symbol: str = "KO") -> StockPredictor:
    """Legacy: Create StockPredictor with Alpaca only (for backward compatibility)"""
    dataFetcher = AlpacaDataFetcher(apiKey, secretKey)
    return StockPredictor(dataFetcher, symbol)

def createYahooPredictor(symbol: str = "KO") -> StockPredictor:
    """NEW: Create StockPredictor with Yahoo Finance only"""
    dataFetcher = YahooDataFetcher()
    return StockPredictor(dataFetcher, symbol)

def createGBMMarket(initialPrice: float = 100.0, volatility: float = 0.2, 
                    expectedYearlyReturn: float = 0.08, numTradingDays: int = TRADING_DAYS_PER_YEAR,
                    seed: int = 13) -> GBMMarket:
    """Factory function to create GBM Market"""
    return GBMMarket(initialPrice, volatility, expectedYearlyReturn, numTradingDays, seed)

def createMLMarket(predictor: StockPredictor, initialPrice: float = 100.0, 
                   numTradingDays: int = TRADING_DAYS_PER_YEAR) -> MLMarket:
    """Factory function to create ML Market"""
    return MLMarket(initialPrice, 0.2, 0.08, numTradingDays, 13, predictor)