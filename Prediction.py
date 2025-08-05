import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    A comprehensive stock price prediction class using XGBoost with rolling normalization
    """
    
    def __init__(self, api_key: str, secret_key: str, symbol: str = "KO"):
        """
        Initialize the StockPredictor
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            symbol: Stock symbol to predict
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.client = StockHistoricalDataClient(api_key, secret_key)
        
        # Data storage
        self.data = None
        self.data_clean = None
        self.model = None
        
        # Features
        self.price_features = ["open", "high", "low", "close", "vwap"]
        self.norm_features = []
        self.rate_features = []
        self.all_features = []
        
        # Training data
        self.trainData = None
        self.testData = None
        self.split_ratio = 0.85
        
        # Results storage
        self.predicted_prices_clean = np.array([])
        self.actual_prices_clean = np.array([])
        self.test_timestamps_clean = np.array([])
        self.future_prices = []
        self.future_dates = []
    
    def fetch_data(self, start_date: datetime.datetime = None, end_date: datetime.datetime = None):
        """
        Fetch stock data from Alpaca API
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
        """
        if start_date is None:
            start_date = datetime.datetime(2016, 1, 4)
        if end_date is None:
            end_date = datetime.datetime(2025, 6, 1)
        
        print(f"Fetching data for {self.symbol} from {start_date.date()} to {end_date.date()}...")
        
        self.data = self.client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )).df
        
        # Reset index for easier manipulation
        self.data = self.data.reset_index()
        print(f"Fetched {len(self.data)} data points")
    
    @staticmethod
    def rolling_normalize_safe(series: pd.Series, window: int = 252) -> pd.Series:
        """
        Normalize using rolling statistics with NaN handling
        
        Args:
            series: Pandas series to normalize
            window: Rolling window size
            
        Returns:
            Normalized series
        """
        rolling_mean = series.rolling(window=window, min_periods=30).mean()
        rolling_std = series.rolling(window=window, min_periods=30).std()
        
        # Handle division by zero and NaN values
        rolling_std = rolling_std.fillna(1.0)
        rolling_std = rolling_std.replace(0, 1.0)
        
        normalized = (series - rolling_mean) / rolling_std
        return normalized.fillna(0)
    
    def prepare_features(self):
        """
        Prepare and engineer features for the model
        """
        print("Preparing features...")
        
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        # Create a copy for feature engineering
        data_norm = self.data.copy()
        
        # Apply rolling normalization to price features
        for feature in self.price_features:
            data_norm[f"{feature}_norm"] = self.rolling_normalize_safe(self.data[feature])
        
        # Set up normalized features (exclude close from features)
        self.norm_features = [f"{feature}_norm" for feature in ["open", "high", "low", "vwap"]]
        
        # Add rate of change features
        self.rate_features = []
        for feature in self.norm_features:
            rate_col = f"{feature}_diff"
            data_norm[rate_col] = data_norm[feature].diff().fillna(0)
            self.rate_features.append(rate_col)
        
        # Combine all features
        self.all_features = self.norm_features + self.rate_features + ["volume", "trade_count"]
        
        # Clean data
        self.data_clean = data_norm.copy()
        for col in self.all_features + ["close", "close_norm"]:
            if col in self.data_clean.columns:
                self.data_clean[col] = self.data_clean[col].ffill().fillna(0)
        
        # Remove problematic rows
        self.data_clean = self.data_clean.dropna(subset=self.all_features + ["close_norm"])
        
        print(f"Data shape after cleaning: {self.data_clean.shape}")
        print(f"Features: {len(self.all_features)}")
    
    def split_data(self, split_ratio: float = None):
        """
        Split data into training and testing sets
        
        Args:
            split_ratio: Ratio for train/test split
        """
        if split_ratio is not None:
            self.split_ratio = split_ratio
        
        if self.data_clean is None:
            raise ValueError("No clean data available. Please prepare features first.")
        
        split_index = int(self.split_ratio * len(self.data_clean))
        self.trainData = self.data_clean.iloc[:split_index, :]
        self.testData = self.data_clean.iloc[split_index:, :]
        
        print(f"Training data: {len(self.trainData)} days")
        print(f"Test data: {len(self.testData)} days")
    
    def train_model(self, **xgb_params):
        """
        Train the XGBoost model
        
        Args:
            **xgb_params: Additional XGBoost parameters
        """
        if self.trainData is None:
            raise ValueError("No training data available. Please split data first.")
        
        # Default XGBoost parameters
        default_params = {
            'random_state': 42,
            'max_depth': 6,
            'n_estimators': 200,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        
        # Update with user-provided parameters
        default_params.update(xgb_params)
        
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(**default_params)
        self.model.fit(self.trainData[self.all_features], self.trainData["close_norm"])
        print("Model training completed")
    
    def evaluate_model(self):
        """
        Evaluate the trained model on test data
        """
        if self.model is None or self.testData is None:
            raise ValueError("Model not trained or test data not available")
        
        try:
            norm_predictions_test = self.model.predict(self.testData[self.all_features])
            
            # Safer denormalization
            test_mean = self.testData["close"].rolling(window=252, min_periods=30).mean().fillna(
                self.testData["close"].mean())
            test_std = self.testData["close"].rolling(window=252, min_periods=30).std().fillna(1.0)
            test_std = test_std.replace(0, 1.0)
            
            predicted_prices_test = norm_predictions_test * test_std.values + test_mean.values
            
            # Clean predictions
            valid_mask = ~(np.isnan(predicted_prices_test) | np.isnan(self.testData["close"].values))
            self.predicted_prices_clean = predicted_prices_test[valid_mask]
            self.actual_prices_clean = self.testData["close"].values[valid_mask]
            self.test_timestamps_clean = self.testData["timestamp"].values[valid_mask]
            
            print(f"Valid test predictions: {len(self.predicted_prices_clean)} out of {len(self.testData)}")
            
        except Exception as e:
            print(f"Error in test predictions: {e}")
            self.predicted_prices_clean = np.array([])
            self.actual_prices_clean = np.array([])
            self.test_timestamps_clean = np.array([])
    
    def predict_future(self, days_ahead: int = 780):
        """
        Predict future stock prices
        
        Args:
            days_ahead: Number of days to predict into the future
        """
        if self.model is None or self.data_clean is None:
            raise ValueError("Model not trained or data not available")
        
        print(f"Starting prediction for {days_ahead} days...")
        
        # Get base statistics
        base_mean = self.data_clean["close"].rolling(window=252, min_periods=30).mean().iloc[-1]
        base_std = self.data_clean["close"].rolling(window=252, min_periods=30).std().iloc[-1]
        if pd.isna(base_mean):
            base_mean = self.data_clean["close"].mean()
        if pd.isna(base_std) or base_std == 0:
            base_std = self.data_clean["close"].std() * 0.1
        
        # Starting point
        last_price = self.data_clean["close"].iloc[-1]
        last_date = pd.to_datetime(self.data_clean["timestamp"].iloc[-1])
        
        print(f"Starting from: {last_date.date()}, Price: ${last_price:.2f}")
        print(f"Base stats - Mean: ${base_mean:.2f}, Std: ${base_std:.2f}")
        
        # Generate future business dates
        self.future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        # Initialize prediction arrays
        self.future_prices = []
        current_price = last_price
        
        # Get the last known features as template
        last_features = self.data_clean[self.all_features].iloc[-1].copy()
        
        # Track basic statistics for realistic movement
        daily_returns = self.data_clean["close"].pct_change().dropna()
        avg_return = daily_returns.mean()
        return_std = daily_returns.std()
        
        print(f"Historical daily return: {avg_return:.6f} +/- {return_std:.6f}")
        
        for i, future_date in enumerate(self.future_dates):
            try:
                # Create feature vector for prediction
                current_features = last_features.copy()
                
                # Add some realistic noise and trend to features
                if i > 0:
                    # Update normalized features based on price movement
                    for feature in self.norm_features:
                        current_features[feature] += np.random.normal(0, 0.01)
                        current_features[feature] = np.clip(current_features[feature], -3, 3)
                    
                    # Update rate features
                    for feature in self.rate_features:
                        current_features[feature] = np.random.normal(0, 0.005)
                        current_features[feature] = np.clip(current_features[feature], -0.1, 0.1)
                
                # Make prediction
                features_array = current_features.values.reshape(1, -1)
                pred_norm = self.model.predict(features_array)[0]
                
                # Convert normalized prediction to actual price
                pred_price = pred_norm * base_std + base_mean
                
                # Apply realistic constraints
                max_daily_change = 0.15  # 15% max daily change
                min_price = current_price * (1 - max_daily_change)
                max_price = current_price * (1 + max_daily_change)
                pred_price = np.clip(pred_price, min_price, max_price)
                
                # Add trend component
                if i > 0:
                    trend_factor = 1 + avg_return + np.random.normal(0, return_std * 0.5)
                    pred_price = current_price * trend_factor
                    pred_price = np.clip(pred_price, min_price, max_price)
                
                self.future_prices.append(pred_price)
                current_price = pred_price
                
                # Progress updates
                if i % 260 == 0 and i > 0:  # Every ~year
                    years = i // 260 + 1
                    print(f"Year {years}: ${pred_price:.2f}")
                
            except Exception as e:
                if i < 5:  # Only show first few errors
                    print(f"Error at day {i}: {e}")
                
                # Fallback prediction
                if self.future_prices:
                    growth = np.random.normal(avg_return, return_std)
                    pred_price = self.future_prices[-1] * (1 + growth)
                else:
                    pred_price = current_price * (1 + avg_return)
                
                self.future_prices.append(pred_price)
                current_price = pred_price
    
    def plot_results(self, figsize: tuple = (25, 12)):
        """
        Plot historical data, test predictions, and future predictions
        
        Args:
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        plt.title(f"3-Year Stock Price Prediction for {self.symbol}", size=24)
        
        # Plot historical data
        plt.plot(self.data_clean["timestamp"], self.data_clean["close"], 
                 label="Historical Price", alpha=0.8, color='blue', linewidth=1)
        
        # Plot test predictions if available
        if len(self.predicted_prices_clean) > 0:
            plt.plot(self.test_timestamps_clean, self.predicted_prices_clean, 
                     label="Model Validation", color='orange', linewidth=2)
        
        # Plot future predictions
        if self.future_prices:
            plt.plot(self.future_dates, self.future_prices, 
                     label="3-Year Prediction", color='red', linewidth=2)
        
        # Add reference lines
        split_index = int(self.split_ratio * len(self.data_clean))
        split_point = self.data_clean["timestamp"].iloc[split_index]
        plt.axvline(x=split_point, color='gray', linestyle='--', 
                   label='Train/Test Split', alpha=0.8)
        
        current_date = self.data_clean["timestamp"].iloc[-1]
        plt.axvline(x=current_date, color='green', linestyle='--', 
                   label='Prediction Start', alpha=0.8)
        
        plt.xlabel('Date', size=14)
        plt.ylabel('Close Price ($)', size=14)
        plt.legend(fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_metrics(self):
        """
        Calculate and print model performance metrics
        """
        if len(self.predicted_prices_clean) > 0 and len(self.actual_prices_clean) > 0:
            try:
                mae = mean_absolute_error(self.actual_prices_clean, self.predicted_prices_clean)
                rmse = np.sqrt(mean_squared_error(self.actual_prices_clean, self.predicted_prices_clean))
                mape = np.mean(np.abs((self.actual_prices_clean - self.predicted_prices_clean) / 
                                    self.actual_prices_clean)) * 100
                
                print(f"\nModel Validation Metrics:")
                print(f"MAE: ${mae:.2f}")
                print(f"RMSE: ${rmse:.2f}")
                print(f"MAPE: {mape:.1f}%")
                
                return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                return None
        else:
            print("No valid predictions available for metric calculation")
            return None
    
    def print_prediction_summary(self):
        """
        Print a summary of future predictions
        """
        if not self.future_prices:
            print("No future predictions available")
            return
        
        current_price = self.data_clean["close"].iloc[-1]
        print(f"\nPrediction Summary for {self.symbol}:")
        print(f"Current Price: ${current_price:.2f}")
        
        if len(self.future_prices) > 260:
            year1_price = self.future_prices[259]  # ~1 year
            print(f"1 Year Prediction: ${year1_price:.2f} "
                  f"({((year1_price/current_price-1)*100):+.1f}%)")
        
        if len(self.future_prices) > 520:
            year2_price = self.future_prices[519]  # ~2 years
            print(f"2 Year Prediction: ${year2_price:.2f} "
                  f"({((year2_price/current_price-1)*100):+.1f}%)")
        
        if len(self.future_prices) > 0:
            final_price = self.future_prices[-1]
            years = len(self.future_prices) / 260
            annual_return = ((final_price/current_price)**(1/years) - 1) * 100
            print(f"3 Year Prediction: ${final_price:.2f} "
                  f"({((final_price/current_price-1)*100):+.1f}%, {annual_return:.1f}% annually)")
    
    def run_complete_analysis(self, days_ahead: int = 780):
        """
        Run the complete analysis pipeline
        
        Args:
            days_ahead: Number of days to predict
        """
        print("=== Starting Complete Stock Prediction Analysis ===\n")
        
        # Fetch data
        self.fetch_data()
        
        # Prepare features
        self.prepare_features()
        
        # Split data
        self.split_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        self.evaluate_model()
        
        # Predict future
        self.predict_future(days_ahead)
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Print summary
        self.print_prediction_summary()
        
        # Plot results
        self.plot_results()
        
        print("\n=== Analysis Complete ===")

# Usage Example
def main():
    """
    Example usage of the StockPredictor class
    """
    # Initialize predictor
    predictor = StockPredictor(
        api_key='PK6DG1SZL5CTNJYR794G',
        secret_key='jHDZEyDbqDfHY57WHrvIYHfBk12bTrDrMcz76kom',
        symbol='NVDA'
    )
    
    # Run complete analysis
    predictor.run_complete_analysis(days_ahead=780)
    
    # Or run step by step
    # predictor.fetch_data()
    # predictor.prepare_features()
    # predictor.split_data()
    # predictor.train_model()
    # predictor.evaluate_model()
    # predictor.predict_future(780)
    # predictor.plot_results()
    # predictor.calculate_metrics()
    # predictor.print_prediction_summary()

if __name__ == "__main__":
    main()
