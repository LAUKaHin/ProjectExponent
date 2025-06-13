from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Market import Market

class Action(Enum):
    """Trading actions enumeration - matches COMP2012 enum Action"""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"

class Strategy(ABC):
    """Abstract base class for trading strategies - matches COMP2012 Strategy class"""
    
    def __init__(self, name: str = ""):
        """Constructor matching Strategy(const string& name)"""
        self.name = name
    
    def get_name(self) -> str:
        """Return strategy name - matches string getName() const"""
        return self.name
    
    @abstractmethod
    def decide_action(self, market: 'Market', index: int, current_holding: float) -> Action:
        """
        Abstract method to decide trading action
        Matches: Action decideAction(Market *market, int index, double currentHolding) const = 0
        """
        pass
    
    def calculate_moving_average(self, market: 'Market', index: int, window: int) -> float:
        """
        Calculate moving average of stock prices over specified window
        Matches: double calculateMovingAverage(Market *market, int index, int window) const
        
        Following COMP2012 specification:
        - Check if window falls beyond array bounds, if yes start from index 0
        - Calculate average of most recent prices within defined window
        """
        # Check if window falls beyond array bounds
        start = max(0, index - window + 1)
        
        # Sum prices within the window
        prices_sum = 0.0
        count = 0
        for i in range(start, index + 1):
            prices_sum += market.get_price(i)
            count += 1
        
        # Return average
        return prices_sum / count if count > 0 else 0.0

class MeanReversionStrategy(Strategy):
    """
    Mean reversion trading strategy
    Matches COMP2012 MeanReversionStrategy class
    """
    
    def __init__(self, name: str = "", window: int = 0, threshold: int = 0):
        """Constructor matching MeanReversionStrategy(const string &name, int window, int threshold)"""
        super().__init__(name)
        self.window = window
        self.threshold = threshold
    
    def decide_action(self, market: 'Market', index: int, current_holding: float) -> Action:
        """
        Decide action based on mean reversion logic
        Matches: Action decideAction(Market *market, int index, double currentHolding) const override
        
        Logic: If price is below MA*(1-threshold%), BUY. If above MA*(1+threshold%), SELL.
        Cannot buy if already holding (currentHolding == 1)
        Cannot sell if not holding (currentHolding == 0)
        """
        ma = self.calculate_moving_average(market, index, self.window)
        current_price = market.get_price(index)
        
        # Calculate thresholds
        lower_threshold = ma * (1 - self.threshold / 100.0)
        upper_threshold = ma * (1 + self.threshold / 100.0)
        
        if current_price <= lower_threshold and current_holding != 1:
            return Action.BUY
        elif current_price >= upper_threshold and current_holding != 0:
            return Action.SELL
        else:
            return Action.HOLD
    
    @staticmethod
    def generate_strategy_set(base_name: str, min_window: int, max_window: int, window_step: int,
                            min_threshold: int, max_threshold: int, threshold_step: int) -> list:
        """
        Generate set of MeanReversionStrategy objects with varying parameters
        Matches: static MeanReversionStrategy **generateStrategySet(...)
        
        Creates strategies for each combination of window sizes and thresholds
        Names follow pattern: baseName + "_" + window + "_" + threshold
        """
        strategies = []
        
        for window in range(min_window, max_window + 1, window_step):
            for threshold in range(min_threshold, max_threshold + 1, threshold_step):
                strategy_name = f"{base_name}_{window}_{threshold}"
                strategy = MeanReversionStrategy(strategy_name, window, threshold)
                strategies.append(strategy)
        
        return strategies

class TrendFollowingStrategy(Strategy):
    """
    Trend following strategy based on moving average crossovers
    Matches COMP2012 TrendFollowingStrategy class
    """
    
    def __init__(self, name: str = "", short_window: int = 0, long_window: int = 0):
        """Constructor matching TrendFollowingStrategy(const string &name, int shortWindow, int longWindow)"""
        super().__init__(name)
        self.short_moving_average_window = short_window
        self.long_moving_average_window = long_window
    
    def decide_action(self, market: 'Market', index: int, current_holding: float) -> Action:
        """
        Decide action based on moving average crossover
        Matches: Action decideAction(Market *market, int index, double currentHolding) const override
        
        Logic: If short MA > long MA, BUY. If short MA < long MA, SELL.
        Cannot buy if already holding (currentHolding == 1)
        Cannot sell if not holding (currentHolding == 0)
        """
        short_ma = self.calculate_moving_average(market, index, self.short_moving_average_window)
        long_ma = self.calculate_moving_average(market, index, self.long_moving_average_window)
        
        if short_ma > long_ma and current_holding != 1:
            return Action.BUY
        elif short_ma < long_ma and current_holding != 0:
            return Action.SELL
        else:
            return Action.HOLD
    
    @staticmethod
    def generate_strategy_set(name: str, min_short_window: int, max_short_window: int, step_short_window: int,
                            min_long_window: int, max_long_window: int, step_long_window: int) -> list:
        """
        Generate set of TrendFollowingStrategy objects with varying window sizes
        Matches: static TrendFollowingStrategy **generateStrategySet(...)
        
        Creates strategies for each combination of short and long window sizes
        Names follow pattern: baseName + "_" + shortWindow + "_" + longWindow
        """
        strategies = []
        
        for short_window in range(min_short_window, max_short_window + 1, step_short_window):
            for long_window in range(min_long_window, max_long_window + 1, step_long_window):
                strategy_name = f"{name}_{short_window}_{long_window}"
                strategy = TrendFollowingStrategy(strategy_name, short_window, long_window)
                strategies.append(strategy)
        
        return strategies

class WeightedTrendFollowingStrategy(TrendFollowingStrategy):
    """
    Weighted trend following strategy using exponential weights
    Matches COMP2012 WeightedTrendFollowingStrategy class
    """
    
    def __init__(self, name: str = "", short_window: int = 0, long_window: int = 0):
        """Constructor matching WeightedTrendFollowingStrategy(const string &name, int shortWindow, int longWindow)"""
        super().__init__(name, short_window, long_window)
    
    def _calculate_exponential_weight(self, index: int) -> float:
        """
        Calculate exponential weight for given index
        Matches: double calculateExponentialWeight(int index) const
        
        Uses growth factor of 1.1 (+10%), starting with weight 1.0
        More recent prices have greater influence
        """
        return 1.1 ** index
    
    def calculate_moving_average(self, market: 'Market', index: int, window: int) -> float:
        """
        Calculate weighted moving average using exponential weights
        Matches: double calculateMovingAverage(Market *market, int index, int window) const override
        
        Uses exponential weights determined by calculateExponentialWeight
        Sums weighted prices and divides by total weight
        """
        # Check if window falls beyond array bounds
        start = max(0, index - window + 1)
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i in range(start, index + 1):
            weight = self._calculate_exponential_weight(i - start)
            weighted_sum += market.get_price(i) * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    @staticmethod
    def generate_strategy_set(name: str, min_short_window: int, max_short_window: int, step_short_window: int,
                            min_long_window: int, max_long_window: int, step_long_window: int) -> list:
        """
        Generate set of WeightedTrendFollowingStrategy objects with varying window sizes
        Matches: static WeightedTrendFollowingStrategy **generateStrategySet(...)
        
        Creates strategies for each combination of short and long window sizes
        Names follow pattern: baseName + "_" + shortWindow + "_" + longWindow
        """
        strategies = []
        
        for short_window in range(min_short_window, max_short_window + 1, step_short_window):
            for long_window in range(min_long_window, max_long_window + 1, step_long_window):
                strategy_name = f"{name}_{short_window}_{long_window}"
                strategy = WeightedTrendFollowingStrategy(strategy_name, short_window, long_window)
                strategies.append(strategy)
        
        return strategies