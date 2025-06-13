"""
Analysis Package
Clean import structure without excessive error handling
"""

# ✅ CLEAN: Direct imports in correct order
from .StabilityAnalysis import StabilityAnalyzer

from .GradingSystem import (
    SimplifiedGradingSystem,
    HKUSTStrictPercentileSystem, 
    GenAIStockEvaluator,
    EnhancedGradingWithGenAI
)

from .StockAnalyzer import (
    ComprehensiveStockAnalyzer,
    EnhancedUnifiedTradingSystem
)

from .SP500Analysis import (
    SP500ComprehensiveAnalyzer
    # ❌ REMOVED: SP500DataProvider (now in SP500Data.py)
)

from .Visualization import (
    VisualizationEngine,
    UserPreferenceManager
)

# Package metadata
__version__ = "0.0.3"
__author__ = "VictorMcTrix"

# Define exports
__all__ = [
    # Core Analysis
    'StabilityAnalyzer',
    
    # Grading Systems
    "SimplifiedGradingSystem",
    'HKUSTStrictPercentileSystem',
    'GenAIStockEvaluator',
    'EnhancedGradingWithGenAI',
    
    # Stock Analysis
    'ComprehensiveStockAnalyzer',
    'EnhancedUnifiedTradingSystem',
    
    # S&P 500 Analysis
    'SP500ComprehensiveAnalyzer',
    
    # Visualization & User Management
    'VisualizationEngine',
    'UserPreferenceManager'
]

print("📦 Analysis Package Loaded Successfully!")
