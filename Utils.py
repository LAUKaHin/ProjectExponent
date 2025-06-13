"""
Utility functions and constants following COMP2012 Assignment 2 pattern
"""
import math

# Constants
TRADING_DAYS_PER_YEAR = 252
EVALUATION_WINDOW = 100

def round_to_decimals(value: float, decimals: int) -> float:
    """
    Round value to specified number of decimal places
    Equivalent to roundToDecimals(price, 3) in C++
    """
    return round(value, decimals)

def generate_z() -> float:
    """
    Generate random number from standard normal distribution
    Equivalent to Market::generateZ() in C++
    """
    import numpy as np
    return np.random.normal(0, 1)