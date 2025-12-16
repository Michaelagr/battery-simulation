"""
Configuration constants for battery simulation.
"""

# Battery defaults
BATTERY_EFFICIENCY = 0.9
DEFAULT_DEPTH_OF_DISCHARGE = 90  # percentage
INTERVAL_HOURS = 0.25  # 15-minute intervals

# Financial constants
EXPORT_REVENUE_FACTOR = 0.9
VOLLLASTSTUNDEN_THRESHOLD = 2500
DEMAND_CHARGE_HIGH = 174  # €/kW/year
DEMAND_CHARGE_LOW = 20    # €/kW/year

# Price analysis defaults
LOW_PRICE_PERCENTILE = 30
HIGH_PRICE_PERCENTILE = 70
MIN_ARBITRAGE_SPREAD = 0.005  # €/kWh
NEGATIVE_PRICE_THRESHOLD = -0.01  # €/kWh

# Performance settings
ENABLE_CHART_OPTIMIZATION = True
MIN_POINTS_FOR_RESAMPLING = 10000
RESAMPLE_FREQUENCY = "1H"

# Feature flags
ENABLE_ARBITRAGE = True  # Enable energy arbitrage (load shifting) by default

# Battery configurations
BATTERY_CONFIGS = {
    "Small": {"capacity": 90, "power": 92},
    "Medium": {"capacity": 215, "power": 100},
    "Large": {"capacity": 500, "power": 250}
}