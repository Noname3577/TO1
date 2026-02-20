# Configuration for MT5 Auto Trading
# Edit these settings as needed

# Trading Settings
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_LOT_SIZE = 0.01
DEFAULT_SL_PIPS = 50
DEFAULT_TP_PIPS = 100
DEVIATION = 20  # Price deviation for order execution
MAGIC_NUMBER = 234000  # Unique identifier for orders from this bot

# Account Settings (Optional - Leave empty to use current logged-in account)
# If you want to auto-login to MT5, fill in these details:
MT5_LOGIN = None  # Your MT5 account number (int) or None
MT5_PASSWORD = None  # Your MT5 account password (str) or None
MT5_SERVER = None  # Your broker's server name (str) or None

# Example:
# MT5_LOGIN = 12345678
# MT5_PASSWORD = "YourPassword123"
# MT5_SERVER = "YourBroker-Demo"

# UI Settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
AUTO_REFRESH_INTERVAL = 5000  # milliseconds (5 seconds)
ENABLE_AUTO_REFRESH = False

# Risk Management
MAX_LOT_SIZE = 1.0
MIN_LOT_SIZE = 0.01
MAX_POSITIONS = 10  # Maximum number of open positions allowed

# Logging
ENABLE_LOGGING = True
LOG_FILE = "trading_log.txt"
VERBOSE = False  # Set to True for detailed logging
