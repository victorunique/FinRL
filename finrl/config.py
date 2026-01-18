# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2014-01-06"  # bug fix: set Monday right, start date set 2014-01-01 ValueError: all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1658 and the array at index 1 has size 1657
TRAIN_END_DATE = "2020-07-31"

TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2021-10-01"

TRADE_START_DATE = "2021-11-01"
TRADE_END_DATE = "2021-12-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "atr",  # Average True Range (Default 14 periods)
    "atr_20",  # Average True Range (20 periods)
    "macd",  # Moving Average Convergence Divergence
    "macds",  # MACD Signal Line
    "macdh",  # MACD Histogram
    "boll_ub",  # Bollinger Bands Upper Band
    "boll_lb",  # Bollinger Bands Lower Band
    "boll",  # Bollinger Bands Middle Band (Main)
    "rsi_14",  # Relative Strength Index (Default 14 periods)
    "rsi_6",  # Relative Strength Index (6 periods)
    "adx",  # Average Directional Index (Default 14 periods)
    "dx",  # Directional Movement Index
    "close_12_ema",  # Exponential Moving Average (12 periods)
    "close_26_ema",  # Exponential Moving Average (26 periods)
    "close_50_ema",  # Exponential Moving Average (50 periods)
    "close_30_sma",  # Simple Moving Average (30 periods)
    "close_60_sma",  # Simple Moving Average (60 periods)
    "kdjk",  # Stochastic Oscillator K value
    "kdjd",  # Stochastic Oscillator D value
    "kdjj",  # Stochastic Oscillator J value
    "cci",  # Commodity Channel Index (Default 14 periods)
    "cci_20",  # Commodity Channel Index (20 periods)
    "tr",  # True Range
]


# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}


# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = "Asia/Jakarta"  # LQ45
TIME_ZONE_SELFDEFINED = "xxx"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)

# parameters for data sources
ALPACA_API_KEY = "xxx"  # your ALPACA_API_KEY
ALPACA_API_SECRET = "xxx"  # your ALPACA_API_SECRET
ALPACA_API_BASE_URL = "https://paper-api.alpaca.markets"  # alpaca url
BINANCE_BASE_URL = "https://data.binance.vision/"  # binance url
