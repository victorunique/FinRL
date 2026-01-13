import pandas as pd
import numpy as np
import datetime
import os
import torch

# FinRL library imports
# yahoo downloader is used to fetch data from Yahoo Finance
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
# FeatureEngineer performs preprocessing and technical indicator calculation
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# The specific environment with stop-loss mechanism we want to use
from finrl.meta.env_stock_trading.env_stocktrading_stoploss import StockTradingEnvStopLoss
# DRLAgent wrapper for Stable Baselines 3 algorithms
from finrl.agents.stablebaselines3.models import DRLAgent
# Configuration files for default tickers and indicators
from finrl import config_tickers
from finrl.config import INDICATORS
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from finrl.agents.stablebaselines3.custom_models import CNN1DFeaturesExtractor

# =======================================================================================
# 1. Configuration Constants
# =======================================================================================

# Define the directory where trained models and results will be saved
TRAINED_MODEL_DIR = "trained_models"
RESULTS_DIR = "results"
PREPROCESSED_DATA_DIR = "preprocessed_data"
os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

# Define the timeframe for training and trading (validation/testing)
# It is important to have a distinct split to prevent data leakage.
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2023-01-01'
TRADE_START_DATE = '2023-01-01'
TRADE_END_DATE = '2024-01-01'

WINDOW_SIZE = 10  # Size of the sliding window for 1D CNN


# Helper function to check relative dates (sanity check)
def check_dates():
    if TRAIN_END_DATE > TRADE_START_DATE:
        print("Warning: Training end date overlaps with Trading start date.")


check_dates()

# Select the list of tickers to trade.
# DOW_30_TICKER is a standard list of 30 major US stocks provided by FinRL.
# You can define your own list: e.g., TICKER_LIST = ['AAPL', 'MSFT', 'GOOG']
TICKER_LIST = config_tickers.DOW_30_TICKER

# =======================================================================================
# 2. Data Fetching and Preprocessing
# =======================================================================================

# Check if preprocessed data is already available
PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data.csv")

if os.path.exists(PREPROCESSED_DATA_FILE):
    print(f"\nLoading preprocessed data from {PREPROCESSED_DATA_FILE}...")
    processed_df = pd.read_csv(PREPROCESSED_DATA_FILE)
else:
    # Check if data is already downloaded
    DATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, "quickstart_data.csv")
    
    if os.path.exists(DATA_FILE):
        print(f"\nLoading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
    else:
        print(f"Fetching data for {len(TICKER_LIST)} tickers from {TRAIN_START_DATE} to {TRADE_END_DATE}...")
    
        # Download data using YahooDownloader
        # This handles fetching the daily OHLCV (Open, High, Low, Close, Volume) data.
        downloader = YahooDownloader(
            start_date=TRAIN_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=TICKER_LIST
        )
        df = downloader.fetch_data()
        print(f"Saving data to {DATA_FILE}...")
        df.to_csv(DATA_FILE, index=False)
    
    print("Data fetched successfully. Starting preprocessing...")
    
    # Initialize FeatureEngineer
    # This class handles adding technical indicators, VIX, and turbulence index.
    # - use_technical_indicator: Adds standard indicators (MACD, RSI, CCI, ADX).
    # - use_vix: Adds the VIX index (volatility index) as a feature, useful for gauging market fear.
    # - use_turbulence: Adds a turbulence index to detect extreme market conditions (market crashes).
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )
    
    # Apply preprocessing
    processed_df = fe.preprocess_data(df)
    
    # Save the preprocessed data
    print(f"Saving preprocessed data to {PREPROCESSED_DATA_DIR}...")
    processed_df.to_csv(PREPROCESSED_DATA_FILE, index=False)

# The shape of the dataframe is important verification
# Expected columns: date, tic, open, high, low, close, volume, <tech_indicators>, vix, turbulence
print(f"Processed data shape: {processed_df.shape}")
print(f"Columns: {processed_df.columns.tolist()}")

# =======================================================================================
# 3. Environment Setup
# =======================================================================================

# Define the stock dimension (number of unique tickers)
stock_dimension = len(processed_df.tic.unique())

# Calculate state space dimension
# The state includes:
# 1. Current available cash (1 value)
# 2. Current share holdings for each stock (stock_dimension values)
# 3. Features for each stock (close price + tech indicators) * stock_dimension
# Note: In the stop-loss env implementation, the get_date_vector logic usually appends features.
# We approximate the state space size here for the config.
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension

# If VIX and Turbulence are used, they are often added as extra columns in the data vector
# The environment automatically handles the vector composition based on 'daily_information_cols'
# but we define 'state_space' for the agent's neural network input layer.

print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Define Environment Key Arguments
# These parameters control the market simulation and the stop-loss logic.
env_kwargs = {
    "hmax": 100,                             # Max number of shares to buy/sell per transaction
    "initial_amount": 1000000,               # Initial capital
    "buy_cost_pct": 0.001,                   # Transaction cost for buying (0.1%)
    "sell_cost_pct": 0.001,                  # Transaction cost for selling (0.1%)
    "print_verbosity": 50,                    # How often to print trading logs (every 5 steps)
    "daily_information_cols": ["open", "close", "high", "low", "volume"] + INDICATORS + ["vix", "turbulence"],
    
    # Specific Stop-Loss Parameters
    "stoploss_penalty": 0.9,                 # Threshold: if price < 0.9 * avg_buy_price, sell immediately.
    "profit_loss_ratio": 2.0,                # Target profit ratio relative to stop loss (risk/reward).
    "turbulence_threshold": None             # During training, we often set this to None to let the agent learn.
                                             # In trading, we might set a threshold (e.g., 0.99 quant) to stop trading during crashes.
}

# =======================================================================================
# 4. Training
# =======================================================================================

# Split the processed data into training set
train_df = data_split(processed_df, TRAIN_START_DATE, TRAIN_END_DATE)

# Create the training environment
# We wrap it in DummyVecEnv and VecFrameStack for the 1D CNN Policy
e_train_gym = DummyVecEnv([lambda: StockTradingEnvStopLoss(df=train_df, **env_kwargs)])
e_train_stacked = VecFrameStack(e_train_gym, n_stack=WINDOW_SIZE)

# Initialize the DRL Agent
agent = DRLAgent(env=e_train_stacked)

# Determine device (CUDA for NVIDIA, MPS for Mac, or CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Configuration for the PPO Algorithm (Proximal Policy Optimization)
# You can pass customized arguments here for policy_kwargs or model_kwargs if needed.
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
    "device": device,
}

POLICY_KWARGS = {
    "features_extractor_class": CNN1DFeaturesExtractor,
    "features_extractor_kwargs": {"features_dim": 128, "n_stack": WINDOW_SIZE},
}

# Get the PPO model instance
model_ppo = agent.get_model(
    "ppo", 
    model_kwargs=PPO_PARAMS, 
    policy_kwargs=POLICY_KWARGS,
    verbose=1
)

# Train the model
# total_timesteps determines how long the agent learns. 
# For quick testing use 10k-50k. For good results use >100k-1M.

MODEL_NAME = "ppo_stoploss_agent"
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_NAME)

if os.path.exists(MODEL_PATH + ".zip"):
    print(f"Loading existing model from {MODEL_PATH}...")
    trained_ppo = PPO.load(MODEL_PATH)
else:
    print("Starting training...")
    trained_ppo = agent.train_model(
        model=model_ppo, 
        tb_log_name="ppo_stoploss",
        total_timesteps=50000  
    )
    print("Training finished!")
    # Save the trained model (optional but recommended)
    trained_ppo.save(MODEL_PATH)

# =======================================================================================
# 5. Backtesting / Prediction
# =======================================================================================

# Split the processed data into trading (testing) set
trade_df = data_split(processed_df, TRADE_START_DATE, TRADE_END_DATE)

# Create the trading environment
# Ideally, we set the turbulence_threshold here to filter out trading on extremely volatile days.
# A common practice is to calculate the 99% quantile of turbulence from the training data.
insample_turbulence = train_df['turbulence'].values
turbulence_threshold = np.quantile(insample_turbulence, 0.99)
print(f"Setting turbulence threshold to: {turbulence_threshold}")

env_kwargs['turbulence_threshold'] = turbulence_threshold
# Create the trading environment with stacking
e_trade_gym = DummyVecEnv([lambda: StockTradingEnvStopLoss(df=trade_df, **env_kwargs)])
e_trade_stacked = VecFrameStack(e_trade_gym, n_stack=WINDOW_SIZE)

# Run prediction manually since DRLAgent.DRL_prediction doesn't handle external VecEnv well
print("Running backtest on trade/validation data...")

obs = e_trade_stacked.reset()
done = False
max_steps = len(trade_df.index.unique()) - 1
for i in range(len(trade_df.index.unique())):
    action, _states = trained_ppo.predict(obs, deterministic=True)
    obs, rewards, dones, info = e_trade_stacked.step(action)
    
    if i == max_steps - 1 or dones[0]:
        print("hit end!")
        break

df_account_value = e_trade_stacked.envs[0].save_asset_memory()
df_actions = e_trade_stacked.envs[0].save_action_memory()

# =======================================================================================
# 6. Performance Analysis (Simple)
# =======================================================================================

final_value = df_account_value.iloc[-1]['total_assets']
initial_value = env_kwargs['initial_amount']
return_pct = ((final_value - initial_value) / initial_value) * 100

print(f"Initial Portfolio Value: {initial_value}")
print(f"Final Portfolio Value:   {final_value:.2f}")
print(f"Total Return:            {return_pct:.2f}%")

# Save results to CSV
df_account_value.to_csv(os.path.join(RESULTS_DIR, "account_value_ppo.csv"))
df_actions.to_csv(os.path.join(RESULTS_DIR, "daily_actions_ppo.csv"))

print(f"Results saved to {RESULTS_DIR}")
