import pandas as pd
import numpy as np
import datetime
import os
import torch
import argparse
import random

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

# Define the timeframe for training and trading (validation/testing)
TRAIN_START_DATE = '2022-08-01'
TRAIN_END_DATE = '2025-01-01'
TRADE_START_DATE = '2025-01-01'
TRADE_END_DATE = '2025-08-01'

WINDOW_SIZE = 60  # Size of the sliding window for 1D CNN


# Helper function to check relative dates (sanity check)
def check_dates():
    if TRAIN_END_DATE > TRADE_START_DATE:
        print("Warning: Training end date overlaps with Trading start date.")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with StopLoss environment")
    
    # -----------------------------------------------------------------------------------
    # Risk/Reward Parameters (TUNABLE)
    # -----------------------------------------------------------------------------------
    parser.add_argument("--profit_loss_ratio", type=float, default=2.0, 
                        help="Target profit relative to stop loss risk (Default: 2.0). Lower this to 1.0 or 1.5 for more aggressive profit taking.")
    parser.add_argument("--stoploss_penalty", type=float, default=0.9, 
                        help="Stop loss threshold (Default: 0.9 = 10% loss). Lower to 0.85 for 15% tolerance.")
    parser.add_argument("--cash_penalty", type=float, default=0.1, 
                        help="Penalty portion for holding cash (Default: 0.1). Set to 0.0 to allow 100% investment.")
    
    # -----------------------------------------------------------------------------------
    # Training Hyperparameters
    # -----------------------------------------------------------------------------------
    parser.add_argument("--ent_coef", type=float, default=0.01, 
                        help="Entropy coefficient (Default: 0.01). Increase to 0.02-0.03 for more exploration.")
    parser.add_argument("--learning_rate", type=float, default=0.00025, 
                        help="Learning rate for PPO (Default: 0.00025)")
    parser.add_argument("--total_timesteps", type=int, default=200000, 
                        help="Total training timesteps (Default: 200000)")
    
    # -----------------------------------------------------------------------------------
    # Model Naming & Misc
    # -----------------------------------------------------------------------------------
    parser.add_argument("--model_name", type=str, default="ppo_stoploss_agent", 
                        help="Name of the model file to save (without extension)")
    
    # -----------------------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (Default: 42)")

    args = parser.parse_args()

    # Set seeds for reproducibility
    print(f"Setting global seed to {args.seed}...")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Ensure directories exist
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

    check_dates()

    # Select the list of tickers to trade.
    TICKER_LIST = config_tickers.DOW_30_TICKER

    # =======================================================================================
    # 2. Data Fetching and Preprocessing
    # =======================================================================================
    # Check if preprocessed data is already available
    PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data.csv")

    if os.path.exists(PREPROCESSED_DATA_FILE):
        print(f"\nLoading preprocessed data from {PREPROCESSED_DATA_FILE}...")
        processed_df = pd.read_csv(PREPROCESSED_DATA_FILE)
        processed_df['date'] = pd.to_datetime(processed_df['date'])
    else:
        # Check if data is already downloaded (raw)
        DATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, "quickstart_data.csv")
        
        if os.path.exists(DATA_FILE):
            print(f"\nLoading data from {DATA_FILE}...")
            df = pd.read_csv(DATA_FILE)
            df['date'] = pd.to_datetime(df['date'])
        else:
            print(f"Fetching data for {len(TICKER_LIST)} tickers from {TRAIN_START_DATE} to {TRADE_END_DATE}...")
            downloader = YahooDownloader(
                start_date=TRAIN_START_DATE,
                end_date=TRADE_END_DATE,
                ticker_list=TICKER_LIST
            )
            df = downloader.fetch_data()
            print(f"Saving data to {DATA_FILE}...")
            df.to_csv(DATA_FILE, index=False)
        
        print("Data fetched successfully. Starting preprocessing...")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=INDICATORS,
            use_vix=True,
            use_turbulence=False,
            user_defined_feature=False
        )
        processed_df = fe.preprocess_data(df)
        print(f"Saving preprocessed data to {PREPROCESSED_DATA_FILE}...")
        processed_df.to_csv(PREPROCESSED_DATA_FILE, index=False)

    print(f"Processed data shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")

    # =======================================================================================
    # 3. Environment Setup
    # =======================================================================================
    stock_dimension = len(processed_df.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # Define Environment Key Arguments using ARGPARSE variables
    env_kwargs = {
        "hmax": 100,                             
        "initial_amount": 1000000,               
        "buy_cost_pct": 0.001,                   
        "sell_cost_pct": 0.001,                  
        "print_verbosity": 1000,                    
        "discrete_actions": False,
        "daily_information_cols": ["open", "close", "high", "low", "volume"] + INDICATORS + [col for col in ["vix", "turbulence"] if col in processed_df.columns],
        
        # --- Tuned Parameters from Args ---
        "stoploss_penalty": args.stoploss_penalty,
        "profit_loss_ratio": args.profit_loss_ratio,
        "cash_penalty_proportion": args.cash_penalty,
        
        "turbulence_threshold": None             
    }

    print(f"Environment Config: profit_loss_ratio={args.profit_loss_ratio}, stoploss_penalty={args.stoploss_penalty}, cash_penalty={args.cash_penalty}")

    # =======================================================================================
    # 4. Training
    # =======================================================================================
    train_df = data_split(processed_df, TRAIN_START_DATE, TRAIN_END_DATE)

    e_train_gym = DummyVecEnv([lambda: StockTradingEnvStopLoss(df=train_df, **env_kwargs)])
    e_train_stacked = VecFrameStack(e_train_gym, n_stack=WINDOW_SIZE)

    agent = DRLAgent(env=e_train_stacked)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": args.ent_coef,  # Tuned Parameter
        "learning_rate": args.learning_rate,
        "batch_size": 128,
        "device": device,
    }

    POLICY_KWARGS = {
        "features_extractor_class": CNN1DFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128, "n_stack": WINDOW_SIZE},
    }

    model_ppo = agent.get_model(
        "ppo", 
        model_kwargs=PPO_PARAMS, 
        policy_kwargs=POLICY_KWARGS,
        verbose=1
    )

    MODEL_NAME = args.model_name
    MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_NAME)

    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_PATH}...")
        trained_ppo = PPO.load(MODEL_PATH)
    else:
        print(f"Starting training for {args.total_timesteps} timesteps...")
        trained_ppo = agent.train_model(
            model=model_ppo, 
            tb_log_name="ppo_stoploss",
            total_timesteps=args.total_timesteps
        )
        print("Training finished!")
        trained_ppo.save(MODEL_PATH)

    # =======================================================================================
    # 5. Backtesting / Prediction
    # =======================================================================================
    trade_df = data_split(processed_df, TRADE_START_DATE, TRADE_END_DATE)

    if "turbulence" in train_df.columns:
        insample_turbulence = train_df['turbulence'].values
        turbulence_threshold = np.quantile(insample_turbulence, 0.99)
        print(f"Setting turbulence threshold to: {turbulence_threshold}")
        env_kwargs['turbulence_threshold'] = turbulence_threshold
    else:
        print("Turbulence column not found, skipping turbulence threshold.")
        env_kwargs['turbulence_threshold'] = None
    
    e_trade_gym = DummyVecEnv([lambda: StockTradingEnvStopLoss(df=trade_df, **env_kwargs)])
    e_trade_stacked = VecFrameStack(e_trade_gym, n_stack=WINDOW_SIZE)

    print("Running backtest on trade/validation data...")

    obs = e_trade_stacked.reset()
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
    # We prefix the filename with the model name to avoid overwriting baseline results
    result_prefix = args.model_name
    df_account_value.to_csv(os.path.join(RESULTS_DIR, f"{result_prefix}_account_history.csv"))
    df_actions.to_csv(os.path.join(RESULTS_DIR, f"{result_prefix}_action_history.csv"))

    print(f"Results saved to {RESULTS_DIR} with prefix {result_prefix}")


if __name__ == "__main__":
    main()
