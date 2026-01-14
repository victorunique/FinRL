import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from finrl import config_tickers

# =======================================================================================
# Configuration (Must match quickstart_ppo_stoploss.py)
# =======================================================================================
PREPROCESSED_DATA_DIR = "preprocessed_data"
RESULTS_DIR = "results"
INITIAL_AMOUNT = 1000000
TRANSACTION_COST_PCT = 0.001  # 0.1%


def calculate_max_drawdown(series):
    """
    Calculates Maximum Drawdown (MDD) of a value series.
    MDD = (Trough Value - Peak Value) / Peak Value
    """
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_sortino_ratio(series, risk_free_rate=0):
    """
    Calculates Sortino Ratio based on daily returns.
    Sortino = (Mean Excess Return) / (Downside Deviation)
    """
    returns = series.pct_change().dropna()
    mean_return = returns.mean()
    
    # Downside deviation: Standard deviation of returns below the target (risk_free_rate)
    downside_returns = returns[returns < risk_free_rate]
    if len(downside_returns) == 0:
        return np.nan
        
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.inf
        
    # Annualize (assuming 252 trading days)
    # Sortino is usually annualized
    sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(252)
    return sortino


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Buy & Hold Strategy")
    parser.add_argument("--model_name", type=str, default="ppo_stoploss_agent", 
                        help="Name of the model to compare against (default: ppo_stoploss_agent)")
    args = parser.parse_args()

    # 1. Load PPO Agent Results (to get Trade Dates)
    ppo_account_file = os.path.join(RESULTS_DIR, f"{args.model_name}_account_history.csv")
    if not os.path.exists(ppo_account_file):
        print(f"Error: {ppo_account_file} not found. Needs PPO results to determine trade period.")
        return

    print(f"Loading PPO results from {ppo_account_file} to determine trade dates...")
    ppo_df = pd.read_csv(ppo_account_file)
    ppo_df['date'] = pd.to_datetime(ppo_df['date'])
    
    # Determine Trade Start and End Dates from PPO results
    start_date = ppo_df['date'].min()
    end_date = ppo_df['date'].max()
    print(f"Dynamic Trade Period: {start_date} to {end_date}")

    # 2. Load Preprocessed Data
    data_file = os.path.join(PREPROCESSED_DATA_DIR, "preprocessed_data.csv")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run quickstart_ppo_stoploss.py to generate data first.")
        return

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])

    # 3. Filter for Trading Period
    # Detect Timezone in Data
    if df['date'].dt.tz is not None:
        print(f"Data is timezone-aware: {df['date'].dt.tz}")
        if start_date.tz is None:
            start_date = start_date.tz_localize(df['date'].dt.tz)
            end_date = end_date.tz_localize(df['date'].dt.tz)
    else:
        print("Data is timezone-naive")
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
            end_date = end_date.tz_localize(None)

    # Note: PPO results are inclusive of the end date usually, so we use <=
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    trade_df = df[mask].copy()
    
    if trade_df.empty:
        print(f"No data found for the range {start_date} to {end_date}")
        return

    # 3. Pivot Data to get Close Prices matrix (Index=Date, Columns=Ticker)
    # We use 'close' price for portfolio valuation
    df_close = trade_df.pivot(index='date', columns='tic', values='close')
    
    # Handle any missing values (forward fill then backward fill)
    df_close = df_close.ffill().bfill()
    
    tickers = df_close.columns.tolist()
    num_tickers = len(tickers)
    print(f"Calculating Buy & Hold performance for {num_tickers} tickers: {tickers}")

    # 4. Simulate Buy and Hold
    # Strategy: Buy equal dollar amount of each stock at the first available price.
    
    # Prices at the start of the trading period
    start_prices = df_close.iloc[0]
    
    # Allocate capital equally
    allocation_per_ticker = INITIAL_AMOUNT / num_tickers
    
    # Calculate shares bought (accounting for transaction cost)
    # invested_amount = shares * price
    # total_cost = invested_amount + (invested_amount * cost_pct) = invested_amount * (1 + cost_pct)
    # allocated = invested_amount * (1 + cost_pct)
    # invested_amount = allocated / (1 + cost_pct)
    # shares = invested_amount / price
    
    shares_bought = (allocation_per_ticker / (1 + TRANSACTION_COST_PCT)) / start_prices
    
    # Calculate Portfolio Value Over Time
    # Value = Shares * Current Price
    portfolio_value_per_ticker = df_close.multiply(shares_bought, axis=1)
    
    # Total Account Value = Sum of all ticker values + Residual Cash (negligible/zero here effectively)
    # We can assume residual cash is kept constant or ignore it for this approximation as it's very small.
    # Let's calculate the minimal residual cash just to be precise with the initial amount.
    actual_invested = (shares_bought * start_prices).sum()
    actual_cost = actual_invested * TRANSACTION_COST_PCT
    residual_cash = INITIAL_AMOUNT - (actual_invested + actual_cost)
    
    # Baseline Account Value Series
    baseline_account_value = portfolio_value_per_ticker.sum(axis=1) + residual_cash
    
    # 5. Load PPO Agent Results for Comparison (already loaded)
    ppo_values = None
    if ppo_df is not None:
        # Ensure alignment (sometimes PPO results might have slight date mismatches or different index)
        ppo_df_indexed = ppo_df.set_index('date')
        
        # We might need to handle duplicate index if any
        if ppo_df_indexed.index.duplicated().any():
            ppo_df_indexed = ppo_df_indexed[~ppo_df_indexed.index.duplicated(keep='first')]

        # Reindex ppo to match baseline index to ensure fair comparison on same days
        # Fill missing with ffill (hold value)
        ppo_values = ppo_df_indexed['total_assets'].reindex(baseline_account_value.index, method='ffill')
    
    # 6. Performance Metrics
    
    # --- Metrics for Baseline (Buy & Hold) ---
    baseline_final = baseline_account_value.iloc[-1]
    baseline_return = ((baseline_final - INITIAL_AMOUNT) / INITIAL_AMOUNT) * 100
    baseline_mdd = calculate_max_drawdown(baseline_account_value)
    baseline_sortino = calculate_sortino_ratio(baseline_account_value)

    # --- Metrics for PPO Agent ---
    ppo_return = float('nan')
    ppo_mdd = float('nan')
    ppo_sortino = float('nan')

    if ppo_values is not None:
        ppo_final = ppo_values.iloc[-1]
        ppo_return = ((ppo_final - INITIAL_AMOUNT) / INITIAL_AMOUNT) * 100
        ppo_mdd = calculate_max_drawdown(ppo_values)
        ppo_sortino = calculate_sortino_ratio(ppo_values)
    
    print("\n" + "=" * 50)
    print(f"{'Metric':<20} | {'Buy & Hold':<12} | {'PPO Agent':<12}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {baseline_return:>11.2f}% | {ppo_return:>11.2f}%")
    print(f"{'Max Drawdown':<20} | {baseline_mdd:>11.2%} | {ppo_mdd:>11.2%}")
    print(f"{'Sortino Ratio':<20} | {baseline_sortino:>12.2f} | {ppo_sortino:>12.2f}")
    print("=" * 50 + "\n")

    # 7. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Baseline
    plt.plot(baseline_account_value.index, baseline_account_value.values, label='Buy & Hold', color='gray', linestyle='--')
    
    # Plot PPO if available
    if ppo_values is not None:
        plt.plot(ppo_values.index, ppo_values.values, label='PPO Agent', color='blue')

    plt.title('Strategy Performance Comparison: PPO vs Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Account Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, f"{args.model_name}_baseline_comparison.png")
    plt.savefig(plot_path)
    print(f"Comparison chart saved to {plot_path}")
    
    # Save baseline data
    baseline_csv_path = os.path.join(RESULTS_DIR, f"{args.model_name}_baseline_account_value.csv")
    baseline_account_value.to_csv(baseline_csv_path)
    print(f"Baseline data saved to {baseline_csv_path}")


if __name__ == "__main__":
    main()
