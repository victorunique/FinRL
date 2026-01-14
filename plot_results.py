import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import argparse


def parse_list_string(list_str):
    try:
        # The CSV saves lists as strings like "[1.23, 4.56]"
        # We need to handle cases where it might be space separated inside or standard comma
        # But 'ast.literal_eval' usually works for standard python list repr
        return ast.literal_eval(list_str)
    except Exception as e:
        print(f"Error parsing: {list_str} -> {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Plot Trading Results")
    parser.add_argument("--model_name", type=str, default="ppo_stoploss_agent", 
                        help="Name of the model to plot (default: ppo_stoploss_agent)")
    args = parser.parse_args()

    results_dir = "results"
    account_file = os.path.join(results_dir, f"{args.model_name}_account_history.csv")
    actions_file = os.path.join(results_dir, f"{args.model_name}_action_history.csv")

    if not os.path.exists(account_file) or not os.path.exists(actions_file):
        print(f"Error: Files not found in {results_dir}")
        return

    # Load data
    df_account = pd.read_csv(account_file)
    df_actions = pd.read_csv(actions_file)

    # Parse dates
    df_account['date'] = pd.to_datetime(df_account['date'])
    df_actions['date'] = pd.to_datetime(df_actions['date'])

    # Merge to ensure we have aligned data
    # We strip time info if necessary, but data seems to match exactly in previous `head` output
    df = pd.merge(df_account, df_actions, on='date', how='inner')
    
    # Process actions/transactions
    # user asked for 'actions' but referenced values that turned out to be scaled actions.
    # 'transactions' contains actual shares bought/sold.
    # We will use 'transactions' to determine Buy vs Sell.
    
    # Check format of transactions column - it is a string representation of a list
    df['transactions_list'] = df['transactions'].apply(parse_list_string)
    
    # Calculate net transaction volume for the step (sum of all tickers)
    # Positive = Buy, Negative = Sell
    df['net_transaction'] = df['transactions_list'].apply(lambda x: sum(x) if isinstance(x, list) else 0)

    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Plot Total Assets
    plt.plot(df['date'], df['total_assets'], label='Total Assets', color='blue', alpha=0.7)
    
    # Overlay Actions
    # Buys
    buys = df[df['net_transaction'] > 0]
    plt.scatter(buys['date'], buys['total_assets'], color='green', marker='^', label='Buy', s=50, zorder=5)
    
    # Sells
    sells = df[df['net_transaction'] < 0]
    plt.scatter(sells['date'], sells['total_assets'], color='red', marker='v', label='Sell', s=50, zorder=5)

    plt.title('Portfolio Value & Trading Actions')
    plt.xlabel('Date')
    plt.ylabel('Total Assets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(results_dir, f"{args.model_name}_backtest_plot.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
