import os
import pandas as pd
import argparse


def merge_data(input_dir, output_file, tic):
    """
    Merges all CSV files in the input_dir into a single CSV file.
    add 'tic' column with the provided ticker symbol.
    calculates 'day' column (weekday).
    Sorts by date.
    """
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not all_files:
        print(f"No CSV files found in {input_dir}")
        return

    df_list = []
    
    print(f"Found {len(all_files)} files. Processing...")

    for file in all_files:
        try:
            df = pd.read_csv(file)
            
            # Ensure required columns exist (based on observation of spy_2020_08_trades.csv)
            # data/spy_202008_202507/spy_2020_08_trades.csv columns:
            # date,open,high,low,close,volume,average,barCount
            
            # We need to map to: date,close,high,low,open,volume,tic,day
            
            # "date" column already exists.
            
            # Add 'tic'
            df['tic'] = tic
            
            # Convert date to datetime to extract day of week
            # We accept whatever format is there, hopefully pandas handles it parsing or we just parse it
            # The sample showed: 2020-08-03 09:30:00-04:00
            df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Add 'day' (0=Monday, 6=Sunday)
            df['day'] = df['date'].dt.dayofweek
            
            # Ensure volume is integer
            if 'volume' in df.columns:
                df['volume'] = df['volume'].astype(int)
            
            # Select and reorder columns
            # Target: date,close,high,low,open,volume,tic,day
            # Note: The target sample kept full timestamp in date column?
            # User said: "allow the ... first column to save full timestame, rather than only the date."
            # So we keep the datetime objects or convert back to string if needed. 
            # Pandas default string representation for datetime usually includes time.
            
            cols_to_keep = ['date', 'close', 'high', 'low', 'open', 'volume', 'tic', 'day']
            
            # Check if all cols exist
            if not all(col in df.columns for col in cols_to_keep):
                print(f"Warning: Missing columns in {file}. Skipping. Found: {df.columns}")
                continue
                 
            df_filtered = df[cols_to_keep]
            df_list.append(df_filtered)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not df_list:
        print("No data processed.")
        return

    print("Concatenating data...")
    final_df = pd.concat(df_list, ignore_index=True)
    
    print("Sorting by date...")
    final_df = final_df.sort_values(by='date')
    
    print(f"Saving to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge raw trading data files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing raw CSV files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged CSV file.")
    parser.add_argument("--tic", type=str, required=True, help="Ticker symbol to assign to the data (e.g. SPY).")
    
    args = parser.parse_args()
    
    merge_data(args.input_dir, args.output_file, args.tic)
