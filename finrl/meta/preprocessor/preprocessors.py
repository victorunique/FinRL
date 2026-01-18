from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MaxAbsScaler
from stockstats import StockDataFrame as Sdf

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class GroupByScaler(BaseEstimator, TransformerMixin):
    """Sklearn-like scaler that scales considering groups of data.

    In the financial setting, this scale can be used to normalize a DataFrame
    with time series of multiple tickers. The scaler will fit and transform
    data for each ticker independently.
    """

    def __init__(self, by, scaler=MaxAbsScaler, columns=None, scaler_kwargs=None):
        """Initializes GoupBy scaler.

        Args:
            by: Name of column that will be used to group.
            scaler: Scikit-learn scaler class to be used.
            columns: List of columns that will be scaled.
            scaler_kwargs: Keyword arguments for chosen scaler.
        """
        self.scalers = {}  # dictionary with scalers
        self.by = by
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs

    def fit(self, X, y=None):
        """Fits the scaler to input data.

        Args:
            X: DataFrame to fit.
            y: Not used.

        Returns:
            Fitted GroupBy scaler.
        """
        # if columns aren't specified, considered all numeric columns
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns
        # fit one scaler for each group
        for value in X[self.by].unique():
            X_group = X.loc[X[self.by] == value, self.columns]
            self.scalers[value] = self.scaler(**self.scaler_kwargs).fit(X_group)
        return self

    def transform(self, X, y=None):
        """Transforms unscaled data.

        Args:
            X: DataFrame to transform.
            y: Not used.

        Returns:
            Transformed DataFrame.
        """
        # apply scaler for each group
        X = X.copy()
        for value in X[self.by].unique():
            select_mask = X[self.by] == value
            X.loc[select_mask, self.columns] = self.scalers[value].transform(
                X.loc[select_mask, self.columns]
            )
        return X


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.ffill().bfill()
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        
        # Drop duplicates first to ensure clean data processing
        if df.duplicated(subset=["date", "tic"]).any():
            print(f"Detected {df.duplicated(subset=['date', 'tic']).sum()} duplicate rows. Dropping them...")
            df = df.drop_duplicates(subset=["date", "tic"])
            
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        
        # Use pivot table to align dates and tickers
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        
        # Forward fill missing values to handle gaps in minute-level data
        merged_closes = merged_closes.ffill().bfill()
        
        # Only drop columns that are still completely empty (if any)
        merged_closes = merged_closes.dropna(axis=1, how='all')
        
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        final_df = pd.DataFrame()
        
        for ticker in unique_ticker:
            # Filter data for current ticker
            ticker_df = df[df.tic == ticker].copy()
            # Convert to StockDataFrame
            stock = Sdf.retype(ticker_df)
            
            # Calculate all indicators
            for indicator in self.tech_indicator_list:
                # Accessing the column calculates it and adds it to 'stock'
                _ = stock[indicator]
            
            # Append to final result
            final_df = pd.concat([final_df, stock], ignore_index=True, axis=0) 
            
        final_df = final_df.sort_values(by=["date", "tic"])
        
        # Restore original column order and add indicators, removing intermediate columns (like macds, boll, etc.)
        columns_to_keep = list(df.columns) + self.tech_indicator_list
        # Remove duplicates while preserving order
        columns_to_keep = list(dict.fromkeys(columns_to_keep))
        # Ensure only existing columns are selected
        columns_to_keep = [col for col in columns_to_keep if col in final_df.columns]
        
        return final_df[columns_to_keep]

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        
        # Ensure date column is datetime objects for reliable manipulation
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Get min and max dates as strings YYYY-MM-DD for YahooDownloader
        # Fetch extra history to account for the shift
        start_date = (df.date.min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (df.date.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Fetch VIX data
        print(f"Fetching VIX data from {start_date} to {end_date}...")
        try:
            df_vix = YahooDownloader(
                start_date=start_date, end_date=end_date, ticker_list=["^VIX"]
            ).fetch_data()
        except Exception as e:
            print(f"Error fetching VIX data: {e}. proceed without VIX.")
            return df
        
        vix = df_vix[["date", "close"]].copy()
        vix.columns = ["date", "vix"]
        
        # Shift VIX by 1 day to use previous day's close (avoid look-ahead bias)
        vix['vix'] = vix['vix'].shift(1)
        # Drop the first row which is now NaN after shift
        vix = vix.dropna()

        # Ensure VIX date is datetime for merging
        vix['date'] = pd.to_datetime(vix['date'])

        # Create a temporary column for merging: just the date part of the timestamp
        # Ensure we just have the date part and no timezone issues for merging
        df['_merge_date'] = df['date'].dt.normalize().dt.tz_localize(None)
        vix['_merge_date'] = vix['date'].dt.normalize().dt.tz_localize(None)

        # Merge on the date-only column
        df = df.merge(vix[['_merge_date', 'vix']], on="_merge_date", how="left")
        
        # Drop the temporary merge column
        df = df.drop(columns=['_merge_date'])
        
        # Sort and reset index
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        
        # Forward fill VIX data for missing days/minutes if necessary
        df['vix'] = df['vix'].ffill().bfill()
        
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data, time_period=252):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year (or time_period)
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        
        # Optimize: Use rolling operations instead of slicing in a loop
        # We need the window of size `time_period` ENDING at i-1 (strictly past)
        # So we shift returns by 1, then take rolling window
        
        # Calculate mean and covariance on the shifted data (historical only)
        # We need rolling mean and covariance of the PAST `time_period` days.
        # Shift(1) means the value at index `i` is the return at `i-1`.
        # Then rolling(window) at index `i` includes `i-window+1` to `i`.
        # Effectively returns[i-window:i] which is what we want.
        
        # NOTE: rolling.cov() returns a MultiIndex DataFrame (date, tic) -> cov
        
        historical_returns = df_price_pivot
        
        # We loop to compute Mahalanobis distance
        # For significantly faster execution, we need to avoid re-calculating cov in tight loop if possible
        # or at least avoid dataframe slicing.
        
        # Because dropna(axis=1) in the original code is dynamic (drops columns with *any* NaN in the window),
        # standard rolling functions are slightly different (they handle pair-wise NaNs).
        # However, for performance, we will assume reasonable data quality or standard pairwise covariance.
        
        # Pre-calculate rolling mean and covariance
        # We look at strict past, so we use shift(1)
        shifted_returns = historical_returns.shift(1)
        
        # rolling mean: (N, M)
        rolling_mean = shifted_returns.rolling(window=time_period).mean()
        
        # rolling cov: (N*M, M) - MultiIndex
        rolling_cov = shifted_returns.rolling(window=time_period).cov()
        
        # Loop is still needed to perform the matrix multiplication and inversion validly per step,
        # but now we just lookup the pre-calculated matrices instead of slicing and reducing.
        
        for i in range(start, len(unique_date)):
            current_date = unique_date[i]
            
            # Current returns: (1, M)
            current_row = df_price_pivot.loc[current_date]
            
            # Historical Mean estimate used for centering: (M,)
            # Corresponds to index i in rolling_mean because we shifted the input
            mu = rolling_mean.loc[current_date]
            
            # Covariance matrix: (M, M)
            # rolling_cov index is (date, tic), we select by date
            cov_mat = rolling_cov.loc[current_date]
            
            # Robustness: Filter out assets that have NaNs in this window (mimic original dropna)
            # In the rolling result, if an asset has insufficient data, its mean/cov might be NaN.
            valid_tics = cov_mat.dropna(axis=0, how='all').dropna(axis=1, how='all').index
            # Further intersection with current valid data
            valid_tics = valid_tics.intersection(current_row.dropna().index)
            # Ensure mu is also valid
            valid_tics = valid_tics.intersection(mu.dropna().index)
            
            if len(valid_tics) == 0:
                turbulence_index.append(0)
                continue
                
            # Filter matrices
            current_temp = current_row[valid_tics] - mu[valid_tics]
            cov_temp = cov_mat.loc[valid_tics, valid_tics]
            
            # Inverse covariance
            try:
                # Pseudo-inverse
                inv_cov_temp = np.linalg.pinv(cov_temp.values)
                
                # Mahalanobis distance: (x-u) * S^-1 * (x-u)^T
                temp = current_temp.values.dot(inv_cov_temp).dot(current_temp.values.T)
                
                if temp > 0:
                    count += 1
                    if count > 2:
                        turbulence_temp = temp
                    else:
                        turbulence_temp = 0
                else:
                    turbulence_temp = 0
            except Exception:
                turbulence_temp = 0
                
            turbulence_index.append(turbulence_temp)

        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index
