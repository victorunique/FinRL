from __future__ import annotations

import random
import time
from copy import deepcopy

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv

matplotlib.use("Agg")


class StockTradingEnvMinute(gym.Env):
    """
    A minute-level trading environment optimized for scalping/day-trading.
    It focuses on immediate PnL (Profit and Loss) rather than long-term accumulated average return.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        date_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        print_verbosity=10,
        initial_amount=1e6,
        daily_information_cols=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        episode_length=-1,
    ):
        self.df = df
        self.stock_col = "tic"
        self.assets = sorted(df[self.stock_col].unique())
        self.dates = df[date_col_name].sort_values().unique()
        self.random_start = random_start
        self.episode_length = episode_length
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(date_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stoploss_penalty = stoploss_penalty
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)
        self.daily_information_cols = daily_information_cols
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.daily_information_cols)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.assets),))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        self.episode = -1  # initialize so we can call reset
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        self.cash_penalty_proportion = cash_penalty_proportion
        if self.cache_indicator_data:
            print("caching data...")
            # Optimization: Unified 3D Cache
            # 1. Reset index to get date as a column for pivoting
            temp_df = self.df.reset_index()

            # 2. Pivot table to reshape data: Index='date', Columns='tic'
            # We want a 3D array: (Dates, Assets, Features)
            # First, pivot to (Date, Asset) with MultiIndex columns (Feature)
            pivot_df = temp_df.pivot(index='date', columns='tic', values=self.daily_information_cols)

            # 3. Features are currently the top level of columns. We want them as the last dimension.
            # Current columns: (Feature, Tic) -> we need (Tic, Feature) to align with our loops if we flattened
            # BUT for 3D array (Date, Asset, Feature), we need to check how pivot returns.
            # pivot_df.columns is MultiIndex (Feature, Tic).
            # Let's swap scale to (Tic, Feature) to ensure we sort Tics correctly.
            pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
            
            # 4. Reindex columns to ensure Assets are sorted as per self.assets
            # And Features are sorted/ordered as per self.daily_information_cols
            expected_cols = pd.MultiIndex.from_product(
                [self.assets, self.daily_information_cols],
                names=['tic', 'feature']
            )
            pivot_df = pivot_df.reindex(columns=expected_cols)

            # 5. Reshape to 3D Array
            # The values are currently 2D: (n_dates, n_assets * n_features)
            # We reshape to (n_dates, n_assets, n_features)
            # Since we ordered columns as (Tic1_Feat1, Tic1_Feat2, ... Tic2_Feat1...), the reshape works naturally
            n_dates = len(self.dates)
            n_assets = len(self.assets)
            n_features = len(self.daily_information_cols)
            
            self.cached_data = pivot_df.values.reshape(n_dates, n_assets, n_features)
            
            # Map Feature Name -> Index for fast lookups
            self.col_map = {col: i for i, col in enumerate(self.daily_information_cols)}
            
            print(f"data cached! Shape: {self.cached_data.shape}")
        self.final_asset_memory = None
        self.final_action_memory = None

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self):
        return self.date_index - self.starting_point

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # self.sum_trades = 0
        # self.actual_num_trades = 0
        # self.closing_diff_avg_buy = np.zeros(len(self.assets))
        # self.profit_sell_diff_avg_buy = np.zeros(len(self.assets))
        # self.n_buys = np.zeros(len(self.assets))
        # self.avg_buy_price = np.zeros(len(self.assets))
        if self.random_start:
            # If episode_length is set, we must leave enough room for the episode to finish
            if self.episode_length > 0:
                # Ensure we have enough data: total_len - episode_length
                max_start = len(self.dates) - self.episode_length
                if max_start <= 0:
                    # Fallback if data is shorter than episode_length
                    starting_point = 0
                else:
                    starting_point = random.choice(range(max_start))
            else:
                starting_point = random.choice(range(int(len(self.dates) * 0.5)))

            self.starting_point = starting_point
        else:
            self.starting_point = 0
        
        self.date_index = self.starting_point
        self.step_in_episode = 0

        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [self.initial_amount],
            "asset_value": [0],
            "total_assets": [self.initial_amount],
            "reward": [0],
        }
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(init_state)
        return init_state, {}

    def get_date_vector(self, date, cols=None):
        if self.cached_data is not None:
            # Unified efficient 3D cache lookup
            if cols is None:
                # Return all features for all assets at this date
                # Shape: (n_assets, n_features) -> Flatten to (n_assets * n_features)
                return self.cached_data[date].flatten().tolist()
            else:
                # Return specific features
                # Get indices for the requested columns
                # This assumes 'cols' is a list of strings present in self.daily_information_cols
                col_indices = [self.col_map[c] for c in cols]
                
                # Slice: [Date, All Assets, Specific Features]
                # Result shape: (n_assets, len(cols)) -> Flatten to (n_assets * len(cols))
                return self.cached_data[date, :, col_indices].flatten().tolist()

        else:
            date = self.dates[date]
            if cols is None:
                cols = self.daily_information_cols
            trunc_df = self.df.loc[[date]]
            v = []
            for a in self.assets:
                subset = trunc_df[trunc_df[self.stock_col] == a]
                v += subset.loc[date, cols].tolist()
            assert len(v) == len(self.assets) * len(cols)
            return v

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        self.final_asset_memory = self.save_asset_memory()
        self.final_action_memory = self.save_action_memory()
        return state, reward, True, False, {}

    def log_step(self, reason, terminal_reward=None):
        should_force_print = terminal_reward is not None
        if terminal_reward is None:
            if len(self.account_information["reward"]) > 0:
                terminal_reward = self.account_information["reward"][-1]
            else:
                terminal_reward = 0
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward * 100:0.5f}%",
            f"{(gl_pct - 1) * 100:0.5f}%",
            f"{cash_pct * 100:0.2f}%",
        ]
        self.episode_history.append(rec)
        if (self.current_step + 1) % self.print_verbosity == 0 or should_force_print:
            print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"  # column widths: 8, 10, 15, 7, 10
        print(
            self.template.format(
                "EPISODE",
                "STEPS",
                "TERMINAL_REASON",
                "CASH",
                "TOT_ASSETS",
                "TERMINAL_REWARD_unsc",
                "GAINLOSS_PCT",
                "CASH_PROPORTION",
            )
        )
        self.printed_header = True

    def step(self, actions):
        """
        Modified step function for minute-level trading.
        Key differences:
        1. Reward is calculated AFTER the action is taken.
        2. Reward is the immediate percentage change in portfolio value.
        3. No '1/t' scaling which diluted rewards in the original env.
        """
        # Track previous portfolio value
        begin_cash = self.state_memory[-1][0]
        holdings = self.state_memory[-1][1 : len(self.assets) + 1]

        # 1. Get current valuation (State t)
        current_closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))
        prev_total_assets = begin_cash + np.dot(holdings, current_closings)

        # -----------------------------------------------------------------------
        # EXECUTE ACTION
        # -----------------------------------------------------------------------
        # self.sum_trades += np.sum(np.abs(actions))

        # Print header and log if needed
        if self.printed_header is False:
            self.log_header()
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")

        # Check if we are at the end
        if self.date_index == len(self.dates) - 1:
            return self.return_terminal(reward=0)  # Reward doesn't verify matter at terminal step for PPO update usually

        # Scale actions
        actions = actions * self.hmax
        self.actions_memory.append(actions)

        # Handle missing data (price=0)
        # actions = np.where(current_closings > 0, actions, 0)

        # Normalize actions (Cash Value -> Shares)
        if self.discrete_actions:
            actions = np.where(current_closings > 0, actions // current_closings, 0)
            actions = actions.astype(int)
            actions = np.where(
                actions >= 0,
                (actions // self.shares_increment) * self.shares_increment,
                ((actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            actions = np.where(current_closings > 0, actions / current_closings, 0)

        # Clip sell actions to holdings
        actions = np.maximum(actions, -np.array(holdings))

        # -----------------------------------------------------------------------
        # STOP LOSS & FORCED EXIT LOGIC (Safety)
        # -----------------------------------------------------------------------
        # self.closing_diff_avg_buy = current_closings - (
        #     self.stoploss_penalty * self.avg_buy_price
        # )
        # if begin_cash >= self.stoploss_penalty * self.initial_amount:
        #     actions = np.where(
        #         self.closing_diff_avg_buy < 0, -np.array(holdings), actions
        #     )
        #     if any(np.clip(self.closing_diff_avg_buy, -np.inf, 0) < 0):
        #         self.log_step(reason="STOP LOSS")

        # -----------------------------------------------------------------------
        # UPDATE STATE (CASH & HOLDINGS) from Action Execution
        # -----------------------------------------------------------------------
        # Sells
        sells = -np.clip(actions, -np.inf, 0)
        proceeds = np.dot(sells, current_closings)
        costs = proceeds * self.sell_cost_pct
        coh = begin_cash + proceeds

        # Buys
        buys = np.clip(actions, 0, np.inf)
        spend = np.dot(buys, current_closings)
        costs += spend * self.buy_cost_pct

        # Cash Shortage Logic
        if (spend + costs) > coh:
            if self.patient:
                self.log_step(reason="CASH SHORTAGE")
                actions = np.where(actions > 0, 0, actions)
                spend = 0
                costs = 0
            else:
                self.transaction_memory.append(actions)
                return self.return_terminal(reason="CASH SHORTAGE", reward=0)  # Small penalty for dying

        self.transaction_memory.append(actions)

        # Verify valid trade
        assert (spend + costs) <= coh

        # Update actual holdings and cash
        coh = coh - spend - costs
        holdings_updated = holdings + actions

        # Update Average Buy Price
        # buys_sign = np.sign(buys)
        # self.n_buys += buys_sign
        # safe_n_buys = np.where(self.n_buys > 0, self.n_buys, 1)
        # self.avg_buy_price = np.where(
        #     buys_sign > 0,
        #     self.avg_buy_price + ((current_closings - self.avg_buy_price) / safe_n_buys),
        #     self.avg_buy_price,
        # )
        # self.n_buys = np.where(holdings_updated > 0, self.n_buys, 0)
        # self.avg_buy_price = np.where(holdings_updated > 0, self.avg_buy_price, 0)

        # Update log counters
        # self.actual_num_trades = np.sum(np.abs(np.sign(actions)))

        # -----------------------------------------------------------------------
        # TIME STEP (t -> t+1)
        # -----------------------------------------------------------------------
        self.date_index += 1
        self.step_in_episode += 1

        # Check Truncation (Episode Length Limit)
        truncated = False
        if self.episode_length > 0 and self.step_in_episode >= self.episode_length:
            truncated = True

        # -----------------------------------------------------------------------
        # CALCULATE REWARD based on NEW State (Valuation at t+1)
        # -----------------------------------------------------------------------
        new_closings = np.array(self.get_date_vector(self.date_index, cols=["close"]))

        # New Portfolio Value
        new_asset_value = np.dot(holdings_updated, new_closings)
        new_total_assets = coh + new_asset_value

        # Reward = Percentage Change in Total Assets
        # Multiply by 1000 to make the scale significant for the agent
        reward = ((new_total_assets - prev_total_assets) / prev_total_assets) * 1000

        # Update Memory
        self.account_information["cash"].append(coh)
        self.account_information["asset_value"].append(new_asset_value)
        self.account_information["total_assets"].append(new_total_assets)
        self.account_information["reward"].append(reward)

        next_state = (
            [coh] + list(holdings_updated) + self.get_date_vector(self.date_index)
        )
        self.state_memory.append(next_state)

        return next_state, reward, False, truncated, {}

    def get_sb_env(self):
        def get_self():
            return deepcopy(self)

        e = DummyVecEnv([get_self])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=10):
        def get_self():
            return deepcopy(self)

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="fork")
        obs = e.reset()
        return e, obs

    def save_asset_memory(self):
        if self.current_step == 0:
            if hasattr(self, "final_asset_memory") and self.final_asset_memory is not None:
                return self.final_asset_memory
            return None
        else:
            self.account_information["date"] = self.dates[
                self.starting_point : self.starting_point + len(self.account_information["cash"])
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self):
        if self.current_step == 0:
            if hasattr(self, "final_action_memory") and self.final_action_memory is not None:
                return self.final_action_memory
            return None
        else:
            return pd.DataFrame(
                {
                    "date": self.dates[self.starting_point : self.starting_point + len(self.actions_memory)],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,
                }
            )
