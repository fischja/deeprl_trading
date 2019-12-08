import numpy as np

from features import (attach_empty_training_cols, set_pre_action_cols, set_post_action_cols, get_state_for_policy,
                      attach_features)
from account_management import AccountState


class TradingEnv:
    def __init__(self, dfs, starting_acc_balance, features, allocations, n_start_point_to_ignore):
        self.dfs = dfs
        self.starting_acc_balance = starting_acc_balance
        self.curr_df = None
        self.curr_dates = None
        self.curr_date_idx = None
        self.features = features
        self.actions_to_alloc = {i: allocations[i] for i in range(len(allocations))}
        self.n_start_point_to_ignore = n_start_point_to_ignore

    def reset(self):
        ticker = np.random.choice(list(self.dfs.keys()))
        self.curr_df = self.dfs[ticker].copy()
        attach_features(df=self.curr_df)
        # print(self.curr_df.head(50))
        self.curr_df = self.curr_df.iloc[self.n_start_point_to_ignore:].copy()

        # print(f'\n######## reset: {ticker}, {self.curr_df.index.date[0]} to {self.curr_df.index.date[-1]}, {len(self.curr_df)}')

        # print((self.curr_df.isnull().any()) | ((self.curr_df == 'NaN').any()))

        attach_empty_training_cols(df=self.curr_df)
        self.curr_dates = self.curr_df.index.date
        self.curr_date_idx = 0

        day = self.curr_dates[0]
        pre_account = AccountState(fixed=self.starting_acc_balance, market_price=self.curr_df.at[day, 'Close'])
        set_pre_action_cols(df=self.curr_df, day=day, pre_account=pre_account)
        state = get_state_for_policy(df=self.curr_df, day=day, features=self.features)
        return state

    def step(self, action):
        day = self.curr_dates[self.curr_date_idx]

        target_alloc = self.actions_to_alloc[action]
        units = self.curr_df.at[day, 'pre_account'].get_units_to_trade(target_alloc=target_alloc)
        post_account = self.curr_df.at[day, 'pre_account'].trade(units=units)
        set_post_action_cols(df=self.curr_df, day=day, post_account=post_account, target_alloc=target_alloc)

        self.curr_date_idx += 1

        next_day = self.curr_dates[self.curr_date_idx]
        pre_account = self.curr_df.at[day, 'post_account'].update(market_price=self.curr_df.at[next_day, 'Close'])
        set_pre_action_cols(df=self.curr_df, day=next_day, pre_account=pre_account)

        day_total = self.curr_df.at[day, 'post_total']
        next_day_total = self.curr_df.at[next_day, 'pre_total']
        reward = (next_day_total - day_total) / day_total
        self.curr_df.at[day, 'reward'] = reward

        state = get_state_for_policy(df=self.curr_df, day=day, features=self.features)
        done = False
        if self.curr_date_idx >= len(self.curr_dates) - 1 or post_account.total() < self.curr_df.at[day, 'Close']:
            done = True
        else:
            state = get_state_for_policy(df=self.curr_df, day=next_day, features=self.features)

        return state, reward, done
