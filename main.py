import pandas as pd
from pathlib import Path
from data_management import download_dfs, save_dfs, load_dfs, filter_dfs, train_test_split
from runs import train, test, test_baseline


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


base_data_path = Path('./data/')
accepted_tickers = pd.read_csv('./s&p_500_tickers.txt', header=None)[0].values
max_days_break = pd.Timedelta(days=7)
max_first_date = pd.Timestamp('1998-01-01')
min_first_date = pd.Timestamp('1993-01-01')
min_last_date = pd.Timestamp('2019-10-31')
train_test_split_date = pd.Timestamp('2015-01-01')
min_ts_len = 3000

# dfs = download_dfs(tickers=accepted_tickers)
# save_dfs(dfs=dfs, base_data_path=base_data_path)
loaded_dfs = load_dfs(base_data_path=base_data_path, accepted_tickers=accepted_tickers)
filtered_dfs = filter_dfs(dfs=loaded_dfs,
                          min_ts_len=min_ts_len,
                          max_days_break=max_days_break,
                          max_first_date=max_first_date,
                          min_last_date=min_last_date,
                          min_first_date=min_first_date,
                          n_rows_to_remove_start=300)

train_dfs, test_dfs = train_test_split(dfs=filtered_dfs, split_date=train_test_split_date)

n_start_point_to_ignore_train = 2000
n_start_point_to_ignore_test = 100

allocations = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]
features = ['trix', 'rsi', 'cci', 'aroon', 'perc_bb']

model_path = Path('.\\models\\input=[trix, rsi, cci, aroon, perc_bb] output=[-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6].pt')
#results_path = Path('.\\testing_results\\results1.csv')

train(allocations=allocations,
      features=features,
      model_path=model_path,
      dfs=train_dfs,
      n_start_point_to_ignore=n_start_point_to_ignore_train)

test(allocations=allocations,
     features=features,
     model_path=model_path,
     dfs=test_dfs,
     results_path=results_path,
     n_start_point_to_ignore=n_start_point_to_ignore_test)

results_path = Path('.\\testing_results\\-0.1.csv')
test_baseline(allocation=-0.1,
              dfs=test_dfs,
              results_path=results_path,
              n_start_point_to_ignore=n_start_point_to_ignore_test)

