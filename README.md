`account_management.py` - contains the immutable `AccountState` class. All instance methods of `AccountState` return a new instance.

`data_management.py`- provides functions for downloading (`download_dfs`), saving (`save_dfs`), loading (`load_dfs`), splitting (`train_test_split`) and filtering (`filter_dfs`) historic data 

`features.py`- provides functions for attaching/retrieving features to/from historic data 

`main.py`- entry point for training, testing and running buy and hold strategies

`ppo.py`- contains `PPO` class, which wraps the functionality of proximal policy optimization

`rl_env.py` - contains the `TradingEnv` class, which wraps a simplified trading environment in the OpenAI Gym environment API

`runs.py`- contains functions for training agents (`train`), testing agents (`test`) and running buy and hold strategies (`test_baseline`)

`testing_results.py` - visualizes the results of testing located in the `testing_results` folder

`training_results.py` - visualizes the results of training located in the `training_results` folder
