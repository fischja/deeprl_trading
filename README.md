## Deep Reinforcement Learning Seminar Project
The repo contains the code accompanying a masters level seminar project on deep reinforcement learning. The objective of our project was to train a reinforcement learning agent based on a deep neural network policy to profitably trade shares of S&P 500 companies in a simplified trading environment. 


#### Running Calculations
- Run the `main.py` file. It contains 3 sections identifiable by comments (`TRAINING PPO`, `TESTING PPO`, `TESTING BUY AND HOLD`). Comment out the sections you to do not want to run and adjust the parameters accordingly. 


#### Running Result Visualizations
- Run `training_results.py` to visualize the result files in the `training_results` folder 
- Run `testing_results.py` to visualize the result files in the `testing_results` folder


#### Modules
- `account_management.py` - contains the immutable `AccountState` class. All instance methods of `AccountState` return a new instance.

- `data_management.py`- provides functions for downloading (`download_dfs`), saving (`save_dfs`), loading (`load_dfs`), splitting (`train_test_split`) and filtering (`filter_dfs`) historic data 

- `features.py`- provides functions for attaching/retrieving features to/from historic data. Based on [bukosabino/ta](https://github.com/bukosabino/ta)

- `main.py`- entry point for training and testing the PPO agents, as well as running testing baselines (buy and hold strategies). Adjust calls to `train`, `test`, `test_baseline` as needed.

- `ppo.py`- contains `PPO` class, which wraps the functionality of proximal policy optimization. Based on [seungeunrho/minimalRL](https://github.com/seungeunrho/minimalRL) 

- `rl_env.py` - contains the `TradingEnv` class, which wraps a simplified trading environment in the OpenAI Gym environment API

- `runs.py`- contains functions for training agents (`train`), testing agents (`test`) and running buy and hold strategies (`test_baseline`)

- `testing_results.py` - visualizes the results of testing located in the `testing_results` folder

- `training_results.py` - visualizes the results of training located in the `training_results` folder
