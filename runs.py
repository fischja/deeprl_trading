import pandas as pd
import torch
from torch.distributions import Categorical
import numpy as np
from rl_env import TradingEnv
from ppo import PPO


K_epoch = 3
T_horizon = 100
starting_acc_balance = 10000.0


def train(allocations, features, model_path, dfs, n_start_point_to_ignore):
    env = TradingEnv(dfs=dfs,
                     starting_acc_balance=starting_acc_balance,
                     features=features,
                     allocations=allocations,
                     n_start_point_to_ignore=n_start_point_to_ignore)
    model = PPO(dim_state=len(features), dim_actions=len(allocations))
    print_interval = 1

    for n_epi in range(1, 600):
        daily_perc_returns = []
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                try:
                    a = m.sample().item()
                except RuntimeError:
                    print(prob)
                    print(m)

                s_prime, r, done = env.step(a)
                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime

                daily_perc_returns.append(r * 100)

                if done:
                    break
            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("mean: {:.3f}%, std: {:.3f}%, episode: {}, npoints: {}".format(np.mean(daily_perc_returns),
                                                                                 np.std(daily_perc_returns),
                                                                                 n_epi,
                                                                                 len(daily_perc_returns)))
            daily_perc_returns = []

    torch.save(model.state_dict(), model_path)


def test(allocations, features, model_path, dfs, results_path, n_start_point_to_ignore):
    env = TradingEnv(dfs=dfs,
                     starting_acc_balance=starting_acc_balance,
                     features=features,
                     allocations=allocations,
                     n_start_point_to_ignore=n_start_point_to_ignore)
    model = PPO(dim_state=len(features), dim_actions=len(allocations))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print_interval = 1
    all_daily_perc_returns = []

    for n_epi in range(1, 1001):
        daily_perc_returns = []
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                try:
                    a = m.sample().item()
                except RuntimeError:
                    print(prob)
                    print(m)

                s_prime, r, done = env.step(a)
                s = s_prime

                daily_perc_returns.append(r * 100)
                all_daily_perc_returns.append(r * 100)

                if done:
                    break

        if n_epi % print_interval == 0 and n_epi != 0:
            print("mean: {:.3f}%, std: {:.3f}%, episode: {}, npoints: {}".format(np.mean(daily_perc_returns),
                                                                                 np.std(daily_perc_returns),
                                                                                 n_epi,
                                                                                 len(daily_perc_returns)))
            daily_perc_returns = []

    df = pd.DataFrame({'data': all_daily_perc_returns})
    df.to_csv(results_path, header=False)


def test_baseline(allocation, dfs, results_path, n_start_point_to_ignore):
    all_daily_perc_returns = []

    counter = 0
    for k, v in dfs.items():
        counter += 1
        print(f'{counter}/{len(dfs)} dfs completed.')

        df = {k: v}
        env = TradingEnv(dfs=df,
                         starting_acc_balance=starting_acc_balance,
                         features=['trix'],
                         allocations=[allocation],
                         n_start_point_to_ignore=n_start_point_to_ignore)
        env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                s_prime, r, done = env.step(0)
                all_daily_perc_returns.append(r * 100)

                if done:
                    break

    df = pd.DataFrame({'data': all_daily_perc_returns})
    df.to_csv(results_path, header=False)
