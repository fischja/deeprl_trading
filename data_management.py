import yfinance as yf
import pandas as pd
import numpy as np


def download_dfs(tickers):
    print('\n######## downloading data ########\n')
    dfs = {}
    n_data_points = 0
    for i in range(len(tickers)):
        ticker = tickers[i]
        tk = yf.Ticker(ticker=ticker)
        df = tk.history(period='max', interval='1d', auto_adjust=True)[['Open', 'High', 'Low', 'Close', 'Volume']]

        if df.empty:
            print(f'discarded {ticker}: data unavailable.')
            continue

        n_data_points += len(df)
        print(f'({i+1}/{len(tickers)}) downloaded: {ticker}, ({len(df)} points).')
        dfs[ticker] = df

    print(f'\ndownloaded {len(dfs)} dfs with total {n_data_points} data points.\n')
    return dfs


def save_dfs(dfs, base_data_path):
    print('\n######## saving data ########\n')
    for ticker in dfs.keys():
        path = base_data_path.joinpath(f'{ticker}.csv')
        dfs[ticker].to_csv(path)


def load_dfs(base_data_path, accepted_tickers):
    print('\n######## loading data ########\n')
    dfs = {}
    for f in base_data_path.iterdir():
        if f.is_file() and f.stem in accepted_tickers:
            df = pd.read_csv(f, header=0, index_col='Date')
            df.index = pd.to_datetime(df.index)
            dfs[f.stem] = df

    return dfs


def train_test_split(dfs, split_date):
    train_dfs = {}
    test_dfs = {}
    for ticker in dfs.keys():
        train_dfs[ticker] = dfs[ticker][:split_date].copy()
        test_dfs[ticker] = dfs[ticker][split_date:].copy()

    n_train_points = sum([len(df) for df in train_dfs.values()])
    n_test_points = sum([len(df) for df in test_dfs.values()])
    print(f'n_train_points={n_train_points}, n_test_points={n_test_points}')
    return train_dfs, test_dfs


def filter_dfs(dfs, min_ts_len, max_days_break, max_first_date, min_first_date, min_last_date, n_rows_to_remove_start):
    print('\n######## start filtering dfs ########\n')
    init_n_keys = len(dfs)
    init_n_points = sum([len(df) for df in dfs.values()])
    tickers_to_del = []
    for ticker in dfs.keys():
        dfs[ticker] = dfs[ticker].iloc[n_rows_to_remove_start:]
        dfs[ticker] = dfs[ticker].loc[min_first_date:]

        dfs[ticker] = dfs[ticker].replace(['NaN', 'NaT'], np.nan)
        dfs[ticker] = dfs[ticker].dropna(axis=0)

        if len(dfs[ticker]) < min_ts_len:
            print(f'-- discarding {ticker}: n_points ({len(dfs[ticker])}) < min ({min_ts_len})).')
            tickers_to_del.append(ticker)
            continue

        if not _date_range_ok(df=dfs[ticker], ticker=ticker, max_first_date=max_first_date, min_last_date=min_last_date):
            tickers_to_del.append(ticker)
            continue

        if not _data_continuity_ok(df=dfs[ticker], ticker=ticker, max_days_break=max_days_break):
            tickers_to_del.append(ticker)
            continue

        if not _price_range_ok(df=dfs[ticker], ticker=ticker):
            tickers_to_del.append(ticker)
            continue

        if not _volume_range_ok(df=dfs[ticker], ticker=ticker):
            tickers_to_del.append(ticker)
            continue

    for ticker in tickers_to_del:
        del(dfs[ticker])

    total_len = sum([len(df) for df in dfs.values()])
    print(f'\ninitially: {init_n_keys} dfs ({init_n_points} data points).')
    print(f'remaining: {init_n_keys} - {len(tickers_to_del)} = {len(dfs)} dfs ({total_len} data points).')

    # sanity check
    for ticker, df in dfs.items():
        if not _date_range_ok(df=df, ticker=ticker, max_first_date=max_first_date, min_last_date=min_last_date):
            raise ValueError

        if not _data_continuity_ok(df=df, ticker=ticker, max_days_break=max_days_break):
            raise ValueError

        if not _price_range_ok(df=df, ticker=ticker):
            raise ValueError

        if not _volume_range_ok(df=df, ticker=ticker):
            raise ValueError

    return dfs


def _date_range_ok(df, ticker, max_first_date, min_last_date):
    dates = pd.DataFrame(df.index)
    first_date = dates.min()[0]
    last_date = dates.max()[0]

    if first_date > max_first_date or last_date < min_last_date:
        print(f'-- discarding {ticker}: range=({first_date.date()}) to ({last_date.date()}).')
        return False
    return True


def _data_continuity_ok(df, ticker, max_days_break):
    dates = pd.DataFrame(df.index)
    max_diff = dates.diff().max()[0]
    if max_diff > max_days_break:
        print(f'discarded {ticker}: max_diff ({max_diff}) > allowed ({max_days_break}).')
        return False
    return True


def _price_range_ok(df, ticker):
    if (df[['Open', 'High', 'Low', 'Close']] < 0.001).any().any():
        print(f'discarded {ticker}: contains prices below min acceptable.')
        return False

    if (df[['Open', 'High', 'Low', 'Close']] > 3000.0).any().any():
        print(f'discarded {ticker}: contains prices above max acceptable.')
        return False

    return True


def _volume_range_ok(df, ticker):
    if (df['Volume'] < 1000).any().any():
        print(f'discarded {ticker}: contains volume below min acceptable.')
        return False

    if (df['Volume'] > 1e10).any().any():
        print(f'discarded {ticker}: contains volume above max acceptable.')
        return False

    return True



