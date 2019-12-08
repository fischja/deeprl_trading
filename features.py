import pandas as pd
import numpy as np


def get_state_for_policy(df, day, features):
    state = df.loc[day, features].values.astype(dtype=np.float32)
    return state


def attach_features(df):
    """
    _attach_adx(df=df, periods=14)
    """
    _attach_sma(df=df)
    _attach_trix(df=df, periods=15)
    _attach_rsi(df=df, periods=14)
    _attach_cci(df=df, periods=20, c=0.015)
    _attach_aroon(df=df, periods=15)
    _attach_bollinger_bands(df=df)
    _attach_stoch(df=df, k_periods=14, d_periods=3)
    _attach_macd(df=df, signal_periods=9, fast_periods=12, slow_periods=26)


def set_pre_action_cols(df, day, pre_account):
    df.at[day, 'pre_total'] = pre_account.total()
    df.at[day, 'pre_fixed'] = pre_account.fixed
    df.at[day, 'pre_floating'] = pre_account.floating
    df.at[day, 'pre_alloc'] = pre_account.alloc()
    df.at[day, 'pre_units'] = pre_account.units
    df.at[day, 'pre_account'] = pre_account


def set_post_action_cols(df, day, post_account, target_alloc):
    df.at[day, 'policy_alloc'] = target_alloc

    df.at[day, 'post_total'] = post_account.total()
    df.at[day, 'post_fixed'] = post_account.fixed
    df.at[day, 'post_floating'] = post_account.floating
    df.at[day, 'post_alloc'] = post_account.alloc()
    df.at[day, 'post_units'] = post_account.units
    df.at[day, 'post_account'] = post_account


def attach_empty_training_cols(df):
    df['pre_fixed'] = np.nan
    df['pre_floating'] = np.nan
    df['pre_total'] = np.nan
    df['pre_alloc'] = np.nan
    df['pre_units'] = np.nan

    df['policy_alloc'] = np.nan

    df['post_fixed'] = np.nan
    df['post_floating'] = np.nan
    df['post_total'] = np.nan
    df['post_alloc'] = np.nan
    df['post_units'] = np.nan

    df['pre_account'] = np.nan
    df['post_account'] = np.nan

    df['reward'] = np.nan


def _attach_sma(df):
    sma_5 = df['Close'].rolling(window=5).mean()
    sma_10 = df['Close'].rolling(window=10).mean()
    sma_30 = df['Close'].rolling(window=30).mean()

    df['sma_5/10'] = (sma_5 - sma_10) / sma_10
    df['sma_5/30'] = (sma_5 - sma_30) / sma_30


def _attach_bollinger_bands(df):
    bb_middle = df['Close'].rolling(window=20).mean()
    price_2x_std = 2 * df['Close'].rolling(window=20).std()

    upper = bb_middle + price_2x_std
    lower = bb_middle - price_2x_std
    df['perc_bb'] = ((df['Close'] - lower) / (upper - lower)) - 0.5


def _attach_rsi(df, periods=14):
    delta = df['Close'].diff(periods=1)

    up = delta.mask(delta < 0, 0)
    down = delta.mask(delta > 0, 0).abs()

    up_ewm = up.ewm(com=periods - 1, min_periods=periods).mean()
    down_ewm = down.ewm(com=periods - 1, min_periods=periods).mean()

    rs = up_ewm / down_ewm
    rsi = 100 - 100 / (1 + rs)
    rsi.loc[rsi.isna() & rsi.index.isin(rsi.index[periods:])] = 100

    # scale into range [-0.5, 0.5]
    df['rsi'] = (rsi / 100) - 0.5


def _get_min_max(x1, x2, f='min'):
    if not np.isnan(x1) and not np.isnan(x2):
        if f == 'max':
            return max(x1, x2)
        elif f == 'min':
            return min(x1, x2)
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    else:
        return np.nan


def _attach_trix(df, periods=15):
    single_smoothed = df['Close'].ewm(span=periods, min_periods=periods).mean()
    double_smoothed = single_smoothed.ewm(span=periods, min_periods=periods).mean()
    triple_smoothed = double_smoothed.ewm(span=periods, min_periods=periods).mean()
    trix = 100 * triple_smoothed.pct_change(periods=1)
    df['trix'] = trix


def _attach_stoch(df, k_periods=14, d_periods=3):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full
    """
    min_low = df['Low'].rolling(k_periods, min_periods=d_periods).min()
    max_high = df['High'].rolling(k_periods, min_periods=d_periods).max()

    k = 100 * (df['Close'] - min_low) / (max_high - min_low)
    d = k.rolling(d_periods, min_periods=d_periods).mean()

    df['stoch'] = (k - d) / 100


def _attach_cci(df, periods=20, c=0.015):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    mean_abs_deviation = lambda x: np.mean(np.abs(x-np.mean(x)))
    cci = ((typical_price - typical_price.rolling(periods, min_periods=periods).mean())
           / (c * typical_price.rolling(periods, min_periods=0).apply(mean_abs_deviation, True)))
    df['cci'] = cci / 200


def _attach_aroon(df, periods=15):
    """
    https://school.stockcharts.com/doku.php?id=technical_indicators:aroon_oscillator
    """
    aroon_up = df['Close'].rolling(periods, min_periods=periods).apply(lambda x: float(np.argmax(x) + 1) / periods * 100,
                                                                 raw=True)
    aroon_down = df['Close'].rolling(periods, min_periods=0).apply(lambda x: float(np.argmin(x) + 1) / periods * 100,
                                                             raw=True)
    df['aroon'] = (aroon_up - aroon_down) / 100


def _attach_adx(df, periods=14):
    """
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    """
    cs = df['Close'].shift(1)
    pdm = df['High'].combine(cs, lambda x1, x2: _get_min_max(x1, x2, 'max'))
    pdn = df['Low'].combine(cs, lambda x1, x2: _get_min_max(x1, x2, 'min'))
    tr = pdm - pdn

    trs_initial = np.zeros(periods - 1)
    trs = np.zeros(len(df['Close']) - (periods - 1))
    trs[0] = tr.dropna()[0:periods].sum()
    tr = tr.reset_index(drop=True)
    for i in range(1, len(trs)-1):
        trs[i] = trs[i-1] - (trs[i-1] / float(periods)) + tr[periods + i]

    up = df['High'] - df['High'].shift(1)
    dn = df['Low'].shift(1) - df['Low']
    pos = abs(((up > dn) & (up > 0)) * up)
    neg = abs(((dn > up) & (dn > 0)) * dn)

    dip_mio = np.zeros(len(df['Close']) - (periods - 1))
    dip_mio[0] = pos.dropna()[0:periods].sum()

    pos = pos.reset_index(drop=True)
    for i in range(1, len(dip_mio)-1):
        dip_mio[i] = dip_mio[i-1] - (dip_mio[i-1] / float(periods)) + pos[periods + i]

    din_mio = np.zeros(len(df['Close']) - (periods - 1))
    din_mio[0] = neg.dropna()[0:periods].sum()

    neg = neg.reset_index(drop=True)
    for i in range(1, len(din_mio)-1):
        din_mio[i] = din_mio[i-1] - (din_mio[i-1] / float(periods)) + neg[periods + i]

    dip = np.zeros(len(trs))
    for i in range(len(trs)):
        dip[i] = 100 * (dip_mio[i]/trs[i])

    din = np.zeros(len(trs))
    for i in range(len(trs)):
        din[i] = 100 * (din_mio[i]/trs[i])

    dx = 100 * np.abs((dip - din) / (dip + din))

    adx = np.zeros(len(trs))
    adx[periods] = dx[0:periods].mean()

    for i in range(periods + 1, len(adx)):
        adx[i] = ((adx[i-1] * (periods - 1)) + dx[i - 1]) / float(periods)

    adx = np.concatenate((trs_initial, adx), axis=0)
    adx = pd.Series(data=adx, index=df['Close'].index)

    df['adx'] = adx


def _attach_macd(df, signal_periods=9, fast_periods=12, slow_periods=26):
    fast = df['Close'].ewm(span=fast_periods, min_periods=fast_periods).mean()
    slow = df['Close'].ewm(span=slow_periods, min_periods=slow_periods).mean()
    macd_line = (fast - slow) / slow
    df['macd'] = macd_line

    """   
    macd_signal = macd_line.ewm(span=signal_periods, min_periods=signal_periods).mean()
    macd_trend = macd_line - macd_signal

    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_trend'] = macd_trend
    """