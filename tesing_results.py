from pathlib import Path
import pandas as pd
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import column
import itertools
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np


pd.set_option('display.float_format', lambda x: '%.5f' % x)

p3 = Path('.\\testing_results')
files = [x for x in p3.iterdir() if x.is_file()]
dfs = {}

for f in files:
    df = pd.read_csv(f, header=None)[1].reset_index(drop=True)
    print()
    print(f.stem)
    print(df.describe())
    print(df.mean() / df.std())

    df = df[(df > -4) & (df < 4)]
    df.plot.hist(by=[0], bins=80)
    plt.xticks(np.arange(-4, 5, 1))
    #plt.show()
    dfs[f.stem] = df

exit()





min_max_len = min([len(df) for df in dfs.values()])

for k in dfs.keys():
    dfs[k] = dfs[k].iloc[:min_max_len]

p1 = figure(width=1400, height=1000)

sorted_dfs = sorted([(k, v) for k, v in dfs.items()], key=lambda x: x[1].loc[min_max_len - 1, 'mean_ema'],
                    reverse=True)

for k, v in sorted_dfs:
    p1.line(x=v.index.values, y=v['mean_ema'].values, legend=k,
            color=next(colors), line_dash=next(line_dash), alpha=1, line_width=2)

p1.title.text = 'mean of daily account value % changes'
p1.legend.location = "top_left"
p1.legend.click_policy = 'hide'
p1.xaxis.axis_label = 'episode number'
p1.yaxis.axis_label = 'mean (%)'


p2 = figure(width=1400, height=1000)

sorted_dfs = sorted([(k, v) for k, v in dfs.items()], key=lambda x: x[1].loc[min_max_len - 1, 'std_ema'],
                    reverse=True)
for k, v in sorted_dfs:
    p2.line(x=v.index.values, y=v['std_ema'].values, legend=k,
            color=next(colors), line_dash=next(line_dash), alpha=1, line_width=2)

p2.title.text = 'standard deviation of daily account value % changes'
p2.legend.location = "top_left"
p2.legend.click_policy = 'hide'
p2.xaxis.axis_label = 'episode number'
p2.yaxis.axis_label = 'standard deviation (%)'

p3 = figure(width=1400, height=1000)

sorted_dfs = sorted([(k, v) for k, v in dfs.items()], key=lambda x: auc(x=x[1].index, y=x[1]['sharpe_ratio_ema']),
                    reverse=True)
for k, v in sorted_dfs:
    p3.line(x=v.index.values, y=v['sharpe_ratio_ema'].values, legend=k,
            color=next(colors), line_dash=next(line_dash), alpha=1, line_width=2)

p3.title.text = 'sharpe ratio of daily account value % changes'
p3.legend.location = "top_left"
p3.legend.click_policy = 'hide'
p3.xaxis.axis_label = 'episode number'
p3.yaxis.axis_label = 'sharpe ratio'

layout = column([p1, p2, p3])
show(layout)

# output_file('curr_results.html')
# save(layout)
