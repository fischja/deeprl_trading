from pathlib import Path
import pandas as pd
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import column
import itertools
from sklearn.metrics import auc
from bokeh.models import Legend


colors = itertools.cycle(['red', 'green', 'blue'])
line_dash = itertools.cycle(['solid', 'dotted'])

p3 = Path('.\\selected_training_results')
files = [x for x in p3.iterdir() if x.is_file()]
dfs = {}

for f in files:
    df = pd.read_csv(f, header=[0]).reset_index(drop=True)
    df['mean_ema'] = df['mean'].ewm(span=100, min_periods=10).mean()
    df['std_ema'] = df['std'].ewm(span=100, min_periods=10).mean()
    df['sharpe_ratio_ema'] = df['mean_ema'] / df['std_ema']
    dfs[f.stem] = df

min_max_len = min([len(df) for df in dfs.values()])

for k in dfs.keys():
    dfs[k] = dfs[k].iloc[:min_max_len]

p1 = figure(width=900, height=600, toolbar_location=None)

sorted_dfs = sorted([(k, v) for k, v in dfs.items()], key=lambda x: x[1].loc[min_max_len - 1, 'mean_ema'],
                    reverse=True)

lines = []
for k, v in sorted_dfs:
    lines.append(p1.line(x=v.index.values, y=v['mean_ema'].values,
                         color=next(colors), line_dash=next(line_dash), alpha=1, line_width=2))


p1.title.text = 'mean of daily account value % changes'
p1.legend.location = "top_left"
p1.legend.click_policy = 'hide'
p1.legend.background_fill_alpha = 1.0
p1.xaxis.axis_label = 'episode number'
p1.yaxis.axis_label = 'mean (%)'

p2 = figure(width=900, height=600, toolbar_location=None)

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
p2.legend.background_fill_alpha = 1.0


p3 = figure(width=900, height=600, toolbar_location=None)

sorted_dfs = sorted([(k, v) for k, v in dfs.items()], key=lambda x: auc(x=x[1][200:].index, y=x[1].loc[200:,'sharpe_ratio_ema']),
                    reverse=True)
for k, v in sorted_dfs:
    p3.line(x=v.index.values, y=v['sharpe_ratio_ema'].values,
            color=next(colors), line_dash=next(line_dash), alpha=1, line_width=2)

p3.title.text = 'sharpe ratio of daily account value % changes'
p3.legend.location = "top_left"
p3.legend.click_policy = 'hide'
p3.xaxis.axis_label = 'episode number'
p3.yaxis.axis_label = 'sharpe ratio'
p3.legend.background_fill_alpha = 1.0

layout = column([p1, p2, p3])
show(layout)

# output_file('curr_results.html')
# save(layout)
