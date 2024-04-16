import wntr
from wntr.epanet.io import BinFile
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tabulate
import sys

reader = BinFile()

res = reader.read(sys.argv[1])
cor = reader.read(sys.argv[2])


def metrics(df1, df2):
    diffs = df1 - df2
    diffs = diffs.abs()
    max_ind_diff = diffs.max().max()
    sum_diffs = diffs.sum()
    max_sum_diff = sum_diffs.max()
    rel_err = diffs / df2.replace({0: np.nan})
    rel_err.replace([np.inf, -np.inf], np.nan, inplace=True)
    max_rel_err = rel_err.max().max()
    mse = mean_squared_error(df1.values.flatten(), df2.values.flatten())
    mae = mean_absolute_error(df1.values.flatten(), df2.values.flatten())
    return max_ind_diff, max_sum_diff, max_rel_err, mse, mae

check = ["node", "link"]
exclude = ["quality"]

data = []

for x in check:
    res_x = getattr(res, x)
    cor_x = getattr(cor, x)
    for key in res_x.keys():
        if key in exclude: continue

        row = [f"{x}/{key}"]
        row.extend(metrics(res_x[key], cor_x[key]))
        data.append(row)

print(tabulate.tabulate(data, headers=["", "Max Ind. Diff.", "Max Sum Diff.", "Max Rel. Err.", "MSE", "MAE"]))
