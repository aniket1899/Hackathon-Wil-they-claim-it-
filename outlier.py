import numpy as np

def iqr(col):
    per_25 = np.percentile(col, 25)
    per_75 = np.percentile(col, 75)
    iqr = per_75 - per_25
    low = per_25 - (1.5*iqr)
    up  = per_75 + (1.5*iqr)
    out = [i for i in col if i < low or i > up]
    return low, up


def winsor(data, col):
    outlier_thresh = 3
    mean = data[col].mean()
    stdev = data[col].std()
    data[f"{col}+_zscores"] = (data[col] - mean) / stdev
    data = data[abs(data[f"{col}+_zscores"])<=3]
    return data.drop(f"{col}+_zscores", axis=1)

