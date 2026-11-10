import numpy as np
import pandas as pd
from scipy import stats

COVID_START = pd.Period("2020-03", freq="M")
COVID_END = pd.Period("2023-05", freq="M")

def period_of(p):
    if p < COVID_START:
        return "Pre-COVID"
    return "During-COVID" if p <= COVID_END else "Post-COVID"

# ---- point-forecast functions: f(train_array, h) -> forecast for month n-1+h ----
def naive(train, h):
    return train[-1]

def snaive(train, h, season=12):
    k = int(np.ceil(h / season))
    return train[len(train) - 1 + h - season * k]

def drift(train, h):
    return train[-1] + h * (train[-1] - train[0]) / (len(train) - 1)

BASELINES = {"naive": naive, "snaive": snaive, "drift": drift}

def rolling_origin(series, methods, horizons, min_train, commodity="", segment=""):
    """Expanding-window evaluation. `series` is a pandas Series (log index, PeriodIndex)."""
    y = series.to_numpy(dtype=float)
    idx = series.index
    N = len(y)
    rec = []
    for h in horizons:
        for n in range(min_train, N - h + 1):
            train = y[:n]
            t = n - 1 + h                                   # position of the target month
            scale = np.mean(np.abs(np.diff(train)))         # in-sample one-step naive MAE (for MASE)
            for name, f in methods.items():
                rec.append({"commodity": commodity, "segment": segment, "method": name,
                            "h": h, "origin_n": n, "target": str(idx[t]),
                            "target_period": period_of(idx[t]),
                            "y_true": y[t], "y_pred": float(f(train, h)), "scale": scale})
    return pd.DataFrame(rec)

def add_errors(F):
    F = F.copy()
    F["err"] = F["y_pred"] - F["y_true"]
    F["ae"] = F["err"].abs()
    F["se"] = F["err"] ** 2
    F["ase"] = F["ae"] / F["scale"]
    # MAPE is computed on the index level (back-transformed from logs), so it is an exact % error
    F["ape"] = (np.exp(F["y_pred"]) - np.exp(F["y_true"])).abs() / np.exp(F["y_true"])
    return F

def summarize(F, by):
    g = add_errors(F).groupby(by)
    return g.agg(n=("ae", "size"),
                 MAE=("ae", lambda x: round(x.mean() * 100, 2)),
                 RMSE=("se", lambda x: round(np.sqrt(x.mean()) * 100, 2)),
                 MAPE=("ape", lambda x: round(x.mean() * 100, 2)),
                 MASE=("ase", lambda x: round(x.mean(), 2))).reset_index()

def dm_test(e_model, e_bench, h, power=2):
    """Diebold-Mariano test (Bartlett long-run variance, Harvey-Leybourne-Newbold correction).
    Negative statistic = model has lower loss than the benchmark. Returns (stat, two-sided p)."""
    d = np.abs(np.asarray(e_model)) ** power - np.abs(np.asarray(e_bench)) ** power
    n = len(d)
    dbar = d.mean()
    lrv = np.mean((d - dbar) ** 2)
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / h) * gk
    corr = (n + 1 - 2 * h + h * (h - 1) / n) / n
    if lrv <= 0 or corr <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / n) * np.sqrt(corr)
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)

def compare_to_benchmark(F, benchmark="naive", min_n=10,
                         keys=("commodity", "segment", "target_period", "h")):
    """Relative MAE / RMSE / MAPE and DM p-value of each method vs a benchmark, per cell."""
    F = add_errors(F)
    rows = []
    for key, g in F.groupby(list(keys)):
        b = g[g["method"] == benchmark].sort_values("origin_n")
        if len(b) == 0:
            continue
        for m, gm in g[g["method"] != benchmark].groupby("method"):
            gm = gm.sort_values("origin_n")
            n = len(gm)
            row = dict(zip(keys, key))
            row.update(method=m, n=n)
            row["rel_MAE"] = round(gm["ae"].mean() / b["ae"].mean(), 2)
            row["rel_RMSE"] = round(np.sqrt(gm["se"].mean()) / np.sqrt(b["se"].mean()), 2)
            row["rel_MAPE"] = round(gm["ape"].mean() / b["ape"].mean(), 2)
            if n >= min_n:
                s, p = dm_test(gm["err"].to_numpy(), b["err"].to_numpy(),
                               h=int(key[list(keys).index("h")]))
                row["DM_p"] = round(p, 3)
            else:
                row["DM_p"] = np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out["beats_" + benchmark] = ((out["rel_MAE"] < 1) & (out["rel_RMSE"] < 1) & (out["DM_p"] < 0.10))
    return out