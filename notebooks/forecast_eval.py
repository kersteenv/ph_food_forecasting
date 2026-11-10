import numpy as np
import pandas as pd
from pathlib import Path
from study_config import WINDOWS

KEYS = ["window", "market_id", "commodity_id"]
EPS = 0.01          # PHP / percentage points, keeps ratios finite when a forecast is exact

def series_metrics(df, models):
    """Per series-window MAE, RMSE (PHP) and MAPE (%) for each model; series with a missing forecast are dropped."""
    out = []
    for name, col in models.items():
        err = np.exp(df[col]) - df["price"]
        o = df.assign(ae=err.abs(), se=err ** 2, ape=err.abs() / df["price"])
        g = (o.groupby(KEYS + ["category", "commodity", "admin1"])
               .agg(n_test=("ae", "size"), MAE=("ae", "mean"),
                    RMSE=("se", lambda x: np.sqrt(x.mean())), MAPE=("ape", lambda x: x.mean() * 100)).reset_index())
        g["model"] = name
        out.append(g.dropna(subset=["MAE"]))
    return pd.concat(out, ignore_index=True)

def obs_errors(df, models):
    out = []
    for name, col in models.items():
        err = np.exp(df[col]) - df["price"]
        out.append(df.assign(model=name, ape=err.abs() / df["price"] * 100)[["window", "h", "model", "ape"]])
    return pd.concat(out, ignore_index=True)

def boot_mean(df, col, B=1000, seed=42):
    """95% interval of exp(mean log ratio), resampling whole markets."""
    rng = np.random.default_rng(seed)
    cl = df.groupby("market_id")[col].apply(lambda s: s.to_numpy())
    vals = [np.mean(np.concatenate([cl.iloc[i] for i in rng.integers(0, len(cl), len(cl))])) for _ in range(B)]
    return np.exp(np.percentile(vals, [2.5, 97.5]))

def compare_gm(sm, a, b, by=None, windows=None):
    """Geometric-mean error ratio of model a to model b per series (< 1 favours a), tie-robust."""
    ma, mb = sm[sm["model"] == a].set_index(KEYS), sm[sm["model"] == b].set_index(KEYS)
    meta = sm.drop_duplicates(KEYS).set_index(KEYS)
    rows = []
    for w in (windows or list(WINDOWS)):
        idx = [i for i in ma.index.intersection(mb.index) if i[0] == w]
        if not idx:
            continue
        for metric in ["MAE", "RMSE", "MAPE"]:
            ra, rb = ma.loc[idx, metric].to_numpy(), mb.loc[idx, metric].to_numpy()
            df = pd.DataFrame({"market_id": [i[1] for i in idx], "lr": np.log((ra + EPS) / (rb + EPS)),
                               "tie": ra == rb, "grp": meta.loc[idx, by].to_numpy() if by else "all"})
            for g, d in df.groupby("grp"):
                lo, hi = boot_mean(d, "lr")
                rows.append({"window": w, "group": g, "metric": metric, "series": len(d),
                             "ties_%": round(d["tie"].mean() * 100, 1),
                             "gm_ratio": round(float(np.exp(d["lr"].mean())), 3), "ci": f"{lo:.3f}-{hi:.3f}",
                             "verdict": f"{a} better" if hi < 1 else (f"{b} better" if lo > 1 else "no clear winner")})
    return pd.DataFrame(rows)

def overall(cmp):
    """One verdict per window (and group): all three metrics must agree, otherwise 'mixed'."""
    return (cmp.groupby(["window", "group"])["verdict"]
               .agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed across metrics"))

def load_variant(folder):
    """Read the per-window SARIMA result files of one variant; returns the log and a long forecast table."""
    folder = Path(folder)
    parts = [pd.read_csv(folder / f"sarima_{w}.csv") for w in WINDOWS if (folder / f"sarima_{w}.csv").exists()]
    sar = pd.concat(parts, ignore_index=True)
    sar["fail_reason"] = sar["fail_reason"].fillna("")
    long = sar.melt(id_vars=KEYS, value_vars=[f"fc{s}" for s in range(1, 13)], var_name="h", value_name="f")
    long["h"] = long["h"].str[2:].astype(int)
    return sar, long