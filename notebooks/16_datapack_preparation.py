#!/usr/bin/env python
"""
Step 16 - Data pack for the dashboard (data preparation only, no dashboard code).

Builds the 13 tables of the Step 15 data contract in outputs/tables/ (dash_*.csv),
refits SARIMA on data through the latest month for forward forecasts (Jul 2026 - Jun 2027),
and writes a verification manifest.

Run from the notebooks/ folder:
    python 16_datapack_preparation.py --limit 20 --workers 4     # trial (writes to outputs/tables_trial)
    python 16_datapack_preparation.py --workers 11               # full run
"""
import argparse
import datetime as dt
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from study_config import *                       # period definitions, JUMP, CORE, ...
from forecast_eval import KEYS, compare_gm, series_metrics

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

# ---- forward-forecast settings -------------------------------------------------------------
FWD_START = pd.Period("2020-05", freq="M")       # training window starts here (Segment B)
HORIZON = 12                                     # months ahead
MIN_COV, MAX_GAP = 0.80, 3                       # eligibility, same spirit as Step 11
GRID12 = [(p, q, P, Q) for p in (0, 1) for q in (0, 1) for (P, Q) in ((0, 0), (1, 0), (0, 1))]
MAX_JUMP = 2.0                                   # stability screen (log points), same as Step 12
SIGMA_FLOOR = 0.005                              # floor for a series' monthly log-change SD
EMP_WINDOW = "Post-pandemic"                     # holdout window used to calibrate empirical intervals
MIN_MKT, MIN_SER = 3, 5                          # "reliable" cell rule, same as Step 10B
LEVELS = (80, 95)


# =============================================================================================
# Generic helpers
# =============================================================================================
class Out:
    """Where tables are written, and a manifest of what was written."""

    def __init__(self, folder, gzip_large):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.gzip_large = gzip_large
        self.files = {}

    def write(self, df, name, large=False):
        fname = f"{name}.csv.gz" if (large and self.gzip_large) else f"{name}.csv"
        for other in (f"{name}.csv", f"{name}.csv.gz"):      # never leave both versions behind
            if other != fname and (self.folder / other).exists():
                (self.folder / other).unlink()
        df.to_csv(self.folder / fname, index=False)
        self.files[name] = fname
        print(f"  wrote {fname:<34} {len(df):>9,} rows x {df.shape[1]:>2} cols")
        return df


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================================
# Part A - tables that need no model fitting
# =============================================================================================
def build_direct_tables(out, d):
    """The six tables that already exist in some form: re-written under their contract names."""
    prices = d[["date", "period", "segment", "admin1", "admin2", "market_id", "market", "category",
                "commodity_id", "commodity", "unit", "price", "lr", "flag_jump"]]
    out.write(prices, "dash_prices", large=True)

    out.write(pd.read_csv(TAB / "09_series_stats.csv"), "dash_series_stats")

    mk = pd.read_csv(TAB / "09B_market_profile.csv").drop(columns=["n_coords"], errors="ignore")
    out.write(mk, "dash_markets")
    out.write(pd.read_csv(TAB / "09B_commodity_coverage.csv"), "dash_commodities")

    paired = pd.read_csv(PROC / "forecast_results_paired.csv")
    hold = paired[["window", "market_id", "commodity_id", "category", "commodity", "market", "admin1",
                   "ym", "h", "price", "f_naive", "f_snaive", "f_sarima"]].copy()
    for m in ("naive", "snaive", "sarima"):
        hold[f"{m}_php"] = np.exp(hold[f"f_{m}"]).round(2)
    out.write(hold, "dash_forecast_holdout", large=True)

    out.write(pd.read_csv(TAB / "12_series_metrics.csv"), "dash_forecast_metrics")
    return paired


def chain_index(dd, by, base_month):
    """Chain-linked index: mean month-to-month log change of the SAME series, cumulated (Step 9 method)."""
    m = dd[dd["lr"].notna()].groupby([by, "ym"])["lr"].mean().unstack(by)
    start = pd.DataFrame(0.0, index=[base_month], columns=m.columns)
    m = pd.concat([start, m.fillna(0)]).sort_index()
    return 100 * np.exp(m.cumsum())


def build_chain_table(out, d):
    d = d.copy()
    d["ym"] = d["date"].dt.to_period("M")
    parts = []
    for level, col, seg, base in [("category", "category", "A", PRE_START),
                                  ("category", "category", "B", PAN_START),
                                  ("region", "admin1", "B", PAN_START)]:
        idx = chain_index(d[d["segment"] == seg], col, base)
        long = idx.reset_index().melt(id_vars="index", var_name="group", value_name="index_value")
        long = long.rename(columns={"index": "month"})
        long["month"] = long["month"].astype(str)
        long["level"], long["segment"] = level, seg
        parts.append(long[["level", "group", "segment", "month", "index_value"]])
    chain = pd.concat(parts, ignore_index=True)
    chain["index_value"] = chain["index_value"].round(3)
    out.write(chain, "dash_chain_index")
    end = chain[(chain["segment"] == "B") & (chain["month"] == "2026-06")]
    log("  chain index, June 2026 (May 2020 = 100): " +
        "; ".join(f"{r.group} {r.index_value:.1f}" for r in end[end["level"] == "category"].itertuples()))


def boot_ci(m, B=1000, seed=42):
    """95% interval of the median % change, resampling whole markets (identical to Step 10)."""
    rng = np.random.default_rng(seed)
    cl = m.groupby("market_id")["pct_chg"].apply(np.array)
    vals = [np.median(np.concatenate([cl.iloc[i] for i in rng.integers(0, len(cl), len(cl))])) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def build_period_tables(out):
    frames = []
    for k in ["pre_pan", "pan_post", "pre_post"]:
        m = pd.read_csv(TAB / f"10_matched_series_{k}.csv")
        b = (pd.read_csv(TAB / f"13_calendar_balanced_{k}.csv")[["market_id", "commodity_id", "pct_chg"]]
             .rename(columns={"pct_chg": "pct_chg_calendar_balanced"}))
        m = m.merge(b, on=["market_id", "commodity_id"], how="left")
        m.insert(0, "comparison", k)
        frames.append(m[["comparison", "market_id", "commodity_id", "category", "commodity", "market",
                         "admin1", "p1", "p2", "pct_chg", "pct_chg_calendar_balanced"]])
    ps = pd.concat(frames, ignore_index=True)
    out.write(ps.round({"p1": 3, "p2": 3, "pct_chg": 3, "pct_chg_calendar_balanced": 3}), "dash_period_series")

    dims = [("overall", []), ("category", ["category"]), ("region", ["admin1"]),
            ("commodity", ["commodity"]), ("market", ["market", "admin1"]),
            ("category_region", ["admin1", "category"])]
    rows = []
    for comp, df in ps.groupby("comparison"):
        for dim, cols in dims:
            groups = [((), df)] if not cols else list(df.groupby(cols))
            for key, g in groups:
                key = key if isinstance(key, tuple) else (key,)
                lab = {"overall": "All", "category": key[0] if cols else "", "region": key[0] if cols else "",
                       "commodity": key[0] if cols else "", "market": key[0] if cols else "",
                       "category_region": f"{key[0]} | {key[1]}" if len(key) > 1 else ""}[dim]
                n_mkt, n_ser = g["market_id"].nunique(), len(g)
                reliable = (n_mkt >= MIN_MKT) and (n_ser >= MIN_SER)
                lo = hi = np.nan
                if comp == "pan_post" and reliable:
                    lo, hi = boot_ci(g)
                rows.append({"comparison": comp, "dimension": dim, "group": lab,
                             "category": g["category"].iloc[0] if dim in ("category", "commodity", "category_region") else "",
                             "region": g["admin1"].iloc[0] if dim in ("region", "market", "category_region") else "",
                             "commodity": g["commodity"].iloc[0] if dim == "commodity" else "",
                             "market": g["market"].iloc[0] if dim == "market" else "",
                             "series": n_ser, "markets": n_mkt,
                             "median_pct": round(g["pct_chg"].median(), 2),
                             "share_rising_pct": round((g["pct_chg"] > 0).mean() * 100, 1),
                             "median_pct_cal_balanced": round(g["pct_chg_calendar_balanced"].median(), 2),
                             "ci_low": round(lo, 2) if lo == lo else np.nan,
                             "ci_high": round(hi, 2) if hi == hi else np.nan,
                             "reliable": reliable})
    summ = pd.DataFrame(rows)
    out.write(summ, "dash_period_summary")
    ov = summ[(summ["comparison"] == "pan_post") & (summ["dimension"] == "overall")].iloc[0]
    log(f"  Pandemic -> Post-pandemic overall: median {ov.median_pct}% [{ov.ci_low}, {ov.ci_high}], {ov.series} series")


def build_forecast_summary(out, paired):
    models = {"snaive": "f_snaive", "naive": "f_naive", "sarima": "f_sarima"}
    sm = series_metrics(paired, models)
    rows = []

    obs = []
    for name, col in models.items():
        ape = (np.exp(paired[col]) - paired["price"]).abs() / paired["price"] * 100
        obs.append(pd.DataFrame({"window": paired["window"], "model": name, "h": paired["h"],
                                 "category": paired["category"], "commodity": paired["commodity"], "ape": ape}))
    obs = pd.concat(obs, ignore_index=True)
    obs["all"] = "All"
    obs["horizon_band"] = pd.cut(obs["h"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
    sm["all"] = "All"

    for dim, col in [("overall", "all"), ("category", "category"), ("commodity", "commodity")]:
        o = obs.groupby(["window", "model", col])["ape"].agg(n_obs="size", pooled_MAPE="mean").reset_index()
        s = sm.groupby(["window", "model", col]).agg(
            n_series=("MAE", "size"), median_series_MAPE=("MAPE", "median"),
            median_series_MAE_php=("MAE", "median"), median_series_RMSE_php=("RMSE", "median")).reset_index()
        m = o.merge(s, on=["window", "model", col])
        if dim == "commodity":
            m = m[m["n_series"] >= 10]
        for r in m.itertuples(index=False):
            grp = getattr(r, col)
            for metric in ("pooled_MAPE", "median_series_MAPE", "median_series_MAE_php", "median_series_RMSE_php"):
                rows.append({"record_type": "accuracy", "window": r.window, "model": r.model, "vs": "",
                             "dimension": dim, "group": grp, "metric": metric, "value": round(getattr(r, metric), 3),
                             "ci_low": np.nan, "ci_high": np.nan, "verdict": "", "n_series": r.n_series,
                             "n_obs": r.n_obs, "ties_pct": np.nan})
    o = obs.groupby(["window", "model", "horizon_band"])["ape"].agg(n_obs="size", pooled_MAPE="mean").reset_index()
    for r in o.itertuples(index=False):
        rows.append({"record_type": "accuracy", "window": r.window, "model": r.model, "vs": "",
                     "dimension": "horizon_band", "group": r.horizon_band, "metric": "pooled_MAPE",
                     "value": round(r.pooled_MAPE, 3), "ci_low": np.nan, "ci_high": np.nan, "verdict": "",
                     "n_series": np.nan, "n_obs": r.n_obs, "ties_pct": np.nan})

    for a, b in [("sarima", "snaive"), ("sarima", "naive"), ("naive", "snaive")]:
        for dim, by in [("overall", None), ("category", "category")]:
            c = compare_gm(sm, a, b, by=by)
            for r in c.to_dict("records"):
                lo, hi = (float(x) for x in r["ci"].split("-"))
                rows.append({"record_type": "decision", "window": r["window"], "model": a, "vs": b,
                             "dimension": dim, "group": "All" if dim == "overall" else r["group"], "metric": r["metric"],
                             "value": r["gm_ratio"], "ci_low": lo, "ci_high": hi, "verdict": r["verdict"],
                             "n_series": r["series"], "n_obs": np.nan, "ties_pct": r["ties_%"]})
    summ = pd.DataFrame(rows)
    out.write(summ, "dash_forecast_summary")
    ov = summ[(summ["record_type"] == "accuracy") & (summ["dimension"] == "overall") & (summ["metric"] == "pooled_MAPE")]
    log("  pooled MAPE (%) by window/model: " + "; ".join(f"{r.window[:3]} {r.model} {r.value:.2f}" for r in ov.itertuples()))


def build_about(out, d, trial=False):
    last = d["date"].max().to_period("M")
    rows = [
        ("study_title", "Forecasting Philippine Food Prices Before, During, and After the COVID-19 Pandemic: A Time-Series Approach"),
        ("data_source", "WFP food prices via HDX (Philippines); retail prices only"),
        ("data_through", str(last)),
        ("generated_on", dt.date.today().isoformat()),
        ("price_unit", "Nominal Philippine pesos (PHP) per commodity unit (KG for most commodities); not adjusted for inflation"),
        ("period_pre_pandemic", f"{PRE_START} to {PRE_END}"),
        ("period_pandemic", f"{PAN_START} to {PAN_END}"),
        ("period_post_pandemic", f"{POST_START} to {POST_END}"),
        ("excluded_months", "Jun 2019 to Apr 2020 are excluded from all comparisons (Jun 2019-Feb 2020 too thin; Mar-Apr 2020 no data)"),
        ("series_definition", "A series is one commodity in one market (market_id x commodity_id)"),
        ("matched_series_rule", f"Period comparisons use series with at least 6 months in BOTH periods compared; prices are period means"),
        ("pre_pandemic_coverage", "Before 2020 only 17 markets (one per region) report; only 3 markets (Metro Manila, Davao City, Zamboanga City) also report later"),
        ("market_coverage_changes", "About 80 markets begin reporting in May 2020 and 14 of the original 17 stop; market-level trends refer to the market's own reporting era"),
        ("chain_index", "Chain-linked index: average month-to-month log change of the same series, cumulated; base month = 100; Segment A starts Jan 2016, Segment B starts May 2020; the two cannot be joined"),
        ("flag_rule", f"A monthly change larger than {JUMP} log points in absolute value is flagged (not removed)"),
        ("reliable_rule", f"A comparison cell is 'reliable' with at least {MIN_MKT} markets and {MIN_SER} series; confidence intervals resample whole markets"),
        ("forecast_models", "Seasonal Naive, SARIMA (12-model grid, AIC choice, optional seasonal terms), and plain Naive as a reference"),
        ("forecast_windows", "; ".join(f"{w}: train {a} to {b}, test {c} to {e}" for w, (a, b, c, e) in WINDOWS.items())),
        ("forecast_finding", "SARIMA is more accurate than Seasonal Naive but no better than plain Naive; forecasts are indicative only"),
        ("forward_forecasts", f"Origin {last}; horizon {HORIZON} months; series with >= {int(MIN_COV * 100)}% of months observed since May 2020, no gap over {MAX_GAP} months, observed in the latest month"),
        ("interval_method", "'_model' intervals come from the SARIMA fit; '_emp' intervals scale each series' volatility by empirical error quantiles from the Post-pandemic holdout. Prefer '_emp'; see coverage check files"),
        ("monitoring_rules", f"alert: monthly change beyond {JUMP} log points or more than 3 typical monthly swings of the series; watch: more than 2; typical swing = SD of that series' monthly log changes (needs 12 or more changes)"),
        ("causal_caveat", "Differences between periods are descriptive, not estimates of a pandemic effect"),
    ]
    if trial:
        rows.append(("trial_run", "THIS IS A TRIAL OUTPUT - NOT FOR USE"))
    out.write(pd.DataFrame(rows, columns=["key", "value"]), "dash_about")


# =============================================================================================
# Part B - forward forecasting (model refit)
# =============================================================================================
def fit_sarima_auto(y, steps=HORIZON):
    """Pick the SARIMA specification by AIC; return forecasts and intervals (log scale). Module level for joblib."""
    from statsmodels.tsa.arima.model import ARIMA
    best, spec = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q, P, Q in GRID12:
            try:
                m = ARIMA(y, order=(p, 1, q), seasonal_order=(P, 0, Q, 12), trend="n").fit()
            except Exception:
                continue
            if np.isfinite(m.aic) and (best is None or m.aic < best.aic):
                best, spec = m, (p, q, P, Q)
        if best is None:
            return None, "no model converged"
        pf = best.get_forecast(steps)
        res = {"spec": spec, "fc": np.asarray(pf.predicted_mean, dtype=float)}
        for lvl in LEVELS:
            try:
                ci = np.asarray(pf.conf_int(alpha=1 - lvl / 100), dtype=float)
                res[f"lo{lvl}"], res[f"hi{lvl}"] = ci[:, 0], ci[:, 1]
            except Exception:
                res[f"lo{lvl}"] = res[f"hi{lvl}"] = np.full(steps, np.nan)
    if not np.isfinite(res["fc"]).all():
        return None, "non-finite forecast"
    if np.abs(res["fc"] - y[-1]).max() > MAX_JUMP:
        return None, "unstable forecast"
    return res, ""


def fit_one(key, y):
    res, why = fit_sarima_auto(y)
    return key, res, why


def flatten(key, res, why):
    r = {"market_id": key[0], "commodity_id": key[1], "spec": str(res["spec"]) if res else "", "fail_reason": why}
    for name in ("fc", "lo80", "hi80", "lo95", "hi95"):
        for h in range(HORIZON):
            r[f"{name}_{h + 1}"] = res[name][h] if res else np.nan
    return r


def build_forward_inputs(d):
    """Training series for the forward forecasts: May 2020 to the latest month, small gaps interpolated."""
    d = d.copy()
    d["ym"] = d["date"].dt.to_period("M")
    last_month = d["ym"].max()
    months = pd.period_range(FWD_START, last_month, freq="M")
    ids = ["market_id", "commodity_id"]
    lp = d.pivot_table(index=ids, columns="ym", values="log_price", aggfunc="first").reindex(columns=months)
    arr = lp.to_numpy()
    pres = ~np.isnan(arr)

    def longest_gap(row):
        best = cur = 0
        for x in row:
            cur = 0 if x else cur + 1
            best = max(best, cur)
        return best

    gaps = np.array([longest_gap(r) for r in pres])
    cov = pres.sum(axis=1) / len(months)
    ok = (cov >= MIN_COV) & (gaps <= MAX_GAP) & pres[:, -1]
    train = {}
    for i in np.where(ok)[0]:
        x = arr[i]
        valid = ~np.isnan(x)
        f = int(np.where(valid)[0][0])
        idx = np.arange(f, len(x))
        train[lp.index[i]] = np.interp(idx, idx[valid[f:]], x[f:][valid[f:]])
    meta = (d.groupby(ids).agg(category=("category", "first"), commodity=("commodity", "first"),
                               market=("market", "first"), admin1=("admin1", "first")).reset_index())
    log(f"  forward-eligible series: {len(train):,} of {len(lp):,} with data since {FWD_START} "
        f"(last month {last_month}, {len(months)} training months)")
    return train, meta, last_month


def run_forward_fits(train, fit_dir, workers, limit):
    fit_dir.mkdir(parents=True, exist_ok=True)
    f = fit_dir / "forward_fits.csv"
    done = set()
    if f.exists():
        prev = pd.read_csv(f)
        done = set(zip(prev["market_id"], prev["commodity_id"]))
    todo = [(k, y) for k, y in train.items() if k not in done]
    if limit:
        todo = todo[:limit]
    log(f"  SARIMA refits to run: {len(todo):,} (already done: {len(done):,}); workers: {workers}")
    batch = 200
    for i in range(0, len(todo), batch):
        res = Parallel(n_jobs=workers)(delayed(fit_one)(k, y) for k, y in todo[i:i + batch])
        pd.DataFrame([flatten(k, r, w) for k, r, w in res]).to_csv(f, mode="a", header=not f.exists(), index=False)
        log(f"    fits {min(i + batch, len(todo))}/{len(todo)}")
    return pd.read_csv(f)


def window_sigma(panel, window):
    """SD of monthly log changes in each series' training data."""
    tr = panel[(panel["role"] == "train") & (panel["window"] == window)].sort_values("ym")
    s = tr.groupby(["market_id", "commodity_id"])["log_price"].apply(lambda x: np.std(np.diff(x.to_numpy()), ddof=1))
    return s.clip(lower=SIGMA_FLOOR).rename("sigma").reset_index()


def calibrate(paired, panel, window):
    """Empirical quantiles of normalised holdout errors z = (actual - forecast) / sigma, by model and horizon."""
    p = paired[paired["window"] == window].merge(window_sigma(panel, window), on=["market_id", "commodity_id"])
    cal = {}
    for model in ("sarima", "naive"):
        z = (p["log_price"] - p[f"f_{model}"]) / p["sigma"]
        for h, zh in z.groupby(p["h"]):
            q = {}
            for lvl in LEVELS:
                a = (1 - lvl / 100) / 2
                q[f"lo{lvl}"], q[f"hi{lvl}"] = np.quantile(zh, a), np.quantile(zh, 1 - a)
            cal[(model, int(h))] = q
    return cal


def calibration_check(paired, panel, out):
    """Calibrate on the Pandemic window, test on the Post-pandemic window (out-of-window coverage)."""
    cal = calibrate(paired, panel, "Pandemic")
    p = paired[paired["window"] == EMP_WINDOW].merge(window_sigma(panel, EMP_WINDOW), on=["market_id", "commodity_id"])
    rows = []
    for model in ("sarima", "naive"):
        z = (p["log_price"] - p[f"f_{model}"]) / p["sigma"]
        for lvl in LEVELS:
            lo = p["h"].map(lambda h: cal[(model, int(h))][f"lo{lvl}"])
            hi = p["h"].map(lambda h: cal[(model, int(h))][f"hi{lvl}"])
            inside = (z >= lo) & (z <= hi)
            band = pd.cut(p["h"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
            for b, g in inside.groupby(band):
                rows.append({"model": model, "nominal_pct": lvl, "horizon_band": b, "coverage_pct": round(g.mean() * 100, 1), "n": len(g)})
            rows.append({"model": model, "nominal_pct": lvl, "horizon_band": "all", "coverage_pct": round(inside.mean() * 100, 1), "n": len(inside)})
    chk = pd.DataFrame(rows)
    chk.to_csv(out.folder / "16_interval_calibration_check.csv", index=False)
    log("  empirical intervals, calibrated on the Pandemic window and tested on the Post-pandemic window (coverage %):")
    print(chk[chk["horizon_band"] == "all"].to_string(index=False))
    return chk


def model_interval_check(panel, workers, n, out, seed=1):
    """Refit a sample of Post-pandemic-window series and measure the coverage of SARIMA's own intervals."""
    if n <= 0:
        return None
    tr = panel[(panel["role"] == "train") & (panel["window"] == EMP_WINDOW)].sort_values("ym")
    arrays = tr.groupby(["market_id", "commodity_id"])["log_price"].apply(lambda s: s.to_numpy())
    keys = list(arrays.index)
    rng = np.random.default_rng(seed)
    pick = [keys[i] for i in rng.choice(len(keys), size=min(n, len(keys)), replace=False)]
    log(f"  model-interval coverage check: refitting {len(pick)} Post-pandemic-window series")
    res = Parallel(n_jobs=workers)(delayed(fit_one)(k, arrays[k]) for k in pick)
    te = panel[(panel["role"] == "test") & (panel["window"] == EMP_WINDOW)]
    rows = []
    for k, r, why in res:
        if r is None:
            continue
        t = te[(te["market_id"] == k[0]) & (te["commodity_id"] == k[1])]
        for lvl in LEVELS:
            lo, hi = r[f"lo{lvl}"][t["h"].to_numpy() - 1], r[f"hi{lvl}"][t["h"].to_numpy() - 1]
            inside = (t["log_price"].to_numpy() >= lo) & (t["log_price"].to_numpy() <= hi)
            for h, ins in zip(t["h"], inside):
                rows.append({"nominal_pct": lvl, "h": int(h), "inside": bool(ins)})
    if not rows:
        log("  (no successful fits for the coverage check)")
        return None
    r = pd.DataFrame(rows)
    r["horizon_band"] = pd.cut(r["h"], [0, 3, 6, 12], labels=["1-3", "4-6", "7-12"]).astype(str)
    chk = (r.groupby(["nominal_pct", "horizon_band"])["inside"].agg(coverage_pct=lambda x: round(x.mean() * 100, 1), n="size").reset_index())
    allc = r.groupby("nominal_pct")["inside"].agg(coverage_pct=lambda x: round(x.mean() * 100, 1), n="size").reset_index()
    allc["horizon_band"] = "all"
    chk = pd.concat([chk, allc], ignore_index=True)
    chk.to_csv(out.folder / "16_model_interval_coverage.csv", index=False)
    log("  SARIMA's own intervals, coverage on the Post-pandemic holdout (%):")
    print(allc.to_string(index=False))
    return chk


def build_forecast_future(out, train, meta, fits, paired, panel, last_month):
    cal = calibrate(paired, panel, EMP_WINDOW)
    hold = pd.read_csv(TAB / "12_series_metrics.csv")
    hold = hold[hold["window"] == EMP_WINDOW].pivot_table(index=["market_id", "commodity_id"], columns="model", values="MAPE").add_prefix("holdout_mape_").reset_index()
    sigma = {k: max(float(np.std(np.diff(y), ddof=1)), SIGMA_FLOOR) for k, y in train.items()}
    fits = fits.copy()
    fits["fail_reason"] = fits["fail_reason"].fillna("")
    fit_by_key = {(r["market_id"], r["commodity_id"]): r for r in fits.to_dict("records")}
    meta_by_key = {(r["market_id"], r["commodity_id"]): r for r in meta.to_dict("records")}
    rows = []
    for k, y in train.items():
        f = fit_by_key.get(k)
        m = meta_by_key[k]
        ok = f is not None and f["fail_reason"] == ""
        for h in range(1, HORIZON + 1):
            naive = y[-1]
            snaive = y[-12 + (h - 1)]
            row = {"market_id": k[0], "commodity_id": k[1], "category": m["category"], "commodity": m["commodity"],
                   "market": m["market"], "admin1": m["admin1"], "forecast_origin": str(last_month),
                   "ym": str(last_month + h), "h": h, "last_price": round(float(np.exp(y[-1])), 2),
                   "naive_php": np.exp(naive), "snaive_php": np.exp(snaive)}
            s = sigma[k]
            for lvl in LEVELS:
                c = cal[("naive", h)]
                row[f"naive_lo{lvl}_emp"], row[f"naive_hi{lvl}_emp"] = np.exp(naive + c[f"lo{lvl}"] * s), np.exp(naive + c[f"hi{lvl}"] * s)
            if ok:
                fc = f[f"fc_{h}"]
                row["sarima_php"] = np.exp(fc)
                for lvl in LEVELS:
                    c = cal[("sarima", h)]
                    row[f"sarima_lo{lvl}_model"], row[f"sarima_hi{lvl}_model"] = np.exp(f[f"lo{lvl}_{h}"]), np.exp(f[f"hi{lvl}_{h}"])
                    row[f"sarima_lo{lvl}_emp"], row[f"sarima_hi{lvl}_emp"] = np.exp(fc + c[f"lo{lvl}"] * s), np.exp(fc + c[f"hi{lvl}"] * s)
                row["sarima_spec"], row["sarima_status"] = f["spec"], "ok"
            else:
                row["sarima_status"] = (f["fail_reason"] if f is not None else "not fitted")
            row["indicative"] = True
            rows.append(row)
    fut = pd.DataFrame(rows).merge(hold, on=["market_id", "commodity_id"], how="left")
    num = [c for c in fut.columns if c.endswith(("_php", "_model", "_emp"))]
    fut[num] = fut[num].round(2)
    for c in ("holdout_mape_sarima", "holdout_mape_snaive", "holdout_mape_naive"):
        if c not in fut.columns:
            fut[c] = np.nan
    fut[[c for c in fut.columns if c.startswith("holdout_")]] = fut[[c for c in fut.columns if c.startswith("holdout_")]].round(2)
    out.write(fut, "dash_forecast_future")
    n_ok = fut[fut["h"] == 1]["sarima_status"].eq("ok").sum()
    log(f"  forward forecasts: {fut.groupby(['market_id', 'commodity_id']).ngroups:,} series x {HORIZON} months; SARIMA ok for {n_ok:,}")
    if n_ok:
        s = fut[(fut["h"] == 1) & fut["sarima_status"].eq("ok")]
        log(f"  SARIMA chose the random walk (0,0,0,0) in {(s['sarima_spec'] == '(0, 0, 0, 0)').mean() * 100:.1f}% of series")
    return fut


# =============================================================================================
# Part C - monitoring
# =============================================================================================

def add_model_monitoring(mon, paired, panel, latest_month):
    """Compare the latest actual price with the forecast made 12 months earlier (the last holdout month)."""
    cal = calibrate(paired, panel, "Pandemic")                      # out-of-window calibration
    sig = window_sigma(panel, EMP_WINDOW)
    p = paired[(paired["window"] == EMP_WINDOW) & (paired["ym"] == latest_month)].merge(sig, on=["market_id", "commodity_id"])
    keep = p[["market_id", "commodity_id"]].copy()
    for model in ("sarima", "naive"):
        z = (p["log_price"] - p[f"f_{model}"]) / p["sigma"]
        lo = p["h"].map(lambda h: cal[(model, int(h))]["lo95"])
        hi = p["h"].map(lambda h: cal[(model, int(h))]["hi95"])
        keep[f"prev_forecast_{model}_php"] = np.exp(p[f"f_{model}"]).round(2)
        keep[f"prev_error_pct_{model}"] = ((np.exp(p["log_price"] - p[f"f_{model}"]) - 1) * 100).round(2)
        keep[f"model_alert_{model}"] = np.select([z > hi, z < lo], ["above 95% interval", "below 95% interval"], default="within interval")
    mon = mon.merge(keep, on=["market_id", "commodity_id"], how="left")
    for model in ("sarima", "naive"):
        mon[f"model_alert_{model}"] = mon[f"model_alert_{model}"].fillna("no forecast")
    return mon

def build_monitoring(out, d, paired=None, panel=None):
    d = d.copy()
    d["mnum"] = d["date"].dt.year * 12 + d["date"].dt.month - 1
    latest = int(d["mnum"].max())
    last = d.sort_values("date").groupby(["market_id", "commodity_id"]).tail(1)
    last = last[last["mnum"] >= latest - 2].copy()
    pm = d.set_index(["market_id", "commodity_id", "mnum"])["price"]

    def look(off):
        idx = pd.MultiIndex.from_arrays([last["market_id"], last["commodity_id"], last["mnum"] - off])
        return pm.reindex(idx).to_numpy()

    last["prev_price"], last["price_12m_ago"] = look(1), look(12)
    last["change_vs_prev_month_pct"] = (last["price"] / last["prev_price"] - 1) * 100
    last["change_vs_12m_pct"] = (last["price"] / last["price_12m_ago"] - 1) * 100
    sd = d.groupby(["market_id", "commodity_id"])["lr"].agg(sd="std", n="count").reset_index()
    sd["vol_pct"] = np.where(sd["n"] >= 12, sd["sd"] * 100, np.nan)
    last = last.merge(sd[["market_id", "commodity_id", "vol_pct"]], on=["market_id", "commodity_id"], how="left")
    last["z_score"] = (last["lr"] * 100) / last["vol_pct"]
    big = last["flag_jump"].fillna(False).astype(bool)
    z = last["z_score"].abs()
    last["alert_level"] = np.select([big | (z >= 3), z >= 2], ["alert", "watch"], default="normal")
    last["alert_reason"] = np.select(
        [big, z >= 3, z >= 2],
        [f"monthly change beyond {JUMP} log points", "more than 3 typical monthly swings", "more than 2 typical monthly swings"],
        default="")
    last["latest_month"] = [str(pd.Period(year=int(m // 12), month=int(m % 12) + 1, freq="M")) for m in last["mnum"]]
    last["months_behind"] = latest - last["mnum"]
    cols = ["market_id", "commodity_id", "category", "commodity", "market", "admin1", "latest_month", "months_behind",
            "price", "prev_price", "change_vs_prev_month_pct", "price_12m_ago", "change_vs_12m_pct",
            "vol_pct", "z_score", "flag_jump", "alert_level", "alert_reason"]
    mon = last[cols].rename(columns={"price": "latest_price"})
    mon = mon.round({"latest_price": 2, "prev_price": 2, "price_12m_ago": 2, "change_vs_prev_month_pct": 2,
                     "change_vs_12m_pct": 2, "vol_pct": 2, "z_score": 2})
    if paired is not None and panel is not None:
        mon = add_model_monitoring(mon, paired, panel, str(pd.Period(year=latest // 12, month=latest % 12 + 1, freq="M")))
    out.write(mon, "dash_monitoring")
    log(f"  monitoring: {len(mon):,} series; price alert levels {mon['alert_level'].value_counts().to_dict()}")
    if "model_alert_naive" in mon.columns:
        log(f"  model monitoring (latest price vs the forecast made 12 months earlier), Naive: {mon['model_alert_naive'].value_counts().to_dict()}")
        log(f"  model monitoring, SARIMA: {mon['model_alert_sarima'].value_counts().to_dict()}")


# =============================================================================================
# Part D - verification
# =============================================================================================
REQUIRED = {
    "dash_prices": (["market_id", "commodity_id", "date"], ["date", "period", "segment", "admin1", "market_id", "market", "category", "commodity_id", "commodity", "unit", "price", "lr", "flag_jump"], 207601),
    "dash_series_stats": (["market_id", "commodity_id"], ["category", "commodity", "admin1", "market", "n_obs", "trend_pct_yr", "vol_pct"], 2725),
    "dash_chain_index": (["level", "group", "segment", "month"], ["index_value"], None),
    "dash_markets": (["market_id"], ["market", "admin1", "latitude", "longitude", "era", "eligible_series"], 97),
    "dash_commodities": (["commodity"], ["category", "unit", "observations", "markets", "eligible_series", "tier"], 64),
    "dash_period_series": (["comparison", "market_id", "commodity_id"], ["category", "commodity", "market", "admin1", "p1", "p2", "pct_chg", "pct_chg_calendar_balanced"], 2689),
    "dash_period_summary": (["comparison", "dimension", "group"], ["series", "markets", "median_pct", "ci_low", "ci_high", "reliable"], None),
    "dash_forecast_holdout": (["window", "market_id", "commodity_id", "ym"], ["h", "price", "f_naive", "f_snaive", "f_sarima"], 49864),
    "dash_forecast_metrics": (["window", "market_id", "commodity_id", "model"], ["n_test", "MAE", "RMSE", "MAPE"], 12486),
    "dash_forecast_summary": (["record_type", "window", "model", "vs", "dimension", "group", "metric"], ["value", "verdict", "n_series"], None),
    "dash_forecast_future": (["market_id", "commodity_id", "ym"], ["h", "naive_php", "snaive_php", "sarima_php", "sarima_lo95_emp", "sarima_hi95_emp", "indicative"], None),
    "dash_monitoring": (["market_id", "commodity_id"], ["latest_month", "latest_price", "change_vs_prev_month_pct", "change_vs_12m_pct", "alert_level", "model_alert_sarima", "model_alert_naive"], None),
    "dash_about": (["key"], ["value"], None),
}


def verify(out, trial):
    log("Verification of the 13 contract tables:")
    rows = []
    for name, (key, cols, expected) in REQUIRED.items():
        fname = out.files.get(name)
        status, nrows, ncols, size, uniq, notes = "OK", 0, 0, 0.0, None, []
        if fname is None or not (out.folder / fname).exists():
            rows.append({"table": name, "file": "", "rows": 0, "columns": 0, "size_mb": 0, "key_unique": None, "status": "MISSING", "notes": ""})
            continue
        df = pd.read_csv(out.folder / fname, low_memory=False)
        nrows, ncols, size = len(df), df.shape[1], (out.folder / fname).stat().st_size / 1e6
        miss = [c for c in key + cols if c not in df.columns]
        if miss:
            status = "FAIL"; notes.append(f"missing columns {miss}")
        else:
            k = df[key].fillna("").astype(str)
            uniq = not k.duplicated().any()
            if not uniq:
                status = "FAIL"; notes.append("duplicate keys")
            if nrows == 0:
                status = "FAIL"; notes.append("empty")
        if status == "OK" and expected is not None and nrows != expected and not trial:
            status = "CHECK"; notes.append(f"expected {expected:,} rows")
        rows.append({"table": name, "file": fname, "rows": nrows, "columns": ncols, "size_mb": round(size, 2),
                     "key_unique": uniq, "status": status, "notes": "; ".join(notes)})
    man = pd.DataFrame(rows)
    man.to_csv(out.folder / "16_datapack_manifest.csv", index=False)
    print(man.to_string(index=False))
    counts = man["status"].value_counts().to_dict()
    log(f"Status counts: {counts}")
    big = man[man["size_mb"] > 50]
    if len(big):
        log(f"WARNING: files above 50 MB (GitHub warns): {big['file'].tolist()}")
    return counts


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=-2, help="parallel workers for model fitting (default: all cores but one)")
    ap.add_argument("--limit", type=int, default=None, help="trial run: fit only this many series; writes to outputs/tables_trial")
    ap.add_argument("--skip-fit", action="store_true", help="reuse saved forward fits, do not refit")
    ap.add_argument("--coverage-n", type=int, default=200, help="series refitted for the model-interval coverage check (0 = skip)")
    ap.add_argument("--no-gzip", action="store_true", help="write the two large tables as plain CSV")
    args = ap.parse_args()
    trial = args.limit is not None

    out = Out(ROOT / "outputs" / ("tables_trial" if trial else "tables"), gzip_large=not args.no_gzip)
    fit_dir = PROC / ("sarima_forward_trial" if trial else "sarima_forward")
    t0 = time.time()
    log(f"Step 16 data pack | output folder: {out.folder} | trial: {trial}")

    d = pd.read_csv(PROC / "retail_clean_flagged.csv", parse_dates=["date"])
    panel = pd.read_csv(PROC / "forecast_panel.csv")

    log("Part A: tables that need no refit")
    paired = build_direct_tables(out, d)
    build_chain_table(out, d)
    build_period_tables(out)
    build_forecast_summary(out, paired)
    build_about(out, d, trial)

    log("Part B: forward forecasts (SARIMA refit through the latest month)")
    train, meta, last_month = build_forward_inputs(d)
    if args.skip_fit and (fit_dir / "forward_fits.csv").exists():
        fits = pd.read_csv(fit_dir / "forward_fits.csv")
    else:
        fits = run_forward_fits(train, fit_dir, args.workers, args.limit)
    if trial:                                   # keep a trial output small and consistent
        keep = set(zip(fits["market_id"], fits["commodity_id"]))
        train = {k: v for k, v in train.items() if k in keep}
    calibration_check(paired, panel, out)
    model_interval_check(panel, args.workers, min(args.coverage_n, args.limit) if trial else args.coverage_n, out)
    build_forecast_future(out, train, meta, fits, paired, panel, last_month)

    log("Part C: monitoring")
    build_monitoring(out, d, paired, panel)

    log("Part D: verification")
    counts = verify(out, trial)

    c = pd.read_csv(TAB / "15_dashboard_data_contract.csv") if (TAB / "15_dashboard_data_contract.csv").exists() else None
    if c is not None:
        info = {}
        for t, fn in out.files.items():
            df_t = pd.read_csv(out.folder / fn, low_memory=False)
            info[t] = (fn, f"{len(df_t):,}", ", ".join(df_t.columns))
        c["file"] = c["table"].map(lambda t: info[t][0] if t in info else "")
        c["rows_now"] = c["table"].map(lambda t: info[t][1] if t in info else "n/a")
        c["columns"] = c["table"].map(lambda t: info[t][2] if t in info else "")
        c["source"] = c["source"].astype(str) + " | generated by 16_datapack_preparation.py"
        c.to_csv(out.folder / "16_dashboard_data_contract_final.csv", index=False)
    log(f"Done in {(time.time() - t0) / 60:.1f} min")
    if counts.get("MISSING", 0) or counts.get("FAIL", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()