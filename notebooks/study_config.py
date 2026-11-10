import pandas as pd

def M(s):
    return pd.Period(s, freq="M")

# ---- Study-wide definitions (single source of truth) ----
PRE_START, PRE_END = M("2016-01"), M("2019-05")        # Pre-pandemic
PAN_START, PAN_END = M("2020-05"), M("2023-05")        # Pandemic
POST_START, POST_END = M("2023-06"), M("2026-06")      # Post-pandemic
# Excluded: June 2019 - Feb 2020 (too thin) and March - April 2020 (no data) = June 2019 - April 2020

JUMP = 0.5               # anomaly flag: |month-to-month log change| > 0.5 (same rule as Step 4)
MIN_MONTHS_TREND = 24    # a series needs >= 24 observed months to get a trend estimate
MIN_RETURNS_VOL = 12     # ... and >= 12 consecutive-month changes to get a volatility estimate

CORE = ["Rice (regular, milled)", "Meat (chicken, whole)", "Onions (red)",
        "Meat (pork)", "Meat (beef, chops with bones)"]

# ---- Forecast evaluation windows (Step 11): (train_start, train_end, test_start, test_end) ----
WINDOWS = {
    "Pre-pandemic":  ("2016-01", "2018-05", "2018-06", "2019-05"),
    "Pandemic":      ("2020-05", "2022-05", "2022-06", "2023-05"),
    "Post-pandemic": ("2020-05", "2025-06", "2025-07", "2026-06"),
}
# eligibility: >= 80% of training months observed, no run of > 3 missing months, >= 9 of 12 test months observed
ELIG = dict(min_train_cov=0.80, max_gap=3, min_test_obs=9)

def period_label(p):
    if p <= PRE_END:
        return "Pre-pandemic"
    if p < PAN_START:
        return "Excluded"
    if p <= PAN_END:
        return "Pandemic"
    return "Post-pandemic"