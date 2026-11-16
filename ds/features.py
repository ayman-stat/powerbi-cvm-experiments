"""Data loading and feature engineering for the CVM notebooks (imported by the notebooks and by tests/test_features.py).

Mirrors etl/build_data.py, the production ETL behind the Power BI report, so the notebooks reproduce its numbers:

* Online Retail II: the same cleaning rules, the same order table and the same snapshot(t) churn features and label
  (no order in the 90 days from t). One deliberate change: a customer's country is the mode of their lines *before* t.
  The ETL takes the mode over the whole history, a small look-ahead (12 of 5,875 customers ever change country) that
  leaves the published AUC unchanged; notebook 02 shows both.
* Hillstrom e-mail test: the same arm labels, design matrix and cross-fitted logistic T-learner, plus the
  self-normalised inverse-propensity policy value used for the policy table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / ".raw"

# --------------------------------------------------------------------------------------------- Online Retail II
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002", "D", "CRUK", "S", "B", "PADS"}
FEATS = ["recency_days", "tenure_days", "orders_365", "orders_90", "orders_all", "spend_365", "spend_90", "spend_prev_90",
         "avg_basket", "products_365", "cancel_lines_365", "avg_gap_days", "is_uk"]
FEAT_LABEL = {"recency_days": "Days since last order", "tenure_days": "Customer tenure, days", "orders_365": "Orders, last 12 months",
              "orders_90": "Orders, last 90 days", "orders_all": "Orders, all time", "spend_365": "Spend, last 12 months",
              "spend_90": "Spend, last 90 days", "spend_prev_90": "Spend, prior 90 days", "avg_basket": "Average basket",
              "products_365": "Distinct products, 12 months", "cancel_lines_365": "Cancelled lines, 12 months",
              "avg_gap_days": "Average days between orders", "is_uk": "UK customer"}
SNAPSHOTS = {"train": "2011-03-01", "valid": "2011-06-01", "test": "2011-09-01"}
HORIZON, LOOKBACK = 90, 365
# the production champion (etl/build_data.py)
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=42)


def load_raw_lines(raw: Path = RAW) -> pd.DataFrame:
    """Every order line of both workbook sheets (the ETL caches them in .raw/onlineretail/lines.pkl).
    The sheets overlap on 1-9 Dec 2010; that week is kept from the second sheet only."""
    cache = Path(raw) / "onlineretail" / "lines.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    xl = Path(raw) / "onlineretail" / "online_retail_II.xlsx"
    a = pd.read_excel(xl, sheet_name="Year 2009-2010", dtype={"Invoice": str, "StockCode": str})
    b = pd.read_excel(xl, sheet_name="Year 2010-2011", dtype={"Invoice": str, "StockCode": str})
    return pd.concat([a[a["InvoiceDate"] < "2010-12-01"], b], ignore_index=True)


def clean_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ETL's cleaning rules, one step at a time so the notebook can show what each rule removes.
    Cancellations (invoice 'C...') are kept: they carry negative quantities, so revenue is net of returns."""
    steps = [("Raw lines (both sheets, overlap week de-duplicated)", len(raw))]
    df = raw.rename(columns={"Customer ID": "CustomerID"})
    df = df[df["CustomerID"].notna()]
    steps.append(("Drop lines without a customer ID", len(df)))
    df = df[df["Price"] > 0]
    steps.append(("Drop zero and negative prices (bad debt, free samples)", len(df)))
    code = df["StockCode"].astype(str)
    df = df[~code.str.upper().isin(NON_PRODUCT) & ~code.str.startswith("gift_")].copy()
    steps.append(("Drop non-product codes (postage, fees, manual adjustments, vouchers)", len(df)))
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["IsCancel"] = df["Invoice"].astype(str).str.startswith("C")
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Day"] = df["InvoiceDate"].dt.normalize()
    t = pd.DataFrame(steps, columns=["step", "rows"]).set_index("step")
    t["removed"] = (-t["rows"].diff()).fillna(0).astype(int)
    t["share_of_raw"] = t["rows"] / t["rows"].iloc[0]
    return df, t


def build_orders(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per (customer, purchase invoice); cancellations are not orders."""
    return (lines[~lines["IsCancel"]].groupby(["CustomerID", "Invoice"])
            .agg(Day=("Day", "min"), Month=("Month", "min")).reset_index())


def customer_country(lines: pd.DataFrame) -> pd.Series:
    """Most frequent country on a customer's lines (ties -> alphabetical first, as in the ETL)."""
    return lines.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0])


def snapshot(lines: pd.DataFrame, t, orders: pd.DataFrame | None = None, horizon: int = HORIZON,
             country: pd.Series | None = None) -> pd.DataFrame:
    """Features of every customer with an order in the 365 days before t, from data strictly before t, and the label
    churn = 1 if they place no order in [t, t + horizon). `country` overrides the history-only country (the ETL passes
    the whole-history mode)."""
    t = pd.Timestamp(t)
    orders = build_orders(lines) if orders is None else orders
    hist = lines[lines["Day"] < t]
    o = orders[orders["Day"] < t]
    o365 = o[o["Day"] >= t - pd.Timedelta(days=LOOKBACK)]
    base = o365["CustomerID"].unique()
    g = o.groupby("CustomerID")
    f = pd.DataFrame(index=pd.Index(base, name="CustomerID"))
    f["recency_days"] = (t - g["Day"].max()).dt.days.reindex(base)
    f["tenure_days"] = (t - g["Day"].min()).dt.days.reindex(base)
    f["orders_365"] = o365.groupby("CustomerID").size().reindex(base).fillna(0)
    f["orders_90"] = o[o["Day"] >= t - pd.Timedelta(days=90)].groupby("CustomerID").size().reindex(base).fillna(0)
    f["orders_all"] = g.size().reindex(base)
    h365 = hist[hist["Day"] >= t - pd.Timedelta(days=LOOKBACK)]
    f["spend_365"] = h365.groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    f["spend_90"] = hist[hist["Day"] >= t - pd.Timedelta(days=90)].groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    prev = hist[(hist["Day"] >= t - pd.Timedelta(days=180)) & (hist["Day"] < t - pd.Timedelta(days=90))]
    f["spend_prev_90"] = prev.groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    f["avg_basket"] = f["spend_365"] / f["orders_365"].clip(lower=1)
    f["products_365"] = h365.groupby("CustomerID")["StockCode"].nunique().reindex(base).fillna(0)
    f["cancel_lines_365"] = h365[h365["IsCancel"]].groupby("CustomerID").size().reindex(base).fillna(0)
    gaps = o.sort_values("Day").groupby("CustomerID")["Day"].apply(lambda s: s.diff().dt.days.mean())
    f["avg_gap_days"] = gaps.reindex(base).fillna(f["tenure_days"])
    ctry = customer_country(hist) if country is None else country
    f["is_uk"] = (ctry.reindex(base) == "United Kingdom").astype(int).values
    f["churn"] = churn_label(orders, base, t, horizon)
    return f


def churn_label(orders: pd.DataFrame, customers, t, horizon: int = HORIZON) -> np.ndarray:
    """1 if the customer places no order in [t, t + horizon) days."""
    t = pd.Timestamp(t)
    nxt = orders.loc[(orders["Day"] >= t) & (orders["Day"] < t + pd.Timedelta(days=horizon)), "CustomerID"].unique()
    return (~pd.Index(customers).isin(nxt)).astype(int)


def same_season_orders(orders: pd.DataFrame, customers, t, horizon: int = HORIZON) -> np.ndarray:
    """Orders in the label window one year earlier, [t - 365, t - 365 + horizon): a seasonal-buyer signal that uses only
    history before t."""
    t0 = pd.Timestamp(t) - pd.Timedelta(days=LOOKBACK)
    o = orders[(orders["Day"] >= t0) & (orders["Day"] < t0 + pd.Timedelta(days=horizon))]
    return o.groupby("CustomerID").size().reindex(pd.Index(customers)).fillna(0).values


def segments(f: pd.DataFrame) -> pd.DataFrame:
    """Segments used for robustness checks: market, value tier (the ETL's spend tiers) and tenure band."""
    v = f["spend_365"]
    return pd.DataFrame({
        "market": np.where(f["is_uk"] == 1, "UK", "Outside UK"),
        "value_tier": np.where(v >= v.quantile(0.8), "Top 20%", np.where(v >= v.quantile(0.5), "Middle 30%", "Bottom 50%")),
        "tenure_band": pd.cut(f["tenure_days"], [-1, 180, 365, 10_000], labels=["Under 6 months", "6-12 months", "Over 12 months"]).astype(str),
    }, index=f.index)


def signed_log1p(x):
    """log(1 + |x|) with the sign kept: tames heavy-tailed spend that can be negative (net of returns)."""
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def shift_to_rate(p, target_rate: float) -> np.ndarray:
    """Calibration-in-the-large: add the one logit offset that makes the mean prediction equal `target_rate`.
    Ranking (AUC) is unchanged; use it when the base rate moves (season, sampling) but the ranking still holds."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))



class RFMRule(ClassifierMixin, BaseEstimator):
    """The CVM team's classic rule: recency and frequency quintiles (edges learned on the training base), each of the
    25 R x F cells scored by its training churn rate, shrunk towards the base rate by `prior_weight` customers."""

    def __init__(self, prior_weight: float = 20.0):
        self.prior_weight = prior_weight

    def _cells(self, X):
        r = np.searchsorted(self.r_edges_, np.asarray(X["recency_days"], float), side="right")
        f = np.searchsorted(self.f_edges_, np.asarray(X["orders_365"], float), side="right")
        return r * 10 + f

    def fit(self, X, y):
        q = [0.2, 0.4, 0.6, 0.8]
        self.r_edges_ = np.unique(np.quantile(X["recency_days"], q))
        self.f_edges_ = np.unique(np.quantile(X["orders_365"], q))
        y = np.asarray(y)
        self.classes_, self.base_ = np.array([0, 1]), float(y.mean())
        g = pd.DataFrame({"c": self._cells(X), "y": y}).groupby("c")["y"].agg(["sum", "count"])
        self.rate_ = ((g["sum"] + self.prior_weight * self.base_) / (g["count"] + self.prior_weight)).to_dict()
        return self

    def predict_proba(self, X):
        p = np.array([self.rate_.get(c, self.base_) for c in self._cells(X)])
        return np.c_[1 - p, p]


def rfm_scores(f: pd.DataFrame) -> pd.DataFrame:
    """Recency and frequency quintile scores (5 = best) and the ETL's named RFM segment."""
    r = pd.qcut(-f["recency_days"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    fr = pd.qcut(f["orders_365"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return pd.DataFrame({"R": r, "F": fr, "segment": [rfm_segment(a, b) for a, b in zip(r, fr)]}, index=f.index)


def rfm_segment(r: int, f: int) -> str:
    if r >= 4 and f >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "Promising"
    if r <= 2 and f >= 4:
        return "Can't lose"
    if r <= 2 and f >= 2:
        return "At risk"
    if r == 3:
        return "Needs attention"
    return "Hibernating"


def monthly_flows(orders: pd.DataFrame, start="2009-12-01", end="2011-11-01", report_from="2010-03-01") -> pd.DataFrame:
    """Customer bridge per month on the ETL's rolling-3-month active definition (bought in m-2..m):
    New (first purchase month), Retained (active now and last month), Reactivated (active now, not last month)
    and Churned (active last month, not now)."""
    months = pd.date_range(start, end, freq="MS")
    grid = orders.groupby(["CustomerID", "Month"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0)
    bought = (grid > 0).values
    first = orders.groupby("CustomerID")["Month"].min().reindex(grid.index).values
    active = np.zeros_like(bought)
    for j in range(len(months)):
        active[:, j] = bought[:, max(0, j - 2): j + 1].any(axis=1)
    prev = np.concatenate([np.zeros((len(bought), 1), bool), active[:, :-1]], axis=1)
    rows = []
    for j, m in enumerate(months):
        if m < pd.Timestamp(report_from):
            continue
        started = first <= np.datetime64(m)
        new = first == np.datetime64(m)
        a, pa = active[:, j] & started, prev[:, j] & started
        rows.append({"month": m, "New": int(new.sum()), "Retained": int((a & pa & ~new).sum()),
                     "Reactivated": int((a & ~pa & ~new).sum()), "Churned": int((pa & ~a & ~new).sum()),
                     "Active": int(a.sum())})
    return pd.DataFrame(rows).set_index("month")


def cohort_matrix(orders: pd.DataFrame, last_month="2011-11-01") -> pd.DataFrame:
    """Share of each first-purchase cohort that buys again k months later (k = 0 is the cohort month itself)."""
    o = orders[orders["Month"] <= pd.Timestamp(last_month)][["CustomerID", "Month"]].drop_duplicates()
    first = o.groupby("CustomerID")["Month"].min().rename("Cohort")
    o = o.join(first, on="CustomerID")
    o["k"] = (o["Month"].dt.year - o["Cohort"].dt.year) * 12 + o["Month"].dt.month - o["Cohort"].dt.month
    counts = o.groupby(["Cohort", "k"])["CustomerID"].nunique().unstack()
    return counts.div(counts[0], axis=0)


# --------------------------------------------------------------------------------------------- Hillstrom e-mail test
ARMS = ["Control", "Mens offer", "Womens offer"]
TREATMENTS = ["Mens offer", "Womens offer"]
P_ARM = 1 / 3                                  # randomisation probability of each arm


def load_hillstrom(raw: Path = RAW) -> pd.DataFrame:
    h = pd.read_csv(Path(raw) / "hillstrom" / "hillstrom.csv")
    h["Arm"] = h["segment"].map({"Mens E-Mail": "Mens offer", "Womens E-Mail": "Womens offer", "No E-Mail": "Control"})
    h["ZipCode"] = h["zip_code"].replace({"Surburban": "Suburban"})
    h["Purchased"] = np.select([(h["mens"] == 1) & (h["womens"] == 1), h["mens"] == 1, h["womens"] == 1],
                               ["Both", "Mens only", "Womens only"], "Neither")
    h["RecencyBand"] = pd.cut(h["recency"], [0, 2, 4, 6, 9, 12],
                              labels=["1-2 months", "3-4 months", "5-6 months", "7-9 months", "10-12 months"]).astype(str)
    return h


def uplift_design(h: pd.DataFrame) -> pd.DataFrame:
    """The ETL's design matrix for the uplift models (pre-treatment covariates only)."""
    X = pd.get_dummies(h[["recency", "history", "mens", "womens", "newbie", "ZipCode", "channel"]], columns=["ZipCode", "channel"], dtype=int)
    X["log_history"] = np.log1p(h["history"])
    X["mens_and_womens"] = h["mens"] * h["womens"]
    return X


def cross_fit_arm_models(X: pd.DataFrame, y, arm, make_model, arms=ARMS, n_splits: int = 5, seed: int = 7) -> dict:
    """T-learner with cross-fitting: one outcome model per arm, and every row is scored by models that never saw it.
    Returns {arm: out-of-fold P(outcome | x, arm)}."""
    from sklearn.model_selection import KFold
    y, arm = np.asarray(y), np.asarray(arm)
    pred = {a: np.zeros(len(X)) for a in arms}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        for a in arms:
            idx = tr[arm[tr] == a]
            m = make_model().fit(X.iloc[idx], y[idx])
            pred[a][te] = m.predict_proba(X.iloc[te])[:, 1] if hasattr(m, "predict_proba") else m.predict(X.iloc[te])
    return pred


def cross_fit_transformed_outcome(X: pd.DataFrame, y, arm, treatment: str, make_model, n_splits: int = 5, seed: int = 7) -> np.ndarray:
    """Transformed-outcome (class-transformation) learner for one treatment vs control: with a 50/50 split inside the
    pair, Z = 2 * y * (+1 treated / -1 control) has E[Z | x] = uplift(x), so one regression of Z on x estimates uplift
    directly. Cross-fitted on the same folds as the T-learner; every row (any arm) gets an out-of-fold score."""
    from sklearn.model_selection import KFold
    y, arm = np.asarray(y, float), np.asarray(arm)
    pair = np.isin(arm, [treatment, "Control"])
    z = 2 * y * np.where(arm == treatment, 1.0, -1.0)
    out = np.zeros(len(X))
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        idx = tr[pair[tr]]
        out[te] = make_model().fit(X.iloc[idx], z[idx]).predict(X.iloc[te])
    return out


def auuc_draws(y, treated, scores: dict, n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Bootstrap AUUC of several scores on the same customer resamples (so differences between scores are paired)."""
    y, w = np.asarray(y, float), np.asarray(treated).astype(bool)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        rows.append({k: auuc(uplift_curve(y[ix], w[ix], np.asarray(s)[ix], n_points=50)) for k, s in scores.items()})
    return pd.DataFrame(rows)


def capture_at(curve: pd.DataFrame, share: float) -> float:
    """Share of all incremental outcomes (at 100% targeting) captured by the top `share` of customers. Random = share."""
    k = (curve["share"] - share).abs().idxmin()
    return float(curve.loc[k, "incremental"] / curve["incremental"].iloc[-1])


def targeting_curve(score, offer, arm, outcome, shares) -> np.ndarray:
    """The ETL's targeting simulator: send `offer` to the top share by `score`, no e-mail to the rest, and estimate the
    total incremental outcome vs no e-mail for the whole population by the self-normalised IPW (Hajek) estimator."""
    order = np.argsort(-np.asarray(score, float))           # numpy's default sort, as in the ETL (it decides the order of tied scores)
    arm, offer, y = np.asarray(arm)[order], np.asarray(offer)[order], np.asarray(outcome, float)[order]
    hit_top, hit_rest = arm == offer, arm == "Control"
    top_n, top_y = np.r_[0, np.cumsum(hit_top)], np.r_[0, np.cumsum(hit_top * y)]
    rest_n, rest_y = np.r_[np.cumsum((hit_rest * 1)[::-1])[::-1], 0], np.r_[np.cumsum((hit_rest * y)[::-1])[::-1], 0]
    base = y[hit_rest].mean()
    ks = np.round(np.asarray(shares) * len(y)).astype(int)
    return ((top_y[ks] + rest_y[ks]) / (top_n[ks] + rest_n[ks]) - base) * len(y)


def policy_value(arm, assign, outcome) -> float:
    """Self-normalised (Hajek) inverse-propensity estimate of the mean outcome if everyone got `assign`.
    Equal propensities (1/3) cancel, so it is the outcome mean over customers whose random arm matches the policy."""
    w = (np.asarray(arm) == np.asarray(assign)) / P_ARM
    return float(np.sum(w * np.asarray(outcome, float)) / np.sum(w))


def policy_bootstrap(arm, assign, outcome, rng, n: int = 400) -> np.ndarray:
    """Bootstrap draws of policy value minus the control mean (the ETL's interval; pass its RNG to reproduce it)."""
    arm, a, y = np.asarray(arm), np.asarray(assign), np.asarray(outcome, float)
    vals = np.empty(n)
    for b in range(n):
        ix = rng.integers(0, len(arm), len(arm))
        wa, wc = arm[ix] == a[ix], arm[ix] == "Control"
        vals[b] = np.sum(wa * y[ix]) / wa.sum() - np.sum(wc * y[ix]) / wc.sum()
    return vals


def uplift_curve(y, treated, score, n_points: int = 100) -> pd.DataFrame:
    """Cumulative uplift curve on a randomised sample: target the top share by score; incremental outcomes =
    (treated rate - control rate among the targeted) x targeted count. Also the Qini form (Radcliffe)."""
    y, w, s = np.asarray(y, float), np.asarray(treated).astype(bool), np.asarray(score, float)
    order = np.argsort(-s, kind="stable")
    y, w = y[order], w[order]
    nt, nc = np.cumsum(w), np.cumsum(~w)
    yt, yc = np.cumsum(y * w), np.cumsum(y * ~w)
    ks = np.unique(np.linspace(0, len(y), n_points + 1).astype(int))[1:]
    rt = yt[ks - 1] / np.maximum(nt[ks - 1], 1)
    rc = yc[ks - 1] / np.maximum(nc[ks - 1], 1)
    out = pd.DataFrame({"share": ks / len(y), "n": ks, "uplift_rate": rt - rc, "incremental": (rt - rc) * ks,
                        "qini": yt[ks - 1] - yc[ks - 1] * nt[ks - 1] / np.maximum(nc[ks - 1], 1)})
    return pd.concat([pd.DataFrame({"share": [0.0], "n": [0], "uplift_rate": [0.0], "incremental": [0.0], "qini": [0.0]}), out],
                     ignore_index=True)


def auuc(curve: pd.DataFrame, col: str = "incremental") -> float:
    """Area under the cumulative uplift curve minus the random-targeting diagonal, normalised by the population size
    (units: incremental outcomes per customer, averaged over target shares). 0 = no better than random."""
    x, yv = curve["share"].values, curve[col].values / curve["n"].values.max()
    rand = x * yv[-1]
    return float(np.sum((yv[1:] - rand[1:] + yv[:-1] - rand[:-1]) / 2 * np.diff(x)))
