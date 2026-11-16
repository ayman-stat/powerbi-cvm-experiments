"""Project tests on tiny synthetic frames (no raw data needed): cleaning rules, snapshot features without look-ahead,
the churn label window, the lifecycle bridge and the uplift / policy estimators."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ds"))
import features as F  # noqa: E402

T = pd.Timestamp("2011-03-01")


def raw_lines(rows):
    """rows: (invoice, customer, date, qty, price, stockcode, country)."""
    return pd.DataFrame([{"Invoice": inv, "StockCode": sc, "Description": "item", "Quantity": q,
                          "InvoiceDate": pd.Timestamp(d), "Price": p, "Customer ID": c, "Country": ctry}
                         for inv, c, d, q, p, sc, ctry in rows])


BASE_ROWS = [
    ("1001", 1.0, "2010-06-01 10:00", 2, 5.0, "A1", "United Kingdom"),
    ("1002", 1.0, "2011-01-15 10:00", 1, 10.0, "A2", "United Kingdom"),
    ("C1003", 1.0, "2011-01-20 10:00", -1, 10.0, "A2", "United Kingdom"),     # cancellation of the January line
    ("1004", 2.0, "2010-12-01 09:00", 5, 2.0, "B1", "France"),
    ("1005", 3.0, "2010-02-28 09:00", 1, 3.0, "B1", "Germany"),               # last order > 365 days before T
]


def build(rows):
    lines, _ = F.clean_lines(raw_lines(rows))
    return lines


def test_clean_lines_drops_missing_ids_bad_prices_and_fee_codes():
    rows = BASE_ROWS + [("2001", None, "2011-01-01", 1, 5.0, "A1", "United Kingdom"),
                        ("2002", 1.0, "2011-01-01", 1, 0.0, "A1", "United Kingdom"),
                        ("2003", 1.0, "2011-01-01", 1, 5.0, "POST", "United Kingdom"),
                        ("2004", 1.0, "2011-01-01", 1, 5.0, "gift_0001_10", "United Kingdom")]
    lines, steps = F.clean_lines(raw_lines(rows))
    assert len(lines) == len(BASE_ROWS)
    assert steps["removed"].tolist() == [0, 1, 1, 2]
    assert lines.loc[lines["Invoice"] == "C1003", "Revenue"].iloc[0] == -10.0


def test_snapshot_features_ignore_everything_after_the_snapshot_date():
    lines = build(BASE_ROWS)
    before = F.snapshot(lines, T)
    # a big order on T itself and one later, from a new country: features must not move, only the label can
    later = build(BASE_ROWS + [("3001", 2.0, "2011-03-01 08:00", 100, 50.0, "Z9", "United Kingdom"),
                               ("3002", 2.0, "2011-04-10 08:00", 100, 50.0, "Z9", "United Kingdom"),
                               ("3003", 2.0, "2011-04-11 08:00", 100, 50.0, "Z9", "United Kingdom")])
    after = F.snapshot(later, T)
    pd.testing.assert_frame_equal(before[F.FEATS], after[F.FEATS])
    assert before.loc[2, "churn"] == 1 and after.loc[2, "churn"] == 0
    assert after.loc[2, "is_uk"] == 0                       # country comes from history, not from the later UK orders


def test_churn_label_window_is_t_to_t_plus_90_days_half_open():
    orders = pd.DataFrame({"CustomerID": [1, 2, 3, 4], "Invoice": ["a", "b", "c", "d"],
                           "Day": [T, T + pd.Timedelta(days=89), T + pd.Timedelta(days=90), T - pd.Timedelta(days=1)]})
    assert F.churn_label(orders, [1, 2, 3, 4], T, 90).tolist() == [0, 0, 1, 1]
    assert F.churn_label(orders, [3], T, 120).tolist() == [0]


def test_snapshot_base_and_rfm_features():
    f = F.snapshot(build(BASE_ROWS), T)
    assert set(f.index) == {1, 2}                           # customer 3 has not ordered in the 365 days before T
    c1 = f.loc[1]
    assert c1["orders_365"] == 2 and c1["orders_all"] == 2  # the cancellation is not an order
    assert c1["spend_365"] == pytest.approx(10.0)          # 2*5 + 10 - 10: revenue is net of returns
    assert c1["cancel_lines_365"] == 1
    assert c1["recency_days"] == (T - pd.Timestamp("2011-01-15")).days
    assert c1["avg_gap_days"] == (pd.Timestamp("2011-01-15") - pd.Timestamp("2010-06-01")).days
    assert f.loc[2, "avg_gap_days"] == f.loc[2, "tenure_days"]   # one order: gap falls back to tenure


def test_monthly_flows_balance_the_customer_bridge():
    rng = np.random.default_rng(1)
    months = pd.date_range("2009-12-01", "2011-11-01", freq="MS")
    rows = [(c, m) for c in range(300) for m in months if rng.random() < 0.25]
    orders = pd.DataFrame(rows, columns=["CustomerID", "Month"])
    fl = F.monthly_flows(orders)
    prev_active = fl["Retained"] + fl["Churned"]             # last month's base = stayers + leavers
    assert (fl["Active"] == fl["Retained"] + fl["New"] + fl["Reactivated"]).all()
    assert (fl["Active"].shift(1).iloc[1:] == prev_active.iloc[1:]).all()


def test_policy_value_of_a_single_arm_policy_is_that_arms_mean():
    rng = np.random.default_rng(0)
    arm = rng.choice(F.ARMS, 3000)
    spend = rng.exponential(1.0, 3000) * (arm == "Mens offer") + rng.exponential(0.5, 3000)
    v = F.policy_value(arm, np.array(["Mens offer"] * 3000), spend)
    assert v == pytest.approx(spend[arm == "Mens offer"].mean())


def test_uplift_curve_finds_a_known_heterogeneous_effect():
    rng = np.random.default_rng(3)
    n = 20000
    x = rng.normal(size=n)
    treated = rng.random(n) < 0.5
    y = (rng.random(n) < 0.10 + 0.15 * treated * (x > 0)).astype(int)     # only x > 0 responds
    good, rand = F.uplift_curve(y, treated, x), F.uplift_curve(y, treated, rng.normal(size=n))
    assert F.auuc(good) == pytest.approx(0.01875, abs=0.004)   # analytic value: 0.075 x (0.125 + 0.125)
    assert abs(F.auuc(rand)) < 0.006
    assert good["incremental"].iloc[-1] == pytest.approx(rand["incremental"].iloc[-1])   # same total at 100%


def test_shift_to_rate_hits_the_target_mean_and_keeps_the_ranking():
    rng = np.random.default_rng(5)
    p = rng.beta(4, 2, 2000)                                 # mean about 0.67, like a spring-trained churn score
    q = F.shift_to_rate(p, 0.5)
    assert q.mean() == pytest.approx(0.5, abs=1e-6)
    assert (np.argsort(p) == np.argsort(q)).all()


def test_rfm_rule_scores_cells_by_their_training_churn_rate():
    rng = np.random.default_rng(2)
    n = 4000
    X = pd.DataFrame({"recency_days": rng.integers(1, 365, n), "orders_365": rng.integers(1, 20, n)})
    y = (rng.random(n) < np.where(X["recency_days"] > 180, 0.8, 0.2)).astype(int)
    p = F.RFMRule(prior_weight=0).fit(X, y).predict_proba(X)[:, 1]
    assert p[X["recency_days"] > 250].mean() > 0.7 and p[X["recency_days"] < 100].mean() < 0.3


def test_targeting_curve_is_zero_with_nobody_and_the_arm_effect_with_everybody():
    rng = np.random.default_rng(4)
    n = 9000
    arm = rng.choice(F.ARMS, n)
    y = rng.exponential(1.0, n) + (arm == "Mens offer") * 0.5
    offer = np.array(["Mens offer"] * n)
    c = F.targeting_curve(rng.random(n), offer, arm, y, [0.0, 1.0])
    assert c[0] == pytest.approx(0.0, abs=1e-9)
    assert c[1] == pytest.approx((y[arm == "Mens offer"].mean() - y[arm == "Control"].mean()) * n)
