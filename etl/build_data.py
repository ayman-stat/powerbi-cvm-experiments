"""ETL + ML for the CVM Experiments report. Run with the project venv:  ..\\.venv\\Scripts\\python etl\\build_data.py

Two real public retail datasets (no single public dataset has both lifecycle history and a randomised test):
  1. UCI Online Retail II (CC BY 4.0) - every transaction of a UK online gift retailer, Dec 2009 - Dec 2011.
     -> customer-month lifecycle flows (new / retained / reactivated / churned), cohorts, and an out-of-time churn model.
  2. MineThatData E-Mail Analytics challenge (Kevin Hillstrom, 2008) - 64,000 customers randomised into a
     Mens e-mail, a Womens e-mail and a no-e-mail control; two-week visit, conversion and spend.
     -> A/B readout, segment uplift, a cross-fitted T-learner uplift model and next-best-offer policy evaluation.
"""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW, OUT, ML = os.path.join(ROOT, ".raw"), os.path.join(ROOT, "data"), os.path.join(ROOT, "ml")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ML, exist_ok=True)
SEED = 42
metrics = {}


def save(df, name, fmt="%.6g"):
    df.to_csv(os.path.join(OUT, name), index=False, float_format=fmt)
    print(f"  {name:32s} {len(df):>8,} rows")


# =============================================================================================== 1. Online Retail II
def load_retail():
    cache = os.path.join(RAW, "onlineretail", "lines.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    xl = os.path.join(RAW, "onlineretail", "online_retail_II.xlsx")
    a = pd.read_excel(xl, sheet_name="Year 2009-2010", dtype={"Invoice": str, "StockCode": str})
    b = pd.read_excel(xl, sheet_name="Year 2010-2011", dtype={"Invoice": str, "StockCode": str})
    # the two sheets overlap on 1-9 Dec 2010: keep that week from the second sheet only
    a = a[a["InvoiceDate"] < "2010-12-01"]
    df = pd.concat([a, b], ignore_index=True)
    pickle.dump(df, open(cache, "wb"))
    return df


print("Online Retail II")
lines = load_retail()
n_raw = len(lines)
lines = lines.rename(columns={"Customer ID": "CustomerID"})
lines = lines[lines["CustomerID"].notna() & (lines["Price"] > 0)]
NON_PRODUCT = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001", "TEST002", "D", "CRUK", "S", "B", "PADS"}
lines = lines[~lines["StockCode"].astype(str).str.upper().isin(NON_PRODUCT) & ~lines["StockCode"].astype(str).str.startswith("gift_")]
lines["CustomerID"] = lines["CustomerID"].astype(int)
lines["IsCancel"] = lines["Invoice"].str.startswith("C")
lines["Revenue"] = lines["Quantity"] * lines["Price"]          # cancellations carry negative quantity -> net revenue
lines["Month"] = lines["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
lines["Day"] = lines["InvoiceDate"].dt.normalize()
metrics["retail_raw_lines"] = n_raw
metrics["retail_clean_lines"] = len(lines)
assert lines["InvoiceDate"].min() >= pd.Timestamp("2009-12-01") and lines["InvoiceDate"].max() < pd.Timestamp("2011-12-10")

orders = lines[~lines["IsCancel"]].groupby(["CustomerID", "Invoice"]).agg(Day=("Day", "min"), Month=("Month", "min")).reset_index()
cm = (lines.groupby(["CustomerID", "Month"]).agg(Revenue=("Revenue", "sum")).reset_index()
      .merge(orders.groupby(["CustomerID", "Month"]).size().rename("Orders").reset_index(), how="left").fillna({"Orders": 0}))
cm = cm[(cm["Orders"] > 0) | (cm["Revenue"] != 0)]
first = orders.groupby("CustomerID")["Month"].min().rename("FirstMonth")
country = lines.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0])

# ---- lifecycle flows on a rolling-3-month active definition, reported Mar 2010 - Nov 2011 (Dec 2011 is a partial month)
months = pd.period_range("2009-12", "2011-11", freq="M").to_timestamp()
REPORT = months[months >= "2010-03-01"]
bought = orders.groupby(["CustomerID", "Month"]).size().unstack(fill_value=0).reindex(columns=months, fill_value=0) > 0
active = bought.T.rolling(3, min_periods=1).max().T.astype(bool)             # purchased in m-2..m
prev_active = active.shift(1, axis=1, fill_value=False)
rows = []
for cid in bought.index:
    fm = first[cid]
    for m in REPORT:
        if m < fm:
            continue
        a, pa, bt = active.at[cid, m], prev_active.at[cid, m], bought.at[cid, m]
        if m == fm:
            flow = "New"
        elif a and pa:
            flow = "Retained"
        elif a and not pa:
            flow = "Reactivated"
        elif pa and not a:
            flow = "Churned"
        else:
            continue                                                       # dormant: outside the base, no row
        rows.append((cid, m, flow, int(a), int(bt)))
fcm = pd.DataFrame(rows, columns=["CustomerKey", "MonthStart", "Flow", "IsActive", "Purchased"])
fcm = fcm.merge(cm.rename(columns={"CustomerID": "CustomerKey", "Month": "MonthStart"}), how="left").fillna({"Revenue": 0, "Orders": 0})
fcm["MonthsSinceFirst"] = ((fcm["MonthStart"].dt.year - fcm["CustomerKey"].map(first).dt.year) * 12
                           + fcm["MonthStart"].dt.month - fcm["CustomerKey"].map(first).dt.month)
fcm["Revenue"] = fcm["Revenue"].round(2)
fcm["Orders"] = fcm["Orders"].astype(int)

# cohort grid: every purchase month for every customer (classic cohort retention), cohorts Dec 2009 - Nov 2011
coh = cm[cm["Orders"] > 0].rename(columns={"CustomerID": "CustomerKey", "Month": "MonthStart"})
coh = coh[coh["MonthStart"] <= "2011-11-01"].copy()
coh["CohortMonth"] = coh["CustomerKey"].map(first)
coh["MonthIndex"] = (coh["MonthStart"].dt.year - coh["CohortMonth"].dt.year) * 12 + coh["MonthStart"].dt.month - coh["CohortMonth"].dt.month
coh = coh[["CustomerKey", "MonthStart", "CohortMonth", "MonthIndex", "Revenue"]]

# ---- churn model: features at a snapshot, label = no purchase in the next 90 days
def snapshot(t):
    t = pd.Timestamp(t)
    hist = lines[lines["Day"] < t]
    o = orders[orders["Day"] < t]
    o365 = o[o["Day"] >= t - pd.Timedelta(days=365)]
    base = o365["CustomerID"].unique()
    g = o.groupby("CustomerID")
    f = pd.DataFrame(index=pd.Index(base, name="CustomerID"))
    f["recency_days"] = (t - g["Day"].max()).dt.days.reindex(base)
    f["tenure_days"] = (t - g["Day"].min()).dt.days.reindex(base)
    f["orders_365"] = o365.groupby("CustomerID").size().reindex(base).fillna(0)
    f["orders_90"] = o[o["Day"] >= t - pd.Timedelta(days=90)].groupby("CustomerID").size().reindex(base).fillna(0)
    f["orders_all"] = g.size().reindex(base)
    h365 = hist[hist["Day"] >= t - pd.Timedelta(days=365)]
    f["spend_365"] = h365.groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    f["spend_90"] = hist[hist["Day"] >= t - pd.Timedelta(days=90)].groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    f["spend_prev_90"] = hist[(hist["Day"] >= t - pd.Timedelta(days=180)) & (hist["Day"] < t - pd.Timedelta(days=90))].groupby("CustomerID")["Revenue"].sum().reindex(base).fillna(0)
    f["avg_basket"] = f["spend_365"] / f["orders_365"].clip(lower=1)
    f["products_365"] = h365.groupby("CustomerID")["StockCode"].nunique().reindex(base).fillna(0)
    f["cancel_lines_365"] = h365[h365["IsCancel"]].groupby("CustomerID").size().reindex(base).fillna(0)
    gaps = o.sort_values("Day").groupby("CustomerID")["Day"].apply(lambda s: s.diff().dt.days.mean())
    f["avg_gap_days"] = gaps.reindex(base).fillna(f["tenure_days"])
    f["is_uk"] = (country.reindex(base) == "United Kingdom").astype(int).values
    nxt = orders[(orders["Day"] >= t) & (orders["Day"] < t + pd.Timedelta(days=90))]["CustomerID"].unique()
    f["churn"] = (~f.index.isin(nxt)).astype(int)
    return f


FEATS = ["recency_days", "tenure_days", "orders_365", "orders_90", "orders_all", "spend_365", "spend_90", "spend_prev_90",
         "avg_basket", "products_365", "cancel_lines_365", "avg_gap_days", "is_uk"]
s_train, s_valid, s_test = snapshot("2011-03-01"), snapshot("2011-06-01"), snapshot("2011-09-01")
assert pd.Timestamp("2011-09-01") + pd.Timedelta(days=90) <= lines["Day"].max() + pd.Timedelta(days=1)
params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
clf = HistGradientBoostingClassifier(**params).fit(s_train[FEATS], s_train["churn"])
metrics["churn_auc_valid"] = round(roc_auc_score(s_valid["churn"], clf.predict_proba(s_valid[FEATS])[:, 1]), 4)
clf = HistGradientBoostingClassifier(**params).fit(pd.concat([s_train, s_valid])[FEATS], pd.concat([s_train, s_valid])["churn"])
p_test = clf.predict_proba(s_test[FEATS])[:, 1]
metrics["churn_auc_test"] = round(roc_auc_score(s_test["churn"], p_test), 4)
metrics["churn_auc_recency_only_test"] = round(roc_auc_score(s_test["churn"], s_test["recency_days"]), 4)
metrics["churn_base_rate_test"] = round(s_test["churn"].mean(), 4)
print(f"  churn AUC valid {metrics['churn_auc_valid']}  test {metrics['churn_auc_test']}  (recency-only {metrics['churn_auc_recency_only_test']})")

risk = s_test.copy()
risk["ChurnScore"] = p_test
risk["RiskDecile"] = 10 - pd.qcut(risk["ChurnScore"].rank(method="first"), 10, labels=False)          # 1 = highest risk
risk["RiskBand"] = pd.cut(risk["ChurnScore"], [0, .3, .6, 1.0001], labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
v = risk["spend_365"]
risk["ValueTier"] = np.where(v >= v.quantile(.8), "Top 20%", np.where(v >= v.quantile(.5), "Middle 30%", "Bottom 50%"))
r_s = pd.qcut(-risk["recency_days"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
f_s = pd.qcut(risk["orders_365"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm(r, f):
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


risk["RFMSegment"] = [rfm(a, b_) for a, b_ in zip(r_s, f_s)]
dec = risk.groupby("RiskDecile").agg(n=("churn", "size"), churn=("churn", "mean"), score=("ChurnScore", "mean"))
metrics["churn_top_decile_rate"] = round(dec.loc[1, "churn"], 4)
metrics["churn_top_decile_lift"] = round(dec.loc[1, "churn"] / risk["churn"].mean(), 2)
top3 = risk[risk["RiskDecile"] <= 3]
metrics["churn_top30_capture"] = round(top3["churn"].sum() / risk["churn"].sum(), 4)
churn_scores = risk.reset_index().rename(columns={"CustomerID": "CustomerKey", "churn": "ActualChurn", "spend_365": "Value365",
                                                  "recency_days": "RecencyDays", "orders_365": "Orders365"})
churn_scores = churn_scores[["CustomerKey", "ChurnScore", "RiskDecile", "RiskBand", "ActualChurn", "Value365", "RecencyDays", "Orders365",
                             "ValueTier", "RFMSegment"]]
churn_scores["ChurnScore"] = churn_scores["ChurnScore"].round(4)
churn_scores["Value365"] = churn_scores["Value365"].round(2)
imp_rows = []
from sklearn.inspection import permutation_importance  # noqa: E402
pi = permutation_importance(clf, s_test[FEATS], s_test["churn"], scoring="roc_auc", n_repeats=5, random_state=SEED)
for fname, val in sorted(zip(FEATS, pi.importances_mean), key=lambda z: -z[1]):
    imp_rows.append((fname, round(val, 4)))
FEAT_LABEL = {"recency_days": "Days since last order", "tenure_days": "Customer tenure", "orders_365": "Orders, last 12 months",
              "orders_90": "Orders, last 90 days", "orders_all": "Orders, all time", "spend_365": "Spend, last 12 months",
              "spend_90": "Spend, last 90 days", "spend_prev_90": "Spend, prior 90 days", "avg_basket": "Average basket",
              "products_365": "Distinct products", "cancel_lines_365": "Cancelled lines", "avg_gap_days": "Average days between orders",
              "is_uk": "UK customer"}
importance = pd.DataFrame([(FEAT_LABEL[a], b_) for a, b_ in imp_rows], columns=["Feature", "AUCDrop"])

customers = pd.DataFrame({"CustomerKey": first.index, "FirstMonth": first.values})
customers["Country"] = customers["CustomerKey"].map(country)
customers["Market"] = np.where(customers["Country"] == "United Kingdom", "United Kingdom",
                               np.where(customers["Country"].isin(["Germany", "France", "EIRE", "Netherlands", "Spain", "Belgium", "Switzerland",
                                                                    "Portugal", "Italy", "Sweden", "Denmark", "Norway", "Finland", "Austria",
                                                                    "Poland", "Cyprus", "Greece", "Channel Islands", "Iceland", "Lithuania",
                                                                    "Malta", "Czech Republic", "European Community"]), "Rest of Europe", "Rest of world"))
customers["CohortYear"] = customers["FirstMonth"].dt.year.astype(str)
customers["CohortLabel"] = customers["FirstMonth"].dt.strftime("%b %y")
customers["CohortSort"] = customers["FirstMonth"].dt.year * 100 + customers["FirstMonth"].dt.month

# =============================================================================================== 2. Hillstrom e-mail test
print("Hillstrom e-mail test")
h = pd.read_csv(os.path.join(RAW, "hillstrom", "hillstrom.csv"))
assert len(h) == 64000 and set(h["segment"]) == {"Mens E-Mail", "Womens E-Mail", "No E-Mail"}
h.insert(0, "CustomerKey", np.arange(1, len(h) + 1))
h["Arm"] = h["segment"].map({"Mens E-Mail": "Mens offer", "Womens E-Mail": "Womens offer", "No E-Mail": "Control"})
h["ZipCode"] = h["zip_code"].replace({"Surburban": "Suburban"})
h["Purchased"] = np.select([(h["mens"] == 1) & (h["womens"] == 1), h["mens"] == 1, h["womens"] == 1], ["Both", "Mens only", "Womens only"], "Neither")
h["Customer type"] = np.where(h["newbie"] == 1, "New (last 12 months)", "Existing")
h["RecencyBand"] = pd.cut(h["recency"], [0, 2, 4, 6, 9, 12], labels=["1-2 months", "3-4 months", "5-6 months", "7-9 months", "10-12 months"]).astype(str)
h["HistoryBand"] = h["history_segment"].str.slice(3)
X = pd.get_dummies(h[["recency", "history", "mens", "womens", "newbie", "ZipCode", "channel"]], columns=["ZipCode", "channel"], dtype=int)
X["log_history"] = np.log1p(h["history"])
X["mens_and_womens"] = h["mens"] * h["womens"]
# Uplift target = visit (15% base rate). Spend is too sparse (1% convert) to rank individuals reliably - tested
# gradient boosting and logistic T-learners on spend, conversion and visit; only visit uplift ranks out of sample.
# Revenue is then measured at policy level, where randomisation makes the estimate unbiased.
y = h["visit"].values
pred = {k: np.zeros(len(h)) for k in ["Control", "Mens offer", "Womens offer"]}
kf = KFold(5, shuffle=True, random_state=7)
for tr, te in kf.split(X):                                   # cross-fitting: every customer is scored by models that never saw them
    for arm in pred:
        idx = tr[h["Arm"].values[tr] == arm]
        m = LogisticRegression(max_iter=3000, C=0.5).fit(X.iloc[idx], y[idx])
        pred[arm][te] = m.predict_proba(X.iloc[te])[:, 1]
h["UpliftMens"] = pred["Mens offer"] - pred["Control"]
h["UpliftWomens"] = pred["Womens offer"] - pred["Control"]
h["BestUplift"] = h[["UpliftMens", "UpliftWomens"]].max(axis=1)
h["ModelOffer"] = np.where(h["UpliftMens"] >= h["UpliftWomens"], "Mens offer", "Womens offer")
h["UpliftDecile"] = 10 - pd.qcut(h["BestUplift"].rank(method="first"), 10, labels=False)          # 1 = most persuadable
P = 1 / 3                                                     # randomisation probability of each arm


def policy_value(assign):
    """Inverse-propensity estimate of spend per customer under an assignment (unbiased because arms were randomised)."""
    w = (h["Arm"].values == assign) / P
    return float(np.sum(w * h["spend"].values) / np.sum(w)), w            # self-normalised (Hajek) estimator


base_spend = h.loc[h["Arm"] == "Control", "spend"].mean()
rng = np.random.default_rng(SEED)


def boot_ci(assign, n=400):
    vals = []
    a = np.asarray(assign)
    arm, sp = h["Arm"].values, h["spend"].values
    for _ in range(n):
        ix = rng.integers(0, len(h), len(h))
        wa, wc = (arm[ix] == a[ix]), (arm[ix] == "Control")
        v = np.sum(wa * sp[ix]) / wa.sum() - np.sum(wc * sp[ix]) / wc.sum()
        vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


# segment rule from the experiment itself: best offer per purchase history (what a CVM team would deploy first)
seg_rule = h.groupby(["Purchased"]).apply(lambda g: pd.Series({a: g.loc[g["Arm"] == a, "spend"].mean() for a in ["Control", "Mens offer", "Womens offer"]}),
                                          include_groups=False)
seg_best = seg_rule[["Mens offer", "Womens offer"]].idxmax(axis=1)
policies = {
    "No e-mail": np.array(["Control"] * len(h)),
    "All: Womens offer": np.array(["Womens offer"] * len(h)),
    "All: Mens offer": np.array(["Mens offer"] * len(h)),
    "Rule: by purchase history": h["Purchased"].map(seg_best).values,
    "Model: next-best offer": h["ModelOffer"].values,
}
pol_rows = []
for k, (name, a) in enumerate(policies.items()):
    v, _ = policy_value(a)
    lo, hi = boot_ci(a) if name != "No e-mail" else (0.0, 0.0)
    pol_rows.append((name, k + 1, round(v, 4), round(v - base_spend, 4), round(lo, 4), round(hi, 4), 0 if k == 0 else 1000))
policy = pd.DataFrame(pol_rows, columns=["Policy", "PolicyOrder", "SpendPerCustomer", "IncrementalPerCustomer", "CILow", "CIHigh", "ContactsPer1000"])
policy[["IncPer1000", "CILowPer1000", "CIHighPer1000"]] = policy[["IncrementalPerCustomer", "CILow", "CIHigh"]] * 1000

# targeting curve: e-mail only the top-k% by predicted uplift (model offer), control for the rest
order = np.argsort(-h["BestUplift"].values)
curve = [(0, 0, 0.0, 0.0, 0.0, 0.0)]
rand_order = np.random.default_rng(SEED).permutation(len(h))
for share in np.round(np.arange(0.05, 1.0001, 0.05), 2):
    k = int(round(share * len(h)))
    assign = np.array(["Control"] * len(h), dtype=object)
    assign[order[:k]] = h["ModelOffer"].values[order[:k]]
    rnd = np.array(["Control"] * len(h), dtype=object)
    rnd[rand_order[:k]] = "Mens offer"
    arm_ = h["Arm"].values
    inc = lambda a, col: (np.sum((arm_ == a) * h[col].values) / np.sum(arm_ == a)  # noqa: E731
                          - np.sum((arm_ == "Control") * h[col].values) / np.sum(arm_ == "Control")) * len(h)
    curve.append((int(round(share * 100)), k, round(inc(assign, "spend"), 2), round(inc(rnd, "spend"), 2),
                  round(inc(assign, "visit"), 1), round(inc(rnd, "visit"), 1)))
curve = pd.DataFrame(curve, columns=["TargetPct", "Contacts", "ModelIncRevenue", "RandomIncRevenue", "ModelIncVisits", "RandomIncVisits"])
metrics["uplift_inc_at_30pct"] = float(curve.loc[curve["TargetPct"] == 30, "ModelIncRevenue"].iat[0])
metrics["uplift_inc_at_100pct"] = float(curve.loc[curve["TargetPct"] == 100, "ModelIncRevenue"].iat[0])
# decile check on held-out predictions: observed uplift per predicted decile
def _dec(g):
    t_ = g[g["Arm"].values == g["ModelOffer"].values]
    c_ = g[g["Arm"] == "Control"]
    return pd.Series({"Customers": len(g), "PredictedVisitUplift": g["BestUplift"].mean(),
                      "ObservedVisitUplift": t_["visit"].mean() - c_["visit"].mean(),
                      "ObservedSpendUplift": t_["spend"].mean() - c_["spend"].mean()})


dchk = h.groupby("UpliftDecile").apply(_dec, include_groups=False).reset_index()
metrics["uplift_top_decile_visit_pp"] = round(dchk.loc[0, "ObservedVisitUplift"] * 100, 1)
metrics["uplift_avg_visit_pp"] = round((h.loc[h["Arm"] != "Control", "visit"].mean() - h.loc[h["Arm"] == "Control", "visit"].mean()) * 100, 1)
print(dchk.round(3).to_string(index=False))
print(policy[["Policy", "IncPer1000", "CILowPer1000", "CIHighPer1000"]].round(1).to_string(index=False))
print(curve.iloc[[2, 4, 6, 10, 20]].to_string(index=False))

hill = h[["CustomerKey", "Arm", "recency", "RecencyBand", "HistoryBand", "history", "Purchased", "ZipCode", "Customer type", "channel",
          "visit", "conversion", "spend", "UpliftMens", "UpliftWomens", "BestUplift", "ModelOffer", "UpliftDecile"]].rename(columns={
    "recency": "RecencyMonths", "history": "History12m", "channel": "Channel", "visit": "Visited", "conversion": "Converted", "spend": "Spend",
    "Customer type": "CustomerType"})
for c in ["UpliftMens", "UpliftWomens", "BestUplift"]:
    hill[c] = hill[c].round(4)

# =============================================================================================== write
print("Writing")
save(customers.rename(columns={"FirstMonth": "CohortMonth"}), "dim_retail_customer.csv")
save(fcm, "fact_customer_month.csv")
save(coh, "fact_cohort_activity.csv")
save(pd.DataFrame({"Flow": ["Opening base", "New", "Reactivated", "Churned", "Retained"], "FlowOrder": [0, 1, 2, 3, 4],
                   "Direction": ["Base", "Gain", "Gain", "Loss", "Base"]}), "dim_flow.csv")
save(churn_scores, "fact_churn_scores.csv")
save(importance, "churn_feature_importance.csv")
save(hill, "fact_email_test.csv")
save(policy, "uplift_policies.csv")
save(curve, "uplift_targeting_curve.csv")
save(dchk.round(4), "uplift_decile_check.csv")
save(pd.DataFrame({"Arm": ["Mens offer", "Womens offer"], "ArmOrder": [1, 2]}), "dim_test_arm.csv")
save(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]), "model_metrics.csv", fmt="%.10g")
json.dump(metrics, open(os.path.join(ML, "metrics.json"), "w"), indent=2)
print(json.dumps(metrics, indent=2))
