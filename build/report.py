"""Report definition (PBIR) for the CVM Experiments report.

Identity: COMMAND - a dark CVM command centre. Violet = treatment / the data in focus, slate = control / context,
mint = gains and incremental value, rose = losses. Pill-tab navigation, KPI tiles with sparklines and a left accent.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import Report, write_pbip, C, M, proj, b, i, n, s, props, idprops, color, lit  # noqa: E402
import design as D  # noqa: E402
from design import (S, GUT, use, theme, chrome, kpi_band, bar, combo, line, matrix, table, stacked_column, button_slicer, dropdown,  # noqa: E402
                    in_filter, about_page, P, _title, fill_for)

use(D.COMMAND)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "CVMExperiments"
TITLE = "CVM Command Centre"
SRC_R = "UCI Online Retail II (Chen, 2019), CC BY 4.0: every transaction of a UK online gift retailer, Dec 2009 - Dec 2011."
SRC_H = "MineThatData E-Mail Analytics challenge (Kevin Hillstrom, 2008): 64,000 customers randomised to two e-mail offers or a control."
K = lambda m: M("KPI", m)  # noqa: E731
YM = (C("Month", "Year Month"), "Month")
YMS = C("Month", "Year Month Sort")
TREND = C("Month", "Date")
UPL = C("Uplift by", "Uplift by")
NP = 7


def period_slicers(p, market=True):
    button_slicer(p, "period", S.slicer_x, S.title_y + 6, 300 if market else 480, 76, C("Month", "Period"), "Reporting year", selected=["Dec 10 - Nov 11"],
                  columns=2, single=True)
    if market:
        dropdown(p, "market", S.slicer_x + 320, S.title_y + 6, 160, 76, C("Customer", "Market"), "Market")


def single_buttons(p, key, x, y, w, h, field, value_literal, cols, rows, title=None):
    ent, prop = field["Column"]["Expression"]["SourceRef"]["Entity"], field["Column"]["Property"]
    filt_prop = prop + " Fields" if ent == "Uplift by" else prop
    objs = {"layout": props(columnCount=i(cols), rowCount=i(rows)),
            "value": idprops(fontSize=n(11), fontFamily=s(S.font_b), fontColor=color(S.ink)),
            "fillCustom": [{"properties": {"show": b(True), "fillColor": color(S.panel)}, "selector": {"id": "default"}},
                           {"properties": {"show": b(True), "fillColor": color(S.sel_fill)}, "selector": {"id": "selected"}}],
            "outline": [{"properties": {"show": b(True), "lineColor": color(S.line), "weight": n(1)}, "selector": {"id": "default"}},
                        {"properties": {"show": b(True), "lineColor": color(S.primary), "weight": n(2)}, "selector": {"id": "selected"}}],
            "shapeCustomRectangle": props(tileShape=s("rectangleRounded"), rectangleRoundedCurve=i(8)),
            "selection": props(singleSelect=b(True)),
            "general": [{"properties": {"filter": {"filter": {"Version": 2, "From": [{"Name": "f", "Entity": ent, "Type": 0}],
                "Where": [{"Condition": {"In": {"Expressions": [{"Column": {"Expression": {"SourceRef": {"Source": "f"}}, "Property": filt_prop}}],
                                                "Values": [[{"Literal": {"Value": value_literal}}]]}}}]}}}}]}
    vco = dict(D.NOBOX)
    if title:
        vco["title"] = props(show=b(True), text=s(title), fontSize=n(10), fontColor=color(S.muted), fontFamily=s(S.font))
    return p.add(key, "advancedSlicerVisual", x, y, w, h, query={"Values": [proj(field)]}, objects=objs, vco=vco, alt=title or "Selector")


def build():
    r = Report(os.path.join(ROOT, f"{NAME}.Report"), NAME, f"{NAME}.SemanticModel", theme())
    r.n_pages = NP
    x0, cw, by, bh = S.x0, S.cw, S.body_y, S.body_h
    L = 620
    R = cw - L - GUT

    # ------------------------------------------------------------------ 1 Base health
    p = r.page("base", "Base health", bg=S.page)
    chrome(p, TITLE, "The base fell 42% over the winter, then rebuilt itself on reactivations",
           "Active base = customers who bought in the last 3 months  ·  Dec 2010 - Nov 2011: 4,481 churned, 2,864 reactivated, 1,585 new", f"1 / {NP}", SRC_R)
    period_slicers(p)
    kpi_band(p, "kpi", [(K("Active Base"), "Active base, end of period"), (K("New Customers"), "New customers"), (K("Reactivated"), "Reactivated"),
                        (K("Churned"), "Churned"), (K("Monthly Churn Rate"), "Avg monthly churn rate"), (K("Revenue"), "Net revenue")])
    p.add("bridge", "waterfallChart", x0, by, L, bh,
          query={"Category": [P((C("Flow", "Flow"), "Flow"))], "Y": [P((K("Bridge Value"), "Customers"))]},
          objects={"sentimentColors": props(increaseFill=color(S.accent), decreaseFill=color(S.neg), totalFill=color(S.primary)),
                   "labels": props(show=b(True), color=color(S.ink), fontSize=n(11)),
                   "categoryAxis": props(fontSize=n(11), showAxisTitle=b(False)), "valueAxis": props(show=b(False), showAxisTitle=b(False)),
                   "legend": props(show=b(False))},
          vco=_title("Customer bridge for the period", "Opening base + new + reactivated - churned = closing base"),
          sort=[(C("Flow", "FlowOrder"), "Ascending")], alt="Customer bridge waterfall",
          filters=[in_filter("Flow", "Flow", ["Opening base", "New", "Reactivated", "Churned"], "bridge")])
    top = 330
    stacked_column(p, "flows", x0 + L + GUT, by, R, top, YM,
                   [(K("New Customers"), "New"), (K("Reactivated"), "Reactivated"), (K("Churned (negative)"), "Churned")],
                   "Gains and losses every month", "Mint: new  ·  Violet: reactivated  ·  Rose: churned (below zero)",
                   colors=(S.accent, S.primary, S.neg), sort_cat=YMS)
    p.visuals[-1]["visual"]["objects"]["valueAxis"] = props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted), fontSize=n(10))
    w1 = (R - GUT) * 3 // 5
    combo(p, "base", x0 + L + GUT, by + top + GUT, w1, bh - top - GUT, YM, [(K("Active Base"), "Active base")], [(K("Monthly Churn Rate"), "Churn rate")],
          "Active base and monthly churn rate", "Columns: active base  ·  Line: churned / opening base", col_colors=("#3B3470",), line_colors=(S.neg,), sort_cat=YMS)
    bar(p, "mkt", x0 + L + GUT + w1 + GUT, by + top + GUT, R - w1 - GUT, bh - top - GUT, (C("Customer", "Market"), "Market"),
        (K("Monthly Churn Rate"), "Monthly churn rate"), "Churn by market", "Average monthly churn rate", gradient=("#3A2440", S.neg))

    # ------------------------------------------------------------------ 2 Retention & cohorts
    p = r.page("cohorts", "Retention & cohorts", bg=S.page)
    p.filters.append(in_filter("Customer", "CohortSort", list(range(201001, 201013)), "cohort-page"))
    chrome(p, TITLE, "Only 1 in 5 new customers buys again the next month; 18% are still buying a year later",
           "2010 acquisition cohorts (Jan - Dec), purchase activity by month since first order  ·  Dec 2009 excluded (customers acquired before the data starts)",
           f"2 / {NP}", SRC_R)
    dropdown(p, "market", S.slicer_x + 320, S.title_y + 6, 160, 76, C("Customer", "Market"), "Market")
    kpi_band(p, "kpi", [K("Cohort Size"), K("Month 1 Retention"), K("Month 3 Retention"), K("Month 6 Retention"), K("Month 12 Retention"),
                        K("12-Month Value per Customer")])
    hm_w = 1150
    matrix(p, "heat", x0, by, hm_w, bh, [(C("Customer", "CohortLabel"), "Cohort")], [(K("Retention"), "Retention")],
           "Cohort retention: share of each month's new customers buying N months later",
           "Rows: month of first order  ·  Columns: months since first order  ·  Brighter = higher retention",
           columns=[(C("Cohort Activity", "MonthIndex"), "Month")], gradients=[(K("Retention"), S.panel, S.primary)], font=11,
           filters=[in_filter("Cohort Activity", "MonthIndex", list(range(1, 13)), "heat")])
    rw = cw - hm_w - GUT
    hh = (bh - GUT) // 2
    p.add("curve", "lineChart", x0 + hm_w + GUT, by, rw, hh,
          query={"Category": [P((C("Cohort Activity", "MonthIndex"), "Months since first order"))], "Y": [P((K("Retention"), "Retention"))],
                 "Series": [P((C("Customer", "Market"), "Market"))]},
          objects={"lineStyles": props(showMarker=b(True), strokeWidth=n(2.5))},
          vco=_title("Retention curve by market", "Average of the 2010 cohorts"), sort=[(C("Cohort Activity", "MonthIndex"), "Ascending")],
          alt="Retention curve by market", filters=[in_filter("Cohort Activity", "MonthIndex", list(range(1, 13)), "curve")])
    bar(p, "ltv", x0 + hm_w + GUT, by + hh + GUT, rw, hh, (C("Customer", "CohortLabel"), "Cohort"), (K("12-Month Value per Customer"), "12-month value"),
        "First-year value per new customer", "Net revenue in the first 12 months, GBP", horizontal=False, sort_desc=False,
        sort_cat=C("Customer", "CohortSort"))

    # ------------------------------------------------------------------ 3 Churn risk model
    p = r.page("churn", "Churn risk model", bg=S.page)
    chrome(p, TITLE, "The model's top decile churned at 77%; the top 30% holds 44% of all churners",
           "Gradient boosting on 13 behavioural features  ·  trained on Mar and Jun 2011 snapshots, scored 1 Sep 2011, judged on what happened by 30 Nov",
           f"3 / {NP}", SRC_R)
    dropdown(p, "seg", S.slicer_x + 320, S.title_y + 6, 160, 76, C("Churn Scores", "RFMSegment"), "RFM segment")
    kpi_band(p, "kpi", [K("Customers Scored"), K("Actual Churn Rate"), K("Test AUC"), K("Top Decile Churn"), K("Churn Capture"), K("Value at Risk")])
    lw = 900
    th = 380
    p.add("calib", "clusteredColumnChart", x0, by, lw, th,
          query={"Category": [P((C("Churn Scores", "RiskDecile"), "Risk decile (1 = highest)"))],
                 "Y": [P((K("Actual Churn Rate"), "Actual churn")), P((K("Predicted Churn"), "Predicted"))]},
          objects={"dataPoint": [fill_for(K("Actual Churn Rate"), S.primary), fill_for(K("Predicted Churn"), "#4B5876")],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False)),
                   "valueAxis": props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted))},
          vco=_title("Calibration: predicted vs actual churn by risk decile", "Out-of-time test, one axis  ·  violet: churned by 30 Nov  ·  slate: mean predicted probability"),
          sort=[(C("Churn Scores", "RiskDecile"), "Ascending")], alt="Calibration by risk decile")
    bar(p, "drivers", x0 + lw + GUT, by, cw - lw - GUT, th, (C("Churn Drivers", "Feature"), "Feature"), (K("Driver Importance"), "AUC drop"),
        "What drives the score", "Permutation importance: drop in test AUC when the feature is shuffled")
    bh2 = bh - th - GUT
    matrix(p, "save", x0, by + th + GUT, lw, bh2, [(C("Churn Scores", "ValueTier"), "Value tier")], [(K("Value at Risk"), "Value at risk")],
           "Where the retention budget goes: value at risk", "Rows: last-12-month spend tier  ·  Columns: risk band  ·  Brighter = more revenue at risk",
           columns=[(C("Churn Scores", "RiskBand"), "Risk band")], gradients=[(K("Value at Risk"), S.panel, S.primary)], font=11)
    matrix(p, "rfm", x0 + lw + GUT, by + th + GUT, cw - lw - GUT, bh2, [(C("Churn Scores", "RFMSegment"), "RFM segment")],
           [(K("Customers Scored"), "Customers"), (K("Actual Churn Rate"), "Actual churn"), (K("Predicted Churn"), "Predicted"),
            (K("Value at Risk"), "Value at risk")],
           "Churn by RFM segment", "Data bars: value at risk  ·  churn shaded mint (low) to rose (high)",
           gradients=[(K("Actual Churn Rate"), S.soft_pos, S.soft_neg)], databars=[K("Value at Risk")], sort=[(K("Value at Risk"), "Descending")])

    # ------------------------------------------------------------------ 4 A/B test readout
    p = r.page("abtest", "A/B test readout", bg=S.page)
    chrome(p, TITLE, "The Mens e-mail lifted spend by $0.77 a customer: significant, and worth $16K in two weeks",
           "64,000 customers randomised 1/3 each  ·  outcomes over two weeks  ·  pick the arm to compare with the no-e-mail control", f"4 / {NP}", SRC_H)
    single_buttons(p, "arm", S.slicer_x, S.title_y + 6, 300, 76, C("Test Arm", "Arm"), "'Mens offer'", 2, 1, "Treatment arm")
    dropdown(p, "chan", S.slicer_x + 320, S.title_y + 6, 160, 76, C("Email Test", "Channel"), "Channel")
    kpi_band(p, "kpi", [(K("Visit Rate Uplift"), "Visit uplift (points)"), (K("Conversion Rate Uplift"), "Conversion uplift (points)"),
                        (K("Spend per Customer Uplift"), "Spend uplift per customer"), (K("Incremental Revenue"), "Incremental revenue"),
                        (K("Spend p-value Label"), "p-value (spend)"), (K("T Customers"), "Customers treated")])
    tw = (cw - 2 * GUT) // 3
    th = 330
    for k_, (lab, meas_t, meas_c, sub) in enumerate([("Visit rate", "T Visit Rate", "C Visit Rate", "Visited the site within two weeks"),
                                                     ("Conversion rate", "T Conversion Rate", "C Conversion Rate", "Bought within two weeks"),
                                                     ("Spend per customer", "T Spend per Customer", "C Spend per Customer", "Two-week spend, USD, all customers")]):
        p.add(f"tc{k_}", "clusteredColumnChart", x0 + k_ * (tw + GUT), by, tw, th,
              query={"Category": [P((C("Test Arm", "Arm"), "Arm"))], "Y": [P((K(meas_t), "Treatment")), P((K(meas_c), "Control"))]},
              objects={"dataPoint": [fill_for(K(meas_t), S.primary), fill_for(K(meas_c), S.grey)],
                       "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(True), color=color(S.ink), fontSize=n(12))},
              vco=_title(f"{lab}: treatment vs control", sub), alt=f"{lab} treatment vs control")
    bw = 1150
    matrix(p, "bychan", x0, by + th + GUT, bw, bh - th - GUT, [(C("Email Test", "Channel"), "Channel")],
           [(K("T Customers"), "Treated"), (K("Visit Rate Uplift"), "Visit uplift"), (K("Conversion Rate Uplift"), "Conversion uplift"),
            (K("Spend per Customer Uplift"), "Spend uplift"), (K("Spend Uplift CI"), "95% CI (spend)"), (K("Spend p-value"), "p-value"),
            (K("Readout Verdict"), "Verdict")],
           "Readout by purchase channel", "Uplift = treatment minus control within the channel  ·  p-value shaded mint when small",
           gradients=[(K("Spend per Customer Uplift"), S.panel, S.soft_pos), (K("Spend p-value"), S.soft_pos, S.panel)], font=11)
    cx = x0 + bw + GUT
    cwid = cw - bw - GUT
    D.panel(p, "verdict-bg", cx, by + th + GUT, cwid, bh - th - GUT)
    D.textbox(p, "verdict-h", cx + 24, by + th + GUT + 18, cwid - 48, 26, [[("Experiment verdict", 13, S.ink, True)]])
    for j, (m_, lab, sz) in enumerate([("Readout Verdict", "Decision", 20), ("Spend Uplift CI", "95% confidence interval, spend per customer", 16),
                                       ("Conversion Uplift CI", "95% confidence interval, conversion rate", 16)]):
        p.add(f"v{j}", "cardVisual", cx + 12, by + th + GUT + 56 + j * 88, cwid - 24, 80, query={"Data": [proj(K(m_), display=lab)]},
              objects={"value": idprops(fontSize=n(sz), fontColor=color(S.accent if j == 0 else S.ink), fontFamily=s(S.font_b), horizontalAlignment=s("left")),
                       "label": idprops(show=b(True), fontSize=n(11), fontColor=color(S.muted), position=s("aboveValue"), horizontalAlignment=s("left")),
                       "outline": idprops(show=b(False)),
                       "padding": idprops(leftMargin=i(12), rightMargin=i(8), topMargin=i(0), bottomMargin=i(0))},
              vco=dict(D.NOBOX), alt=lab)

    # ------------------------------------------------------------------ 5 Customised offers
    p = r.page("offers", "Customised offers", bg=S.page)
    chrome(p, TITLE, "Mens buyers respond only to the Mens offer; Womens buyers to either. One offer does not fit all",
           "Uplift vs control within each segment (randomised, so causal)  ·  choose the segment dimension on the right", f"5 / {NP}", SRC_H)
    single_buttons(p, "upl", S.slicer_x, S.title_y - 8, 480, 112, UPL, "'''Email Test''[Purchased]'", 3, 2, None)
    kpi_band(p, "kpi", [(K("Visit Uplift Mens"), "Visit uplift, Mens offer"), (K("Visit Uplift Womens"), "Visit uplift, Womens offer"),
                        (K("Spend Uplift Mens"), "Spend uplift, Mens offer"), (K("Spend Uplift Womens"), "Spend uplift, Womens offer"),
                        (K("Segment Customers"), "Customers in test")])
    mw = 1150
    sh = 300
    matrix(p, "seg", x0, by, mw, sh, [C("Email Test", "Purchased")],
           [(K("Segment Customers"), "Customers"), (K("Visit Uplift Mens"), "Visit uplift, Mens"), (K("Visit Uplift Womens"), "Visit uplift, Womens"),
            (K("Spend Uplift Mens"), "Spend uplift, Mens"), (K("Spend Uplift Womens"), "Spend uplift, Womens"), (K("Best Offer"), "Best offer")],
           "Next-best offer by segment", "Shaded rose (negative) to mint (positive)  ·  best offer maximises spend uplift",
           gradients=[(K("Visit Uplift Mens"), S.soft_neg, S.soft_pos, S.panel, 0), (K("Visit Uplift Womens"), S.soft_neg, S.soft_pos, S.panel, 0),
                      (K("Spend Uplift Mens"), S.soft_neg, S.soft_pos, S.panel, 0), (K("Spend Uplift Womens"), S.soft_neg, S.soft_pos, S.panel, 0)],
           databars=[K("Segment Customers")], field_param=UPL, font=12)
    p.add("segbar", "clusteredBarChart", x0, by + sh + GUT, mw, bh - sh - GUT,
          query={"Category": [P(C("Email Test", "Purchased"))], "Y": [P((K("Spend Uplift Mens"), "Mens offer")), P((K("Spend Uplift Womens"), "Womens offer"))]},
          objects={"dataPoint": [fill_for(K("Spend Uplift Mens"), S.primary), fill_for(K("Spend Uplift Womens"), S.accent)],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(True), color=color(S.ink), fontSize=n(11))},
          vco=_title("Spend uplift per customer: which offer wins in each segment", "Two-week incremental spend vs control, USD"),
          alt="Spend uplift by segment and offer", field_params={"Category": UPL})
    rx = x0 + mw + GUT
    rw = cw - mw - GUT
    ph = 370
    bar(p, "pol", rx, by, rw, ph, (C("Policies", "Policy"), "Policy"), (K("Policy Incremental per 1000"), "Incremental $ per 1,000"),
        "Five policies, evaluated on the randomised test", "Incremental two-week revenue per 1,000 customers (self-normalised IPW)",
        sort_desc=False, sort_cat=C("Policies", "PolicyOrder"), gradient=("#3B3470", S.accent))
    table(p, "polci", rx, by + ph + GUT, rw, bh - ph - GUT, [(C("Policies", "Policy"), "Policy"), (K("Policy Incremental per 1000"), "Per 1,000"),
                                                           (K("Policy CI"), "95% bootstrap interval")],
          "Uncertainty matters", "Personalisation matches mailing everyone the Mens offer within the noise",
          sort=[(C("Policies", "PolicyOrder"), "Ascending")])

    # ------------------------------------------------------------------ 6 Targeting simulator
    p = r.page("targeting", "Targeting simulator", bg=S.page)
    chrome(p, TITLE, "The uplift model's top 10% returns 3.6x the revenue of a random 10%",
           "Cross-fitted T-learner on visit uplift  ·  the ranking carries signal only at the top, so the best share depends on what a contact costs", f"6 / {NP}", SRC_H)
    kpi_band(p, "kpi", [(K("Sim Contacts"), "Contacts"), (K("Sim Incremental Revenue"), "Incremental revenue"), (K("Sim Contact Cost"), "Contact cost"),
                        (K("Sim Net Revenue"), "Net incremental revenue"), (K("Sim ROI"), "Return on contact cost"), (K("Best Target Share"), "Best share at this cost")])
    D.panel(p, "ctrl", x0, by, cw, 92)
    single_buttons(p, "share", x0 + 20, by + 8, 1260, 76, C("Target Share", "Target Share"), "10L", 20, 1, "Share of customers to contact (%), ranked by predicted uplift")
    single_buttons(p, "cost", x0 + 1304, by + 8, cw - 1324, 76, C("Contact Cost", "Contact Cost"), "0.75D", 6, 1, "Cost per contact, USD (offer + channel)")
    cy = by + 92 + GUT
    ch = bh - 92 - GUT
    cwl = 1150
    curve = line(p, "curve", x0, cy, cwl, ch, (C("Targeting Curve", "TargetPct"), "Share of customers e-mailed (%)"),
                 [(K("Curve Model Net"), "Uplift model"), (K("Curve Random Net"), "Random targeting")],
                 "Net incremental revenue by share targeted", "Incremental revenue minus contact cost  ·  the model wins up to ~20%; beyond that random e-mailing does better, so cap at the top decile",
                 colors=(S.primary, S.grey), sort_cat=C("Targeting Curve", "TargetPct"), markers=True)
    p.add("deciles", "clusteredColumnChart", x0 + cwl + GUT, cy, cw - cwl - GUT, ch,
          query={"Category": [P((C("Decile Check", "UpliftDecile"), "Predicted uplift decile"))],
                 "Y": [P((K("Observed Visit Uplift"), "Observed")), P((K("Predicted Visit Uplift"), "Predicted"))]},
          objects={"dataPoint": [fill_for(K("Observed Visit Uplift"), S.primary), fill_for(K("Predicted Visit Uplift"), "#4B5876")],
                   "legend": props(show=b(True), position=s("Top")), "labels": props(show=b(False)),
                   "valueAxis": props(show=b(True), gridlineShow=b(True), labelColor=color(S.muted))},
          vco=_title("Does the ranking hold out of sample?", "Visit uplift by predicted decile, one axis: the top decile is real, the rest is flat"),
          sort=[(C("Decile Check", "UpliftDecile"), "Ascending")], alt="Predicted vs observed uplift by decile")

    # ------------------------------------------------------------------ 7 About
    p = r.page("about", "About & method", bg=S.page)
    chrome(p, TITLE, "How this report is built", "Questions, data, models and definitions behind every number", f"7 / {NP}",
           "UCI Online Retail II (CC BY 4.0) and the MineThatData e-mail test (Hillstrom, 2008).")
    about_page(p, [
        ("Business questions", ["Is the customer base growing, and where do we lose people?", "Who is about to churn, and how much revenue is at risk?",
                                "Did the campaign cause extra revenue, and is it significant?", "Which offer should each customer get, and how many should we contact?"]),
        ("Data & ML", ["Online Retail II: 805,830 clean order lines, 5,852 customers. Lifecycle flows on a 3-month activity window.",
                       "Churn: gradient boosting, out-of-time test AUC 0.76 vs 0.71 for recency alone.",
                       "Hillstrom: 64,000 randomised customers. Cross-fitted logistic T-learner on visit uplift; policies valued by inverse propensity weighting.",
                       "No public dataset has lifecycle history and a randomised test together, so each page names its source."]),
        ("Model & DAX", ["Star schemas for the lifecycle and the experiment in one model; what-if tables for share targeted and contact cost.",
                         "DAX user-defined functions for the two-proportion z-test and the Welch test.",
                         "Field parameter for segment uplift; bridge measure feeding a waterfall."]),
    ], [("Active base", "customers with an order in the last 3 months."), ("Churned", "left the active base this month (3 months without an order)."),
        ("Uplift", "treatment minus control, in points or dollars per customer: the causal effect."),
        ("Incremental revenue", "spend uplift x treated customers."), ("Value at risk", "churn probability x last-12-month spend."),
        ("p-value", "chance of an uplift this large if the e-mail had no effect; < 0.05 = significant.")],
        "Built by Mohamed Ayman, Senior Data Scientist  ·  github.com/ayman-stat  ·  mohamedaymanstat.vercel.app")

    r.write()
    write_pbip(os.path.join(ROOT, f"{NAME}.pbip"), f"{NAME}.Report")


if __name__ == "__main__":
    build()
    print("report written")
