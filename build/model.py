"""Semantic model for the CVM Experiments report (writes TMDL)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "toolkit"))  # vendored copy of the shared design toolkit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_tools"))  # master copy, when present
from pbip import col, meas, m_table, calc_table, measure_table, field_param, rel, write_model, csv_expressions  # noqa: E402

REPO = "powerbi-cvm-experiments"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = "CVMExperiments"

MONTHS = '''VAR _m = FILTER(CALENDAR(DATE(2010, 3, 1), DATE(2011, 11, 30)), DAY([Date]) = 1)
RETURN ADDCOLUMNS(_m,
    "Year", YEAR([Date]),
    "Month", FORMAT([Date], "MMM"),
    "Month Number", MONTH([Date]),
    "Year Month", FORMAT([Date], "MMM yy"),
    "Year Month Sort", YEAR([Date]) * 100 + MONTH([Date]),
    "Period", IF([Date] >= DATE(2010, 12, 1), "Dec 10 - Nov 11", "Mar 10 - Nov 10"),
    "Period Sort", IF([Date] >= DATE(2010, 12, 1), 2, 1))'''

TABLES = [
    calc_table("Month", MONTHS, [
        col("Date", "dateTime", "mmm yyyy", key=True, desc="First day of the month"), col("Year", "int64", "0"), col("Month", sort="Month Number"),
        col("Month Number", "int64", "0", hidden=True), col("Year Month", sort="Year Month Sort"), col("Year Month Sort", "int64", "0", hidden=True),
        col("Period", sort="Period Sort", desc="Reporting year: Dec-Nov. Mar-Nov 2010 is partial (the first three months warm up the 3-month activity window)."),
        col("Period Sort", "int64", "0", hidden=True)], desc="Monthly calendar, Mar 2010 - Nov 2011."),
    m_table("Customer", "dim_retail_customer.csv", [
        col("CustomerKey", "int64", "0", hidden=True, key=True), col("CohortMonth", "dateTime", "mmm yyyy", hidden=True),
        col("Country"), col("Market", desc="United Kingdom, rest of Europe, rest of world"), col("CohortYear", hidden=True),
        col("CohortLabel", sort="CohortSort", desc="Month of first purchase"), col("CohortSort", "int64", "0", hidden=True)],
        desc="Online Retail II customers (UK online gift retailer)."),
    m_table("Flow", "dim_flow.csv", [col("Flow", sort="FlowOrder"), col("FlowOrder", "int64", "0", hidden=True), col("Direction")],
            desc="Lifecycle flow of a customer in a month."),
    m_table("Customer Month", "fact_customer_month.csv", [
        col("CustomerKey", "int64", "0", hidden=True), col("MonthStart", "dateTime", "mmm yyyy", hidden=True), col("Flow", hidden=True),
        col("IsActive", "int64", "0", hidden=True), col("Purchased", "int64", "0", hidden=True), col("Revenue", "double", "#,0", hidden=True),
        col("Orders", "int64", "0", hidden=True), col("MonthsSinceFirst", "int64", "0", hidden=True)],
        desc="Fact: one row per customer per month while in the base (active = bought in the last 3 months) or in the month they churn."),
    m_table("Cohort Activity", "fact_cohort_activity.csv", [
        col("CustomerKey", "int64", "0", hidden=True), col("MonthStart", "dateTime", "mmm yyyy", hidden=True),
        col("CohortMonth", "dateTime", "mmm yyyy", hidden=True), col("MonthIndex", "int64", "0", desc="Months since first purchase"),
        col("Revenue", "double", "#,0", hidden=True)], desc="Fact: one row per customer per purchase month (classic cohort retention)."),
    m_table("Churn Scores", "fact_churn_scores.csv", [
        col("CustomerKey", "int64", "0", hidden=True), col("ChurnScore", "double", "0.0%", hidden=True),
        col("RiskDecile", "int64", "0", desc="1 = highest predicted churn risk"), col("RiskBand", sort="RiskBandOrder"),
        dict(col("RiskBandOrder", "int64", "0", hidden=True), computed=True),
        col("ActualChurn", "int64", "0", hidden=True), col("Value365", "double", "#,0", hidden=True), col("RecencyDays", "int64", "0", hidden=True),
        col("Orders365", "int64", "0", hidden=True), col("ValueTier", sort="ValueTierOrder"), dict(col("ValueTierOrder", "int64", "0", hidden=True), computed=True),
        col("RFMSegment")],
        desc="Out-of-time churn scores at 1 Sep 2011 (model trained on Mar and Jun 2011 snapshots); actual churn observed by 30 Nov 2011.",
        extra_steps='''    Result = Table.AddColumn(Table.AddColumn(Typed, "RiskBandOrder", each if [RiskBand] = "High" then 1 else if [RiskBand] = "Medium" then 2 else 3, Int64.Type),
        "ValueTierOrder", each if [ValueTier] = "Top 20%" then 1 else if [ValueTier] = "Middle 30%" then 2 else 3, Int64.Type)'''),
    m_table("Churn Drivers", "churn_feature_importance.csv", [col("Feature"), col("AUCDrop", "double", "0.000", hidden=True)],
            desc="Permutation importance on the out-of-time test set (drop in AUC when the feature is shuffled)."),
    m_table("Model Metrics", "model_metrics.csv", [col("Metric"), col("Value", "double", "0.####", hidden=True)], desc="ML evaluation metrics.", hidden=True),
    m_table("Email Test", "fact_email_test.csv", [
        col("CustomerKey", "int64", "0", hidden=True), col("Arm", desc="Mens offer, Womens offer or Control (no e-mail), randomised"),
        col("RecencyMonths", "int64", "0", hidden=True), col("RecencyBand", desc="Months since last purchase"), col("HistoryBand", desc="Spend in the past year"),
        col("History12m", "double", "#,0", hidden=True), col("Purchased", desc="Merchandise bought in the past year"), col("ZipCode", desc="Area type"),
        col("CustomerType"), col("Channel", desc="Purchase channel in the past year"),
        col("Visited", "int64", "0", hidden=True), col("Converted", "int64", "0", hidden=True), col("Spend", "double", "#,0.00", hidden=True),
        col("UpliftMens", "double", "0.0%", hidden=True), col("UpliftWomens", "double", "0.0%", hidden=True), col("BestUplift", "double", "0.0%", hidden=True),
        col("ModelOffer", desc="Offer with the higher predicted visit uplift"), col("UpliftDecile", "int64", "0", desc="1 = most persuadable (model)")],
        desc="Fact: MineThatData e-mail test, 64,000 customers randomised 1/3 each; two-week visit, conversion and spend."),
    m_table("Test Arm", "dim_test_arm.csv", [col("Arm", sort="ArmOrder"), col("ArmOrder", "int64", "0", hidden=True)],
            desc="Disconnected selector: which treatment arm the readout compares with the control."),
    m_table("Policies", "uplift_policies.csv", [
        col("Policy", sort="PolicyOrder"), col("PolicyOrder", "int64", "0", hidden=True), col("SpendPerCustomer", "double", "0.000", hidden=True),
        col("IncrementalPerCustomer", "double", "0.000", hidden=True), col("CILow", "double", "0.000", hidden=True), col("CIHigh", "double", "0.000", hidden=True),
        col("ContactsPer1000", "int64", "0", hidden=True), col("IncPer1000", "double", "#,0", hidden=True), col("CILowPer1000", "double", "#,0", hidden=True),
        col("CIHighPer1000", "double", "#,0", hidden=True)],
        desc="Offer policies evaluated on the randomised test by inverse propensity weighting; 95% bootstrap intervals."),
    m_table("Targeting Curve", "uplift_targeting_curve.csv", [
        col("TargetPct", "int64", "0", desc="Share of customers e-mailed, ranked by predicted uplift"), col("Contacts", "int64", "#,0", hidden=True),
        col("ModelIncRevenue", "double", "#,0", hidden=True), col("RandomIncRevenue", "double", "#,0", hidden=True),
        col("ModelIncVisits", "double", "#,0", hidden=True), col("RandomIncVisits", "double", "#,0", hidden=True)],
        desc="Incremental revenue and visits when e-mailing the top-k% by predicted uplift vs a random k%."),
    m_table("Decile Check", "uplift_decile_check.csv", [
        col("UpliftDecile", "int64", "0"), col("Customers", "double", "#,0", hidden=True), col("PredictedVisitUplift", "double", "0.0%", hidden=True),
        col("ObservedVisitUplift", "double", "0.0%", hidden=True), col("ObservedSpendUplift", "double", "0.00", hidden=True)],
        desc="Held-out check: predicted vs observed visit uplift by predicted decile."),
    calc_table("Target Share", "GENERATESERIES(5, 100, 5)", [col("Target Share", "int64", "0", src="Value", desc="Share of the base to e-mail (%)")],
               desc="What-if: share of customers to target, ranked by predicted uplift."),
    calc_table("Contact Cost", 'DATATABLE("Contact Cost", DOUBLE, {{0}, {0.1}, {0.25}, {0.5}, {0.75}, {1}})',
               [col("Contact Cost", "double", "$0.00", desc="Cost per contact (USD): offer plus channel")],
               desc="What-if: fully loaded cost of one contact (offer, fatigue, channel)."),
]

FB, FR, FC, FT, FO, FS, FM = "1 Base", "2 Retention", "3 Churn model", "4 A-B test", "5 Offers", "6 Targeting", "9 Model metrics"
CM = "'Customer Month'"
ARM_T = "TREATAS(VALUES('Test Arm'[Arm]), 'Email Test'[Arm])"
CTRL = "'Email Test'[Arm] = \"Control\""


def arm_pair(label, col_, fmt, folder):
    return [
        meas(f"T {label}", f"CALCULATE(AVERAGE('Email Test'[{col_}]), {ARM_T})", fmt, folder, f"{label} in the selected treatment arm."),
        meas(f"C {label}", f"CALCULATE(AVERAGE('Email Test'[{col_}]), {CTRL})", fmt, folder, f"{label} in the control group."),
        meas(f"{label} Uplift", f"[T {label}] - [C {label}]", fmt, folder, f"Treatment minus control: the causal effect of the e-mail on {label.lower()}."),
    ]


def seg_uplift(arm, col_, fmt, label):
    return meas(f"{label} {arm}", f'''CALCULATE(AVERAGE('Email Test'[{col_}]), 'Email Test'[Arm] = "{arm} offer")
    - CALCULATE(AVERAGE('Email Test'[{col_}]), {CTRL})''', fmt, FO, f"{label} of the {arm} offer vs control within the current segment.")


MEASURES = [
    # ---- base dynamics
    meas("Active Base", f'''VAR _m = MAX('Month'[Date])
RETURN CALCULATE(COUNTROWS({CM}), {CM}[IsActive] = 1, 'Month'[Date] = _m)''', "#,0", FB,
         "Customers who bought in the last 3 months, at the end of the selected period."),
    meas("Opening Base", f'''VAR _m = EDATE(MIN('Month'[Date]), -1)
RETURN CALCULATE(COUNTROWS({CM}), {CM}[IsActive] = 1, REMOVEFILTERS('Month'), 'Month'[Date] = _m)''', "#,0", FB,
         "Active base at the end of the month before the selected period."),
    meas("New Customers", f'CALCULATE(COUNTROWS({CM}), {CM}[Flow] = "New")', "#,0", FB, "First purchase in the month."),
    meas("Reactivated", f'CALCULATE(COUNTROWS({CM}), {CM}[Flow] = "Reactivated")', "#,0", FB, "Returned after 3+ months without a purchase."),
    meas("Churned", f'CALCULATE(COUNTROWS({CM}), {CM}[Flow] = "Churned")', "#,0", FB, "Left the active base: 3 months without a purchase."),
    meas("Churned (negative)", "-[Churned]", "#,0", FB),
    meas("Net Change", "[New Customers] + [Reactivated] - [Churned]", "+#,0;-#,0;0", FB),
    meas("Monthly Churn Rate", f'''AVERAGEX(VALUES('Month'[Date]),
    VAR _d = 'Month'[Date]
    VAR _open = CALCULATE(COUNTROWS({CM}), {CM}[IsActive] = 1, REMOVEFILTERS('Month'), 'Month'[Date] = EDATE(_d, -1))
    RETURN DIVIDE(CALCULATE(COUNTROWS({CM}), {CM}[Flow] = "Churned"), _open))''', "0.0%", FB,
         "Average over months of churned / opening active base."),
    meas("Reactivation Share of Gains", "DIVIDE([Reactivated], [New Customers] + [Reactivated])", "0%", FB),
    meas("Revenue", f"SUM({CM}[Revenue])", "£#,0", FB, "Net revenue (cancellations deducted), GBP."),
    meas("Avg Active Base", "AVERAGEX(VALUES('Month'[Date]), CALCULATE([Active Base]))", "#,0", FB),
    meas("Revenue per Active Customer", "DIVIDE([Revenue], [Avg Active Base] * DISTINCTCOUNT('Month'[Date]))", "£#,0", FB, "Monthly revenue per customer in the active base."),
    meas("Bridge Value", '''SWITCH(SELECTEDVALUE('Flow'[Flow]),
    "Opening base", CALCULATE([Opening Base], REMOVEFILTERS('Flow')),
    "New", [New Customers], "Reactivated", [Reactivated], "Churned", -[Churned])''', "#,0", FB, "Signed values for the customer bridge."),
    # ---- retention / cohorts
    meas("Cohort Customers", "DISTINCTCOUNT('Cohort Activity'[CustomerKey])", "#,0", FR),
    meas("Cohort Size", "CALCULATE(DISTINCTCOUNT('Cohort Activity'[CustomerKey]), REMOVEFILTERS('Cohort Activity'[MonthIndex]), 'Cohort Activity'[MonthIndex] = 0)", "#,0", FR),
    meas("Retention", "DIVIDE([Cohort Customers], [Cohort Size])", "0%", FR, "Share of a cohort buying again N months after the first purchase."),
    meas("Month 1 Retention", "CALCULATE([Retention], 'Cohort Activity'[MonthIndex] = 1)", "0.0%", FR),
    meas("Month 3 Retention", "CALCULATE([Retention], 'Cohort Activity'[MonthIndex] = 3)", "0.0%", FR),
    meas("Month 6 Retention", "CALCULATE([Retention], 'Cohort Activity'[MonthIndex] = 6)", "0.0%", FR),
    meas("Month 12 Retention", "CALCULATE([Retention], 'Cohort Activity'[MonthIndex] = 12)", "0.0%", FR),
    meas("12-Month Value per Customer", '''DIVIDE(CALCULATE(SUM('Cohort Activity'[Revenue]), 'Cohort Activity'[MonthIndex] <= 11), [Cohort Size])''', "£#,0", FR,
         "Net revenue in the first 12 months per new customer."),
    # ---- churn model
    meas("Customers Scored", "COUNTROWS('Churn Scores')", "#,0", FC),
    meas("Actual Churn Rate", "AVERAGE('Churn Scores'[ActualChurn])", "0.0%", FC, "Share with no purchase in the 90 days after scoring."),
    meas("Predicted Churn", "AVERAGE('Churn Scores'[ChurnScore])", "0.0%", FC, "Mean predicted probability."),
    meas("Value at Risk", "SUMX('Churn Scores', 'Churn Scores'[ChurnScore] * 'Churn Scores'[Value365])", "£#,0", FC,
         "Sum of churn probability x last-12-month spend: the revenue a retention team is trying to save."),
    meas("Customer Value", "SUM('Churn Scores'[Value365])", "£#,0", FC),
    meas("Churn Capture", '''DIVIDE(CALCULATE(SUM('Churn Scores'[ActualChurn]), 'Churn Scores'[RiskDecile] <= 3), CALCULATE(SUM('Churn Scores'[ActualChurn]), ALL('Churn Scores')))''',
         "0%", FC, "Share of all churners found in the top 30% of scores."),
    meas("Driver Importance", "SUM('Churn Drivers'[AUCDrop])", "0.000", FC),
    meas("Test AUC", "LOOKUPVALUE('Model Metrics'[Value], 'Model Metrics'[Metric], \"churn_auc_test\")", "0.00", FM, "Out-of-time AUC (scored 1 Sep 2011)."),
    meas("Recency-only AUC", "LOOKUPVALUE('Model Metrics'[Value], 'Model Metrics'[Metric], \"churn_auc_recency_only_test\")", "0.00", FM),
    meas("Top Decile Churn", "LOOKUPVALUE('Model Metrics'[Value], 'Model Metrics'[Metric], \"churn_top_decile_rate\")", "0%", FM),
    # ---- A/B readout
    meas("T Customers", f"CALCULATE(COUNTROWS('Email Test'), {ARM_T})", "#,0", FT),
    meas("C Customers", f"CALCULATE(COUNTROWS('Email Test'), {CTRL})", "#,0", FT),
    *arm_pair("Visit Rate", "Visited", "0.0%", FT),
    *arm_pair("Conversion Rate", "Converted", "0.00%", FT),
    *arm_pair("Spend per Customer", "Spend", "$0.00", FT),
    meas("Incremental Conversions", "[Conversion Rate Uplift] * [T Customers]", "#,0", FT),
    meas("Incremental Revenue", "[Spend per Customer Uplift] * [T Customers]", "$#,0", FT, "Extra two-week spend caused by the e-mail in the treated group."),
    meas("Relative Conversion Lift", "DIVIDE([Conversion Rate Uplift], [C Conversion Rate])", "+0%;-0%", FT),
    meas("Visit p-value", "TwoProportionP([T Visit Rate], [T Customers], [C Visit Rate], [C Customers])", "0.0000", FT, "Two-sided two-proportion z-test."),
    meas("Conversion p-value", "TwoProportionP([T Conversion Rate], [T Customers], [C Conversion Rate], [C Customers])", "0.0000", FT),
    meas("Spend p-value", f'''MeanDifferenceP([T Spend per Customer], CALCULATE(VAR.S('Email Test'[Spend]), {ARM_T}), [T Customers],
    [C Spend per Customer], CALCULATE(VAR.S('Email Test'[Spend]), {CTRL}), [C Customers])''', "0.0000", FT, "Welch test on mean spend (normal approximation)."),
    meas("Spend p-value Label", 'IF([Spend p-value] < 0.001, "< 0.001", FORMAT([Spend p-value], "0.000"))', None, FT, "Readable p-value for cards."),
    meas("Conversion Uplift CI", '''VAR _se = SQRT([T Conversion Rate] * (1 - [T Conversion Rate]) / [T Customers] + [C Conversion Rate] * (1 - [C Conversion Rate]) / [C Customers])
RETURN FORMAT([Conversion Rate Uplift] - 1.96 * _se, "+0.00%;-0.00%") & " to " & FORMAT([Conversion Rate Uplift] + 1.96 * _se, "+0.00%;-0.00%")''', None, FT),
    meas("Spend Uplift CI", f'''VAR _se = SQRT(CALCULATE(VAR.S('Email Test'[Spend]), {ARM_T}) / [T Customers] + CALCULATE(VAR.S('Email Test'[Spend]), {CTRL}) / [C Customers])
RETURN FORMAT([Spend per Customer Uplift] - 1.96 * _se, "+$0.00;-$0.00") & " to " & FORMAT([Spend per Customer Uplift] + 1.96 * _se, "+$0.00;-$0.00")''', None, FT),
    meas("Readout Verdict", '''IF([Spend p-value] < 0.05 && [Spend per Customer Uplift] > 0, "Ship: significant revenue uplift",
    IF([Visit p-value] < 0.05 && [Visit Rate Uplift] > 0, "Engagement up, revenue not yet significant", "No significant effect"))''', None, FT),
    meas("Arm Caption", '"Treatment: " & SELECTEDVALUE(\'Test Arm\'[Arm], "both offers") & " vs control (no e-mail)"', None, FT),
    # ---- customised offers (segment uplift)
    seg_uplift("Mens", "Visited", "+0.0%;-0.0%", "Visit Uplift"),
    seg_uplift("Womens", "Visited", "+0.0%;-0.0%", "Visit Uplift"),
    seg_uplift("Mens", "Spend", "+$0.00;-$0.00", "Spend Uplift"),
    seg_uplift("Womens", "Spend", "+$0.00;-$0.00", "Spend Uplift"),
    meas("Segment Customers", "COUNTROWS('Email Test')", "#,0", FO),
    meas("Best Offer", '''VAR _m = [Spend Uplift Mens]
VAR _w = [Spend Uplift Womens]
RETURN IF(MAX(_m, _w) <= 0, "No e-mail", IF(_m >= _w, "Mens offer", "Womens offer"))''', None, FO, "Revenue-maximising offer for the segment."),
    meas("Best Offer Uplift", "MAX([Spend Uplift Mens], [Spend Uplift Womens])", "+$0.00;-$0.00", FO),
    meas("Policy Incremental per 1000", "SUM('Policies'[IncPer1000])", "$#,0", FO, "Incremental two-week revenue per 1,000 customers (IPW estimate)."),
    meas("Policy CI", '''FORMAT(SUM('Policies'[CILowPer1000]), "$#,0") & " to " & FORMAT(SUM('Policies'[CIHighPer1000]), "$#,0")''', None, FO),
    # ---- targeting simulator
    meas("Target Share Value", "SELECTEDVALUE('Target Share'[Target Share], 10)", "0\"%\"", FS),
    meas("Contact Cost Value", "SELECTEDVALUE('Contact Cost'[Contact Cost], 0.75)", "$0.00", FS),
    meas("Sim Contacts", "LOOKUPVALUE('Targeting Curve'[Contacts], 'Targeting Curve'[TargetPct], [Target Share Value])", "#,0", FS),
    meas("Sim Incremental Revenue", "LOOKUPVALUE('Targeting Curve'[ModelIncRevenue], 'Targeting Curve'[TargetPct], [Target Share Value])", "$#,0", FS),
    meas("Sim Incremental Visits", "LOOKUPVALUE('Targeting Curve'[ModelIncVisits], 'Targeting Curve'[TargetPct], [Target Share Value])", "#,0", FS),
    meas("Sim Contact Cost", "[Sim Contacts] * [Contact Cost Value]", "$#,0", FS),
    meas("Sim Net Revenue", "[Sim Incremental Revenue] - [Sim Contact Cost]", "$#,0;-$#,0", FS),
    meas("Sim ROI", "DIVIDE([Sim Net Revenue], [Sim Contact Cost])", "0%", FS),
    meas("Best Target Share", '''VAR _c = [Contact Cost Value]
VAR _t = TOPN(1, FILTER(ALL('Targeting Curve'), 'Targeting Curve'[TargetPct] > 0), 'Targeting Curve'[ModelIncRevenue] - 'Targeting Curve'[Contacts] * _c, DESC)
RETURN MAXX(_t, 'Targeting Curve'[TargetPct])''', "0\"%\"", FS, "Target share that maximises net incremental revenue at the chosen cost."),
    meas("Curve Model Net", "SUM('Targeting Curve'[ModelIncRevenue]) - SUM('Targeting Curve'[Contacts]) * [Contact Cost Value]", "$#,0", FS),
    meas("Curve Random Net", "SUM('Targeting Curve'[RandomIncRevenue]) - SUM('Targeting Curve'[Contacts]) * [Contact Cost Value]", "$#,0", FS),
    meas("Predicted Visit Uplift", "AVERAGE('Decile Check'[PredictedVisitUplift])", "0.0%", FS),
    meas("Observed Visit Uplift", "AVERAGE('Decile Check'[ObservedVisitUplift])", "0.0%", FS),
    meas("Sim Caption", '''"Target " & FORMAT([Target Share Value], "0") & "% at " & FORMAT([Contact Cost Value], "$0.00") & " per contact. Best share at this cost: " & FORMAT([Best Target Share], "0") & "%"''', None, FS),
]

RELS = [
    rel("Customer Month", "MonthStart", "Month", "Date"), rel("Customer Month", "CustomerKey", "Customer", "CustomerKey"),
    rel("Customer Month", "Flow", "Flow", "Flow"), rel("Cohort Activity", "CustomerKey", "Customer", "CustomerKey"),
    rel("Churn Scores", "CustomerKey", "Customer", "CustomerKey"),
]

FUNCTIONS = [
    dict(name="TwoProportionP", desc="Two-sided p-value of a two-proportion z-test (pooled variance).",
         body="""(p1 : NUMERIC, n1 : NUMERIC, p2 : NUMERIC, n2 : NUMERIC) =>
VAR _p = DIVIDE(p1 * n1 + p2 * n2, n1 + n2)
VAR _se = SQRT(_p * (1 - _p) * (1 / n1 + 1 / n2))
VAR _z = DIVIDE(p1 - p2, _se)
RETURN IF(_se > 0, 2 * (1 - NORM.S.DIST(ABS(_z), TRUE)))"""),
    dict(name="MeanDifferenceP", desc="Two-sided p-value for a difference in means (Welch, normal approximation for large samples).",
         body="""(m1 : NUMERIC, v1 : NUMERIC, n1 : NUMERIC, m2 : NUMERIC, v2 : NUMERIC, n2 : NUMERIC) =>
VAR _se = SQRT(v1 / n1 + v2 / n2)
RETURN IF(_se > 0, 2 * (1 - NORM.S.DIST(ABS(DIVIDE(m1 - m2, _se)), TRUE)))"""),
]


def build(local=True):
    upl = field_param("Uplift by", [("Purchase history", "'Email Test'[Purchased]"), ("Recency", "'Email Test'[RecencyBand]"),
                                    ("Past-year spend", "'Email Test'[HistoryBand]"), ("Channel", "'Email Test'[Channel]"),
                                    ("Area type", "'Email Test'[ZipCode]"), ("Customer type", "'Email Test'[CustomerType]")],
                      desc="Field parameter: the segment dimension for the customised-offer analysis.")
    tables = TABLES + [measure_table("KPI", MEASURES, "All report measures, organised in display folders."), upl]
    exprs = csv_expressions(REPO, os.path.join(ROOT, "data") + "\\", default="Local" if local else "GitHub")
    write_model(os.path.join(ROOT, f"{NAME}.SemanticModel"), NAME, tables, RELS, exprs, functions=FUNCTIONS,
                desc="CVM Experiments: customer lifecycle (UCI Online Retail II) and a randomised offer test (MineThatData) in one model.")


if __name__ == "__main__":
    build(local="--github" not in sys.argv)
    print("model written")
