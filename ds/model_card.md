# Model card: CVM churn score and e-mail uplift model

Two models feed the Power BI report. Both are built in `etl/build_data.py` and reproduced, tested and extended in
`ds/notebooks/`. Numbers below come from `ds/reports/metrics.json`, written by the notebooks.

---

## 1. Churn score (Online Retail II)

| | |
|---|---|
| **Intended use** | Rank active customers once a quarter for a retention contact; size the contact budget. Not for pricing, credit or any individual decision without a human in the loop. |
| **Owner** | Mohamed Ayman (CVM data science) |
| **Data** | UCI Online Retail II (CC BY 4.0): order lines of a UK online gift wholesaler, 1 Dec 2009 - 9 Dec 2011. 805,830 clean lines, 5,875 customers with an ID (about 85% of sales). |
| **Unit and target** | Customer with an order in the 365 days before snapshot date *t*; churn = no order in [t, t + 90 days). |
| **Features** | 13 purchase-history features, all from data strictly before *t*: recency (days since last order, average gap), frequency (orders in 90 / 365 days / all time), spend (90 days, prior 90 days, 12 months, average basket), breadth (distinct products, cancelled lines), tenure, UK flag. No personal data. |
| **Design** | Out of time: train on the 1 Mar 2011 snapshot, validate on 1 Jun 2011, refit on both and test on 1 Sep 2011 (labels mature 29 Nov 2011). A 1 Dec 2010 snapshot adds a second CV fold. Random CV over-states AUC by 3.2 points (0.793 vs 0.761) and is not used. |
| **Production model** | `HistGradientBoostingClassifier` (300 iterations, learning rate 0.05, 15 leaves, 40 rows per leaf, L2 1.0). |

### Performance on the untouched test snapshot (1 Sep 2011, 4,324 customers, 50.5% churn)

| Model | AUC | 95% CI | PR AUC | Brier (raw) |
|---|---|---|---|---|
| Recency only | 0.708 | 0.692 - 0.724 | 0.679 | 0.240 |
| RFM rule (25 R x F cells) | 0.758 | 0.744 - 0.772 | 0.710 | 0.217 |
| Regularised logistic (L2, C = 0.01) | 0.770 | 0.756 - 0.784 | 0.743 | 0.211 |
| **Production champion (GBM)** | **0.761** | 0.747 - 0.775 | 0.720 | 0.215 |
| Tuned LightGBM challenger | 0.773 | 0.759 - 0.786 | 0.744 | 0.210 |

Paired bootstrap on the same customers: challenger vs champion +0.012 (95% CI +0.007 to +0.017); logistic vs champion
+0.009 (+0.002 to +0.016); challenger vs logistic +0.003 (-0.001 to +0.007); champion vs RFM rule +0.003 (-0.006 to +0.011);
champion vs recency +0.053 (+0.041 to +0.064). Top-decile churn 77.1% (lift 1.53); 43.5% of churners in the riskiest 30%.
All published numbers (`ml/metrics.json`) are reproduced exactly and asserted in notebook 02.

**Champion/challenger finding.** The production model is over-fitted (sklearn's automatic early stopping only switches on above 10,000 rows, so all 300 trees are grown on 8,584) and is no better than the RFM rule. The
regularised logistic model matches the tuned LightGBM within noise and is recommended for the next release after one
quarter of shadow scoring. The published numbers stay as the production record until then.

### Calibration

Churn is seasonal: the training snapshots churn at 63%, the autumn test snapshot at 50%. The raw model predicts 64.1%
(calibration ratio 1.27, Brier 0.215). Isotonic recalibration on the June snapshot barely helps (ratio 1.24). The
production fix is an intercept shift to an autumn rate known on the scoring date: the Jun 2011 rate x last year's Sep/Jun
ratio = 52.5%. That gives a ratio of 1.04 and a Brier score of 0.197, with the AUC unchanged.

### Segment performance (test snapshot, calibrated scores)

| Segment | Customers | Churn | Mean prediction | AUC (95% CI) |
|---|---|---|---|---|
| UK | 3,937 | 50.8% | 53.0% | 0.759 (0.740 - 0.774) |
| Outside UK | 387 | 47.6% | 47.1% | 0.779 (0.735 - 0.824) |
| Value: top 20% | 865 | 15.6% | 18.0% | 0.782 (0.741 - 0.824) |
| Value: middle 30% | 1,297 | 42.8% | 47.3% | 0.663 (0.634 - 0.689) |
| Value: bottom 50% | 2,162 | 69.0% | 69.4% | 0.607 (0.582 - 0.635) |
| Tenure under 6 months | 686 | 53.6% | 53.9% | 0.733 (0.693 - 0.767) |
| Tenure 6-12 months | 1,239 | 62.2% | 65.6% | 0.718 (0.692 - 0.749) |
| Tenure over 12 months | 2,399 | 43.5% | 45.4% | 0.762 (0.742 - 0.780) |

Within-segment AUCs are lower than overall because the segment itself carries signal. Ranking is weakest in the
bottom half by value, where most customers have one order.

### Explainability

By share of mean |SHAP|, frequency (37%), spend (33%) and recency (19%) drive the score. Breadth, tenure and country
together add 11%. SHAP and permutation importance agree on the leader (orders in the last 12 months; rank correlation
0.76). Error analysis: false alarms look exactly like the churners beside them (one order, last order about 275 days ago),
which is the error floor of purchase-history data. Missed churners are regular, higher-value customers who stop without
warning.

### Decision use

Rank by **expected margin at risk** (calibrated P(churn) x 30% margin x 12-month spend), not by churn probability. At
a 10% save rate and £10 per contact, contacting the top 42% nets £15.2K a quarter (90% band £11.6K-£18.5K), vs £5.8K
for the best list ranked by risk. Swapping in the better model adds about £0.4K. The save rate dominates (£3.7K at 5%,
£102K at 30%) and must be measured with a random hold-out: about 2,040 customers per arm detect a 10% save rate.

### Limitations and ethics

* Tenure is counted from the start of the data (Dec 2009), so it drifts (PSI 0.72); cap it at 365 days.
* "Churn" is 90 days without an order. Many such customers return, so a save is worth less than the budget assumes.
* The seasonal correction rests on one prior year.
* No protected attributes are used. Country is the only geographic field, and the model ranks customers outside the UK
  at least as well as UK customers.

### Monitoring and retraining

Feature and score PSI at every scoring run (watch > 0.10, act > 0.25). Calibration ratio on matured labels (act
outside 0.9-1.1). AUC below 0.72 (the lower end of today's CI) triggers a refit. Keep a 10-20% random hold-out from
every campaign. Retrain quarterly on the latest two matured snapshots.

---

## 2. E-mail uplift model (Hillstrom test)

| | |
|---|---|
| **Intended use** | Choose which e-mail (Mens, Womens or none) each customer gets, and who to contact when contacts are limited. |
| **Data** | MineThatData / Hillstrom (2008): 64,000 customers randomised 1/3 each to a Mens e-mail, a Womens e-mail or no e-mail; two-week visit, conversion and spend. Randomisation verified: sample-ratio p = 0.90, max \|SMD\| 0.014. |
| **Model** | Cross-fitted logistic T-learner on visit (5 folds; one model per arm; uplift = P(visit \| e-mail) - P(visit \| none)). Next-best offer = the e-mail with the larger predicted uplift. |
| **Why visit** | Only 0.9% of customers buy. Rankings trained on spend reach \|z\| 2.1 at best, against 6.4 for the visit-trained ranking of the Womens e-mail. |
| **Alternatives tested** | LightGBM T-learner and transformed-outcome LightGBM: AUUC within noise of the production model (paired differences -0.0003 to -0.0008). |

### Performance

| | Mens e-mail | Womens e-mail |
|---|---|---|
| AUUC (incremental visits per customer, vs random) | 0.0011 (95% CI -0.0009 to 0.0033) | 0.0063 (0.0044 to 0.0081) |
| Share of extra visits in the top 30% (random = 30%) | 33% | 40% |
| Calibration slope, observed on predicted uplift | 0.35 | 0.90 |

The Mens uplift is close to uniform: one real group (customers who bought both lines, +13.4 points of visits), flat
elsewhere. The Womens e-mail works for womens buyers (+7.4 points) and barely for mens-only buyers (+1.1).

### Policy value (self-normalised IPW, incremental spend per 1,000 customers, 95% bootstrap CI)

| Policy | $ per 1,000 |
|---|---|
| All: Womens offer | 424 (160 - 677) |
| All: Mens offer | 770 (488 - 1,024) |
| Rule: by purchase history | 776 (490 - 1,051) |
| Model: next-best offer | 777 (506 - 1,054) |

Paired difference, model vs Mens to everyone: +$7 per 1,000 (95% CI -$148 to +$171). **Recommendation: send everyone the
Mens e-mail and keep a 10% hold-out.** Use the uplift ranking only when contacts are limited: there, the top 10%
returns 3.6x the spend of a random 10%.

### Limitations

Two-week outcomes only (no long-term or unsubscribe effects). A 2008 test, so effect sizes illustrate the method. The
purchase-history rule is valued in-sample. A test of personalised vs Mens-to-all is not feasible at the observed edge
(about 94 million customers per arm on spend).
