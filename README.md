# Online-shopping-intent-predictor

# 🛒 Online Shopper Purchase Intention Predictor

Predicts whether a website visitor session will end in a purchase, and explains *why* using SHAP — built as an end-to-end ML project covering EDA, preprocessing, model comparison, explainability, and deployment.

**Live app:** [[add your Streamlit Cloud link here once deployed]
](https://online-shopping-intent-predictor-ngxme7ivmaef6xfbrrtjpx.streamlit.app/)
---

## Problem

E-commerce sites want to know, in real time, which visitor sessions are likely to convert — not just to predict, but to understand *what's driving* that prediction, so marketing/product teams can act on it. This project builds a classifier on the [UCI Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) (12,330 sessions, ~15.5% conversion rate) and pairs it with SHAP explainability so every prediction comes with a reason, not just a probability.

## Key EDA Findings

- **PageValues is by far the strongest predictor** of purchase (correlation 0.49 with target — more than double any other numeric feature), confirmed independently through correlation analysis, RFECV, and SHAP.
- **November has the highest conversion rate (~25%)** despite not having the most traffic (May does) — visitors in November convert far better, likely holiday-shopping intent. This is reflected in an engineered `IsHolidaySeason` feature.
- **New visitors convert better (~25%) than returning visitors (~14%)** — a counter-intuitive finding relative to some prior work on this dataset, verified independently in this analysis.
- A chi-square test confirmed `Region` has no significant relationship with purchase (p=0.32) and was dropped; `BounceRates` was dropped for being 0.91-correlated with `ExitRates`.

## Preprocessing

- Removed 125 duplicate rows.
- Dropped `Region` (EDA/chi-square) and `BounceRates` (multicollinearity with `ExitRates`).
- Engineered `IsHolidaySeason` from the monthly conversion-rate finding above.
- Validated feature selection against **RFECV**, which independently agreed with the manual EDA-based drops and additionally flagged 9 sparse one-hot categorical levels with low sample support.
- Built two parallel preprocessed datasets: raw (for tree models, which are invariant to monotonic transforms) and log-transformed + outlier-capped (for linear/distance-based models). Verified empirically that tree models showed ~0.0000–0.0003 AUC difference between the two, confirming the scoping choice was correct.

## Modeling

Compared Logistic Regression, AdaBoost, Random Forest, and XGBoost — baseline and hyperparameter-tuned (`RandomizedSearchCV`, 5-fold CV) — across three class-imbalance strategies (`class_weight`/`scale_pos_weight`, SMOTE, RandomUnderSampler).

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest (tuned, class_weight)** | 0.930 | 0.677 | 0.720 | **0.698** |
| XGBoost (tuned, scale_pos_weight) | **0.938** | 0.557 | 0.848 | 0.672 |
| Voting Classifier (RF+XGB+LR) | 0.934 | 0.580 | 0.785 | 0.667 |
| Logistic Regression (baseline) | 0.927 | 0.560 | 0.825 | 0.667 |
| AdaBoost (baseline) | 0.920 | 0.690 | 0.605 | 0.644 |

**Final model: Random Forest (tuned, `class_weight='balanced'`).** While XGBoost achieved a marginally higher AUC, Random Forest had the best precision-recall balance (highest F1), which better fits the business goal of not over-targeting unlikely buyers. This held even after testing a soft-voting ensemble and sweeping the classification threshold (0.5 was already F1-optimal). Every imbalance-handling alternative (SMOTE, undersampling) underperformed simple class-weighting — both increased recall but at a disproportionate cost to precision, and undersampling additionally lost ~70% of training data.

## Explainability (SHAP)

Used `TreeExplainer` on the final Random Forest model for both global and local explanations.

**Top features by mean |SHAP value|:** PageValues (dominant, 5x the next feature), ExitRates, Month_Nov, ProductRelated_Duration, ProductRelated — consistent with the EDA correlation and RFECV findings above, triangulated across three independent methods.

The dashboard shows a live SHAP waterfall plot for every prediction, so a user sees exactly which features pushed a specific session toward or away from "will purchase" — not just a probability score.

## Dashboard

Built with Streamlit — input session details via form, get a live prediction with probability, and a SHAP waterfall plot explaining the specific prediction.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib/seaborn

## Project Structure

```
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── notebooks/
│   ├── eda_online_shoppers.py
│   ├── preprocessing_online_shoppers.py
│   ├── modeling_online_shoppers.py
│   └── shap_explainability.py
├── final_model.pkl                 # (hosted via GitHub Releases)
├── shap_explainer.pkl              # (hosted via GitHub Releases)
└── shap_sample_data.csv
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Possible Extensions

- Cross-validated final metrics (currently a single 80/20 split) for more robust reporting
- Probability calibration check, given `class_weight` distorts raw predicted probabilities
- Aggregate error analysis across all misclassified sessions, not just individual examples
