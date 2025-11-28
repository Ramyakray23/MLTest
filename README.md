# MLTest

📘 Loan Default Prediction — FinSecure Risk Modeling

This project builds a machine learning model to predict the probability of a loan defaulting using the FinSecure loan dataset. The goal is to help the loan approval team identify high-risk borrowers and reduce default rates.

The model includes:

Full preprocessing pipeline

Logistic Regression baseline

Gradient Boosting model

ROC–AUC evaluation

Fairness analysis across subgroups

Final performance metrics

📂 Project Structure
├── loan_default_model.ipynb   # Main notebook (contains code + outputs)
├── loan_data.csv              # Dataset
├── README.md                  # Project documentation

🎯 Problem Summary

FinSecure wants to replace their manual scorecard system with a data-driven model that predicts whether a loan will be fully paid back (1) or default (0).

We create a new target:

default = 1 - loan_paid_back


This allows the model to predict default probability.

🧹 Data Preprocessing

A complete preprocessing pipeline was built using ColumnTransformer:

🔢 Numerical Features

annual_income

debt_to_income_ratio

credit_score

loan_amount

interest_rate

Processing steps:

Median imputation

Standard scaling

🔤 Categorical Features

gender

marital_status

education_level

employment_status

loan_purpose

grade_subgrade

Processing steps:

Most frequent imputation

One-hot encoding

🤖 Modeling

Two models were trained:

1️⃣ Baseline Model — Logistic Regression

Class-balanced

AUC: 0.91

2️⃣ Final Model — Gradient Boosting (fast optimized version)
GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3
)

📈 Final Test AUC: 0.9162

This model was selected for all downstream analysis.

📉 ROC Curve

The ROC curve was plotted and AUC was computed using:

roc_auc_score(y_test, y_test_proba)


This curve demonstrates strong classifier performance and good separability between defaulters and repayers.

🔍 Threshold Selection

The optimal threshold was chosen by maximizing F1-score, resulting in:

Best Threshold = 0.36
Best F1 = 0.7288

🧭 Subgroup Fairness Analysis

To ensure the model is fair across demographic and purpose-based groups, performance was evaluated across:

📘 1. Education Level
Education Level	AUC	F1	Precision	Recall
PhD	0.9246	0.7301	0.778	0.687
Bachelor's	0.9183	0.7349	0.776	0.697
High School	0.9151	0.7231	0.770	0.681
Master's	0.9109	0.7215	0.777	0.673
Other	0.9162	0.7239	0.806	0.657

📌 All groups show very consistent performance.

📕 2. Loan Purpose
Top 3 Loan Purposes (Best AUC)
Loan Purpose	AUC	F1
Education	0.9299	0.7606
Vacation	0.9200	0.7133
Car	0.9188	0.7244
Bottom 3 Loan Purposes (Lowest AUC)
Loan Purpose	AUC	F1
Medical	0.9030	0.6956
Home	0.9131	0.6880
Business	0.9136	0.7236

📌 Performance remains strong across purposes, but Medical loans show slightly weaker recall.

📝 Conclusions

The Gradient Boosting model performs well (AUC ≈ 0.92) and improves upon the logistic baseline.

Fairness analysis shows balanced performance across education levels.

Loan purpose affects difficulty slightly, but differences are small and manageable.

This model can be confidently used to support loan approval decisions.

🚀 Future Improvements

Hyperparameter tuning (RandomizedSearch / Bayesian Optimization)

SHAP explainability analysis

Cost-sensitive modeling

Deployment with Flask/FastAPI

👨‍💻 Author

Ramyak
Machine Learning Project
FinSecure Loan Default Prediction
