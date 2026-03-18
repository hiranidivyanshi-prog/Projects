# 🛡️ Insurance Claim Fraud Detection

A machine learning project that builds a predictive classification model to identify fraudulent automobile insurance claims.

---

## 📌 Problem Statement

Insurance fraud is a significant challenge in the auto insurance industry. This project leverages machine learning to analyze insurance policy data, customer demographics, and accident details to predict whether a given claim is fraudulent or legitimate.

---

## 📂 Dataset

- **Source:** [GitHub — dsrscientist/Data-Science-ML-Capstone-Projects](https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/Automobile_insurance_fraud.csv)
- **Size:** 1,000 rows × 40 columns (original)
- **Target Column:** `fraud_reported` (Y / N → encoded to 1 / 0)

### Feature Groups

| Category | Columns |
|---|---|
| Policy Info | `months_as_customer`, `age`, `policy_bind_date`, `policy_csl`, `policy_deductable`, `policy_annual_premium`, `umbrella_limit` |
| Customer Details | `insured_zip`, `insured_sex`, `insured_education_level`, `insured_occupation`, `insured_hobbies`, `insured_relationship`, `capital-gains`, `capital-loss` |
| Incident Details | `incident_type`, `collision_type`, `incident_severity`, `authorities_contacted`, `incident_state`, `incident_city`, `incident_hour_of_the_day`, `number_of_vehicles_involved` |
| Claim Details | `total_claim_amount`, `injury_claim`, `property_claim`, `vehicle_claim` |

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected shape, data types, unique values, and descriptive statistics
- Dropped irrelevant columns: `policy_number`, `incident_location`, `_c39` (all-null), `incident_Year` (single unique value)
- Confirmed no duplicate rows

### 2. Data Preprocessing
- Parsed `policy_bind_date` and `incident_date` into separate Day / Month / Year columns, then dropped originals
- Replaced `?` placeholder values in `collision_type`, `property_damage`, and `police_report_available` with `"questionable"`
- Confirmed zero missing values post-cleaning

### 3. Visualization
- Pie charts for categorical distributions
- Count plots for multi-category columns and feature vs. target comparisons
- KDE plots for continuous features split by fraud status
- Correlation heatmap and bar plot of feature correlations with `fraud_reported`

### 4. Encoding
- **LabelEncoder** on the target column (`fraud_reported`)
- **OrdinalEncoder** on all remaining categorical feature columns

### 5. Class Imbalance Handling
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the `fraud_reported` classes

### 6. Feature Scaling
- Applied **StandardScaler** to normalize all feature values

### 7. Model Training & Evaluation
Eight classification models were trained and compared using accuracy score and cross-validation:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Support Vector Classifier (SVC) | RBF kernel |
| Decision Tree Classifier | `max_depth=15` |
| Random Forest Classifier | `max_depth=15` |
| K-Nearest Neighbors | `n_neighbors=15` |
| Extra Trees Classifier | ✅ **Best model selected** |
| XGBoost Classifier | — |
| LightGBM Classifier | — |

### 8. Hyperparameter Tuning
- Used **GridSearchCV (5-fold CV)** to tune the Extra Trees Classifier
- Best parameters found:
  - `criterion='gini'`, `max_depth=30`, `n_estimators=300`, `n_jobs=-2`, `random_state=42`

### 9. Final Evaluation
- **AUC-ROC Score:** ~97%
- Confusion matrix generated to visualize true/false positives and negatives

### 10. Model Saving
- Final model saved as `FinalModel_icf.pkl` using `joblib`

---

## 🧰 Dependencies

```bash
pip install pandas numpy seaborn matplotlib scipy missingno scikit-learn imbalanced-learn xgboost lightgbm joblib
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `missingno` | Missing value visualization |
| `scikit-learn` | Preprocessing, models, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost`, `lightgbm` | Gradient boosting models |
| `joblib` | Model serialization |

---

## 🚀 How to Run

1. Clone or download the repository
2. Install dependencies (see above)
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Insurance_Claim_Fraud_Detection_Project.ipynb
   ```
4. Run all cells in order — the dataset is loaded directly from GitHub, no local download needed

---

## 📊 Output

- Trained fraud detection model: `FinalModel_icf.pkl`
- Binary prediction: `1` = Fraudulent claim, `0` = Legitimate claim

---

## 📝 Notes

- The best random state for train-test splitting was identified through an iterative search across 1–1000
- Feature importance was extracted using a Random Forest to understand which features contribute most to fraud prediction
- SMOTE was essential to prevent model bias toward the majority (non-fraud) class
