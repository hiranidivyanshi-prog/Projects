# 📉 Customer Churn Analysis

A machine learning project that identifies telecom customers at risk of churning, enabling proactive retention strategies before customers leave.

---

## 📌 Problem Statement

Customer churn — when customers stop doing business with a company — is a costly problem. Acquiring a new customer is roughly **five times more expensive** than retaining an existing one. This project builds a predictive model to flag high-risk customers early, allowing targeted retention efforts rather than broad, expensive campaigns.

---

## 📂 Dataset

- **Source:** [GitHub — dsrscientist/DSData](https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv)
- **Size:** ~7,032 rows × 21 columns (after cleaning)
- **Target Column:** `Churn` (Yes / No → encoded to 1 / 0)

### Feature Groups

| Category | Columns |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account Info | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone Services | `PhoneService`, `MultipleLines` |
| Internet Services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## ⚙️ Project Workflow

### 1. Data Collection & Inspection
- Loaded dataset directly from GitHub
- Reviewed shape, dtypes, column names, and missing value matrix
- Dropped `customerID` (unique identifier with no predictive value)

### 2. Data Cleaning
- Converted `TotalCharges` from object to numeric (coercing blanks to NaN)
- Found 11 rows with `tenure == 0` and missing `TotalCharges` — dropped them
- Mapped `SeniorCitizen` from binary integers to `"Yes"` / `"No"` for readability
- Confirmed no remaining nulls after cleaning

### 3. Exploratory Data Analysis (EDA)

Key findings from visualizations:

| Factor | Observation |
|---|---|
| Contract type | ~75% of month-to-month customers churn vs. 13% (1-year) and 3% (2-year) |
| Payment method | Electronic check users have the highest churn rate |
| Internet service | Fiber optic customers churn at a much higher rate than DSL users |
| Demographics | Customers without partners or dependents are more likely to churn |
| Senior citizens | Most senior citizens who use the service tend to churn |
| Add-on services | Absence of Online Security or Tech Support strongly correlates with churn |
| Billing | Paperless billing customers churn more frequently |
| Monthly charges | Higher monthly charges correlate with higher churn |
| Tenure | Newer customers are significantly more likely to churn |

### 4. Preprocessing Pipeline
- Applied **LabelEncoder** to all categorical object columns
- Split data into features `X` and target `y`
- Applied **SMOTE** to address class imbalance (74% No-churn vs. 26% Churn)
- Scaled `tenure`, `MonthlyCharges`, and `TotalCharges` using **StandardScaler**
- Train/test split: **67% train / 33% test** (`random_state=42`)

### 5. Feature Importance
- Used **Random Forest** to rank feature importance
- `tenure`, `MonthlyCharges`, and `TotalCharges` emerged as the most influential predictors

### 6. Model Training & Evaluation

Seven classification models were trained and compared:

| Model | Notes |
|---|---|
| K-Nearest Neighbors | `n_neighbors=11` |
| Support Vector Classifier | `random_state=1` |
| Random Forest Classifier | `n_estimators=500`, `max_leaf_nodes=30` — ✅ **Best individual model** |
| Logistic Regression | Baseline linear model |
| Decision Tree Classifier | Default parameters |
| AdaBoost Classifier | Ensemble boosting |
| Gradient Boosting Classifier | Ensemble boosting |
| **Voting Classifier** | Soft voting ensemble of GBC + LR + AdaBoost |

Each model was evaluated using accuracy score, classification report (precision, recall, F1), confusion matrix, and ROC curve (where applicable).

### 7. Hyperparameter Tuning
- Used **RandomizedSearchCV** (5-fold CV, 10 iterations) on the Random Forest Classifier
- Parameters searched: `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Final tuned model achieved consistent performance with the baseline Random Forest (~80% accuracy)

### 8. Model Saving
- Best model saved as `customer-churn-prediction.obj` using `joblib`
- Includes a final comparison DataFrame of original vs. predicted churn values

---

## 🧰 Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly missingno scikit-learn imbalanced-learn joblib
```

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Static visualizations |
| `plotly` | Interactive charts |
| `missingno` | Missing value visualization |
| `scikit-learn` | Models, preprocessing, metrics |
| `imbalanced-learn` | SMOTE oversampling |
| `joblib` | Model serialization |

---

## 🚀 How to Run

1. Install dependencies (see above)
2. Open the notebook:
   ```bash
   jupyter notebook Customer_Churn_Analysis_Project.ipynb
   ```
3. Run all cells — the dataset loads directly from GitHub, no local file needed

---

## 📊 Output

- Saved model: `customer-churn-prediction.obj`
- Binary prediction: `1` = Customer will churn, `0` = Customer will stay
- Final model accuracy: **~80%**

---

## 📝 Concluding Remarks

The Random Forest Classifier performed best among all models at ~80% accuracy. Key takeaways for reducing churn:

- Offer long-term contract incentives to month-to-month customers
- Promote Tech Support and Online Security add-ons proactively
- Investigate dissatisfaction among fiber optic service users
- Design retention campaigns targeting new customers within the first few months of tenure
