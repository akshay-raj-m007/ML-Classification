# Heart Failure Prediction using Classical and Ensemble Machine Learning Models

## 📌 Project Overview

This project analyzes a clinical heart failure dataset to predict patient mortality using multiple machine learning models. The goal is to compare linear and ensemble methods while following proper ML practices such as:

- Exploratory Data Analysis (EDA)
- Duplicate detection and removal
- Data leakage prevention
- Stratified train-test splitting
- Model comparison using multiple evaluation metrics

---

## 📊 Dataset Summary

- Initial rows: **5000**
- Unique rows after duplicate removal: **1320**
- Duplicate rows removed: **3680 (73.6%)**
- Target variable: `DEATH_EVENT` (1 = death, 0 = survived)

### Key Observations from EDA

- No significant missing values.
- Moderate class imbalance (~69% survived, ~31% death).
- Strong correlations with death:
  - Serum Creatinine (+0.29)
  - Age (+0.22)
  - Ejection Fraction (-0.27)
  - Serum Sodium (-0.25)
- `time` variable showed strong correlation (-0.50) but was removed to prevent data leakage.

---

## 🔧 Preprocessing Steps

- Removed duplicate rows.
- Dropped `time` feature to avoid leakage.
- Performed **80-20 stratified train-test split**.
- Applied **StandardScaler only for Logistic Regression**.
- Tree-based models used raw features.

---

## 🤖 Models Implemented

1. Logistic Regression  
2. Random Forest  
3. Gradient Boosting  
4. XGBoost  

Evaluation metrics used:

- Log Loss
- Accuracy
- Recall (for death class)
- ROC-AUC

---

## 📈 Model Performance Comparison

| Model | Train Log Loss | Test Log Loss | Accuracy | Recall (Death) | ROC-AUC |
|--------|---------------|--------------|----------|---------------|----------|
| Logistic Regression | 0.493 | 0.454 | 0.80 | 0.51 | 0.857 |
| Random Forest | 0.066 | 0.197 | 0.94 | 0.85 | 0.985 |
| Gradient Boosting | 0.134 | 0.192 | 0.94 | 0.84 | 0.983 |
| XGBoost | 0.122 | 0.201 | 0.92 | 0.81 | 0.979 |

---

## 🧠 Key Insights

- Removing duplicates was critical — before removal, models showed unrealistically high performance.
- Logistic Regression underfit the dataset due to linear assumptions.
- Tree-based ensemble models significantly outperformed linear models.
- Random Forest achieved the highest ROC-AUC and recall.
- XGBoost demonstrated strong regularization and balanced bias–variance tradeoff.

---

## 📌 Bias–Variance Perspective

- **Logistic Regression** → High bias, low variance  
- **Random Forest** → Low bias, moderate variance  
- **Gradient Boosting** → Balanced  
- **XGBoost** → Regularized boosting with strong generalization  

---

## 🚀 Future Improvements

- K-fold cross-validation
- Hyperparameter tuning
- Probability calibration
- SHAP-based interpretability
- Threshold optimization for medical risk sensitivity

---

## 📂 Repository Structure
