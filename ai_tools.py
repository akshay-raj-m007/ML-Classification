import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    log_loss
)
from xgboost import XGBClassifier


# Load Data
data = pd.read_csv("heart_failure_clinical_records.csv")
print(data.head())

print("*"*70)
print("Shape:", data.shape)
print("*"*70)
print("Columns:", data.columns)
print("*"*70)
print("Data types:\n", data.dtypes)
print("*"*70)
print("Null values:\n", data.isnull().sum())
print("*"*70)
print("Duplicate rows:", data.duplicated().sum())

data = data.drop_duplicates()

print("*"*70)
print("Descriptive Statistics:\n", data.describe())

print("*"*70)
print("Class Distribution (%):")
print(data['DEATH_EVENT'].value_counts(normalize=True) * 100)


# EDA Plots
plt.figure()
sns.histplot(data['age'], kde=True)
plt.title("Age Distribution")
plt.show()

plt.figure()
sns.histplot(data['ejection_fraction'], kde=True)
plt.title("Ejection Fraction Distribution")
plt.show()

plt.figure()
sns.histplot(data['serum_creatinine'], kde=True)
plt.title("Serum Creatinine Distribution")
plt.show()


# Correlation
corr = data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Train Test Split
X = data.drop(columns=['DEATH_EVENT', 'time'])
y = data['DEATH_EVENT']

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Logistic Regression
print("\n" + "="*50)
print("LOGISTIC REGRESSION")

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(x_train_scaled, y_train)

y_pred_lr = lr.predict(x_test_scaled)
y_prob_lr = lr.predict_proba(x_test_scaled)[:,1]

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_lr)
print(cm)
ConfusionMatrixDisplay(cm).plot()
plt.title("Logistic Regression CM")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("Log Loss:", log_loss(y_test, y_prob_lr))


# Random Forest
print("\n" + "="*50)
print("RANDOM FOREST")

rf = RandomForestClassifier(n_estimators=200, random_state=80)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
y_prob_rf = rf.predict_proba(x_test)[:,1]

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rf)  
print(cm)
ConfusionMatrixDisplay(cm).plot()
plt.title("Random Forest CM")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("Log Loss:", log_loss(y_test, y_prob_rf))


# Gradient Boosting
print("\n" + "="*50)
print("GRADIENT BOOSTING")

gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=80
)

gb.fit(x_train, y_train)

y_pred_gb = gb.predict(x_test)
y_prob_gb = gb.predict_proba(x_test)[:,1]

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_gb)
print(cm)
ConfusionMatrixDisplay(cm).plot()
plt.title("Gradient Boosting CM")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_gb))
print("Train Log Loss:", log_loss(y_train, gb.predict_proba(x_train)[:,1]))
print("Test Log Loss:", log_loss(y_test, y_prob_gb))


# XGBoost
print("\n" + "="*50)
print("XGBOOST")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=80,
    eval_metric='logloss'
)

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)
y_prob_xgb = xgb.predict_proba(x_test)[:,1]

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_xgb)
print(cm)
ConfusionMatrixDisplay(cm).plot()
plt.title("XGBoost CM")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("Train Log Loss:", log_loss(y_train, xgb.predict_proba(x_train)[:,1]))
print("Test Log Loss:", log_loss(y_test, y_prob_xgb))