import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('loan_dataset_700.csv')
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "loan_dataset_700.csv"

df = pd.read_csv(csv_path)

print(df.shape)
print(df.dtypes)
print('\n First 5 Rows')
print(df.head())
print()

#   Column Names
print(df.columns.to_list())

print('\nCheck missing values')
print(df.isnull().sum())

print('\nTarget Distribution')
print(df['loan_paid_back'].value_counts())
print(df['loan_paid_back'].value_counts(normalize=True) * 100)

#   Dataset Information
print('\nDataset Information')
print(df.info())

print('\nCategorical Columns')
categorical_columns = df.select_dtypes(include=['object', 'string']).columns
print(categorical_columns)


#   Loan Amount by Education
plt.figure(figsize=(8,5))
sns.boxplot(x='education_level',y='loan_amount',color="#0880D0",data=df)
plt.title("Loan Amount by Education")
plt.xticks(rotation=30)
plt.show()

#   Encoding
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
    
print("\nEncoded Dataset:")
print(df.head())

print("\nUpdated Data Types:")
print(df.dtypes)


#   Target Distribution

plt.figure(figsize=(6,4))
sns.countplot(x='loan_paid_back', data=df)
plt.title("Loan Paid Back Distribution")
plt.xlabel("Loan Paid Back")
plt.ylabel("Number of Applicants")
plt.show()


#   Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['annual_income'], bins=30, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.show()

#   Credit Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['credit_score'], bins=25, color="#970FC4",kde=True)
plt.title("Credit Score Distribution")
plt.show()


#   Correlation Heatmap
plt.figure(figsize=(12,9))
corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#   Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Features and Target
X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression(solver="liblinear",max_iter=5000,random_state=42)
log_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

print("All Models Trained Successfully!")


# POINT 5 : MODEL EVALUATION

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# Predictions
y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)


# ACCURACY
print("\n========== ACCURACY ==========")

print("Logistic Regression :", accuracy_score(y_test, y_pred_log))
print("Random Forest :", accuracy_score(y_test, y_pred_rf))
print("XGBoost :", accuracy_score(y_test, y_pred_xgb))

# PRECISION
print("\n========== PRECISION ==========")

print("Logistic Regression :", precision_score(y_test, y_pred_log))
print("Random Forest :", precision_score(y_test, y_pred_rf))
print("XGBoost :", precision_score(y_test, y_pred_xgb))

# RECALL

print("\n========== RECALL ==========")

print("Logistic Regression :", recall_score(y_test, y_pred_log))
print("Random Forest :", recall_score(y_test, y_pred_rf))
print("XGBoost :", recall_score(y_test, y_pred_xgb))

# F1 SCORE

print("\n========== F1 SCORE ==========")

print("Logistic Regression :", f1_score(y_test, y_pred_log))
print("Random Forest :", f1_score(y_test, y_pred_rf))
print("XGBoost :", f1_score(y_test, y_pred_xgb))

# CONFUSION MATRIX
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_log
)

plt.title("Logistic Regression")
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_rf
)

plt.title("Random Forest")
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb
)

plt.title("XGBoost")
plt.show()


#   Feature Importance
feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")

print(importance_df)

plt.figure(figsize=(12,6))
plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


#   Prediction of 5 new applicants
new_applicants = pd.DataFrame({

    "age":[30,45,27,50,35],
    "gender":[1,0,1,2,0],
    "marital_status":[0,1,1,0,1],
    "education_level":[2,1,0,2,1],
    "annual_income":[600000,900000,350000,1200000,500000],
    "monthly_income":[50000,75000,29000,100000,42000],
    "employment_status":[1,1,0,1,1],
    "debt_to_income_ratio":[0.30,0.22,0.48,0.18,0.35],
    "credit_score":[720,810,610,790,680],
    "loan_amount":[300000,500000,250000,650000,350000],
    "loan_purpose":[1,2,0,1,2],
    "interest_rate":[11.5,9.8,13.4,8.9,10.6],
    "loan_term":[36,60,24,48,36],
    "installment":[9800,10500,9500,13000,10000],
    "grade_subgrade":[3,1,5,2,4],
    "num_of_open_accounts":[4,7,3,8,5],
    "total_credit_limit":[900000,1400000,500000,1700000,800000],
    "current_balance":[200000,350000,180000,300000,220000],
    "delinquency_history":[0,0,1,0,0],
    "public_records":[0,0,0,0,0],
    "num_of_delinquencies":[0,1,2,0,1]
})

new_scaled = scaler.transform(new_applicants)
prediction = log_model.predict(new_scaled)
probability = log_model.predict_proba(new_scaled)

print("\n========== Prediction Results ==========")

for i in range(len(new_applicants)):
    print(f"\nApplicant {i+1}")
    print("Prediction :", prediction[i])
    print("Probability of Default :", round(probability[i][0],4))
    print("Probability of Loan Paid Back :", round(probability[i][1],4))