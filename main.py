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