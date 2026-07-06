import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('loan_dataset_20000.csv')
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR / "loan_dataset_20000.csv"

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


#   Encoding
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
    
print("\nEncoded Dataset:")
print(df.head())

print("\nUpdated Data Types:")
print(df.dtypes)