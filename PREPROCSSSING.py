import pandas as pd
import numpy as np

df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Dataset shape:", df.shape)
print(df.info())
print(df.head())

print("Missing values per column:\n", df.isnull().sum())
df = df.dropna()

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

df = df[df['age'] > 0]
df = df[df['bmi'] > 0]

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['smoking_history'] = df['smoking_history'].map({
    'never': 0,
    'former': 1,
    'current': 2,
    'not known': 3
})

for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

df.to_csv("diabetes_cleaned.csv", index=False)
print("Cleaned dataset saved as diabetes_cleaned.csv")
