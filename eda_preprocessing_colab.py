# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Step 2: Upload the dataset
print("Uploading dataset file...")
uploaded = files.upload()

# Step 3: Load dataset
print("\nLoading dataset into DataFrame...")
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Step 4: Inspect dataset
print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nDataset info (column names, types, non-null counts):")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nStatistical summary of numeric columns:")
print(df.describe())

# Step 5: Remove duplicates
print("\nRemoving duplicate rows...")
df = df.drop_duplicates()
print("Dataset shape after removing duplicates:", df.shape)

# Step 6: Remove invalid values
print("\nFiltering invalid values (age > 0, bmi > 0)...")
df = df[df['age'] > 0]
df = df[df['bmi'] > 0]
print("Dataset shape after filtering invalid values:", df.shape)

# Step 7: Inspect unique values in smoking_history
print("\nUnique values in smoking_history column:")
unique_smoking = df['smoking_history'].unique()
print(unique_smoking)

print("\nValue counts for smoking_history:")
print(df['smoking_history'].value_counts())

# Step 8: Build mapping dynamically for smoking_history
smoking_map = {val: i for i, val in enumerate(unique_smoking)}
print("\nGenerated smoking map:", smoking_map)
df['smoking_history'] = df['smoking_history'].map(smoking_map)

# Step 9: Encode gender dynamically
print("\nUnique values in gender column:")
unique_gender = df['gender'].unique()
print(unique_gender)

gender_map = {val: i for i, val in enumerate(unique_gender)}
print("Generated gender map:", gender_map)
df['gender'] = df['gender'].map(gender_map)

# Step 10: Save cleaned dataset
print("\nSaving cleaned dataset as Excel file...")
df.to_excel("diabetes_cleaned.xlsx", index=False)
print("Cleaned dataset saved as diabetes_cleaned.xlsx")

# Step 11: Histograms and Bar Charts
print("\nPlotting Age Distribution (Histogram)...")
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

print("\nPlotting BMI Distribution (Histogram)...")
plt.figure(figsize=(6,4))
sns.histplot(df['bmi'], bins=30, kde=True)
plt.title("BMI Distribution")
plt.show()

print("\nPlotting HbA1c Level Distribution (Histogram)...")
plt.figure(figsize=(6,4))
sns.histplot(df['HbA1c_level'], bins=30, kde=True)
plt.title("HbA1c Level Distribution")
plt.show()

print("\nPlotting Blood Glucose Level Distribution (Histogram)...")
plt.figure(figsize=(6,4))
sns.histplot(df['blood_glucose_level'], bins=30, kde=True)
plt.title("Blood Glucose Level Distribution")
plt.show()

print("\nPlotting Smoking History Distribution (Bar Chart)...")
plt.figure(figsize=(6,4))
df['smoking_history'].value_counts().plot(kind='bar')
plt.title("Smoking History Distribution")
plt.show()

print("\nPlotting Gender Distribution (Bar Chart)...")
plt.figure(figsize=(6,4))
df['gender'].value_counts().plot(kind='bar')
plt.title("Gender Distribution")
plt.show()

# Step 12: Download cleaned dataset
print("\nDownloading cleaned dataset to your computer...")
files.download("diabetes_cleaned.xlsx")
