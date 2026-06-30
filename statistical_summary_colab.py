# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Step 2: Upload the cleaned dataset
# We always work on the CLEANED dataset, not raw, because raw data may contain duplicates,
# invalid values, or unencoded categories. Cleaned dataset ensures accurate statistical calculations.
print("Uploading cleaned dataset file...")
uploaded = files.upload()

# Step 3: Load the cleaned dataset (CSV format)
print("\nLoading cleaned dataset into DataFrame...")
df = pd.read_csv("diabetes_cleaned.csv")

# Step 4: Define statistical functions
# Mean, Median, Mode, and IQR are calculated to understand central tendency and spread.

def mean(col):
    # Formula: mean = sum(values) / count(values)
    return sum(col) / len(col)

def median(col):
    # Median requires sorting the list first
    sorted_col = sorted(col)
    n = len(sorted_col)
    if n % 2 == 0:
        return (sorted_col[n//2 - 1] + sorted_col[n//2]) / 2
    else:
        return sorted_col[n//2]

def mode(col):
    # Mode: most frequent value
    freq = {}
    for val in col:
        freq[val] = freq.get(val, 0) + 1
    max_count = max(freq.values())
    modes = [k for k,v in freq.items() if v == max_count]
    return modes

def iqr(col):
    # IQR = Q3 - Q1
    sorted_col = sorted(col)
    n = len(sorted_col)
    q1 = sorted_col[n//4]
    q3 = sorted_col[(3*n)//4]
    return q3 - q1

# Step 5: Apply statistical functions
print("\nCalculating mean, median, mode, and IQR for numeric columns...")
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    arr = df[col].tolist()
    print(f"{col} -> Mean: {mean(arr):.4f}, Median: {median(arr)}, Mode: {mode(arr)}, IQR: {iqr(arr)}")

# Step 6: Min-Max Scaling
# Formula: (x - min) / (max - min)
print("\nApplying Min-Max scaling to numeric columns...")
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    df[col+'_minmax'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    print("Min-Max scaling applied to:", col)

# Step 7: Z-Score Scaling
# Formula: (x - mean) / std
print("\nApplying Z-Score scaling to numeric columns...")
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    m = mean(df[col].tolist())
    std = np.sqrt(sum((x-m)**2 for x in df[col]) / len(df[col]))
    df[col+'_zscore'] = [(x-m)/std for x in df[col]]
    print("Z-Score scaling applied to:", col)

# Step 8: Log Scaling
# Formula: log(1+x)
print("\nApplying Log scaling to numeric columns...")
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    df[col+'_log'] = np.log1p(df[col])
    print("Log scaling applied to:", col)

# Step 9: Binning Age
# pd.cut splits continuous values into bins with labels
print("\nCreating age bins...")
df['age_bins'] = pd.cut(df['age'], bins=[0,20,40,60,80,100], labels=['0-20','21-40','41-60','61-80','81-100'])
print("Age bins created.")

# Step 10: Save final processed dataset
print("\nSaving final processed dataset as CSV file...")
df.to_csv("diabetes_final_processed.csv", index=False)
print("Final dataset saved as diabetes_final_processed.csv")

# Step 11: Visualizations
print("\nPlotting Boxplots of numeric columns...")
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['age','bmi','HbA1c_level','blood_glucose_level']])
plt.title("Boxplots of Numeric Columns")
plt.show()

print("\nPlotting Age Bins Distribution (Bar Chart)...")
plt.figure(figsize=(6,4))
df['age_bins'].value_counts().plot(kind='bar')
plt.title("Age Bins Distribution")
plt.show()

# Step 12: Download final dataset
print("\nDownloading final processed dataset to your computer...")
files.download("diabetes_final_processed.csv")
