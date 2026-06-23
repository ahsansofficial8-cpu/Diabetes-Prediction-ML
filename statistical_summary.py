import pandas as pd
import numpy as np

df = pd.read_csv("diabetes_cleaned.csv")

# === Mean / Median / Mode / IQR ===
def mean(col):
    return sum(col) / len(col)

def median(col):
    sorted_col = sorted(col)
    n = len(sorted_col)
    if n % 2 == 0:
        return (sorted_col[n//2 - 1] + sorted_col[n//2]) / 2
    else:
        return sorted_col[n//2]

def mode(col):
    freq = {}
    for val in col:
        freq[val] = freq.get(val, 0) + 1
    max_count = max(freq.values())
    modes = [k for k,v in freq.items() if v == max_count]
    return modes

def iqr(col):
    sorted_col = sorted(col)
    n = len(sorted_col)
    q1 = sorted_col[n//4]
    q3 = sorted_col[(3*n)//4]
    return q3 - q1

for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    arr = df[col].tolist()
    print(f"{col} → Mean: {mean(arr):.4f}, Median: {median(arr)}, Mode: {mode(arr)}, IQR: {iqr(arr)}")

# === Min-Max Scaling ===
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    df[col+'_minmax'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# === Z-Score Scaling ===
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    m = mean(df[col].tolist())
    std = np.sqrt(sum((x-m)**2 for x in df[col]) / len(df[col]))
    df[col+'_zscore'] = [(x-m)/std for x in df[col]]

# === Log Scaling ===
for col in ['age','bmi','HbA1c_level','blood_glucose_level']:
    df[col+'_log'] = np.log1p(df[col])

# === Binning Age ===
df['age_bins'] = pd.cut(df['age'], bins=[0,20,40,60,80,100], labels=['0-20','21-40','41-60','61-80','81-100'])

df.to_csv("diabetes_final_processed.csv", index=False)
print("Final dataset saved as diabetes_final_processed.csv")