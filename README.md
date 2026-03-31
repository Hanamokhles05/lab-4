# lab-4
data preprocessing
Wrong Data Types: The Date column is a string (Object) instead of DateTime, and Amount was initially a string due to currency symbols.

Potential Outliers: There are very high values in Amount or Boxes Shipped that might skew the analysis.

Inconsistent Formats: Currency symbols ($) and commas in numerical columns prevent mathematical operations.

Task 2: Apply one missing value strategy
Since the dataset might be clean, we can simulate a missing value and then fill it using the Mean Imputation strategy.

Python
# Strategy: Mean Imputation
# Why: We use the mean for numerical data (Amount) to maintain the overall 
# distribution without losing rows (which would happen if we deleted them).

df['Amount'] = df['Amount'].fillna(df['Amount'].mean())
Task 3: Detect and handle outliers using IQR
The Interquartile Range (IQR) helps remove extreme values.

Python
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering the outliers
df_cleaned = df[(df['Amount'] >= lower_bound) & (df['Amount'] <= upper_bound)]
print(f"Removed {len(df) - len(df_cleaned)} outliers.")
Task 4: Normalize numerical features
You need to apply both Min-Max (scales to 0-1) and Z-score (scales to mean=0, std=1).

Python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Min-Max Normalization
min_max_scaler = MinMaxScaler()
df_cleaned['Amount_MinMax'] = min_max_scaler.fit_transform(df_cleaned[['Amount']])

# 2. Z-score Normalization
std_scaler = StandardScaler()
df_cleaned['Amount_Zscore'] = std_scaler.fit_transform(df_cleaned[['Amount']])
Task 5: Apply PCA (Principal Component Analysis)
Apply PCA only if Amount and Boxes Shipped are correlated.

Python
from sklearn.decomposition import PCA

# Check correlation first
correlation = df_cleaned[['Amount', 'Boxes Shipped']].corr()
print("Correlation:\n", correlation)

# If correlation is high, apply PCA
if abs(correlation.iloc[0,1]) > 0.5:
    pca = PCA(n_components=1)
    # Ensure data is standardized before PCA
    features_scaled = std_scaler.fit_transform(df_cleaned[['Amount', 'Boxes Shipped']])
    pca_result = pca.fit_transform(features_scaled)
    print("PCA completed.")
else:
    print("Correlation is low; PCA might not be necessary but can still be applied.")
