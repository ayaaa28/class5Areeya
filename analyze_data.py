import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the data
df = pd.read_csv('data.csv')

print("=" * 80)
print("DATA QUALITY AND BIAS ANALYSIS FOR PRODTAKEN PREDICTION")
print("=" * 80)

# 1. BASIC DATASET INFO
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# 2. MISSING VALUES ANALYSIS
print("\n\n2. MISSING VALUES ANALYSIS")
print("-" * 80)
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

if len(missing_df) > 0:
    print("⚠️  MISSING VALUES DETECTED:")
    print(missing_df)
else:
    print("✓ No missing values found")

# 3. DUPLICATE ROWS
print("\n\n3. DUPLICATE ROWS ANALYSIS")
print("-" * 80)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
if duplicates > 0:
    print(f"⚠️  WARNING: {duplicates} duplicate rows found ({duplicates/len(df)*100:.2f}%)")
else:
    print("✓ No duplicate rows found")

# 4. DATA INCONSISTENCIES
print("\n\n4. DATA INCONSISTENCIES")
print("-" * 80)

# Check Gender column for inconsistencies
print("\nGender values:")
gender_counts = df['Gender'].value_counts()
print(gender_counts)
if 'Fe Male' in df['Gender'].values:
    print("⚠️  WARNING: 'Fe Male' detected - should be 'Female'")

# Check for unusual values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df[col].unique()
    if len(unique_vals) < 20:  # Only show if reasonable number
        print(f"\n{col} unique values ({len(unique_vals)}): {sorted(unique_vals)}")

# 5. TARGET VARIABLE ANALYSIS (CLASS BALANCE)
print("\n\n5. TARGET VARIABLE ANALYSIS - CLASS BALANCE")
print("-" * 80)
target_counts = df['ProdTaken'].value_counts()
target_pct = df['ProdTaken'].value_counts(normalize=True) * 100

print(f"ProdTaken distribution:")
print(f"  0 (Not Taken): {target_counts[0]} ({target_pct[0]:.2f}%)")
print(f"  1 (Taken):     {target_counts[1]} ({target_pct[1]:.2f}%)")

imbalance_ratio = max(target_counts) / min(target_counts)
print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print(f"⚠️  WARNING: Significant class imbalance detected!")
    print(f"   This may lead to biased predictions favoring the majority class.")
else:
    print("✓ Classes are relatively balanced")

# 6. BIAS ANALYSIS BY PROTECTED ATTRIBUTES
print("\n\n6. BIAS ANALYSIS BY PROTECTED ATTRIBUTES")
print("-" * 80)

# Gender bias
print("\nGENDER BIAS ANALYSIS:")
gender_prod = pd.crosstab(df['Gender'], df['ProdTaken'], normalize='index') * 100
print(gender_prod)
gender_diff = abs(gender_prod[1].max() - gender_prod[1].min())
print(f"Max difference in acceptance rate: {gender_diff:.2f}%")
if gender_diff > 10:
    print(f"⚠️  WARNING: Potential gender bias detected (>{gender_diff:.1f}% difference)")

# Age bias
print("\n\nAGE BIAS ANALYSIS:")
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                         labels=['18-25', '26-35', '36-45', '46-55', '56+'])
age_prod = pd.crosstab(df['AgeGroup'], df['ProdTaken'], normalize='index') * 100
print(age_prod)
age_diff = abs(age_prod[1].max() - age_prod[1].min())
print(f"Max difference in acceptance rate across age groups: {age_diff:.2f}%")
if age_diff > 15:
    print(f"⚠️  WARNING: Potential age bias detected (>{age_diff:.1f}% difference)")

# Marital Status bias
print("\n\nMARITAL STATUS BIAS ANALYSIS:")
marital_prod = pd.crosstab(df['MaritalStatus'], df['ProdTaken'], normalize='index') * 100
print(marital_prod)
marital_diff = abs(marital_prod[1].max() - marital_prod[1].min())
print(f"Max difference in acceptance rate: {marital_diff:.2f}%")
if marital_diff > 10:
    print(f"⚠️  WARNING: Potential marital status bias detected (>{marital_diff:.1f}% difference)")

# City Tier bias
print("\n\nCITY TIER BIAS ANALYSIS:")
city_prod = pd.crosstab(df['CityTier'], df['ProdTaken'], normalize='index') * 100
print(city_prod)
city_diff = abs(city_prod[1].max() - city_prod[1].min())
print(f"Max difference in acceptance rate: {city_diff:.2f}%")
if city_diff > 10:
    print(f"⚠️  WARNING: Potential city tier bias detected (>{city_diff:.1f}% difference)")

# 7. OUTLIER DETECTION
print("\n\n7. OUTLIER DETECTION")
print("-" * 80)
numerical_cols = df.select_dtypes(include=[np.number]).columns
numerical_cols = [col for col in numerical_cols if col != 'ProdTaken']

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    
    if len(outliers) > 0:
        print(f"\n{col}:")
        print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print(f"  Range: [{df[col].min()}, {df[col].max()}]")
        print(f"  Expected range (IQR): [{lower_bound:.2f}, {upper_bound:.2f}]")

# 8. CORRELATION WITH TARGET
print("\n\n8. FEATURE CORRELATION WITH TARGET")
print("-" * 80)
# Only use numerical columns for correlation
numerical_df = df.select_dtypes(include=[np.number])
correlations = numerical_df.corr()['ProdTaken'].sort_values(ascending=False)
print(correlations)

# 9. SUMMARY AND RECOMMENDATIONS
print("\n\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

issues_found = []
recommendations = []

# Check for issues
if len(missing_df) > 0:
    issues_found.append(f"Missing values in {len(missing_df)} columns")
    recommendations.append("Impute or remove missing values before training")

if duplicates > 0:
    issues_found.append(f"{duplicates} duplicate rows")
    recommendations.append("Remove duplicate rows to avoid data leakage")

if 'Fe Male' in df['Gender'].values:
    issues_found.append("Data entry errors in Gender column ('Fe Male')")
    recommendations.append("Clean Gender column: replace 'Fe Male' with 'Female'")

if imbalance_ratio > 1.5:
    issues_found.append(f"Class imbalance (ratio: {imbalance_ratio:.2f}:1)")
    recommendations.append("Consider using SMOTE, class weights, or stratified sampling")

if gender_diff > 10:
    issues_found.append(f"Gender bias detected ({gender_diff:.1f}% difference)")
    recommendations.append("Monitor model fairness metrics across gender groups")

if age_diff > 15:
    issues_found.append(f"Age bias detected ({age_diff:.1f}% difference)")
    recommendations.append("Consider age-aware fairness constraints or post-processing")

if marital_diff > 10:
    issues_found.append(f"Marital status bias detected ({marital_diff:.1f}% difference)")
    recommendations.append("Evaluate if marital status should be included as a feature")

if city_diff > 10:
    issues_found.append(f"City tier bias detected ({city_diff:.1f}% difference)")
    recommendations.append("Consider geographic fairness in model evaluation")

print("\n⚠️  ISSUES FOUND:")
if issues_found:
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("  None - data appears clean!")

print("\n💡 RECOMMENDATIONS:")
if recommendations:
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
else:
    print("  No specific recommendations - proceed with standard ML practices")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
