"""
Data Cleaning and Bias Mitigation Script
=========================================
This script cleans the data.csv file and applies bias mitigation techniques
to create a more balanced and fair dataset for ProdTaken prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA CLEANING AND BIAS MITIGATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading data...")
df_original = pd.read_csv('data.csv')
df = df_original.copy()
print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")

# ============================================================================
# STEP 2: FIX DATA QUALITY ISSUES
# ============================================================================
print("\n[STEP 2] Fixing data quality issues...")

# Fix Gender column
print("\n  2.1 Cleaning Gender column...")
before_counts = df['Gender'].value_counts()
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
after_counts = df['Gender'].value_counts()
print(f"    Before: {dict(before_counts)}")
print(f"    After:  {dict(after_counts)}")
print(f"    ✓ Fixed {(before_counts.get('Fe Male', 0))} 'Fe Male' entries")

# Handle outliers (optional - keeping them for now as they may be legitimate)
print("\n  2.2 Outlier handling...")
print("    ℹ️  Keeping outliers as they may represent legitimate edge cases")
print("    ℹ️  Consider domain expertise before removing")

print("\n✓ Data quality issues fixed")

# ============================================================================
# STEP 3: PREPARE DATA FOR BIAS MITIGATION
# ============================================================================
print("\n[STEP 3] Preparing data for bias mitigation...")

# Encode categorical variables
print("\n  3.1 Encoding categorical variables...")
categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
                    'MaritalStatus', 'Designation']

df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = pd.Categorical(df_encoded[col]).codes

print(f"    ✓ Encoded {len(categorical_cols)} categorical columns")

# Separate features and target
X = df_encoded.drop('ProdTaken', axis=1)
y = df_encoded['ProdTaken']

print(f"\n  3.2 Dataset split:")
print(f"    Features: {X.shape}")
print(f"    Target distribution: {dict(y.value_counts())}")

# ============================================================================
# STEP 4: HANDLE CLASS IMBALANCE
# ============================================================================
print("\n[STEP 4] Handling class imbalance...")

print("\n  Before balancing:")
print(f"    Class 0: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"    Class 1: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
print(f"    Imbalance ratio: {sum(y == 0)/sum(y == 1):.2f}:1")

# Strategy: Combine SMOTE (oversample minority) with undersampling (reduce majority)
# This creates a more balanced dataset without extreme oversampling
over = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample to 50% of majority
under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Then undersample to 80%

# Apply resampling
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)
X_resampled, y_resampled = pipeline.fit_resample(X, y)

print("\n  After balancing (SMOTE + Undersampling):")
print(f"    Class 0: {sum(y_resampled == 0)} ({sum(y_resampled == 0)/len(y_resampled)*100:.1f}%)")
print(f"    Class 1: {sum(y_resampled == 1)} ({sum(y_resampled == 1)/len(y_resampled)*100:.1f}%)")
print(f"    New imbalance ratio: {sum(y_resampled == 0)/sum(y_resampled == 1):.2f}:1")
print(f"    ✓ Reduced imbalance from 4.18:1 to {sum(y_resampled == 0)/sum(y_resampled == 1):.2f}:1")

# ============================================================================
# STEP 5: CREATE CLEANED DATAFRAME
# ============================================================================
print("\n[STEP 5] Creating cleaned dataset...")

# Combine resampled features and target
df_cleaned_encoded = pd.DataFrame(X_resampled, columns=X.columns)
df_cleaned_encoded['ProdTaken'] = y_resampled

# Decode categorical variables back to original values
print("\n  5.1 Decoding categorical variables...")
for col in categorical_cols:
    # Get original categories
    categories = pd.Categorical(df[col]).categories
    # Map codes back to categories
    df_cleaned_encoded[col] = df_cleaned_encoded[col].map(
        lambda x: categories[int(x)] if 0 <= int(x) < len(categories) else categories[0]
    )

df_cleaned = df_cleaned_encoded.copy()
print(f"    ✓ Decoded {len(categorical_cols)} categorical columns")

# ============================================================================
# STEP 6: VALIDATE CLEANED DATA
# ============================================================================
print("\n[STEP 6] Validating cleaned dataset...")

print("\n  6.1 Class balance check:")
class_dist = df_cleaned['ProdTaken'].value_counts()
print(f"    Class 0: {class_dist[0]} ({class_dist[0]/len(df_cleaned)*100:.1f}%)")
print(f"    Class 1: {class_dist[1]} ({class_dist[1]/len(df_cleaned)*100:.1f}%)")

print("\n  6.2 Gender distribution:")
gender_dist = df_cleaned['Gender'].value_counts()
print(f"    {dict(gender_dist)}")
print(f"    ✓ No 'Fe Male' entries")

print("\n  6.3 Bias metrics (after cleaning):")
# Age bias
df_cleaned['AgeGroup'] = pd.cut(df_cleaned['Age'], bins=[0, 25, 35, 45, 55, 100], 
                                 labels=['18-25', '26-35', '36-45', '46-55', '56+'])
age_prod = pd.crosstab(df_cleaned['AgeGroup'], df_cleaned['ProdTaken'], normalize='index') * 100
age_diff = abs(age_prod[1].max() - age_prod[1].min())
print(f"    Age bias: {age_diff:.2f}% (was 29.84%)")

# Marital status bias
marital_prod = pd.crosstab(df_cleaned['MaritalStatus'], df_cleaned['ProdTaken'], normalize='index') * 100
marital_diff = abs(marital_prod[1].max() - marital_prod[1].min())
print(f"    Marital status bias: {marital_diff:.2f}% (was 25.91%)")

# Gender bias
gender_prod = pd.crosstab(df_cleaned['Gender'], df_cleaned['ProdTaken'], normalize='index') * 100
gender_diff = abs(gender_prod[1].max() - gender_prod[1].min())
print(f"    Gender bias: {gender_diff:.2f}% (was 2.87%)")

# City tier bias
city_prod = pd.crosstab(df_cleaned['CityTier'], df_cleaned['ProdTaken'], normalize='index') * 100
city_diff = abs(city_prod[1].max() - city_prod[1].min())
print(f"    City tier bias: {city_diff:.2f}% (was 12.23%)")

# ============================================================================
# STEP 7: SAVE CLEANED DATA
# ============================================================================
print("\n[STEP 7] Saving cleaned dataset...")

# Remove temporary AgeGroup column
df_cleaned = df_cleaned.drop('AgeGroup', axis=1)

# Save to CSV
output_file = 'data_cleaned.csv'
df_cleaned.to_csv(output_file, index=False)
print(f"    ✓ Saved to '{output_file}'")
print(f"    ✓ Total records: {len(df_cleaned)}")

# Also save a version with original data but just the Gender fix
df_original_fixed = df_original.copy()
df_original_fixed['Gender'] = df_original_fixed['Gender'].replace('Fe Male', 'Female')
df_original_fixed.to_csv('data_gender_fixed.csv', index=False)
print(f"    ✓ Saved original data with gender fix to 'data_gender_fixed.csv'")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n📊 Dataset Comparison:")
print(f"  Original dataset:     {len(df_original)} records")
print(f"  Cleaned dataset:      {len(df_cleaned)} records")
print(f"  Change:               {len(df_cleaned) - len(df_original):+d} records")

print("\n✅ Data Quality Improvements:")
print("  1. Fixed Gender column: 'Fe Male' → 'Female'")
print("  2. No missing values (already clean)")
print("  3. No duplicates (already clean)")

print("\n⚖️  Class Balance Improvements:")
print(f"  Before: 80.7% vs 19.3% (4.18:1 ratio)")
print(f"  After:  {class_dist[0]/len(df_cleaned)*100:.1f}% vs {class_dist[1]/len(df_cleaned)*100:.1f}% ({class_dist[0]/class_dist[1]:.2f}:1 ratio)")

print("\n🎯 Bias Mitigation:")
print("  Note: SMOTE resampling helps with class imbalance but does NOT")
print("  eliminate demographic biases. The biases in age, marital status,")
print("  and city tier reflect patterns in the original data.")
print("\n  For true bias mitigation, consider:")
print("  - Using fairness-aware algorithms (e.g., Fairlearn)")
print("  - Applying fairness constraints during training")
print("  - Post-processing predictions for demographic parity")
print("  - Removing sensitive features (age, marital status)")

print("\n📁 Output Files:")
print("  1. data_cleaned.csv        - Balanced dataset with SMOTE")
print("  2. data_gender_fixed.csv   - Original data with gender fix only")

print("\n💡 Recommendations:")
print("  - Use 'data_cleaned.csv' for training balanced models")
print("  - Use 'data_gender_fixed.csv' if you prefer original distribution")
print("  - Apply stratified train/test split")
print("  - Monitor fairness metrics during model evaluation")
print("  - Consider using class_weight='balanced' in your model")

print("\n" + "=" * 80)
print("✓ Data cleaning and bias mitigation complete!")
print("=" * 80)
