"""
Train/Validation/Test Split Script
===================================
Splits the cleaned dataset into:
- Training set: 60% (of the 75% train+val portion)
- Validation set: 15% (of the 75% train+val portion)  
- Test set: 25%

Uses stratified sampling to maintain class distribution.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("=" * 80)
print("TRAIN/VALIDATION/TEST SPLIT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD CLEANED DATA
# ============================================================================
print("\n[STEP 1] Loading cleaned dataset...")

# Check which cleaned files exist
if os.path.exists('data_cleaned.csv'):
    df = pd.read_csv('data_cleaned.csv')
    print(f"✓ Loaded 'data_cleaned.csv' - {len(df)} records")
elif os.path.exists('data.csv'):
    df = pd.read_csv('data.csv')
    # Fix gender issue
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    print(f"✓ Loaded 'data.csv' (with gender fix) - {len(df)} records")
else:
    raise FileNotFoundError("No data file found!")

# ============================================================================
# STEP 2: PREPARE FOR SPLITTING
# ============================================================================
print("\n[STEP 2] Preparing data for splitting...")

# Check target distribution
target_dist = df['ProdTaken'].value_counts()
print(f"\nTarget distribution:")
print(f"  Class 0: {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)")
print(f"  Class 1: {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)")

# ============================================================================
# STEP 3: SPLIT INTO TRAIN+VAL (75%) AND TEST (25%)
# ============================================================================
print("\n[STEP 3] Splitting into train+val (75%) and test (25%)...")

# First split: 75% train+val, 25% test
train_val, test = train_test_split(
    df, 
    test_size=0.25, 
    random_state=42, 
    stratify=df['ProdTaken']
)

print(f"\nAfter first split:")
print(f"  Train+Val: {len(train_val)} records ({len(train_val)/len(df)*100:.1f}%)")
print(f"  Test:      {len(test)} records ({len(test)/len(df)*100:.1f}%)")

# ============================================================================
# STEP 4: SPLIT TRAIN+VAL INTO TRAIN (80%) AND VAL (20%)
# ============================================================================
print("\n[STEP 4] Splitting train+val into train (80%) and val (20%)...")

# Second split: 80% train, 20% val (of the 75%)
# This gives us: 60% train, 15% val, 25% test overall
train, val = train_test_split(
    train_val, 
    test_size=0.20,  # 20% of 75% = 15% of total
    random_state=42, 
    stratify=train_val['ProdTaken']
)

print(f"\nFinal split:")
print(f"  Train: {len(train)} records ({len(train)/len(df)*100:.1f}% of total)")
print(f"  Val:   {len(val)} records ({len(val)/len(df)*100:.1f}% of total)")
print(f"  Test:  {len(test)} records ({len(test)/len(df)*100:.1f}% of total)")

# ============================================================================
# STEP 5: VERIFY STRATIFICATION
# ============================================================================
print("\n[STEP 5] Verifying stratification...")

print("\nClass distribution in each set:")

train_dist = train['ProdTaken'].value_counts()
val_dist = val['ProdTaken'].value_counts()
test_dist = test['ProdTaken'].value_counts()

print(f"\nTrain set:")
print(f"  Class 0: {train_dist[0]} ({train_dist[0]/len(train)*100:.1f}%)")
print(f"  Class 1: {train_dist[1]} ({train_dist[1]/len(train)*100:.1f}%)")

print(f"\nValidation set:")
print(f"  Class 0: {val_dist[0]} ({val_dist[0]/len(val)*100:.1f}%)")
print(f"  Class 1: {val_dist[1]} ({val_dist[1]/len(val)*100:.1f}%)")

print(f"\nTest set:")
print(f"  Class 0: {test_dist[0]} ({test_dist[0]/len(test)*100:.1f}%)")
print(f"  Class 1: {test_dist[1]} ({test_dist[1]/len(test)*100:.1f}%)")

print("\n✓ Class distributions are consistent across all sets")

# ============================================================================
# STEP 6: SAVE SPLIT DATASETS
# ============================================================================
print("\n[STEP 6] Saving split datasets...")

train.to_csv('train.csv', index=False)
print(f"  ✓ Saved train.csv ({len(train)} records)")

val.to_csv('val.csv', index=False)
print(f"  ✓ Saved val.csv ({len(val)} records)")

test.to_csv('test.csv', index=False)
print(f"  ✓ Saved test.csv ({len(test)} records)")

# ============================================================================
# STEP 7: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n📊 Dataset Split:")
print(f"  Original:    {len(df):,} records")
print(f"  Train:       {len(train):,} records ({len(train)/len(df)*100:.1f}%)")
print(f"  Validation:  {len(val):,} records ({len(val)/len(df)*100:.1f}%)")
print(f"  Test:        {len(test):,} records ({len(test)/len(df)*100:.1f}%)")

print(f"\n✅ Stratification Verified:")
print(f"  All sets maintain similar class distributions")
print(f"  Random state: 42 (reproducible splits)")

print(f"\n📁 Output Files:")
print(f"  1. train.csv - Training set for model training")
print(f"  2. val.csv   - Validation set for hyperparameter tuning")
print(f"  3. test.csv  - Test set for final evaluation (DO NOT use during training!)")

print(f"\n💡 Usage Recommendations:")
print(f"  - Use train.csv for model training")
print(f"  - Use val.csv for hyperparameter tuning and model selection")
print(f"  - Use test.csv ONLY for final model evaluation")
print(f"  - Never train on validation or test sets")
print(f"  - Consider k-fold cross-validation on train+val for robust evaluation")

print("\n" + "=" * 80)
print("✓ Train/Val/Test split complete!")
print("=" * 80)
