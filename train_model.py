"""
ProdTaken Prediction Model Training
====================================
Trains multiple models to predict ProdTaken using train.csv and val.csv
Includes preprocessing, training, evaluation, and model comparison.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PRODTAKEN PREDICTION MODEL TRAINING")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading data...")

train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

print(f"✓ Train set: {len(train_df)} records")
print(f"✓ Validation set: {len(val_df)} records")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Preprocessing data...")

# Separate features and target
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']

X_val = val_df.drop('ProdTaken', axis=1)
y_val = val_df['ProdTaken']

print(f"\n  Features: {list(X_train.columns)}")
print(f"  Target: ProdTaken")

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n  Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
print("\n  2.1 Encoding categorical variables...")
label_encoders = {}
X_train_encoded = X_train.copy()
X_val_encoded = X_val.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_train_encoded[col] = le.fit_transform(X_train[col])
    X_val_encoded[col] = le.transform(X_val[col])
    label_encoders[col] = le

print(f"    ✓ Encoded {len(categorical_cols)} categorical columns")

# Scale numerical features
print("\n  2.2 Scaling numerical features...")
scaler = StandardScaler()
X_train_encoded[numerical_cols] = scaler.fit_transform(X_train_encoded[numerical_cols])
X_val_encoded[numerical_cols] = scaler.transform(X_val_encoded[numerical_cols])

print(f"    ✓ Scaled {len(numerical_cols)} numerical columns")

# ============================================================================
# STEP 3: TRAIN MULTIPLE MODELS
# ============================================================================
print("\n[STEP 3] Training multiple models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train
    model.fit(X_train_encoded, y_train)
    trained_models[name] = model
    
    # Predict
    y_train_pred = model.predict(X_train_encoded)
    y_val_pred = model.predict(X_val_encoded)
    
    # Get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_val_proba = model.predict_proba(X_val_encoded)[:, 1]
    else:
        y_val_proba = y_val_pred
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Val Accuracy': val_acc,
        'Precision': val_precision,
        'Recall': val_recall,
        'F1-Score': val_f1,
        'ROC-AUC': val_auc
    })
    
    print(f"    Train Accuracy: {train_acc:.4f}")
    print(f"    Val Accuracy:   {val_acc:.4f}")
    print(f"    F1-Score:       {val_f1:.4f}")
    print(f"    ROC-AUC:        {val_auc:.4f}")

# ============================================================================
# STEP 4: MODEL COMPARISON
# ============================================================================
print("\n[STEP 4] Model comparison...")

results_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

# Find best model
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_model = trained_models[best_model_name]

print(f"\n✓ Best model (by F1-Score): {best_model_name}")

# ============================================================================
# STEP 5: DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print(f"\n[STEP 5] Detailed evaluation of {best_model_name}...")

y_val_pred = best_model.predict(X_val_encoded)
y_val_proba = best_model.predict_proba(X_val_encoded)[:, 1] if hasattr(best_model, 'predict_proba') else y_val_pred

# Confusion Matrix
print("\n  5.1 Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(f"\n{cm}")
print(f"\n    True Negatives:  {cm[0,0]}")
print(f"    False Positives: {cm[0,1]}")
print(f"    False Negatives: {cm[1,0]}")
print(f"    True Positives:  {cm[1,1]}")

# Classification Report
print("\n  5.2 Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Not Taken', 'Taken']))

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    print("\n  5.3 Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'Feature': X_train_encoded.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# STEP 6: FAIRNESS EVALUATION
# ============================================================================
print("\n[STEP 6] Fairness evaluation...")

# Add predictions to validation set
val_with_pred = val_df.copy()
val_with_pred['Predicted'] = y_val_pred
val_with_pred['Actual'] = y_val

# Gender fairness
print("\n  6.1 Gender Fairness:")
gender_metrics = val_with_pred.groupby('Gender').agg({
    'Predicted': 'mean',
    'Actual': 'mean'
})
gender_metrics.columns = ['Predicted Rate', 'Actual Rate']
print(gender_metrics)
gender_diff = abs(gender_metrics['Predicted Rate'].max() - gender_metrics['Predicted Rate'].min())
print(f"    Prediction rate difference: {gender_diff:.4f}")
if gender_diff < 0.05:
    print("    ✓ Low gender bias")
elif gender_diff < 0.10:
    print("    ⚠️  Moderate gender bias")
else:
    print("    ⚠️  High gender bias")

# Age group fairness
print("\n  6.2 Age Group Fairness:")
val_with_pred['AgeGroup'] = pd.cut(val_with_pred['Age'], 
                                     bins=[0, 30, 45, 100], 
                                     labels=['Young', 'Middle', 'Senior'])
age_metrics = val_with_pred.groupby('AgeGroup').agg({
    'Predicted': 'mean',
    'Actual': 'mean'
})
age_metrics.columns = ['Predicted Rate', 'Actual Rate']
print(age_metrics)
age_diff = abs(age_metrics['Predicted Rate'].max() - age_metrics['Predicted Rate'].min())
print(f"    Prediction rate difference: {age_diff:.4f}")
if age_diff < 0.10:
    print("    ✓ Low age bias")
elif age_diff < 0.20:
    print("    ⚠️  Moderate age bias")
else:
    print("    ⚠️  High age bias")

# ============================================================================
# STEP 7: SAVE BEST MODEL
# ============================================================================
print("\n[STEP 7] Saving best model...")

import pickle

# Save model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✓ Saved model to 'best_model.pkl'")

# Save preprocessors
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"  ✓ Saved label encoders to 'label_encoders.pkl'")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Saved scaler to 'scaler.pkl'")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'features': list(X_train.columns),
    'metrics': results_df[results_df['Model'] == best_model_name].to_dict('records')[0]
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"  ✓ Saved metadata to 'model_metadata.pkl'")

# ============================================================================
# STEP 8: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"\n📊 Models Trained: {len(models)}")
for name in models.keys():
    print(f"  - {name}")

print(f"\n🏆 Best Model: {best_model_name}")
best_metrics = results_df[results_df['Model'] == best_model_name].iloc[0]
print(f"  Validation Accuracy: {best_metrics['Val Accuracy']:.4f}")
print(f"  Precision:           {best_metrics['Precision']:.4f}")
print(f"  Recall:              {best_metrics['Recall']:.4f}")
print(f"  F1-Score:            {best_metrics['F1-Score']:.4f}")
print(f"  ROC-AUC:             {best_metrics['ROC-AUC']:.4f}")

print(f"\n⚖️  Fairness Metrics:")
print(f"  Gender bias:    {gender_diff:.4f}")
print(f"  Age group bias: {age_diff:.4f}")

print(f"\n📁 Saved Files:")
print(f"  - best_model.pkl        (trained model)")
print(f"  - label_encoders.pkl    (categorical encoders)")
print(f"  - scaler.pkl            (numerical scaler)")
print(f"  - model_metadata.pkl    (model information)")

print(f"\n💡 Next Steps:")
print(f"  1. Evaluate on test.csv for final performance")
print(f"  2. Monitor model predictions for bias")
print(f"  3. Consider ensemble methods for better performance")
print(f"  4. Implement fairness constraints if bias is unacceptable")

print("\n" + "=" * 80)
print("✓ Training complete!")
print("=" * 80)
