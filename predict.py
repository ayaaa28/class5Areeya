"""
Model Prediction Script
========================
Load the trained model and make predictions on new data.
"""

import pandas as pd
import pickle
import numpy as np

def load_model_and_preprocessors():
    """Load the trained model and all preprocessors."""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    return model, label_encoders, scaler, metadata

def preprocess_data(df, label_encoders, scaler, metadata):
    """Preprocess new data using saved encoders and scaler."""
    df_processed = df.copy()
    
    # Encode categorical variables
    for col in metadata['categorical_cols']:
        if col in df_processed.columns:
            df_processed[col] = label_encoders[col].transform(df_processed[col])
    
    # Scale numerical features
    numerical_cols = metadata['numerical_cols']
    df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed

def predict(data_path, output_path='predictions.csv'):
    """
    Make predictions on new data.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with new data
    output_path : str
        Path to save predictions
    
    Returns:
    --------
    DataFrame with predictions
    """
    print("=" * 80)
    print("PRODTAKEN PREDICTION")
    print("=" * 80)
    
    # Load model and preprocessors
    print("\n[1] Loading model and preprocessors...")
    model, label_encoders, scaler, metadata = load_model_and_preprocessors()
    print(f"  ✓ Loaded model: {metadata['model_name']}")
    
    # Load data
    print(f"\n[2] Loading data from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"  ✓ Loaded {len(df)} records")
    
    # Check if target exists (for evaluation mode)
    has_target = 'ProdTaken' in df.columns
    if has_target:
        y_true = df['ProdTaken']
        X = df.drop('ProdTaken', axis=1)
    else:
        X = df.copy()
    
    # Preprocess
    print("\n[3] Preprocessing data...")
    X_processed = preprocess_data(X, label_encoders, scaler, metadata)
    print("  ✓ Data preprocessed")
    
    # Predict
    print("\n[4] Making predictions...")
    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]
    
    # Create output dataframe
    output_df = df.copy()
    output_df['Predicted_ProdTaken'] = predictions
    output_df['Probability_ProdTaken'] = probabilities
    
    # Save predictions
    output_df.to_csv(output_path, index=False)
    print(f"  ✓ Saved predictions to '{output_path}'")
    
    # Print summary
    print("\n[5] Prediction Summary:")
    print(f"  Total records: {len(predictions)}")
    print(f"  Predicted 0 (Not Taken): {sum(predictions == 0)} ({sum(predictions == 0)/len(predictions)*100:.1f}%)")
    print(f"  Predicted 1 (Taken):     {sum(predictions == 1)} ({sum(predictions == 1)/len(predictions)*100:.1f}%)")
    
    # If we have ground truth, evaluate
    if has_target:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        auc = roc_auc_score(y_true, probabilities)
        
        print("\n[6] Evaluation Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Prediction complete!")
    print("=" * 80)
    
    return output_df

if __name__ == "__main__":
    # Example: Predict on test set
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    else:
        # Default: predict on test set
        data_path = 'test.csv'
        output_path = 'test_predictions.csv'
    
    predictions = predict(data_path, output_path)
    print(f"\nPredictions saved to: {output_path}")
