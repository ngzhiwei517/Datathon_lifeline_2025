"""
Training script for XGBoost fetal distress detection model
Loads data, performs feature engineering, trains model, and saves weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import pickle

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,
    'min_child_weight': 3,
    'random_state': RANDOM_STATE,
    'eval_metric': 'mlogloss',
    'use_label_encoder': False
}

def load_and_clean_data(filepath):
    """Load and clean CTG dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records")
    
    # Drop duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} records")
    
    # Drop redundant label columns
    label_cols = ['A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS']
    df = df.drop(columns=[col for col in label_cols if col in df.columns])
    
    return df

def engineer_features(df):
    """Create engineered features"""
    print("Engineering features...")
    df = df.copy()
    
    # Interaction features
    df['AC_per_UC'] = df['AC'] / (df['UC'] + 1)
    df['total_abnormal_var'] = df['ASTV'] + df['ALTV']
    df['ASTV_ALTV_ratio'] = df['ASTV'] / (df['ALTV'] + 1)
    
    # Deceleration features
    df['decel_severity'] = (df['DP'] * 3) + (df['DS'] * 2) + df['DL']
    df['total_decels'] = df['DP'] + df['DS'] + df['DL']
    
    # Binary flags
    df['has_prolonged_decel'] = (df['DP'] > 0).astype(int)
    df['has_severe_decel'] = (df['DS'] > 0).astype(int)
    df['has_movement'] = (df['FM'] > 0).astype(int)
    
    # Heart rate stability
    df['heart_rate_range'] = df['Max'] - df['Min']
    
    print(f"Features created. Total features: {len(df.columns)}")
    return df

def main():
    print("="*70)
    print("TRAINING XGBOOST FETAL DISTRESS DETECTION MODEL")
    print("="*70)
    
    # Load data
    df = load_and_clean_data('CTG.csv')
    
    # Engineer features
    df = engineer_features(df)
    
    # Split features and target
    X = df.drop(columns=['NSP'])
    y = df['NSP']
    
    # Convert target to 0-indexed (1,2,3 -> 0,1,2)
    y = y - 1
    
    print(f"\nClass distribution:")
    print(y.value_counts().sort_index())
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Compute sample weights for class imbalance
    print("\nComputing sample weights for class imbalance...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    print("Training complete!")
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    
    y_pred = model.predict(X_test)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nBalanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Suspect', 'Pathologic']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("(Rows = Actual, Columns = Predicted)")
    
    # Calculate Pathologic recall
    pathologic_total = (y_test == 2).sum()
    pathologic_correct = cm[2, 2]
    pathologic_recall = pathologic_correct / pathologic_total if pathologic_total > 0 else 0
    
    print(f"\nPathologic Recall: {pathologic_recall*100:.1f}%")
    print(f"False Negatives to Normal: {cm[2, 0]}")
    
    # Save model
    model_filename = 'xgboost_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved to: {model_filename}")
    
    # Save feature names for reference
    feature_names = X_train.columns.tolist()
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"✓ Feature names saved to: feature_names.pkl")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()