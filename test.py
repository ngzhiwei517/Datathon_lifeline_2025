"""
Inference script for XGBoost fetal distress detection model
Loads trained model and makes predictions on new data
"""

import pickle
import pandas as pd
import sys

def load_model(model_path='xgboost_model.pkl'):
    """Load trained XGBoost model"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {model_path}")
    return model

def engineer_features(data):
    """Apply same feature engineering as training"""
    data = data.copy()
    
    # Interaction features
    data['AC_per_UC'] = data['AC'] / (data['UC'] + 1)
    data['total_abnormal_var'] = data['ASTV'] + data['ALTV']
    data['ASTV_ALTV_ratio'] = data['ASTV'] / (data['ALTV'] + 1)
    
    # Deceleration features
    data['decel_severity'] = (data['DP'] * 3) + (data['DS'] * 2) + data['DL']
    data['total_decels'] = data['DP'] + data['DS'] + data['DL']
    
    # Binary flags
    data['has_prolonged_decel'] = (data['DP'] > 0).astype(int)
    data['has_severe_decel'] = (data['DS'] > 0).astype(int)
    data['has_movement'] = (data['FM'] > 0).astype(int)
    
    # Heart rate stability
    data['heart_rate_range'] = data['Max'] - data['Min']
    
    return data

def get_feature_order():
    """Return correct feature order for model input"""
    return ['b', 'e', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
            'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
            'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'AC_per_UC',
            'ASTV_ALTV_ratio', 'total_abnormal_var', 'decel_severity',
            'total_decels', 'has_severe_decel', 'has_prolonged_decel',
            'has_movement', 'heart_rate_range']

def main(test_csv_path):
    """Main inference pipeline"""
    print("="*70)
    print("FETAL DISTRESS DETECTION - INFERENCE")
    print("="*70)
    
    # Load model
    model = load_model('xgboost_model.pkl')
    
    # Load test data
    print(f"\nLoading test data from: {test_csv_path}")
    test_data = pd.read_csv(test_csv_path)
    print(f"✓ Loaded {len(test_data)} samples")
    
    # Keep original for output
    test_data_original = test_data.copy()
    
    # Engineer features
    print("\nEngineering features...")
    test_data = engineer_features(test_data)
    print("✓ Feature engineering complete")
    
    # Reorder columns to match training
    print("\nReordering columns...")
    feature_order = get_feature_order()
    test_data = test_data[feature_order]
    print(f"✓ {len(feature_order)} features ready")
    
    # Predict
    print("\nMaking predictions...")
    predictions = model.predict(test_data)
    predictions_clinical = predictions + 1  # Convert 0,1,2 to 1,2,3
    
    # Add predictions to original data
    test_data_original['NSP_predicted'] = predictions_clinical
    
    label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
    test_data_original['NSP_label'] = test_data_original['NSP_predicted'].map(label_map)
    
    print("✓ Predictions complete")
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print("\nSample predictions:")
    print(test_data_original[['LB', 'ASTV', 'ALTV', 'DP', 'NSP_predicted', 'NSP_label']].head(10))
    
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    print(test_data_original['NSP_label'].value_counts())
    print(f"\nTotal samples: {len(test_data_original)}")
    
    # Save predictions
    output_path = test_csv_path.replace('.csv', '_predictions.csv')
    test_data_original.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    print("\nLabel meanings:")
    print("  1 = Normal (healthy baby)")
    print("  2 = Suspect (borderline, needs monitoring)")
    print("  3 = Pathologic (high risk, immediate attention)")

if __name__ == "__main__":
    # Usage: python test.py test_data.csv
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = 'test_sample.csv'  # Default
    
    main(test_file)