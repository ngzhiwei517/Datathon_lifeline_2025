# TM-113: Lifeline - Fetal Distress Detection using CTG Data

For how to train and test the model, please refer part 6, you can click the link to direct to that part

**Team:** TM-113

**Team Members:**
- Ng Zhi Wei (DSAI/Y2)
- Wong Jing En (MAE/Y1)
- Lee Yan Sheng (MAE/Y2)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Our Approach](#our-approach)
   - [1. Data Cleaning](#1-data-cleaning)
   - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
   - [3. Data Preprocessing](#3-data-preprocessing)
   - [4. Exploring Relationships](#4-exploring-relationships)
   - [5. Data Splitting](#5-data-splitting)
   - [6. Feature Engineering](#6-feature-engineering)
   - [7. Model Development](#7-model-development)
   - [8. Model Explainability (SHAP Analysis)](#8-model-explainability-shap-analysis)
   - [9. Cross-Validation](#9-cross-validation)
   - [10. Inference Performance](#10-inference-performance)
5. [Final Results](#final-results)
6. [Training & Testing](#training--testing)


---

## Project Overview

This project develops a machine learning solution to automatically classify fetal health states (Normal, Suspect, Pathologic) from Cardiotocography (CTG) recordings. Built for the MLDA@EEE Datathon 2025, our system aims to support clinicians in detecting fetal distress during labor, potentially preventing complications through early intervention.

---

## Problem Statement

Every year, thousands of expectant mothers undergo CTG monitoring during labor. CTG records fetal heart rate and uterine contractions to detect signs of fetal distress. However, patterns can be subtle and easily missed in busy hospital wards. 

**Our challenge:** Can we build an AI system to automatically identify warning signs that might otherwise go unnoticed?

---

## Dataset

**Source:** [UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography) (2,126 recordings)

**Features:** 21 clinical features including:
- Baseline fetal heart rate (LB)
- Accelerations (AC) and decelerations (DS, DP, DL)
- Variability measures (ASTV, ALTV, MSTV, MLTV)
- Uterine contractions (UC)
- Histogram statistics (Mode, Mean, Median, Variance, etc.)

**Target Classes:**
- **Normal (78%):** 1,648 cases
- **Suspect (14%):** 293 cases
- **Pathologic (8%):** 175 cases

**Challenge:** Severe class imbalance requiring careful handling to avoid missing critical Pathologic cases.

---

## Our Approach

### 1. Data Cleaning

- Identified and removed 10 duplicate rows
- Corrected data types (NSP and CLASS converted to categorical)
- Validated physiological ranges using impossible value detection:
  - Heart rate: 50-240 bpm
  - Counts (AC, FM, UC, decelerations): non-negative
  - Removed rows with physiologically impossible values

### 2. Exploratory Data Analysis

**Target Variable Distribution:**
- Visualized severe class imbalance (78% Normal, 14% Suspect, 8% Pathologic)

**Feature Distributions:**
- Histograms revealed zero-inflated features (DS, DP, DL, FM)
- ASTV showed bimodal distribution indicating clear separation between Normal and at-risk groups
- LB (baseline heart rate) centered around 130-140 bpm (healthy range)

**Outlier Detection:**
- Checked for impossible values in each feature
- Retained clinical outliers that represent genuine emergencies

### 3. Data Preprocessing

- Dropped perfect duplicates
- Dropped redundant alternative labeling columns (A, B, C, D, E, AD, DE, LD, FS, SUSP, CLASS)
- Retained only NSP as the target variable

### 4. Exploring Relationships

**Correlation Analysis (Heatmap):**
**Boxplot Analysis (Progression across classes):**
**Key Insight:** UC provides stress context, but fetal response features (ASTV, ALTV, AC, DP) determine the outcome.

### 5. Data Splitting

- **Train-test split:** 80/20 ratio with stratification
- **Training set:** 1,692 samples (1,318 Normal, 234 Suspect, 140 Pathologic)
- **Test set:** 424 samples (330 Normal, 59 Suspect, 35 Pathologic)

### 6. Feature Engineering

Created 9 new features from EDA insights:

**Interaction Features:**
- `AC_per_UC`: Accelerations per contraction (fetal response quality)
- `total_abnormal_var`: ASTV + ALTV (overall variability burden)
- `ASTV_ALTV_ratio`: Short-term to long-term variability ratio

**Deceleration Aggregation:**
- `decel_severity`: Weighted score (DP×3 + DS×2 + DL×1)
- `total_decels`: Total decelerations (DP + DS + DL)

**Binary Risk Flags (handling zero-inflation):**
- `has_prolonged_decel`: DP > 0
- `has_severe_decel`: DS > 0
- `has_movement`: FM > 0

**Heart Rate Stability:**
- `heart_rate_range`: Max – Min

**Validation:** Engineered features (`ASTV_ALTV_ratio`, `total_abnormal_var`) ranked in XGBoost's top 10 most important features.

### 7. Model Development

**Class Imbalance Handling:**
Applied `compute_sample_weight(class_weight='balanced')` to assign inverse frequency weights. Pathologic samples weighted ~10× higher than Normal to prevent majority class dominance.

**Models Evaluated:**
- Logistic Regression (baseline, interpretable)
- Decision Tree (captures non-linear patterns)
- Random Forest (ensemble, robust)
- XGBoost (gradient boosting, best performance)

### 8. Model Explainability (SHAP Analysis)

Applied **SHAP (SHapley Additive exPlanations)** to interpret XGBoost predictions:

**Global Feature Importance:**
- **ALTV** (0.1229): Top predictor for Pathologic cases
- **Median** (0.1034): Extreme values distinguish Normal vs Pathologic
- **ASTV** (0.0796): Dominates across all classes (total SHAP importance ~1.6)
- **total_abnormal_var** (0.0612): Engineered feature validated as important
- **ASTV_ALTV_ratio** (0.0440): Another engineered feature in top 10

**ASTV-ALTV Synergistic Effect:**
- Threshold at ASTV ≈ 70: below pushes away from Pathologic, above pushes toward
- High ASTV + High ALTV (red dots) create strongest Pathologic signals (SHAP +3.0)
- High ASTV + Low ALTV (blue dots) produce weaker signals
- **Clinical validation:** Both short-term AND long-term variability abnormalities must be present for highest risk

**Case Study - Missed Pathologic Case (#419):**
- **Actual:** Pathologic | **Predicted:** Suspect
- **Features:** ASTV=77 (very high), ALTV=4 (very low), DP=0, AC=0
- **Why missed:** Atypical pattern - Pathologic usually has BOTH high ASTV and high ALTV
- **SHAP waterfall:** Multiple features pushed away (Median=-1.03, ALTV=-0.42, DP=-0.23), only LB=+0.08 pushed toward Pathologic
- **Model reasoning:** Statistically reasonable given conflicting signals; predicted Suspect as safety buffer

**Clinical Implications:**
- **For screening:** Monitor ASTV first (broadest discriminatory power across all classes)
- **For confirming Pathologic:** Focus on Median heart rate (extreme values = immediate concern)
- **For comprehensive assessment:** Combine variability (ASTV+ALTV) + central tendency (Median+Mean) + decelerations (DP)

### 9. Cross-Validation

**5-Fold Stratified Cross-Validation (XGBoost):**
- **Mean Balanced Accuracy:** 91.5% (±1.9%)
- **Mean Macro F1-Score:** 88.9% (±1.9%)
- **Mean Pathologic Recall:** 92.1% (±1.4%)
- **All folds exceeded 85% Pathologic recall** (medical safety threshold)

Confirms robust generalization across diverse patient subsets. Test performance (94.7%) slightly above CV mean but within acceptable variance.

### 10. Inference Performance

**Performance Metrics:**
- **Single prediction:** 6.9 ms average (±4.1 ms)
- **Batch prediction:** 0.025 ms per sample (40,640 predictions/second)
- **Model size:** 0.53 MB (lightweight, CPU-deployable)

**Clinical Context:**
- CTG traces reviewed every 30-60 seconds in practice
- Model response time: **4,000× faster than clinical decision window**
- **Assessment:** Suitable for real-time clinical decision support

---

## Final Results

### Model Performance Comparison

| Model | Balanced Accuracy | Macro F1 | Pathologic Recall |
|-------|-------------------|----------|-------------------|
| **XGBoost** | **94.7%** | **93.1%** | **97.1%** |
| Random Forest | 89.1% | 90.8% | 91.4% |
| Decision Tree | 90.1% | 84.4% | 94.3% |
| Logistic Regression | 82.6% | 78.3% | 77.1% |

### XGBoost Confusion Matrix (Test Set)

|  | Predicted Normal | Predicted Suspect | Predicted Pathologic |
|---|---|---|---|
| **Actual Normal** | 315 | 15 | 0 |
| **Actual Suspect** | 5 | 54 | 0 |
| **Actual Pathologic** | 0 | 1 | 34 |

**Medical Safety Highlights:**
-  **Zero false negatives to Normal** (no Pathologic case misclassified as healthy)
-  **97.1% Pathologic recall** (34/35 critical cases detected)
-  Single missed Pathologic flagged as Suspect (still receives monitoring)

---

## Training & Testing

### Prerequisites

**Required Files:**
- `CTG.csv` - Training dataset (download from datasets folder)
- `train.py` - Training script
- `test.py` - Testing script

**Required Libraries:**
```python
pandas, numpy, scikit-learn, xgboost

## Option 1: Training from Scratch
Google Colab Setup

1. Open Google Colab

Go to colab.research.google.com
Click "New Notebook"


2. Upload Files

Left sidebar → Files icon (📁)
Click "Upload" button
Upload: train.py and CTG.csv


3. Install Dependencies

!pip install xgboost scikit-learn pandas numpy

4. Run Training Script

!python train.py

Training Output:

xgboost_model.pkl - Trained model weights (0.53 MB)
feature_names.pkl - Feature order reference
Training metrics printed to console

Expected Training Time: 2-3 minutes on Colab CPU

## Option 2: Using Pre-trained Model (Recommended)
If you want to skip training and directly test predictions, download our pre-trained model:

xgboost_model.pkl (available in repository)


Testing Instructions
Method A: Command Line Testing (Local/Colab)

2. Upload Required Files to Colab

test.py
xgboost_model.pkl (trained or pre-trained)
Your test CSV file (e.g., test_sample.csv)


2. Install Dependencies (if not already installed)

!pip install xgboost pandas

3. Run Test Script

# Default (uses test_sample.csv)
!python test.py

# Or specify your CSV file
!python test.py your_test_file.csv

4. Output

Predictions displayed in console
New file created: test_sample_predictions.csv (with NSP_predicted and NSP_label columns)

Method B: Interactive Testing in Colab Notebook
Step 1: Upload Files

Left sidebar → Files icon → Upload:

xgboost_model.pkl
Your test CSV file



Step 2: Get CSV File Path

Right-click your CSV file → "Copy path"
Example: /content/test_sample.csv

Step 3: Run Prediction Code

import pickle
import pandas as pd
from google.colab import files

print("="*70)
print("FETAL DISTRESS DETECTION - TESTING")
print("="*70)

# 1. Load trained model
print("\n[1/6] Loading model...")
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("✓ Model loaded successfully")

# 2. Load test data
print("\n[2/6] Loading test data...")
test_file = '/content/test_sample.csv'  # UPDATE THIS PATH
test_data_original = pd.read_csv(test_file)
test_data = test_data_original.copy()
print(f"✓ Loaded {len(test_data)} samples")

# 3. Feature engineering (must match training)
print("\n[3/6] Engineering features...")
test_data['AC_per_UC'] = test_data['AC'] / (test_data['UC'] + 1)
test_data['total_abnormal_var'] = test_data['ASTV'] + test_data['ALTV']
test_data['ASTV_ALTV_ratio'] = test_data['ASTV'] / (test_data['ALTV'] + 1)
test_data['decel_severity'] = (test_data['DP'] * 3) + (test_data['DS'] * 2) + test_data['DL']
test_data['total_decels'] = test_data['DP'] + test_data['DS'] + test_data['DL']
test_data['has_prolonged_decel'] = (test_data['DP'] > 0).astype(int)
test_data['has_severe_decel'] = (test_data['DS'] > 0).astype(int)
test_data['has_movement'] = (test_data['FM'] > 0).astype(int)
test_data['heart_rate_range'] = test_data['Max'] - test_data['Min']
print("✓ 9 engineered features created")

# 4. Reorder columns to match training feature order
print("\n[4/6] Preparing features...")
feature_order = ['b', 'e', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                 'DL', 'DS', 'DP', 'DR', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros',
                 'Mode', 'Mean', 'Median', 'Variance', 'Tendency', 'AC_per_UC',
                 'ASTV_ALTV_ratio', 'total_abnormal_var', 'decel_severity',
                 'total_decels', 'has_severe_decel', 'has_prolonged_decel',
                 'has_movement', 'heart_rate_range']
test_data = test_data[feature_order]
print(f"✓ {len(feature_order)} features ready for prediction")

# 5. Make predictions
print("\n[5/6] Making predictions...")
predictions = model.predict(test_data)
predictions_clinical = predictions + 1  # Convert 0,1,2 → 1,2,3

# Add predictions to original dataframe
test_data_original['NSP_predicted'] = predictions_clinical
label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
test_data_original['NSP_label'] = test_data_original['NSP_predicted'].map(label_map)
print("✓ Predictions complete")

# 6. Display results
print("\n" + "="*70)
print("PREDICTION RESULTS - SAMPLE")
print("="*70)
display_cols = ['LB', 'ASTV', 'ALTV', 'DP', 'NSP_predicted', 'NSP_label']
print(test_data_original[display_cols].head(10).to_string(index=False))

print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)
print(test_data_original['NSP_label'].value_counts())
print(f"\nTotal predictions: {len(test_data_original)}")

print("\n" + "="*70)
print("LABEL DEFINITIONS")
print("="*70)
print("  1 = Normal      → Healthy fetal status")
print("  2 = Suspect     → Borderline, requires monitoring")
print("  3 = Pathologic  → High risk, immediate attention needed")

# 7. Save predictions
output_file = 'predictions_output.csv'
test_data_original.to_csv(output_file, index=False)
print(f"\n✓ Full results saved to: {output_file}")

# 8. Auto-download to your computer
print("\n" + "="*70)
print("DOWNLOADING RESULTS...")
print("="*70)
files.download(output_file)
print("✓ Check your browser's Downloads folder!")

Test Data Requirements
Your test CSV must contain these 21 columns (same as training data):
Core CTG Features:

LB - Baseline fetal heart rate
AC - Accelerations
FM - Fetal movements
UC - Uterine contractions
DL, DS, DP - Deceleration types
ASTV, MSTV, ALTV, MLTV - Variability measures

Histogram Features:

Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency

Additional:

b, e, DR

Note: The model will automatically create 9 engineered features during prediction. Do NOT include these in your test CSV:

AC_per_UC, total_abnormal_var, ASTV_ALTV_ratio, decel_severity, total_decels, has_prolonged_decel, has_severe_decel, has_movement, heart_rate_range


Troubleshooting
Error: "KeyError: column not found"

Ensure your CSV has all 21 required columns with exact names (case-sensitive)

Error: "FileNotFoundError: xgboost_model.pkl"

Verify the model file is uploaded to Colab
Check file name spelling

Predictions seem incorrect:

Verify your CSV uses the same units/scales as training data
Check for missing values (model expects no NaN values)

For more help: Check our repository issues or contact the team.
