# TM-113: Lifeline: Fetal Distress Detection using CTG Data

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Our Approach](#our-approach)
  - [Data Cleaning](#1-data-cleaning)
  - [Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [Data Preprocessing](#3-data-preprocessing)
  - [Exploring Relationships](#4-exploring-relationships)
  - [Data Splitting](#5-data-splitting)
  - [Feature Engineering](#6-feature-engineering)
  - [Model Evaluation](#7-model-evaluation)
- [Key Results](#key-results)
- [Running the Code](#running-the-code)
- [References](#references)
- [Clinical Impact](#clinical-impact)

---

## Project Overview

This project develops a machine learning solution to automatically classify fetal health states (Normal, Suspect, Pathologic) from Cardiotocography (CTG) recordings. Built for the MLDA@EEE Datathon 2025, our system aims to support clinicians in detecting fetal distress during labor, potentially preventing complications through early intervention.

---

## Problem Statement

Every year, thousands of expectant mothers undergo CTG monitoring during labor. CTG records fetal heart rate and uterine contractions to detect signs of fetal distress. However, patterns can be subtle and easily missed in busy hospital wards. Our challenge: **Can we build an AI system to automatically identify warning signs that might otherwise go unnoticed?**

---

## Dataset

**Source:** UCI Cardiotocography Dataset (2,126 recordings)

**Features:** 21 clinical features including:
- Baseline fetal heart rate (LB)
- Accelerations (AC) and decelerations (DS, DP, DL)
- Variability measures (ASTV, ALTV, MSTV, MLTV)
- Uterine contractions (UC)
- Histogram statistics (Mode, Mean, Median, Variance, etc.)

**Target Classes:**
- Normal (78%): 1,648 cases
- Suspect (14%): 293 cases  
- Pathologic (8%): 175 cases

**Challenge:** Severe class imbalance requiring careful handling to avoid missing critical Pathologic cases.

---
## Our Approach
### 1. Data Cleaning
- Identified and removed 10 duplicate rows
- Corrected data types (NSP and CLASS converted to categorical)
- Validated physiological ranges using impossible value detection

---
  ### 2. Exploratory Data Analysis
**Target variable Distribution:**
- Visualized severe class imbalance (78% Normal, 14% Suspect, 8% Pathologic)

**Feature Distributions:**
- Histograms revealed zero-inflated features (DS, DP, DL, FM)
- ASTV showed bimodal distribution indicating clear separation between Normal and at-risk groups
- LB (baseline heart rate) centered around 130-140 bpm (healthy range)
- Checking outlier
- Checking impossible value for each feature (LB not possible less than 50 or more than 240 as a human)


---


### 3. Data Preprocessing
- Drop perfect duplicates
- Dropped redundant alternative labeling columns (A, B, C, D, E, AD, DE, LD, FS, SUSP)

---
### 4. Exploring Relationships
**Correlation Analysis (Heatmap):**
- DP (Prolonged Decelerations): **+0.49** with NSP → strongest predictor  
- ASTV: **+0.47 correlation**  
- ALTV: **+0.42 correlation**  
- AC (Accelerations): **–0.34 correlation** (protective factor)  
- Histogram features (Mode, Mean, Median) highly collinear (**r > 0.89**)  

**Histogram Analysis**
**Boxplot Analysis (Progression across classes):**
- **ASTV:** Normal (median=41) → Suspect (63) → Pathologic (65)  
- **ALTV:** Near zero in Normal, elevated in Suspect/Pathologic  
- **DP:** Almost exclusive to Pathologic (median=1 vs 0 for Normal/Suspect)  
- **AC:** Higher in Normal (median=2) vs Pathologic (median=0) → protective signal  
- **UC:** Similar across classes (response quality > stress frequency)  
---

### 5. Data Splitting
- Train-test split: 80/20 ratio
- Training set: 1,692 samples (1,318 Normal, 234 Suspect, 140 Pathologic)
- Test set: 424 samples (330 Normal, 59 Suspect, 35 Pathologic)

---

### 6. Feature Engineering
Created **9 new features** from EDA insights:

**Interaction Features:**
- `AC_per_UC`: Accelerations per contraction  
- `total_abnormal_var`: ASTV + ALTV  

**Deceleration Aggregation:**
- `decel_severity`: Weighted score (DP×3 + DS×2 + DL×1)  
- `total_decels`: Total decelerations  

**Binary Risk Flags (handling zero-inflation):**
- `has_prolonged_decel` (DP > 0)  
- `has_severe_decel` (DS > 0)  
- `has_movement` (FM > 0)  

**Heart Rate Stability:**
- `heart_rate_range` = Max – Min

---

  ### 7. Model Evaluation

Trained four models with class imbalance handling:


  | Model              | Balanced Acc | Macro F1 | Pathologic Recall |
|--------------------|--------------|----------|-------------------  |
| **XGBoost**        | **0.947**    | **0.931**| **97.1%**           |
| Random Forest      | 0.891        | 0.908    | 91.4%               |
| Decision Tree      | 0.901        | 0.844    | 94.3%               |
| Logistic Regression| 0.826        | 0.783    | 77.1%               |

**Validation:**
- 5-fold CV (XGBoost): **91.7% mean ± 2.3% std**  
- Confirms robust generalization, test score slightly optimistic but reliable  

---

## Key Results

### Feature Importance (XGBoost)
Top predictors matched EDA findings:
1. ALTV
2. Median heartrate
3. ASTV
4. total_abnormal_var
5. Mean heartrate
6. DP
7. AC
8. ASTV_ALTC_ratio

The most influential features identified by XGBoost were closely aligned with patterns observed during our exploratory data analysis (EDA):

ALTV & ASTV (Abnormal Variability Measures): XGBoost ranked these among the top predictors. Our EDA showed a clear progression where Normal cases had low variability, while Suspect and Pathologic groups exhibited significantly higher ASTV/ALTV. This confirms that abnormal variability is a key risk indicator.

XGBoost ranked DP (Prolonged Decelerations) and AC (Accelerations) among its most important features. This matches our EDA, where DP was rare in Normal/Suspect but common in Pathologic, while AC was protective.XGBoost ranked AC as 7, confirming the model learned the inverse relationship (more accelerations → healthier baby).

Engineered Features (total_abnormal_var, ASTV_ALTV_ratio): These new features also appeared among the top predictors. Their high importance shows that adding features based on medical reasoning can improve prediction.

---

## Running the Code (Google Colab)
1. Download the project notebooks (`.ipynb` files) from our repository.  
2. Upload the notebooks to **Google Colab**.  
3. Upload the dataset file `CTG_original_3.csv` (provided in the Datathon materials) into Colab.  
   - You can do this via the file upload button in Colab, or by mounting Google Drive.  

---

## References
- UCI Cardiotocography Dataset: https://archive.ics.uci.edu/dataset/193/cardiotocography
- 
- Medical context: Workshop materials on CTG interpretation and fetal monitoring

### Technical Resources
- Random Forest Explained: *StatQuest - Random Forests Part 1*  
  https://www.youtube.com/watch?v=J4Wdy0Wc_xQ  

- Gradient Boosting Tutorial: *StatQuest - Gradient Boost Part 1*  
  https://www.youtube.com/watch?v=3CC4N4z3GJc  

- Cross-Validation Tutorial: *StatQuest - Cross Validation Explained*  
  https://www.youtube.com/watch?v=fSytzGwwBVw  

- Handling Imbalanced Data: *Scikit-learn documentation on class weights and SMOTE*  
  https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html  
---

## Clinical Impact
In busy delivery wards, this model serves as a reliable second pair of eyes, catching 97% of distressed babies. With 1-2 fetal distress cases per 100 births, our system could prevent complications in thousands of deliveries annually by flagging cases for immediate clinician review.
