# Report

## Table of Contents

1. [Purpose](#purpose)
2. [Data Cleaning and Merging Pipeline](#data-cleaning-and-merging-pipeline)
2. [Models Tried](#models-tried)
 - [RandomForestClassifier](#randomforestclassifier)
 - [LGBMClassifier](#lgbmclassifier)



## Purpose

Project A – Predicting High Fire Risk Areas in Montreal

Objective: The objective of this project is to predict high fire risk areas by month in the city of Montreal, based on historical firefighter intervention data and additional open datasets.



# 2. Data Cleaning and Merging Pipeline


## Description: evaluation_fonciere.py

### 🏗️ evaluation_fonciere.py: Clean and Feature Engineer Property Evaluation Data

#### 📁 File Location
```
dataprep/evaluation_fonciere.py
```

#### 📄 Description
This script processes the [Montréal property evaluation dataset](https://donnees.montreal.ca/dataset/unites-evaluation-fonciere), cleans it, and applies basic feature engineering to prepare it for downstream fire risk modeling.

#### 🚀 How to Run
```bash
python ./dataprep/evaluation_fonciere.py
```

#### 📤 Output
- `datasets/cleaned/eval_cleaned.csv`  
- `datasets/cleaned/eval_cleaned_feat_eng.csv` ← *(Recommended for modeling)*

---

#### 🔍 Summary of Code Workflow

This script performs:
- Cleaning of raw columns
- Imputation of missing values
- Creation of new features
- Normalization of numeric variables

---

#### ✅ Main Steps

1. **Load raw dataset**
2. **Clean columns:**
   - `ANNEE_CONSTRUCTION`, `NOMBRE_LOGEMENT`, `ETAGE_HORS_SOL`, `SUPERFICIE_BATIMENT`
3. **Impute missing values using medians** at borough + building type level
4. **Feature engineering:**
   - `AGE_BATIMENT`
   - `RATIO_SURFACE`
   - `DENSITE_LOGEMENT`
   - `HAS_MULTIPLE_LOGEMENTS`
   - `FIRE_FREQUENCY_ZONE`
5. **Normalize** continuous features using `MinMaxScaler`
6. **Save output** to `eval_cleaned_feat_eng.csv`

---

#### 🧹 Data Cleaning Summary

| Column                | Cleaning Strategy                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|
| `ANNEE_CONSTRUCTION`  | Drop outliers (<1800 or >2025), impute with borough median (fallback: "unknown") |
| `NOMBRE_LOGEMENT`     | Impute with (borough, building-type) median, fallback to borough median          |
| `ETAGE_HORS_SOL`      | Same imputation logic as above                                                   |
| `SUPERFICIE_BATIMENT` | Same imputation logic as above                                                   |

---

#### 🛠️ Feature Engineering Summary

| Feature Name             | Description                                                     |
|--------------------------|-----------------------------------------------------------------|
| `AGE_BATIMENT`           | `2025 - ANNEE_CONSTRUCTION`                                     |
| `RATIO_SURFACE`          | `SUPERFICIE_BATIMENT / SUPERFICIE_TERRAIN`                      |
| `DENSITE_LOGEMENT`       | `NOMBRE_LOGEMENT / SUPERFICIE_BATIMENT`                         |
| `HAS_MULTIPLE_LOGEMENTS` | 1 if more than 1 housing unit, 0 otherwise                      |
| `FIRE_FREQUENCY_ZONE`    | Number of buildings in the same `NO_ARROND_ILE_CUM` as proxy for risk |

> 🔄 All continuous variables are normalized using `MinMaxScaler`.

---

#### 🗃️ Dropped/Excluded Columns

- `_ANNEE_CONSTRUCTION_NUM`
- Temporary columns used for imputation
- Potential future features like `IS_UNKNOWN_YEAR`, etc.

---

#### 📦 Final Output Columns

```text
[
  'ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE', 'SUITE_DEBUT',
  'MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT', 'ANNEE_CONSTRUCTION',
  'CODE_UTILISATION', 'LETTRE_DEBUT', 'LETTRE_FIN', 'LIBELLE_UTILISATION',
  'CATEGORIE_UEF', 'MATRICULE83', 'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT',
  'NO_ARROND_ILE_CUM', 'AGE_BATIMENT', 'RATIO_SURFACE', 'DENSITE_LOGEMENT',
  'HAS_MULTIPLE_LOGEMENTS', 'FIRE_FREQUENCY_ZONE'
]
```






## Models tried

### RandomForestClassifier
(Located in file [EDA-incident-evaluation-fonciere.ipynb](EDA-incident-evaluation-fonciere.ipynb), for pipeline see [instructions](README.md/#how-to-run-the-data-pipeline))

**Target Variable**

Y = P(Fire | X)

Where X includes:

- `log_terrain`
- `log_batiment`
- `log_etage_hors_sol`
- `log_numbre_de_logement`
- `ANNEE_CONSTRUCTION`
- `density`


**Confusion Matrix**

|               | Predicted False | Predicted True |
|---------------|------------------|----------------|
| **Actual False** | 64,073           | 9,540          |
| **Actual True**  | 8,273            | 50,871         |




**Classification Report**

| Label        | Precision | Recall | F1-score | Support |
|--------------|-----------|--------|----------|---------|
| **False**    | 0.89      | 0.87   | 0.88     | 73,613  |
| **True**     | 0.84      | 0.86   | 0.85     | 59,144  |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.87     | 132,757 |
| **Macro avg**| 0.86      | 0.87   | 0.86     | 132,757 |
| **Weighted avg** | 0.87  | 0.87   | 0.87     | 132,757 |


ROC AUC: 0.936

### LGBMClassifier
(Located in file [Model-building.ipynb](Model-building.ipynb), for pipeline see [instructions](README.md/#how-to-run-the-data-pipeline))

**Target Variable**

Y = P(Month of Fire | X)

Where X includes:

- `ETAGE_HORS_SOL`
- `NOMBRE_LOGEMENT`
- `ANNEE_CONSTRUCTION`
- `SUPERFICIE_TERRAIN`
- `SUPERFICIE_BATIMENT`
- `LONGITUDE`
- `LATITUDE`

**Note**: Properties with no recorded fire were assigned to month 13 to indicate the absence of fire incidents

| Metric         | Precision | Recall | F1-score | Support |
|----------------|-----------|--------|----------|---------|
| **Accuracy**   |           |        | 0.676    | 132,757 |
| **Macro avg**  | 0.457     | 0.348  | 0.388    | 132,757 |
| **Weighted avg** | 0.629   | 0.676  | 0.640    | 132,757 |


I then tried to train only on months 1-12:

| Metric           | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| **Accuracy**     |           |        | 0.434    | 58,954  |
| **Macro avg**    | 0.439     | 0.433  | 0.434    | 58,954  |
| **Weighted avg** | 0.437     | 0.434  | 0.433    | 58,954  |


