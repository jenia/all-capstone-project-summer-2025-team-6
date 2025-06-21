# Report

## Purpose

Project A – Predicting High Fire Risk Areas in Montreal

Objective: The objective of this project is to predict high fire risk areas by month in the city of Montreal, based on historical firefighter intervention data and additional open datasets.

## Models tried

### RandomForestClassifier

**Input data used**: *evaluation_with_fire_and_coordinates_and_date.csv`*

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

**Input data used**: *evaluation_with_fire_and_coordinates_and_date.csv*

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



## 1. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

1. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 
dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned we add additional feature enginnering for evaluation fonciere like Age batiment, ratio surface, densite logement, fire frequency zone we get **eval_cleaned_feat_eng.csv**


**To run this code**

 **python ./dataprep/evaluation_fonciere.py 
output ===> eval_cleaned.csv (to drop later if it doesn't crash the code of others)+ for feature engineering use eval_cleaned_feat_eng.csv**

🔍 **Summary of Code Workflow** 
This script performs data cleaning and feature engineering on the Montréal property evaluation dataset (uniteevaluationfonciere.csv). The goal is to prepare a high-quality dataset for fire risk modeling by cleaning critical variables, imputing missing values intelligently, and generating informative derived features.

✅ **Main Steps** 
Read raw data from CSV. Clean key columns: ANNEE_CONSTRUCTION, NOMBRE_LOGEMENT, ETAGE_HORS_SOL, and SUPERFICIE_BATIMENT. Impute missing values using borough-level and building-type-level medians. Engineer new features: AGE_BATIMENT RATIO_SURFACE DENSITE_LOGEMENT HAS_MULTIPLE_LOGEMENTS FIRE_FREQUENCY_ZONE Normalize numeric features using MinMaxScaler. Export the cleaned and enriched dataset to eval_cleaned_feat_eng.csv. 

🧹** Data Cleaning & Feature Engineering Summary **

📌 **Data Cleaning Summary **

| Column                | Cleaning Strategy                                                                 |
|-----------------------|------------------------------------------------------------------------------------|
| `ANNEE_CONSTRUCTION`  | - Removed unrealistic values (<1800 or >2025)  |
|                       |  - Imputed by borough median for "Logement/Immeuble"  |
|                       |    - Fallback: `"unknown"`                            |
| `NOMBRE_LOGEMENT`     | - Imputed using median by `(borough, building type)`  |
|                        |   - Fallback: borough-level median |
| `ETAGE_HORS_SOL`      | - Imputed using median by `(borough, building type)`  |
|                       |    - Fallback: borough-level median |
| `SUPERFICIE_BATIMENT` | - Imputed using median by `(borough, building type)`  |
|                       |    - Fallback: borough-level median |


🏗️ **Feature Engineering Summary**
 

| Feature Name             | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `AGE_BATIMENT`           | Building age = 2025 - construction year                                      |
| `RATIO_SURFACE`          | `SUPERFICIE_BATIMENT` / `SUPERFICIE_TERRAIN`                                 |
| `DENSITE_LOGEMENT`       | `NOMBRE_LOGEMENT` / `SUPERFICIE_BATIMENT`                                    |
| `HAS_MULTIPLE_LOGEMENTS` | Binary flag: 1 if more than one housing unit, 0 otherwise                    |
| `FIRE_FREQUENCY_ZONE`    | Proxy for fire risk: number of buildings in the same `NO_ARROND_ILE_CUM`     |

 
 All continuous features were normalized using MinMaxScaler.


🗃️ **Dropped or Excluded Columns**

 Temporary or unused columns were dropped before final export:

_ANNEE_CONSTRUCTION_NUM (intermediate)

Future features like IS_UNKNOWN_YEAR, IS_MIXED_USE, etc.

Any columns used only during cleaning/imputation logic


 📦 **Final Output Columns**
 
  The following columns are present in the output file eval_cleaned_feat_eng.csv:


 [ 'ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE', 'SUITE_DEBUT', 'MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT', 'ANNEE_CONSTRUCTION', 'CODE_UTILISATION', 'LETTRE_DEBUT', 'LETTRE_FIN', 'LIBELLE_UTILISATION', 'CATEGORIE_UEF', 'MATRICULE83', 'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT', 'NO_ARROND_ILE_CUM', 'AGE_BATIMENT', 'RATIO_SURFACE', 'DENSITE_LOGEMENT', 'HAS_MULTIPLE_LOGEMENTS', 'FIRE_FREQUENCY_ZONE' ]

