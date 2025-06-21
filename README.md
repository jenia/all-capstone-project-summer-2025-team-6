

# all-capstone-project-summer-2025-team-6
## Table of Contents

- [Data Pipeline Diagram](#data-pipeline-diagram)
- [How to Run the Data Pipeline](#how-to-run-the-data-pipeline)
- [Testing Temporal Fire Risk Modeling](#testing-temporal-fire-risk-modeling)
  - [Data Cleaning and Merging Pipeline](#data-cleaning-and-merging-pipeline)
    - [Description: evaluation_fonciere.py](#description-evaluation_foncierepy)
## [Data pipeline diagram](https://docs.google.com/drawings/d/1JSGUZZg9EYoyRtfRQbYmxvmRRgAAAtKCh4ktoKaSbEA/edit)

![img.png](images/img.png)
### How to run the data pipeline:

You need to run these 3 files:

```commandline
python ./dataprep/evaluation_fonciere.py
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
python ./dataprep/main_evaluation_fonciere.py
```
You must run the `python ./dataprep/main_evaluation_fonciere.py` to get the file *evaluation_with_fire_and_coordinates_and_date.csv*
I did not commit it because it's 100MB big.

## Testing temporal fire risk modeling
`datamodel/` contains a Python script for testing temporal fire risk modeling.  
Please note that this is an early test — the results are not yet precise.  
To improve the model, additional feature engineering is required.

The script `Monthly_fire_Risk_prediction-test.py` runs a monthly fire risk prediction test.  
You can execute it from the project root with:


# Data Cleaning and Merging Pipeline


# 1. Description: evaluation_fonciere.py

## 🏗️ evaluation_fonciere.py: Clean and Feature Engineer Property Evaluation Data

### 📁 File Location
```
dataprep/evaluation_fonciere.py
```

### 📄 Description
This script processes the [Montréal property evaluation dataset](https://donnees.montreal.ca/dataset/unites-evaluation-fonciere), cleans it, and applies basic feature engineering to prepare it for downstream fire risk modeling.

### 🚀 How to Run
```bash
python ./dataprep/evaluation_fonciere.py
```

### 📤 Output
- `datasets/cleaned/eval_cleaned.csv`  
- `datasets/cleaned/eval_cleaned_feat_eng.csv` ← *(Recommended for modeling)*

---

### 🔍 Summary of Code Workflow

This script performs:
- Cleaning of raw columns
- Imputation of missing values
- Creation of new features
- Normalization of numeric variables

---

### ✅ Main Steps

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

### 🧹 Data Cleaning Summary

| Column                | Cleaning Strategy                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|
| `ANNEE_CONSTRUCTION`  | Drop outliers (<1800 or >2025), impute with borough median (fallback: "unknown") |
| `NOMBRE_LOGEMENT`     | Impute with (borough, building-type) median, fallback to borough median          |
| `ETAGE_HORS_SOL`      | Same imputation logic as above                                                   |
| `SUPERFICIE_BATIMENT` | Same imputation logic as above                                                   |

---

### 🛠️ Feature Engineering Summary

| Feature Name             | Description                                                     |
|--------------------------|-----------------------------------------------------------------|
| `AGE_BATIMENT`           | `2025 - ANNEE_CONSTRUCTION`                                     |
| `RATIO_SURFACE`          | `SUPERFICIE_BATIMENT / SUPERFICIE_TERRAIN`                      |
| `DENSITE_LOGEMENT`       | `NOMBRE_LOGEMENT / SUPERFICIE_BATIMENT`                         |
| `HAS_MULTIPLE_LOGEMENTS` | 1 if more than 1 housing unit, 0 otherwise                      |
| `FIRE_FREQUENCY_ZONE`    | Number of buildings in the same `NO_ARROND_ILE_CUM` as proxy for risk |

> 🔄 All continuous variables are normalized using `MinMaxScaler`.

---

### 🗃️ Dropped/Excluded Columns

- `_ANNEE_CONSTRUCTION_NUM`
- Temporary columns used for imputation
- Potential future features like `IS_UNKNOWN_YEAR`, etc.

---

### 📦 Final Output Columns

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




TODO: A diagram would be useful here and discussion of results.

```bash
python ./datamodel/Monthly_fire_risk_prediction-test.py

1. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned.

2. datamerge/merge_evaluationfonciere_adresses.py : We merge eval_cleaned.csv and adresses.csv and we do some feature engineering in datamerge/merge_evaluationfonciere_adresses.py we output the file merged_evaluationfonciere_adresses.csv  in datasets/merged

 

3. dataprep/interventions_HAS_FIRE_binary_analysis.py:        We use the datasets:

donneesouvertes-interventions-sim.csv and donneesouvertes-interventions-sim2020.csv saved in datasets/raw from the website https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal    we isolated records labeled as fire-related specific  in DESCRIPTION_GROUPE using categories AUTREFEU, INCENDIE   we output datasets/cleaned/interventions_cleaned_with_has_fire.csv

4.datamerge/merged_interventions_evaluationfonciere_adresses.py  : merge  datasets/cleaned/interventions_cleaned_with_has_fire.csv  with datasets/merged/merged_evaluationfonciere_adresses.csv    The output is merged_interventions_evaluationfonciere_adresses_binary_analysis_1.csv  which has some feature engineering but we can add more for sure



Remark: I have some mistakes to be fixed in the read_csv if you are using visual studio since i was working in jupyternotebook all the time. Check all this if you want to use the github same enviroment folder only to located the files



