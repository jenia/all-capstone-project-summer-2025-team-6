

# all-capstone-project-summer-2025-team-6
## Table of Contents

- [Data Pipeline Diagram](#data-pipeline-diagram)
- [How to Run the Data Pipeline](#how-to-run-the-data-pipeline)
- [Models](#models)
  - [Testing Temporal Fire Risk Modeling](#testing-temporal-fire-risk-modeling)
    - [Data Cleaning and Merging Pipeline](#data-cleaning-and-merging-pipeline)

## [Data pipeline diagram](https://docs.google.com/drawings/d/1JSGUZZg9EYoyRtfRQbYmxvmRRgAAAtKCh4ktoKaSbEA/edit)

![img.png](images/img.png)
### How to run the data pipeline:

You need to run these 3 files:

```commandline
python ./dataprep/evaluation_fonciere.py   ===> eval_cleaned.csv  +feature engineering for evaluation fonciere ===> eval_cleaned_feat_eng.csv
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py===> interventions_cleaned_with_has_fire.csv
python ./dataprep/main_evaluation_fonciere.py ===> evaluation_with_fire_and_coordinates_and_date.csv
```
You must run the `python ./dataprep/main_evaluation_fonciere.py` to get the file `evaluation_with_fire_and_coordinates_and_date.csv`
I did not commit it because it's 100MB big.

# Models
We evaluated multiple models. This subsection describes how to run them and summarizes their performance
## Testing temporal fire risk modeling
`datamodel/` contains a Python script for testing temporal fire risk modeling.  
Please note that this is an early test — the results are not yet precise.  
To improve the model, additional feature engineering is required.

The script `Monthly_fire_Risk_prediction-test.py` runs a monthly fire risk prediction test.  
You can execute it from the project root with:
### Data Cleaning and Merging Pipeline


#### Description  : evaluation_fonciere.py 

1. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned with additional feature enginnering we get **eval_cleaned_feat_eng.csv**
```commandline python ./dataprep/evaluation_fonciere.py ===> eval_cleaned.csv

(to drop later if it doesn't crash the code of others)+ for feature engineering use eval_cleaned_feat_eng.csv

🔍 Summary of Code Workflow This script performs data cleaning and feature engineering on the Montréal property evaluation dataset (uniteevaluationfonciere.csv). The goal is to prepare a high-quality dataset for fire risk modeling by cleaning critical variables, imputing missing values intelligently, and generating informative derived features.

✅ Main Steps: Read raw data from CSV. Clean key columns: ANNEE_CONSTRUCTION, NOMBRE_LOGEMENT, ETAGE_HORS_SOL, and SUPERFICIE_BATIMENT. Impute missing values using borough-level and building-type-level medians. Engineer new features: AGE_BATIMENT RATIO_SURFACE DENSITE_LOGEMENT HAS_MULTIPLE_LOGEMENTS FIRE_FREQUENCY_ZONE Normalize numeric features using MinMaxScaler. Export the cleaned and enriched dataset to eval_cleaned_feat_eng.csv. 🧹 Data Cleaning & Feature Engineering Summary 📌 Data Cleaning Summary Column Cleaning Strategy ANNEE_CONSTRUCTION - Removed unrealistic values (<1800 or >2025) - Imputed by borough median for "Logement/Immeuble"
- Fallback: "unknown" | | NOMBRE_LOGEMENT | - Imputed using median by (borough, building type)

Fallback: borough-level median | | ETAGE_HORS_SOL | - Imputed using median by (borough, building type)
Fallback: borough-level median | | SUPERFICIE_BATIMENT | - Imputed using median by (borough, building type)
Fallback: borough-level median |
🏗️ Feature Engineering Summary Feature Name Description AGE_BATIMENT Building age = 2025 - construction year RATIO_SURFACE SUPERFICIE_BATIMENT / SUPERFICIE_TERRAIN DENSITE_LOGEMENT NOMBRE_LOGEMENT / SUPERFICIE_BATIMENT HAS_MULTIPLE_LOGEMENTS Binary flag: 1 if more than one housing unit, 0 otherwise FIRE_FREQUENCY_ZONE Proxy for fire risk: number of buildings in the same NO_ARROND_ILE_CUM All continuous features were normalized using MinMaxScaler.

🗃️ Dropped or Excluded Columns Temporary or unused columns were dropped before final export:

_ANNEE_CONSTRUCTION_NUM (intermediate) Future features like IS_UNKNOWN_YEAR, IS_MIXED_USE, etc. Any columns used only during cleaning/imputation logic 📦 Final Output Columns The following columns are present in the output file eval_cleaned_feat_eng.csv:

```text [ 'ID_UEV', 'CIVIQUE_DEBUT', 'CIVIQUE_FIN', 'NOM_RUE', 'SUITE_DEBUT', 'MUNICIPALITE', 'ETAGE_HORS_SOL', 'NOMBRE_LOGEMENT', 'ANNEE_CONSTRUCTION', 'CODE_UTILISATION', 'LETTRE_DEBUT', 'LETTRE_FIN', 'LIBELLE_UTILISATION', 'CATEGORIE_UEF', 'MATRICULE83', 'SUPERFICIE_TERRAIN', 'SUPERFICIE_BATIMENT', 'NO_ARROND_ILE_CUM', 'AGE_BATIMENT', 'RATIO_SURFACE', 'DENSITE_LOGEMENT', 'HAS_MULTIPLE_LOGEMENTS', 'FIRE_FREQUENCY_ZONE' ]




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



