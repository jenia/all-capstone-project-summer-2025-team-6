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

