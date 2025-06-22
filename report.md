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


## Description: main_evaluation_feat_eng.py

### 🔥 Detailed Summary of `main_evaluation_feat_eng.py`

This script processes building and fire incident data to produce a geospatially enriched dataset for **fire risk modeling**. It includes spatial joins, temporal feature extraction, and fire frequency calculations at the zone level.

---

#### 🧠 Goal of the Script

Produce a feature-rich dataset at the building level that combines:

- Property evaluation data  
- Fire incident history  
- Geographic information (coordinates, zones)  
- Temporal and structural features

📁 **Final output file**: `evaluation_fire_coordinates_date_feat_eng_2.csv`

---

##### 🗂️ 1. Load Datasets

- `eval_cleaned_feat_eng.csv` — Pre-cleaned building/property data
- `adresses.csv` — Street-level address coordinates
- `interventions_cleaned_with_has_fire.csv` — Fire incident reports

All file paths are dynamically resolved.

---

### 🧹 2. Preprocessing

#### 🔸 Evaluation Data
- Cleans `CIVIQUE_DEBUT` (street number) and standardizes street names
- Copies original version for later merging

#### 🔸 Address Data
- Combines `GENERIQUE` + `SPECIFIQUE` into `NOM_RUE_CLEAN`
- Converts address numbers for matching

#### 🔸 Coordinate Assignment
- Merges buildings and addresses to assign `LATITUDE` and `LONGITUDE`
- Filters out rows without coordinates

#### 🔸 Incident Data
- Filters rows where `DESCRIPTION_GROUPE` contains "INCENDIE"
- Converts date/time and builds GeoDataFrame
- Buffers incidents with a 100m radius to capture nearby buildings

---

### 🗺️ 3. Spatial Join

- Matches each building to nearby fire incidents using `gpd.sjoin()`
- Assigns:
  - `fire = True`
  - `fire_date`
  - `NOMBRE_UNITES` (fire truck count)
  - `CASERNE` (station name)

---

### 🔁 4. Merge Fire Info Back

- Merges fire records with **all buildings** using `ID_UEV`
- Fills nulls for non-fire rows
- Re-attaches coordinates from `addr_df`

---

### 🕒 5. Time Features

- Extracts:
  - `fire_month`, `fire_year`, `year_month`
  - `fire_season` (Winter, Spring, Summer, Fall)

---

### 🌍 6. Fire Zone Aggregates

- Computes per-zone fire counts for year 2024
- Computes:
  - `FIRE_COUNT_LAST_YEAR_ZONE`
  - `FIRE_RATE_ZONE` = fire count / buildings
- Applies `MinMaxScaler` to produce normalized versions

---

### 🔎 7. Missing Coordinates

- Flags rows missing latitude/longitude: `missing_coords`
- Compares fire rates between missing vs present coordinates

---

### 🎯 8. Feature Selection

#### ✅ Kept Features
- Structural: `AGE_BATIMENT`, `DENSITE_LOGEMENT`, `RATIO_SURFACE`
- Target: `fire`, `had_fire`, `fire_date`
- Time: `fire_month`, `fire_year`, `fire_season`, `year_month`
- Spatial: `NO_ARROND_ILE_CUM`, `LATITUDE`, `LONGITUDE`
- Fire Stats: `FIRE_COUNT_LAST_YEAR_ZONE`, `FIRE_RATE_ZONE`, and normalized versions

#### ❌ Dropped Features
- Redundant address fields: `CIVIQUE_DEBUT`, `ADDR_DE`, etc.
- Internal metadata: `MATRICULE83`, `CASERNE`, etc.
- Raw year field: `ANNEE_CONSTRUCTION` (keep `AGE_BATIMENT` instead)

---

### 💾 9. Save Output

- Final cleaned file is saved to:
  ```
  ./datasets/cleaned/evaluation_fire_coordinates_date_feat_eng_2.csv
  ```

---

### ✅ Summary Stats

- ~664K total buildings
- ~295K matched to fire incidents
- ~202K rows missing coordinates (optional to drop)
- ~41 clean modeling-ready features



## Description : dense_panel_building_month.py

# 🔍 Fire Risk Monthly Panel Construction — Detailed Summary

This script builds a **dense monthly panel** of buildings with engineered features and fire occurrence labels. It is designed to support predictive fire risk modeling at the building-month level.

---

## 📁 File Paths and Setup

- Uses `Path.cwd().parents[0]` to define the project root
- **Input**: `evaluation_fire_coordinates_date_feat_eng_2.csv`
- **Output**: `building_month_fire_panel_feat_eng.csv`
- Verifies that input file and output directory exist

---

## 📥 Load and Preprocess Dataset

- Loads the CSV into a pandas DataFrame
- Parses `fire_date` into `datetime` format
- Creates a `month` feature (monthly period)
- Drops rows with missing `ID_UEV`, `LATITUDE`, or `LONGITUDE`
- Converts DataFrame to a projected `GeoDataFrame` using EPSG:32188

---

## 🏢 Create Dense Panel: Building × Month

1. Extracts unique buildings (`ID_UEV`, LAT/LON)
2. Constructs a **cartesian product** of all buildings and all available months
3. Merges back LAT/LON to complete geolocation info

---

## 🔥 Fire Incident Labeling

- Filters the GeoDataFrame to get fire records (`fire == True`)
- Flags each building-month with `HAS_FIRE_THIS_MONTH = 1` if fire occurred
- Fills all other records with `0`

---

## 🧱 Merge Static Building Features

- Static columns include:
  - `MUNICIPALITE`, `ETAGE_HORS_SOL`, `NOMBRE_LOGEMENT`, `AGE_BATIMENT`
  - `CODE_UTILISATION`, `CATEGORIE_UEF`, `SUPERFICIE_TERRAIN`, `SUPERFICIE_BATIMENT`
  - `NO_ARROND_ILE_CUM`, `RATIO_SURFACE`, `DENSITE_LOGEMENT`, `HAS_MULTIPLE_LOGEMENTS`
  - `FIRE_COUNT_LAST_YEAR_ZONE`, `BUILDING_COUNT`, `FIRE_RATE_ZONE`, etc.
- Removes duplicates based on `ID_UEV`
- Merges static features into the panel

---

## 🧠 Feature Engineering: Temporal Fire History

Adds the following **lag and cumulative features**:
- `fire_last_1m`, `fire_last_2m`, `fire_last_3m`: fires in past 1–3 months
- `fire_cumcount`: cumulative fires since beginning
- `fire_rolling_3m`, `fire_rolling_6m`, `fire_rolling_12m`: rolling fire counts
- `has_fire_last_month`: binary lag indicator
- `months_since_last_fire`: time since last recorded fire (999 if none)

---

## 🕒 Time Features

Adds:
- `month_num`: numerical month
- `year`: calendar year

---

## 💾 Save Panel

Saves the final panel to:
```python
datamodel/building_month_fire_panel_feat_eng.csv




# 3. Models tried

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


