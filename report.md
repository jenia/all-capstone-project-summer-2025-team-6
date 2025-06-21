# Report

## Table of Contents

1. [Purpose](#purpose)
2. [Models Tried](#models-tried)
 - [RandomForestClassifier](#randomforestclassifier)
 - [LGBMClassifier](#lgbmclassifier)



## Purpose

Project A – Predicting High Fire Risk Areas in Montreal

Objective: The objective of this project is to predict high fire risk areas by month in the city of Montreal, based on historical firefighter intervention data and additional open datasets.

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


