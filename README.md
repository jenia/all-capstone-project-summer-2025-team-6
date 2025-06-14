

# all-capstone-project-summer-2025-team-6

## How to run:
```commandline
python evaluation_fonciere.py
```

This will clean up the evaluation fonciere data, located in these folders:

```python
ORIGINAL_FILE_NAME_EVAL = '../datasets/raw/uniteevaluationfonciere.csv'
DESTINATION_FILE_NAME = '../datasets/cleaned/eval_cleaned.csv'
```
# Data Cleaning and Merging Pipeline

1. dataprep/interventions.py:                                                                       We use the datasets:

donneesouvertes-interventions-sim.csv and donneesouvertes-interventions-sim2020.csv saved in datasets/raw from the website https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal

We clean in dataprep/interventions.py and we output the dataset:  interventions_cleaned.csv in datasets/cleaned.


2. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned.

3. datamerge/merge_evaluationfonciere_adresses.py : We merge eval_cleaned.csv and adresses.csv and we do some feature engineering in datamerge/merge_evaluationfonciere_adresses.py we output the file merged_evaluationfonciere_adresses.csv  in datasets/merged

 

4. dataprep/interventions_HAS_FIRE_binary_analysis.py:  to the dataset in datasets/cleaned/interventions_cleaned.csv   we add a HAS_FIRE column to categorize fire incidents  in DESCRIPTION_GROUPE The elements AUTREFEU, INCENDIE and Alarmes-incendies as  1,  0 otherwise   we output datasets/cleaned/interventions_cleaned_with_has_fire.csv

5.datamerge/interventions+eval+adresses  : merge  datasets/cleaned/interventions_cleaned_with_has_fire.csv  with merged_evaluationfonciere_adresses.csv  in datasets/merged    

