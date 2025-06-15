

# all-capstone-project-summer-2025-team-6

## How to run:
### evaluation_fonciere.py
```commandline
python ./dataprep/evaluation_fonciere.py
```

This will clean up the evaluation fonciere data, located in these folders:

```python
ORIGINAL_FILE_NAME_EVAL = './datasets/raw/uniteevaluationfonciere.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/eval_cleaned.csv'
```
### interventions.py
```commandline
python ./dataprep/interventions.py
```

This will clean up the interventions data, located in these folders

```python
ORIGINAL_FILE_NAME_2023_2025 = './datasets/raw/donneesouvertes-interventions-sim.csv'
ORIGINAL_FILE_NAME_2022_BEFORE = './datasets/raw/donneesouvertes-interventions-sim2020.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/interventions_cleaned.csv'
```

### interventions_HAS_FIRE_binary_analysis.py

```commandline
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
```

This will mark the interventions with fire, located in these folders

```python
ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED = './datasets/cleaned/interventions_cleaned.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/interventions_cleaned_with_has_fire.csv'
```

# Data Cleaning and Merging Pipeline

1. dataprep/interventions.py:                                                                       We use the datasets:

donneesouvertes-interventions-sim.csv and donneesouvertes-interventions-sim2020.csv saved in datasets/raw from the website https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal

We clean in dataprep/interventions.py and we output the dataset:  interventions_cleaned.csv in datasets/cleaned.


2. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned.

3. datamerge/merge_evaluationfonciere_adresses.py : We merge eval_cleaned.csv and adresses.csv and we do some feature engineering in datamerge/merge_evaluationfonciere_adresses.py we output the file merged_evaluationfonciere_adresses.csv  in datasets/merged

 

4. dataprep/interventions_HAS_FIRE_binary_analysis.py:  to the dataset in datasets/cleaned/interventions_cleaned.csv   we isolated records labeled as fire-related specific  in DESCRIPTION_GROUPE using categories AUTREFEU, INCENDIE   we output datasets/cleaned/interventions_cleaned_with_has_fire.csv

5.datamerge/merged_interventions_evaluationfonciere_adresses.py  : merge  datasets/cleaned/interventions_cleaned_with_has_fire.csv  with datasets/merged/merged_evaluationfonciere_adresses.csv  



Remark: I have some mistakes to be fixed in the read_csv if you are using visual studio since i was working in jupyternotebook all the time. Check all this if you want to use the github same enviroment folder only to located the files