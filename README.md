

# all-capstone-project-summer-2025-team-6

## How to run:
You need to run these 4 files:

```commandline
python ./dataprep/evaluation_fonciere.py
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
// TODO: Eugene, I think I should use the new DESCRIPTION_GROUPE column from interventions_cleaned_with_has_fire.csv
python ./dataprep/main_evaluation_fonciere.py===>I did this look at added code merged_interventions_evaluationfonciere_adresses.py
```

- evaluation_fonciere.py
```commandline
python ./dataprep/evaluation_fonciere.py
```

This will clean up the evaluation fonciere data, located in these folders:

```python
ORIGINAL_FILE_NAME_EVAL = './datasets/raw/uniteevaluationfonciere.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/eval_cleaned.csv'
```
- interventions_HAS_FIRE_binary_analysis.py
```commandline
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
```

This will clean up the interventions data, located in these folders



- interventions_HAS_FIRE_binary_analysis.py

```commandline
python ./dataprep/interventions_HAS_FIRE_binary_analysis.py
```

and  will mark the interventions with fire, located in these folders

```python
ORIGINAL_FILE_NAME_2023_2025 = './datasets/raw/donneesouvertes-interventions-sim.csv'
ORIGINAL_FILE_NAME_2022_BEFORE = './datasets/raw/donneesouvertes-interventions-sim2020.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/interventions_cleaned_with_has_fire.csv'
```

- main_evaluation_fonciere.py

```commandline
python ./dataprep/main_evaluation_fonciere.py

```

This will mark the evaluations fonciere with fire or not

```commandline
eval_df = pd.read_csv("./datasets/cleaned/eval_cleaned.csv", dtype=str)
addr_df = pd.read_csv("./datasets/cleaned/adresses.csv", dtype=str)
inc_df = pd.read_csv("./datasets/cleaned/interventions_cleaned_with_has_fire.csv")
// TODO: Eugene, I think I should use the new DESCRIPTION_GROUPE column from interventions_cleaned_with_has_fire.csv
OUTPUT_FILE = "./datasets/cleaned/evaluation_with_fire_and_coordinates.csv"

```
# Data Cleaning and Merging Pipeline




1. dataprep/evaluation_fonciere.py:  We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned.

2. datamerge/merge_evaluationfonciere_adresses.py : We merge eval_cleaned.csv and adresses.csv and we do some feature engineering in datamerge/merge_evaluationfonciere_adresses.py we output the file merged_evaluationfonciere_adresses.csv  in datasets/merged

 

3. dataprep/interventions_HAS_FIRE_binary_analysis.py:        We use the datasets:

donneesouvertes-interventions-sim.csv and donneesouvertes-interventions-sim2020.csv saved in datasets/raw from the website https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal    we isolated records labeled as fire-related specific  in DESCRIPTION_GROUPE using categories AUTREFEU, INCENDIE   we output datasets/cleaned/interventions_cleaned_with_has_fire.csv

4.datamerge/merged_interventions_evaluationfonciere_adresses.py  : merge  datasets/cleaned/interventions_cleaned_with_has_fire.csv  with datasets/merged/merged_evaluationfonciere_adresses.csv    The output is merged_interventions_evaluationfonciere_adresses_binary_analysis_1.csv  which has some feature engineering but we can add more for sure



Remark: I have some mistakes to be fixed in the read_csv if you are using visual studio since i was working in jupyternotebook all the time. Check all this if you want to use the github same enviroment folder only to located the files
