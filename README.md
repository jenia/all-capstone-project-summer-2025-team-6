

# all-capstone-project-summer-2025-team-6

1. We use the datasets:

donneesouvertes-interventions-sim.csv and donneesouvertes-interventions-sim2020.csv saved in datasets/raw from the website https://donnees.montreal.ca/en/dataset/interventions-service-securite-incendie-montreal

We clean in dataprep/interventions.py and we output the dataset:  interventions_cleaned.csv in datasets/cleaned.


2. We use the dataset uniteevaluationfonciere.csv from the website https://donnees.montreal.ca/dataset/unites-evaluation-fonciere 

We clean in evaluation_fonciere.py and we output the dataset eval_cleaned.csv in datasets/cleaned.

3. We merge eval_cleaned.csv and adresses.csv and we do some feature engineering in datamerge/eval+adresses+feat_eng.py we output the file eval_with_coords_feat_eng.csv  

4. to the dataset in datasets/cleaned/interventions_cleaned.csv   we add a HAS_FIRE column to categorize fire incidents  in DESCRIPTION_GROUPE The elements AUTREFEU, INCENDIE and Alarmes-incendies as  1 0 otherwise   we output datasets/cleaned/interventions_cleaned_with_has_fire.csv'


