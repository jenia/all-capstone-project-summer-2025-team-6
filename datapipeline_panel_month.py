# Ugly script to run all dataprep and data merge scripts
import os
import pandas as pd

#Using the un-enriched file for now. Until eval_enriched is ready 
EVAL_COORDS_PATH = os.path.join('.','datasets','merged','eval_with_coord.csv')
# manuel steps to avoid duplicating the first part of pipeline
#running the "no panel pipeline"
print("running the no panel pipeline first")
os.system(f"python {os.path.join('.','datapipeline_no_panel.py')}")
df = pd.read_csv(os.path.join('.','datasets','cleaned','evaluation_with_fire_and_coordinates_and_date.csv'))
print(df.columns)
dropped_columns=['fire_date','fire']
dropped_columns=[col for col in dropped_columns if col in df.columns]
df.drop(dropped_columns,axis=1,inplace=True)
df.drop_duplicates(subset='ID_UEV',inplace=True)
df.to_csv(EVAL_COORDS_PATH)
print(df.columns)
print(df.shape)
#restarting pipeline from eval_with_coord.csv
steps = [
    os.path.join('dataprep','panelize_per_month.py') #base_panel_month.csv
]
for step in steps:
    print(f"Executing {step}")
    os.system(f"python {step}")  
    print("\n\n")



