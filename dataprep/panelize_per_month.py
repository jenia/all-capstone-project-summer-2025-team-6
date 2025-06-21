import pandas as pd 
import os
import geopandas as gpd
from shapely.geometry import Point
import numpy as np


SOURCE_FILE = os.path.join('.','datasets','merged','eval_with_coord.csv')
DESTINATION_FILE = os.path.join('.','datasets','merged','base_panel_month.csv')
INTERVENTIONS_FILE_PATH = os.path.join('.','datasets','cleaned','interventions_cleaned.csv')
print(f"Loading dataset from {SOURCE_FILE}...")
df = pd.read_csv(SOURCE_FILE)
print(f"dataset loaded - shape {df.shape}")


#build geo dataframe from evaluation df
eval_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["LONGITUDE"].astype(float),
                                 df["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)

# Create list of months (1-12)
months = list(range(1, 13))

# Duplicate each row 12 times and add month column
df = df.loc[df.index.repeat(12)].assign(month=np.tile(months, len(df))).reset_index(drop=True)


# %%
incidents=pd.read_csv(INTERVENTIONS_FILE_PATH,parse_dates=['CREATION_DATE_TIME'])
print(incidents.columns)
incidents=incidents[incidents['DESCRIPTION_GROUPE']=='INCENDIE']
print(f"Incident categories kept: {incidents['DESCRIPTION_GROUPE'].unique()}")
# -- Project both to meters for spatial operations ---
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = gpd.GeoDataFrame(
    incidents,
    geometry=gpd.points_from_xy(incidents["LONGITUDE"], incidents["LATITUDE"]),
    crs="EPSG:4326"
)

# --- Project both to meters for spatial operations ---

incident_gdf = incident_gdf.to_crs(epsg=32188)


df['HAS_FIRE_IN_MONTH']=False #sets all values to False by default
print(f"Starting incident matching")
print(f"df['HAS_FIRE_IN_MONTH'] contains {df['HAS_FIRE_IN_MONTH'].value_counts()}")
for m in months:
    
    monthly_incidents_gdf=incident_gdf[incident_gdf['CREATION_DATE_TIME'].dt.month==m].copy()
    print(monthly_incidents_gdf.head(5))
    print(f"month {m} : {monthly_incidents_gdf.shape}")
    
    
    # --- Buffer fire incidents by 100 meters ---
    monthly_incidents_gdf["buffer"] = monthly_incidents_gdf.geometry.buffer(100)
    incident_buffer_gdf = monthly_incidents_gdf.set_geometry("buffer")
    
    # --- Spatial join: find properties within 100m of a fire incident ---
    joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')
    # print(f"joined length : {joined.shape}")
    # --- Use unique matched ID_UEV set to mark fires ---
    matched_ids = set(joined["ID_UEV"])
    # assign HAS_FIRE_IN_MONTH VALUE if there has been a fire in this month
    df["HAS_FIRE_IN_MONTH"] = (df["ID_UEV"].isin(matched_ids) & (df['month']==m)) | (df['HAS_FIRE_IN_MONTH']==True)
    print(f"Finished processing month {m}")
    print(f"df['HAS_FIRE_IN_MONTH'] contains {df['HAS_FIRE_IN_MONTH'].value_counts()}")


print(f"Saving file to {DESTINATION_FILE}")
print(f"Resulting dataframe's shape: {df.shape}")
df.to_csv(DESTINATION_FILE,index=False)
print(f"File saved")