#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os


# In[20]:


DIRECTORY = r'G:\.shortcut-targets-by-id\1uExmPmKnHKKlOfMdT70cXpwXvdf9aVEC\Capstone Project summer 2025- Team6\datasets\cleaned'
ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED_WITH_HAS_FIRE = 'interventions_cleaned_with_has_fire.csv'
ORIGINAL_FILE_NAME_MERGED_EVALUATIONFONCIERE_ADRESSES = 'merged_evaluationfonciere_adresses.csv'
DESTINATION_FILE_NAME = "merged_interventions_evaluationfonciere_adresses_binary_analysis_1.csv"


# In[21]:


#df_interventions = pd.read_csv(os.path.join(DIRECTORY,ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED_WITH_HAS_FIRE))
#df_evaluationfonciere=pd.read_csv(os.path.join(DIRECTORY,ORIGINAL_FILE_NAME_MERGED_EVALUATIONFONCIERE_ADRESSES))


# In[22]:


df_interventions = pd.read_csv(ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED_WITH_HAS_FIRE)
df_evaluationfonciere=pd.read_csv(ORIGINAL_FILE_NAME_MERGED_EVALUATIONFONCIERE_ADRESSES)


# In[23]:


# --- Convert to GeoDataFrames ---
gdf_eval = gpd.GeoDataFrame(
    df_evaluationfonciere,
    geometry=gpd.points_from_xy(df_evaluationfonciere["LONGITUDE"].astype(float),
                                 df_evaluationfonciere["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)

gdf_interv = gpd.GeoDataFrame(
    df_interventions,
    geometry=gpd.points_from_xy(df_interventions["LONGITUDE"].astype(float),
                                 df_interventions["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)


# In[24]:


# --- Reproject both to metric CRS (Montreal local: EPSG:32188) ---
gdf_eval = gdf_eval.to_crs("EPSG:32188")
gdf_interv = gdf_interv.to_crs("EPSG:32188")



# In[25]:


# --- Buffer fire incidents by 100 meters ---
gdf_interv["buffer"] = gdf_interv.geometry.buffer(100)
gdf_interv_buffer = gdf_interv.set_geometry("buffer")


# In[26]:


# --- Spatial join: find properties within 100m of a fire incident ---
joined = gpd.sjoin( gdf_eval, gdf_interv_buffer, predicate='within', how='inner')


# In[27]:


# Mark affected properties
affected_ids = set(joined["ID_UEV"])
df_evaluationfonciere["HAS_FIRE"] = df_evaluationfonciere["ID_UEV"].isin(affected_ids).astype(int)


# In[28]:


columns_to_keep = [
    "ID_UEV", "LONGITUDE", "LATITUDE", "NO_ARROND_ILE_CUM",
    "ANNEE_CONSTRUCTION_NUM", "AGE_BATIMENT", "LOGEMENT_DENSITY",
    "TERRAIN_RATIO", "IS_CONDO", "IS_PARKING", "HAS_FIRE"
]
df_model = df_evaluationfonciere[columns_to_keep]


# In[29]:


# Extract time components for temporal aggregation
df_interventions["CREATION_DATE_TIME"] = pd.to_datetime(df_interventions["CREATION_DATE_TIME"])
df_interventions["YEAR_MONTH"] = df_interventions["CREATION_DATE_TIME"].dt.to_period("M")


# In[30]:


df_model.to_csv("merged_interventions_evaluationfonciere_adresses_binary_analysis_1.csv", index=False)


# In[31]:


df_model=pd.read_csv("merged_interventions_evaluationfonciere_adresses_binary_analysis_1.csv")


# In[32]:


df_model.head()


# In[33]:


# Count of buildings with at least one fire
num_buildings_with_fire = df_model[df_model["HAS_FIRE"] == 1]["ID_UEV"].nunique()
print(f"🔥 Houses (buildings) with fire: {num_buildings_with_fire}")


# In[34]:


# Step 1: Total number of unique buildings
total_buildings = df_model["ID_UEV"].nunique()

# Step 2: Buildings without fire
no_fire = total_buildings - 211487

# Step 3: Display results
print(f"✅ Buildings without fire: {no_fire}")
print(f"📊 Percentage with fire: {100 * 211487 / total_buildings:.2f}%")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




