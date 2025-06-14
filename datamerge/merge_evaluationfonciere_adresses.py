#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import geopandas as gpd


# In[ ]:





# In[37]:


DIRECTORY = r'G:\.shortcut-targets-by-id\1uExmPmKnHKKlOfMdT70cXpwXvdf9aVEC\Capstone Project summer 2025- Team6\datasets'
ORIGINAL_FILE_NAME_EVAL_CLEANED = 'eval_cleaned.csv'
ORIGINAL_FILE_NAME_ADRESSES = 'adresses.csv'
DESTINATION_FILE_NAME = "merged_evaluationfonciere_adresses.csv"


# In[38]:


eval_df = pd.read_csv(ORIGINAL_FILE_NAME_EVAL_CLEANED)


# In[39]:


addr_df = pd.read_csv(ORIGINAL_FILE_NAME_ADRESSES, dtype=str)


# In[40]:


#CIVIQUE_DEBUT is already numeric


# In[41]:


# --- Clean and prepare eval_df ---
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].astype(int)

eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\s+\(.*)?$")[0].str.lower().str.strip()


# In[42]:


# --- Clean and prepare addr_df ---
addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = (
    addr_df["GENERIQUE"].str.lower().str.strip() + " " +
    addr_df["SPECIFIQUE"].str.lower().str.strip()
)


# In[43]:


# --- Merge eval_df with addr_df to get coordinates ---
eval_with_coords = pd.merge(eval_df, addr_df,
                            left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                            right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                            how="left")


# In[44]:


eval_with_coords.head()


# In[45]:


eval_with_coords.to_csv("eval_with_coords.csv", index=False)


# In[46]:


eval_with_coords.info()


# In[47]:


# Check for missing coordinate values
missing_coords = eval_with_coords[eval_with_coords["LONGITUDE"].isna() | eval_with_coords["LATITUDE"].isna()]

# Summary of missing coordinates
print(f"🧭 Missing coordinates: {len(missing_coords):,} rows")
print(f"✅ Matched coordinates: {len(eval_with_coords) - len(missing_coords):,} rows")
print(f"📊 Percentage matched: {(1 - len(missing_coords) / len(eval_with_coords)) * 100:.2f}%")

# Optionally display the first few rows with missing coordinates
missing_coords.head()


# In[48]:


# --- Remove rows without coordinates before spatial join ---
eval_with_coords = eval_with_coords.dropna(subset=["LONGITUDE", "LATITUDE"])


# # Convert to GeoDataFrame

# In[49]:


import geopandas as gpd
from shapely.geometry import Point

# Convert to float first (if not already)
eval_with_coords["LONGITUDE"] = eval_with_coords["LONGITUDE"].astype(float)
eval_with_coords["LATITUDE"] = eval_with_coords["LATITUDE"].astype(float)

# Create GeoDataFrame
eval_gdf = gpd.GeoDataFrame(
    eval_with_coords,
    geometry=gpd.points_from_xy(eval_with_coords["LONGITUDE"], eval_with_coords["LATITUDE"]),
    crs="EPSG:4326"
)


# # Feature Engineering

# | Feature            | Description                                | Code Hint                               |
# | ------------------ | ------------------------------------------ | --------------------------------------- |
# | `AGE_BATIMENT`     | Building age in years                      | `2025 - year`                           |
# | `LOGEMENT_DENSITY` | Units per m²                               | `NOMBRE_LOGEMENT / SUPERFICIE_BATIMENT` |
# | `TERRAIN_RATIO`    | `SUPERFICIE_TERRAIN / SUPERFICIE_BATIMENT` | Density indicator                       |
# | `IS_CONDO`         | Flag for `CATEGORIE_UEF == "Condominium"`  | Binary feature                          |
# | `IS_PARKING`       | Flag from `LIBELLE_UTILISATION`            | Identify low-risk structures            |
# 

# In[50]:


# --- Load the merged dataset ---
import pandas as pd

eval_df = pd.read_csv("eval_with_coords.csv")

# --- Feature: AGE_BATIMENT ---
# Convert ANNEE_CONSTRUCTION to numeric (coerce errors like 'unknown' to NaN)
eval_df["ANNEE_CONSTRUCTION_NUM"] = pd.to_numeric(eval_df["ANNEE_CONSTRUCTION"], errors='coerce')
eval_df["AGE_BATIMENT"] = 2025 - eval_df["ANNEE_CONSTRUCTION_NUM"]

# --- Feature: LOGEMENT_DENSITY ---
# Avoid division by zero
eval_df["LOGEMENT_DENSITY"] = eval_df["NOMBRE_LOGEMENT"] / eval_df["SUPERFICIE_BATIMENT"].replace(0, pd.NA)

# --- Feature: TERRAIN_RATIO ---
eval_df["TERRAIN_RATIO"] = eval_df["SUPERFICIE_TERRAIN"] / eval_df["SUPERFICIE_BATIMENT"].replace(0, pd.NA)

# --- Feature: IS_CONDO ---
eval_df["IS_CONDO"] = (eval_df["CATEGORIE_UEF"].str.lower() == "condominium").astype(int)

# --- Feature: IS_PARKING ---
eval_df["IS_PARKING"] = eval_df["LIBELLE_UTILISATION"].str.lower().str.contains("stationnement").fillna(False).astype(int)

# --- Preview the result ---
eval_df[["AGE_BATIMENT", "LOGEMENT_DENSITY", "TERRAIN_RATIO", "IS_CONDO", "IS_PARKING"]].head()


# | Feature                | Interpretation for Fire Risk Modeling                                                  |
# | ---------------------- | -------------------------------------------------------------------------------------- |
# | **`AGE_BATIMENT`**     | Older buildings may have outdated wiring or materials → **higher risk**                |
# | **`LOGEMENT_DENSITY`** | Higher density may indicate more occupants per m² → **faster fire spread**             |
# | **`TERRAIN_RATIO`**    | A low ratio means more building footprint relative to lot size → **less buffer space** |
# | **`IS_CONDO`**         | Condos may have stricter fire codes → **possibly lower risk**                          |
# | **`IS_PARKING`**       | Parking areas (interior/exterior) usually have **lower fire risk**                     |
# 

# In[51]:


# Add previously engineered features
eval_df["ANNEE_CONSTRUCTION_NUM"] = pd.to_numeric(eval_df["ANNEE_CONSTRUCTION"], errors='coerce')
eval_df["AGE_BATIMENT"] = 2025 - eval_df["ANNEE_CONSTRUCTION_NUM"]
eval_df["LOGEMENT_DENSITY"] = eval_df["NOMBRE_LOGEMENT"] / eval_df["SUPERFICIE_BATIMENT"].replace(0, pd.NA)
eval_df["TERRAIN_RATIO"] = eval_df["SUPERFICIE_TERRAIN"] / eval_df["SUPERFICIE_BATIMENT"].replace(0, pd.NA)
eval_df["IS_CONDO"] = (eval_df["CATEGORIE_UEF"].str.lower() == "condominium").astype(int)
eval_df["IS_PARKING"] = eval_df["LIBELLE_UTILISATION"].str.lower().str.contains("stationnement").fillna(False).astype(int)

# Save the enriched dataset
output_path = "eval_with_coords_feat_eng.csv"
eval_df.to_csv(output_path, index=False)

output_path


# In[53]:


import pandas as pd

# Load the cleaned evaluation foncière dataset
eval_df = pd.read_csv("eval_with_coords_feat_eng.csv")

# List of columns to drop (redundant or irrelevant for spatial analysis)
columns_to_drop = [
    'MATRICULE83', 'ID_ADRESSE', 'TEXTE', 'ADDR_DE', 'ADDR_A', 'X', 'Y',
    'NOM_RUE_CLEAN', 'ORIENTATION', 'LIEN', 'HAUTEUR', 'GENERIQUE', 'ANGLE',
    'LETTRE_DEBUT', 'LETTRE_FIN'
]

# Drop the columns
eval_df_reduced = eval_df.drop(columns=columns_to_drop)

# Optional: Save reduced version
eval_df_reduced.to_csv("merged_evaluationfonciere_adresses.csv", index=False)

# Check shape and column names
print("✅ Reduced dataset shape:", eval_df_reduced.shape)
print("🧾 Remaining columns:\n", eval_df_reduced.columns.tolist())


# In[ ]:





# In[ ]:




