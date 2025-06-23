#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Set working directory to feat_eng/dataprep_feat_eng
os.chdir(os.path.dirname(__file__))

# Paths
DATA_DIR = os.path.abspath(os.path.join("..", "datasets_feat_eng", "cleaned_feat_eng"))
eval_file = os.path.join(DATA_DIR, "eval_cleaned_feat_eng_1.csv")
addr_file = os.path.join(DATA_DIR, "adresses.csv")
inc_file = os.path.join(DATA_DIR, "interventions_cleaned_with_has_fire.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "evaluation_fire_coordinates_date_feat_eng.csv")

# Load datasets
eval_df = pd.read_csv(eval_file, dtype=str)
addr_df = pd.read_csv(addr_file, dtype=str)
inc_df = pd.read_csv(inc_file)

# Filter fire incidents
inc_df = inc_df[inc_df["DESCRIPTION_GROUPE"].str.contains("INCENDIE", case=False, na=False)]

# Clean eval_df
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\\s+\\(.*)?$")[0].str.lower().str.strip()
original_eval_df = eval_df.copy()

# Clean addr_df
addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = addr_df["GENERIQUE"].str.lower().str.strip() + " " + addr_df["SPECIFIQUE"].str.lower().str.strip()

# Merge to get coordinates
eval_with_coords = pd.merge(eval_df, addr_df,
                            left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                            right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                            how="left").dropna(subset=["LONGITUDE", "LATITUDE"])

# Convert to GeoDataFrame
eval_gdf = gpd.GeoDataFrame(eval_with_coords,
                            geometry=gpd.points_from_xy(eval_with_coords["LONGITUDE"].astype(float),
                                                        eval_with_coords["LATITUDE"].astype(float)),
                            crs="EPSG:4326").to_crs(epsg=32188)

# Prepare incident GeoDataFrame
inc_df["CREATION_DATE_TIME"] = pd.to_datetime(inc_df["CREATION_DATE_TIME"], errors='coerce')
incident_gdf = gpd.GeoDataFrame(inc_df,
                                geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
                                crs="EPSG:4326").to_crs(epsg=32188)
incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")

# Spatial join
detected = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')
detected = detected.rename(columns={"CREATION_DATE_TIME": "fire_date"})
detected["fire"] = True

# Extract relevant fields
fire_records = detected[["ID_UEV", "fire_date", "NOMBRE_UNITES", "CASERNE"]].copy()
fire_records["fire"] = True

# Merge fire info back
final_df = pd.merge(original_eval_df, fire_records, on="ID_UEV", how="left")
final_df["fire"] = final_df["fire"].fillna(False)
final_df["fire_date"] = pd.to_datetime(final_df["fire_date"])

# Merge coordinates again for rows lost previously
addr_df_subset = addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]]
final_df = pd.merge(final_df,
                    addr_df_subset,
                    left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                    right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                    how="left")

# Save merged data
final_df.to_csv(os.path.join(DATA_DIR, "evaluation_fire_coordinates_date_feat_eng.csv"), index=False)

# Add zone-level features
final_df["NO_ARROND_ILE_CUM"] = final_df["NO_ARROND_ILE_CUM"].astype(str)
fires_2024 = final_df[(final_df["fire"] == True) & (final_df["fire_date"].dt.year == 2024)]

fire_count_by_zone = fires_2024.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="FIRE_COUNT_LAST_YEAR_ZONE")
building_count_by_zone = final_df.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="BUILDING_COUNT")

final_df = final_df.merge(fire_count_by_zone, on="NO_ARROND_ILE_CUM", how="left")
final_df = final_df.merge(building_count_by_zone, on="NO_ARROND_ILE_CUM", how="left")
final_df["FIRE_COUNT_LAST_YEAR_ZONE"] = final_df["FIRE_COUNT_LAST_YEAR_ZONE"].fillna(0)
final_df["FIRE_RATE_ZONE"] = (final_df["FIRE_COUNT_LAST_YEAR_ZONE"] / final_df["BUILDING_COUNT"]).fillna(0)

# Normalize
scaler = MinMaxScaler()
final_df[["FIRE_COUNT_LAST_YEAR_ZONE_NORM", "FIRE_RATE_ZONE_NORM"]] = scaler.fit_transform(
    final_df[["FIRE_COUNT_LAST_YEAR_ZONE", "FIRE_RATE_ZONE"]]
)

# Save again
final_df.to_csv(os.path.join(DATA_DIR, "eval_fire_coordinates_date_feat_eng_1.csv"), index=False)

# Add FIRE_RISK_LEVEL_ZONE
final_df["FIRE_RISK_LEVEL_ZONE"] = final_df["FIRE_RATE_ZONE"].apply(
    lambda r: "High" if r >= 0.1 else "Medium" if r >= 0.03 else "Low"
)

# Optional: Print summary
print("✅ Saved final dataset with risk levels and normalized features")
