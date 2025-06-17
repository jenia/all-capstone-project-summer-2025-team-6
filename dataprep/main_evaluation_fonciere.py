import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
# --- Load datasets ---
cleaned_folder=os.path.join('.','datasets','cleaned')
eval_df = pd.read_csv(os.path.join(cleaned_folder,'eval_cleaned.csv'), dtype=str)
addr_df = pd.read_csv(os.path.join(cleaned_folder,"adresses.csv"), dtype=str)
inc_df = pd.read_csv(os.path.join(cleaned_folder,"interventions_cleaned_with_has_fire.csv"))
OUTPUT_FILE = os.path.join(cleaned_folder,"evaluation_with_fire_and_coordinates.csv")

# 🔥 Filter only incidents involving fire
inc_df = inc_df[
    (
        inc_df["DESCRIPTION_GROUPE"].str.contains("feu", case=False, na=False)
    ) &
    ~(
        inc_df["DESCRIPTION_GROUPE"].str.contains("sans feu", case=False, na=False)
    )
]

# --- Clean and prepare eval_df ---
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\s+\(.*)?$")[0].str.lower().str.strip()

# --- Clean and prepare addr_df ---
addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = (
    addr_df["GENERIQUE"].str.lower().str.strip() + " " +
    addr_df["SPECIFIQUE"].str.lower().str.strip()
)

# --- Merge eval_df with addr_df to get coordinates ---
eval_with_coords = pd.merge(eval_df, addr_df,
                            left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                            right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                            how="left")

# --- Remove rows without coordinates before spatial join ---
eval_with_coords = eval_with_coords.dropna(subset=["LONGITUDE", "LATITUDE"])

# --- Convert to GeoDataFrame ---
eval_gdf = gpd.GeoDataFrame(
    eval_with_coords,
    geometry=gpd.points_from_xy(eval_with_coords["LONGITUDE"].astype(float),
                                 eval_with_coords["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)

# --- Convert incidents to GeoDataFrame ---
incident_gdf = gpd.GeoDataFrame(
    inc_df,
    geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
    crs="EPSG:4326"
)

# --- Project both to meters for spatial operations ---
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = incident_gdf.to_crs(epsg=32188)

# --- Buffer fire incidents by 100 meters ---
incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")

# --- Spatial join: find properties within 100m of a fire incident ---
joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')

# --- Use unique matched ID_UEV set to mark fires ---
matched_ids = set(joined["ID_UEV"])

# --- Back to original eval_df (including unmatched rows) ---
# Assign fire flag based on ID_UEV
eval_df["fire"] = eval_df["ID_UEV"].isin(matched_ids)

# --- Merge coordinates (if available) back into eval_df ---
eval_df = pd.merge(eval_df, addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]],
                   left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                   right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                   how="left")

# --- Save final result ---
eval_df.to_csv(OUTPUT_FILE, index=False)
