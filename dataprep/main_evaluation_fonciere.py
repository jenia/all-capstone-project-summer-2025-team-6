import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# --- Load datasets ---
eval_df = pd.read_csv("./datasets/cleaned/eval_cleaned.csv", dtype=str)
addr_df = pd.read_csv("./datasets/cleaned/adresses.csv", dtype=str)
inc_df = pd.read_csv("./datasets/cleaned/interventions_cleaned_with_has_fire.csv")
OUTPUT_FILE = "./datasets/cleaned/evaluation_with_fire_and_coordinates.csv"

# 🔥 Filter only incidents involving fire
inc_df = inc_df[
    inc_df["DESCRIPTION_GROUPE"].str.contains("INCENDIE", case=False, na=False)
]

# --- Clean and prepare eval_df ---
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\s+\(.*)?$")[0].str.lower().str.strip()

# ✅ Save original now that NOM_RUE_CLEAN exists
original_eval_df = eval_df.copy()

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
inc_df["CREATION_DATE_TIME"] = pd.to_datetime(inc_df["CREATION_DATE_TIME"], errors='coerce')
incident_gdf = gpd.GeoDataFrame(
    inc_df,
    geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
    crs="EPSG:4326"
)

# --- Project both to meters for spatial operations ---
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = incident_gdf.to_crs(epsg=32188)
incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")

# --- Spatial join: match each home to nearby fires ---
joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')
joined = joined.rename(columns={"CREATION_DATE_TIME": "fire_date"})
joined["fire"] = True

# --- Extract relevant fire info ---
fire_records = joined[["ID_UEV", "fire_date"]].copy()
fire_records["fire"] = True

# --- Merge fire flags and fire dates into full dataset ---
final_df = pd.merge(original_eval_df, fire_records, on="ID_UEV", how="left")
final_df["fire"] = final_df["fire"].fillna(False)
final_df["fire_date"] = pd.to_datetime(final_df["fire_date"])

# --- Add coordinates back (if available) ---
addr_df_subset = addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]]
final_df = pd.merge(final_df,
    addr_df_subset,
    left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
    right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
    how="left"
)

# --- Save full dataset ---
final_df.to_csv(OUTPUT_FILE, index=False)

# --- Summary ---
print("Houses with incident:", final_df["fire"].sum())
print("Houses without incident:", (~final_df["fire"]).sum())
print("Houses total:", len(final_df))
