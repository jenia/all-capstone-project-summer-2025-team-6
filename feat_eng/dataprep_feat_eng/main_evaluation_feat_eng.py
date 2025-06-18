#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# In[2]:


# --- Load datasets ---
eval_df = pd.read_csv("./feat_eng/datasets_feat_eng/cleaned_feat_eng/eval_cleaned_feat_eng_1.csv", dtype=str)

addr_df = pd.read_csv("./datasets/cleaned/adresses.csv", dtype=str)
inc_df = pd.read_csv("./datasets/cleaned/interventions_cleaned_with_has_fire.csv")

OUTPUT_FILE = "./feat_eng/datasets_feat_eng/cleaned_feat_eng/evaluation_fire_coordinates_date_feat_eng_1.csv"


#eval_df = pd.read_csv("eval_cleaned_feat_eng_1.csv", dtype=str)
#addr_df = pd.read_csv("adresses.csv", dtype=str)
#inc_df = pd.read_csv("interventions_cleaned_with_has_fire.csv")
#OUTPUT_FILE = "evaluation_fire_coordinates_date_feat_eng_1.csv"



# In[3]:


# 🔥 Filter only incidents involving fire
inc_df = inc_df[
    inc_df["DESCRIPTION_GROUPE"].str.contains("INCENDIE", case=False, na=False)
]


# In[4]:


# --- Clean and prepare eval_df ---
eval_df["CIVIQUE_DEBUT"] = eval_df["CIVIQUE_DEBUT"].str.strip().astype(int)
eval_df["NOM_RUE_CLEAN"] = eval_df["NOM_RUE"].str.extract(r"^(.*?)(?:\s+\(.*)?$")[0].str.lower().str.strip()


# In[5]:


# ✅ Save original now that NOM_RUE_CLEAN exists
original_eval_df = eval_df.copy()


# In[6]:


# --- Clean and prepare addr_df ---
addr_df["ADDR_DE"] = addr_df["ADDR_DE"].astype(int)
addr_df["NOM_RUE_CLEAN"] = (
    addr_df["GENERIQUE"].str.lower().str.strip() + " " +
    addr_df["SPECIFIQUE"].str.lower().str.strip()
)


# In[7]:


# --- Merge eval_df with addr_df to get coordinates ---
eval_with_coords = pd.merge(eval_df, addr_df,
                            left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
                            right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
                            how="left")


# In[8]:


# --- Remove rows without coordinates before spatial join ---
eval_with_coords = eval_with_coords.dropna(subset=["LONGITUDE", "LATITUDE"])


# In[9]:


# --- Convert to GeoDataFrame ---
eval_gdf = gpd.GeoDataFrame(
    eval_with_coords,
    geometry=gpd.points_from_xy(eval_with_coords["LONGITUDE"].astype(float),
                                 eval_with_coords["LATITUDE"].astype(float)),
    crs="EPSG:4326"
)


# In[10]:


# --- Convert incidents to GeoDataFrame ---
inc_df["CREATION_DATE_TIME"] = pd.to_datetime(inc_df["CREATION_DATE_TIME"], errors='coerce')
incident_gdf = gpd.GeoDataFrame(
    inc_df,
    geometry=gpd.points_from_xy(inc_df["LONGITUDE"], inc_df["LATITUDE"]),
    crs="EPSG:4326"
)


# In[11]:


# --- Project both to meters for spatial operations ---
eval_gdf = eval_gdf.to_crs(epsg=32188)
incident_gdf = incident_gdf.to_crs(epsg=32188)
incident_gdf["buffer"] = incident_gdf.geometry.buffer(100)
incident_buffer_gdf = incident_gdf.set_geometry("buffer")


# In[12]:


# --- Spatial join: match each home to nearby fires ---
joined = gpd.sjoin(eval_gdf, incident_buffer_gdf, predicate='within', how='inner')
joined = joined.rename(columns={"CREATION_DATE_TIME": "fire_date"})
joined["fire"] = True


# In[13]:


# --- Extract relevant fire info ---
#fire_records = joined[["ID_UEV", "fire_date"]].copy()
#fire_records["fire"] = True


# In[15]:


print(joined.columns.tolist())


# In[16]:


# --- Extract relevant fire information for merging ---
fire_records = joined[[
    "ID_UEV",
    "fire_date",
    #"DESCRIPTION_GROUPE",
    #"INCIDENT_TYPE_DESC",
    "NOMBRE_UNITES",
    "CASERNE",
    #"NOM_ARROND",
    #"DIVISION"
]].copy()

fire_records["fire"] = True



# In[ ]:





# In[ ]:





# In[17]:


# --- Merge fire flags and fire dates into full dataset ---
final_df = pd.merge(original_eval_df, fire_records, on="ID_UEV", how="left")
final_df["fire"] = final_df["fire"].fillna(False)
final_df["fire_date"] = pd.to_datetime(final_df["fire_date"])


# In[18]:


# --- Add coordinates back (if available) ---
addr_df_subset = addr_df[["ADDR_DE", "NOM_RUE_CLEAN", "LONGITUDE", "LATITUDE"]]
final_df = pd.merge(final_df,
    addr_df_subset,
    left_on=["CIVIQUE_DEBUT", "NOM_RUE_CLEAN"],
    right_on=["ADDR_DE", "NOM_RUE_CLEAN"],
    how="left"
)


# In[19]:


# --- Save full dataset ---
#final_df.to_csv(OUTPUT_FILE, index=False)
#final_df.to_csv("evaluation_fire_coordinates_date_feat_eng.csv", index=False)


# In[20]:


# --- Summary ---
print("Houses with incident:", final_df["fire"].sum())
print("Houses without incident:", (~final_df["fire"]).sum())
print("Houses total:", len(final_df))


# In[21]:


final_df.head()


# In[22]:


final_df.info()


# # Feature Engineering

# In[ ]:





# In[23]:


# 🔁 Ensure NO_ARROND_ILE_CUM is the same type in both DataFrames
final_df["NO_ARROND_ILE_CUM"] = final_df["NO_ARROND_ILE_CUM"].astype(str)
fires_2024 = final_df[
    (final_df["fire"] == True) & 
    (final_df["fire_date"].notna()) & 
    (final_df["fire_date"].dt.year == 2024)
].copy()
fires_2024["NO_ARROND_ILE_CUM"] = fires_2024["NO_ARROND_ILE_CUM"].astype(str)

# ✅ Count fires per zone
fire_count_by_zone = (
    fires_2024.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="FIRE_COUNT_LAST_YEAR_ZONE")
)

# ✅ Count buildings per zone
building_count_by_zone = (
    final_df.groupby("NO_ARROND_ILE_CUM").size().reset_index(name="BUILDING_COUNT")
)

# ✅ Merge counts into final_df
final_df = final_df.merge(fire_count_by_zone, on="NO_ARROND_ILE_CUM", how="left")
final_df = final_df.merge(building_count_by_zone, on="NO_ARROND_ILE_CUM", how="left")

# ✅ Fill missing values
final_df["FIRE_COUNT_LAST_YEAR_ZONE"] = final_df["FIRE_COUNT_LAST_YEAR_ZONE"].fillna(0)
final_df["FIRE_RATE_ZONE"] = (
    final_df["FIRE_COUNT_LAST_YEAR_ZONE"] / final_df["BUILDING_COUNT"]
).fillna(0)


# In[24]:


final_df.head()


# In[25]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
final_df[["FIRE_COUNT_LAST_YEAR_ZONE_NORM", "FIRE_RATE_ZONE_NORM"]] = scaler.fit_transform(
    final_df[["FIRE_COUNT_LAST_YEAR_ZONE", "FIRE_RATE_ZONE"]]
)


# In[26]:


import pandas as pd

# If final_df is already defined in memory from previous steps:
#final_df.to_csv("eval_fire_coordinates_date_feat_eng_1.csv", index=False)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ File saved as eval_fire_coordinates_date_feat_eng_1.csv")




# In[27]:


final_df.head()


# In[29]:


final_df.info()


# In[ ]:





# In[ ]:





# In[ ]:





#  Categorize FIRE_RATE_ZONE into risk levels

# In[30]:


def assign_risk_level(rate):
    if rate >= 0.1:
        return "High"
    elif rate >= 0.03:
        return "Medium"
    else:
        return "Low"

final_df["FIRE_RISK_LEVEL_ZONE"] = final_df["FIRE_RATE_ZONE"].apply(assign_risk_level)


# In[31]:


import matplotlib.pyplot as plt

zone_summary = final_df.groupby("NO_ARROND_ILE_CUM")[["FIRE_COUNT_LAST_YEAR_ZONE", "FIRE_RATE_ZONE"]].mean()

zone_summary["FIRE_RATE_ZONE"].plot(kind='bar', figsize=(12, 5), title="Average Fire Rate by Borough")
plt.ylabel("Fire Rate")
plt.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import itertools

# 🔁 Step 1: Filter to valid fire events with dates
fires_with_date = final_df[(final_df["fire"] == True) & (final_df["fire_date"].notna())].copy()
fires_with_date["year_month"] = fires_with_date["fire_date"].dt.to_period("M").astype(str)

# ✅ Step 2: Count fires per (borough, month)
fire_counts_monthly = (
    fires_with_date.groupby(["NO_ARROND_ILE_CUM", "year_month"])
    .size()
    .reset_index(name="FIRE_COUNT_ZONE_MONTH")
)

# ✅ Step 3: Count buildings per borough
building_counts = (
    final_df.groupby("NO_ARROND_ILE_CUM")
    .size()
    .reset_index(name="BUILDING_COUNT")
)

# ✅ Step 4: Create all (borough, month) combinations
zones = final_df["NO_ARROND_ILE_CUM"].dropna().unique()
months = fires_with_date["year_month"].dropna().unique()
zone_month_pairs = pd.DataFrame(itertools.product(zones, months), columns=["NO_ARROND_ILE_CUM", "year_month"])

# ✅ Step 5: Merge & compute fire rate
zone_month_stats = zone_month_pairs.merge(fire_counts_monthly, on=["NO_ARROND_ILE_CUM", "year_month"], how="left")
zone_month_stats = zone_month_stats.merge(building_counts, on="NO_ARROND_ILE_CUM", how="left")
zone_month_stats["FIRE_COUNT_ZONE_MONTH"] = zone_month_stats["FIRE_COUNT_ZONE_MONTH"].fillna(0)
zone_month_stats["FIRE_RATE_ZONE_MONTH"] = zone_month_stats["FIRE_COUNT_ZONE_MONTH"] / zone_month_stats["BUILDING_COUNT"]

# ✅ Preview
zone_month_stats.sort_values(by=["NO_ARROND_ILE_CUM", "year_month"]).head()


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load zone_month_stats if not already defined
zone_month_stats = pd.read_csv("zone_month_stats.csv") if "zone_month_stats" not in locals() else zone_month_stats

# Convert year_month to datetime for plotting
zone_month_stats["year_month"] = pd.to_datetime(zone_month_stats["year_month"])

# Compute average FIRE_RATE_ZONE_MONTH by month across all zones
monthly_avg = (
    zone_month_stats.groupby("year_month")["FIRE_RATE_ZONE_MONTH"]
    .mean()
    .reset_index(name="Avg_Fire_Rate")
)

# Plot the trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg, x="year_month", y="Avg_Fire_Rate", marker="o")
plt.title("Average Monthly Fire Rate Across Boroughs")
plt.xlabel("Month")
plt.ylabel("Average Fire Rate")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




