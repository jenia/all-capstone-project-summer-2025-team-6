#!/usr/bin/env python
# coding: utf-8

# In[13]:


#We should add feature engineering and other features like weather, crime, population,...  to get better results


# In[ ]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


# In[14]:


# Load fire dataset
df = pd.read_csv("./datasets/cleaned/evaluation_with_fire_and_coordinates_and_date.csv")
df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df_fire = df[df["fire"] == True].copy()


# In[15]:


# Create GeoDataFrame and project to meters
gdf_fire = gpd.GeoDataFrame(
    df_fire,
    geometry=gpd.points_from_xy(df_fire["LONGITUDE"], df_fire["LATITUDE"]),
    crs="EPSG:4326"
).to_crs("EPSG:32188")


# In[16]:


# Add month and buffer
gdf_fire["month"] = gdf_fire["fire_date"].dt.to_period("M")
gdf_fire["buffer_geometry"] = gdf_fire.geometry.buffer(100)
gdf_fire = gdf_fire.set_geometry("buffer_geometry")
gdf_fire["buffer_id"] = gdf_fire["buffer_geometry"].apply(lambda g: hash(g.wkt))


# In[17]:


# Fire count per buffer per month
fire_counts = gdf_fire.groupby(["buffer_id", "month"]).size().reset_index(name="fire_count")
fire_counts["has_fire"] = (fire_counts["fire_count"] > 0).astype(int)


# In[18]:


# Create full grid of buffer-month combinations
all_months = pd.period_range(start=gdf_fire["month"].min(), end=gdf_fire["month"].max(), freq="M")
all_buffers = gdf_fire["buffer_id"].unique()
grid = pd.MultiIndex.from_product([all_buffers, all_months], names=["buffer_id", "month"])
df_grid = pd.DataFrame(index=grid).reset_index()
df_all = df_grid.merge(fire_counts[["buffer_id", "month", "has_fire"]], how="left")
df_all["has_fire"] = df_all["has_fire"].fillna(0).astype(int)


# In[19]:


# Total historical fires per buffer
hist_fires = fire_counts.groupby("buffer_id")["fire_count"].sum().reset_index()
hist_fires.columns = ["buffer_id", "total_past_fires"]
df_all = df_all.merge(hist_fires, on="buffer_id", how="left").fillna(0)


# In[20]:


# Month feature
df_all["month_num"] = df_all["month"].dt.month


# In[21]:


# Export to CSV
df_all.to_csv("fire_risk_buffer_features.csv", index=False)


# In[22]:


# Train and evaluate models for each month
results = {}
for m in sorted(df_all["month"].unique()):
    df_month = df_all[df_all["month"] == m]
    X = df_month[["month_num", "total_past_fires"]]
    y = df_month["has_fire"]

    if y.sum() == 0 or y.sum() == len(y):
        continue  # skip months with no positive cases

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[str(m)] = report


# In[23]:


# Compile results for each month
monthly_results = pd.DataFrame({
    month: {
        "precision_1": res["1"]["precision"],
        "recall_1": res["1"]["recall"],
        "f1_1": res["1"]["f1-score"],
        "support_1": res["1"]["support"]
    }
    for month, res in results.items()
}).T


# In[24]:


monthly_results.to_csv("./datasets/cleaned/monthly_fire_risk_results.csv")
monthly_results.head()


# In[ ]:





# In[ ]:




