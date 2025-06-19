#!/usr/bin/env python
# coding: utf-8

#to run the code in python: python datamodel/Monthly_fire_risk_prediction-test.py

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ✅ Run this from the ROOT of the project
# Example: C:\Users\mirei\OneDrive\Desktop\all-capstone-project-summer-2025-team-6-main

# 🔹 Paths
INPUT_CSV = os.path.join("datasets", "cleaned", "evaluation_with_fire_and_coordinates_and_date.csv")
OUTPUT_PANEL = os.path.join("datamodel", "building_month_fire_panel.csv")

# 🔸 Load and clean fire dataset
df = pd.read_csv(INPUT_CSV)
df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df["month"] = df["fire_date"].dt.to_period("M")
df = df.dropna(subset=["LONGITUDE", "LATITUDE", "ID_UEV"])
df["geometry"] = df.apply(lambda row: Point(row["LONGITUDE"], row["LATITUDE"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:32188")

# 🔸 Construct panel: building × month
unique_buildings = gdf[["ID_UEV", "LATITUDE", "LONGITUDE"]].drop_duplicates()
all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
panel = pd.MultiIndex.from_product([unique_buildings["ID_UEV"], all_months],
                                   names=["ID_UEV", "month"]).to_frame(index=False)
panel = panel.merge(unique_buildings, on="ID_UEV", how="left")

# 🔸 Label fire presence
fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
fires["HAS_FIRE_THIS_MONTH"] = 1
panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)

# 🔸 Add time-based features
panel["month_num"] = panel["month"].dt.month
panel["year"] = panel["month"].dt.year

# 💾 Save panel
panel.to_csv(OUTPUT_PANEL, index=False)

# 🔸 Load panel for modeling
df = pd.read_csv(OUTPUT_PANEL)
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["ID_UEV", "month"])
df["year"] = df["month"].dt.year

# 🔸 Add lag features
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# 🔸 Split into train and test
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]
features = ["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]
target = "HAS_FIRE_THIS_MONTH"

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# ⚖️ Class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# 🧠 Train XGBoost model
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
