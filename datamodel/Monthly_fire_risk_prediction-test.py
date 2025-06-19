#!/usr/bin/env python
# coding: utf-8

# In[13]:


#We should add feature engineering and other features like weather, crime, population,...  to get better results

# 📦 Imports
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# In[ ]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


# In[14]:


# Load fire dataset
df = pd.read_csv("./datasets/cleaned/evaluation_with_fire_and_coordinates_and_date.csv")


# 🧼 Parse fire_date and create month field
df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df["month"] = df["fire_date"].dt.to_period("M")

# 📍 Convert to GeoDataFrame
df = df.dropna(subset=["LONGITUDE", "LATITUDE", "ID_UEV"])
df["geometry"] = df.apply(lambda row: Point(row["LONGITUDE"], row["LATITUDE"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:32188")

# 🏢 Get all unique buildings
unique_buildings = gdf[["ID_UEV", "LATITUDE", "LONGITUDE"]].drop_duplicates()

# 📅 Get full time range
all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
building_months = pd.MultiIndex.from_product(
    [unique_buildings["ID_UEV"], all_months],
    names=["ID_UEV", "month"]
).to_frame(index=False)

# 🔁 Merge static info
panel = building_months.merge(unique_buildings, on="ID_UEV", how="left")

# 🔥 Assign HAS_FIRE_THIS_MONTH
fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
fires["HAS_FIRE_THIS_MONTH"] = 1
panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)

# 🧠 Add optional time features
panel["month_num"] = panel["month"].dt.month
panel["year"] = panel["month"].dt.year

# 💾 Save result
panel.to_csv("./datamodel/building_month_fire_panel.csv", index=False)

# 🖼️ Preview
panel.sample(5)


# In[4]:


# 📦 Imports
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 📥 Load building + fire dataset
df = pd.read_csv("./datasets/cleaned/evaluation_with_fire_and_coordinates_and_date.csv")
df = df.dropna(subset=["ID_UEV", "LONGITUDE", "LATITUDE"])  # Ensure location info

# 🧼 Parse fire_date and extract month
df["fire_date"] = pd.to_datetime(df["fire_date"], errors="coerce")
df["month"] = df["fire_date"].dt.to_period("M")

# 📍 Convert to GeoDataFrame
df["geometry"] = df.apply(lambda row: Point(row["LONGITUDE"], row["LATITUDE"]), axis=1)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:32188")

# 🏢 Unique buildings
unique_buildings = gdf[["ID_UEV", "LATITUDE", "LONGITUDE"]].drop_duplicates()

# 📅 Time range
all_months = pd.period_range(start=gdf["month"].min(), end=gdf["month"].max(), freq="M")
building_month_grid = pd.MultiIndex.from_product(
    [unique_buildings["ID_UEV"], all_months], names=["ID_UEV", "month"]
).to_frame(index=False)

# 🔥 Fire label: HAS_FIRE_THIS_MONTH = 1 if fire happened to building that month
fires = gdf[gdf["fire"] == True][["ID_UEV", "month"]].drop_duplicates()
fires["HAS_FIRE_THIS_MONTH"] = 1

# 🧩 Merge static info and fire label
panel = building_month_grid.merge(unique_buildings, on="ID_UEV", how="left")
panel = panel.merge(fires, on=["ID_UEV", "month"], how="left")
panel["HAS_FIRE_THIS_MONTH"] = panel["HAS_FIRE_THIS_MONTH"].fillna(0).astype(int)

# 🧠 Add time features
panel["month_num"] = panel["month"].dt.month
panel["year"] = panel["month"].dt.year

# 💾 Save modeling-ready dataset
panel.to_csv("./datamodel/building_month_panel_2.csv", index=False)

# ✅ Preview
panel.sample(5)


# In[5]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the panel dataset
df = pd.read_csv("./datamodel/building_month_fire_panel.csv")
df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
df = df.sort_values(["ID_UEV", "month"])

# Create lag features (fires in last 1, 2, 3 months)
for lag in [1, 2, 3]:
    df[f"FIRE_LAG_{lag}M"] = (
        df.sort_values(["ID_UEV", "month"])
          .groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
          .shift(lag)
          .fillna(0)
    )

# Total fires in past 3 months
df["FIRE_LAG_3_SUM"] = df[["FIRE_LAG_1M", "FIRE_LAG_2M", "FIRE_LAG_3M"]].sum(axis=1)

# Train/test split: Train on 2020–2023, test on 2024
df["year"] = df["month"].dt.year
train = df[df["year"] <= 2023]
test  = df[df["year"] == 2024]

X_train = train[["month_num", "FIRE_LAG_3_SUM"]]
y_train = train["HAS_FIRE_THIS_MONTH"]

X_test = test[["month_num", "FIRE_LAG_3_SUM"]]
y_test = test["HAS_FIRE_THIS_MONTH"]

# Balance for class imbalance
scale = (len(y_train) - y_train.sum()) / y_train.sum()
model = XGBClassifier(scale_pos_weight=scale, eval_metric="logloss", use_label_encoder=False)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
print(report)


# In[6]:


# 📦 Imports
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 📥 Load the building-month panel
df = pd.read_csv("./datamodel/building_month_fire_panel.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["ID_UEV", "month"])

# ➕ Add lag features
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# 📅 Filter train/test based on year
df["year"] = df["month"].dt.year
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

# 🧪 Define features and target
features = ["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]
X_train = train_df[features]
y_train = train_df["HAS_FIRE_THIS_MONTH"]
X_test = test_df[features]
y_test = test_df["HAS_FIRE_THIS_MONTH"]

# ⚖️ Handle class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# 🧠 Train model
model = XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# 🧾 Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[7]:


# 📦 Imports
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 📥 Load panel
df = pd.read_csv("./datamodel/building_month_fire_panel.csv")
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["ID_UEV", "month"])

# ➕ Add lag features (1 to 3 months)
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# 🔍 Define training and testing sets
df["year"] = df["month"].dt.year
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

# 🧪 Define features and target
features = ["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]
X_train = train_df[features]
y_train = train_df["HAS_FIRE_THIS_MONTH"]
X_test = test_df[features]
y_test = test_df["HAS_FIRE_THIS_MONTH"]

# ⚖️ Compute scale_pos_weight for class imbalance
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

# 📊 Evaluate model
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


# In[8]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 📥 Load your lag-enhanced fire panel
df = pd.read_csv("./datamodel/building_month_fire_panel.csv")
df["month"] = pd.to_datetime(df["month"])

# 🔁 Add lag features
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# 🎯 Define target and features
features = ["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]
target = "HAS_FIRE_THIS_MONTH"

# 📆 Split into train/test by time
df["year"] = df["month"].dt.year
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# ⚖️ Handle imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

# 🧠 Train XGBoost model
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=5,
    learning_rate=0.1,
    n_estimators=150,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 🔍 Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# In[9]:


import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler  # Optional

# 📥 Load fire panel
df = pd.read_csv("./datamodel/building_month_fire_panel.csv")
df["month"] = pd.to_datetime(df["month"])
df["year"] = df["month"].dt.year

# 🔁 Add lag features (1 to 3 months)
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# 🧪 Train/test split (time-based)
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]]
y_train = train_df["HAS_FIRE_THIS_MONTH"]
X_test = test_df[["month_num", "fire_last_1m", "fire_last_2m", "fire_last_3m"]]
y_test = test_df["HAS_FIRE_THIS_MONTH"]

# ⚖️ Optional undersampling (uncomment to try)
# rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
# X_train, y_train = rus.fit_resample(X_train, y_train)

# ⚙️ Train model
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    max_depth=6,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# In[ ]:




