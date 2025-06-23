#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


# In[5]:


from pathlib import Path
import os

# 🔧 Project root directory (2 levels up from current script)
ROOT = Path.cwd().parents[1]

# 🔹 Define input/output paths using ROOT
INPUT_CSV = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"
#OUTPUT_PANEL = ROOT / "datasets" / "cleaned" / "building_month_fire_panel_feat_eng.csv"

# 🔍 Optional: check existence
print("[input exists?]", INPUT_CSV.exists(), "➜", INPUT_CSV)
#print("[output dir exists?]", OUTPUT_PANEL.parent.exists(), "➜", OUTPUT_PANEL.parent)


# In[8]:


# 🔸 Load and clean fire dataset
df = pd.read_csv(INPUT_CSV)
df.head()


# In[10]:


# 🔸 Load panel for modeling
df["month"] = pd.to_datetime(df["month"])
df = df.sort_values(["ID_UEV", "month"])
df["year"] = df["month"].dt.year


# In[11]:


# 🔸 Add lag features
for lag in range(1, 4):
    df[f"fire_last_{lag}m"] = (
        df.groupby("ID_UEV")["HAS_FIRE_THIS_MONTH"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )


# In[12]:


features = [
    "MUNICIPALITE", "ETAGE_HORS_SOL", "NOMBRE_LOGEMENT", "AGE_BATIMENT",
    "CODE_UTILISATION", "CATEGORIE_UEF", "SUPERFICIE_TERRAIN", "SUPERFICIE_BATIMENT",
    "NO_ARROND_ILE_CUM", "RATIO_SURFACE", "DENSITE_LOGEMENT", "HAS_MULTIPLE_LOGEMENTS",
    "FIRE_FREQUENCY_ZONE", "FIRE_RATE_ZONE", "FIRE_COUNT_LAST_YEAR_ZONE", "BUILDING_COUNT",
    "FIRE_RATE_ZONE_NORM", "FIRE_COUNT_LAST_YEAR_ZONE_NORM", 
    "fire_last_1m", "fire_last_2m", "fire_last_3m","fire_cumcount","fire_rolling_3m","fire_rolling_6m","fire_rolling_12m",
    "month_num", "year"  #,"season"
]
target = "HAS_FIRE_THIS_MONTH"


# In[13]:


from sklearn.preprocessing import LabelEncoder

for col in ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# In[14]:


# First: convert in the full df
df["CATEGORIE_UEF"] = df["CATEGORIE_UEF"].astype("category")
df["NO_ARROND_ILE_CUM"] = df["NO_ARROND_ILE_CUM"].astype("category")

# Now re-split
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]


# In[ ]:





# In[15]:


print(train_df.columns.tolist())


# In[16]:


# ⚖️ Class imbalance
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()


# In[17]:


from xgboost import XGBClassifier

model = XGBClassifier(
    enable_categorical=True,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)


# In[18]:


# 📊 Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))


# 🔍 Interpretation of Metrics (Threshold = 0.2?)
# Metric	Class 0 (No Fire)	Class 1 (Fire)
# Precision	0.992	0.027
# Recall	0.698	0.603
# F1-score	0.819	0.051
# 
# 🔵 High recall for fire class (1): You're catching 60% of fires, which is good for early detection.
# 
# 🔴 Very low precision for fire class (1): Among the predicted fires, only 2.7% are actually fires.
# 
# ⚖️ Accuracy is misleading (69%) due to imbalance (only 1.3% fires in your data).

# 🎯 Interpretation
# This is typical for imbalanced binary classification:
# 
# Your model is tuned toward catching more fires (high recall).
# 
# But it's imprecise: many "fire" predictions are wrong.
# 
# 

# ✅ What You Did Well
# Feature engineering helped improve recall for rare event (fire).
# 
# Model is no longer "lazy" and defaulting to class 0.
# 
# Good step forward for early warning/risk flagging.
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# tune the threshold and optimize the tradeoff between precision and recall using XGBoost and sklearn:

# In[19]:


#✅ 1. Predict Probabilities on Test Set
# Predict probabilities (for class 1 = fire)
y_probs = model.predict_proba(X_test)[:, 1]



# In[20]:


#📈 2. Plot Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()


# it clearly shows the typical inverse relationship between precision and recall:
# 
# 🔸 Recall is very high (>90%) when the threshold is low (e.g., <0.2), but precision is very low.
# 
# 🔸 Precision slightly improves as threshold increases, but only becomes meaningful after ~0.8 — at the cost of very low recall.

# In[21]:


from sklearn.metrics import classification_report, confusion_matrix

# 🔧 Set threshold manually
threshold = 0.2
y_pred_custom = (y_probs >= threshold).astype(int)

# 📊 Evaluate
print(f"🔍 Classification report at threshold = {threshold}")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom, digits=3))



# ⚠️ What This Tells You
# ✅ You're capturing 95.6% of actual fires (very high recall).
# 
# ❌ But 98.4% of the "fire" predictions are false alarms (precision = 1.6%).
# 
# ❌ Overall accuracy drops to 23% because you're labeling so many buildings as high risk.
# 
# 📈 What to Do Next (Recommended Steps)
# 1. Raise the threshold to improve precision
# Right now, you're labeling nearly every building as fire-prone. A better threshold balances recall and precision. Try:
# 
# 
# **2. Use rebalanced training (under/oversampling) + time-aware features
# Your feature set is improving. Once you merge in more external data (e.g. weather, interventions), you'll get better precision without compromising recall.**

# ✅ Recommendation
# You need a better balance between recall and precision, especially if you're building a resource-planning or early warning system. 

#  🧪 Test Thresholds in [0.3, 0.5] Range
# These thresholds usually yield better trade-offs in imbalanced cases.

# In[22]:


for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    y_pred = (y_probs >= t).astype(int)
    print(f"Threshold: {t}")
    print(classification_report(y_test, y_pred, digits=3))
    print()


# ✅ Recommended Threshold: 0.45 or 0.5
# 🔹 Use 0.5 if:
# You're okay missing ~40% of fires (recall = 60%)
# 
# But you want higher precision and better overall accuracy
# 
# 🔹 Use 0.45 if:
# You want more recall (70%) and still get better precision than 0.3–0.4
# 
# You're still early in prototyping and prefer recall over precision

# 🔥 Fire Risk Modeling: Which Metric Matters Most?
# ✅ Choose Recall (0.45 threshold) if:
# Goal: Identify as many high-risk buildings as possible.
# 
# Why: You prefer to flag more potential risks (even with false alarms).
# 
# Use Case: Fire department wants early alerts, prevention, or targeted inspections.
# 
# Cost of Missing Fires (False Negatives) is high — it's worse to miss a fire than to overpredict.
# 
# Recommended Metric:
# 
# Recall (focus on sensitivity)
# 
# Optionally: F2-score (which emphasizes recall more than precision)
# 
# ✅ Choose Precision/Accuracy (0.5 threshold) if:
# Goal: Only flag buildings where you are confident fire will occur.
# 
# Why: You want to minimize false alarms — perhaps inspections are expensive.
# 
# Use Case: You only act if the risk is very real (limited resources).
# 
# Cost of False Positives is high (e.g., unnecessary inspections waste time/money).
# 
# Recommended Metric:
# 
# Precision
# 
# Overall Accuracy
# 
# 💡 Best Practice in Risk Modeling:
# Since fire is rare and high-cost, it's usually better to favor recall, especially in early stages of modeling:
# 
# ⚠️ “It’s better to catch 70% of fire risks with a few false alarms than to miss half the actual fires.”
# 
# ✅ Conclusion for You:
# Use threshold = 0.45, prioritize recall, and track F2-score to guide further model improvements.

# In[23]:


import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score

# 🔸 Assumes y_test (true labels) and y_probs (predicted probabilities) already exist

# Define thresholds to evaluate
thresholds = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]

# Store results
results = []

for threshold in thresholds:
    y_pred = (y_probs >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    
    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F2 Score": f2
    })

# Create DataFrame to view results
threshold_df = pd.DataFrame(results)
threshold_df = threshold_df.sort_values(by="F2 Score", ascending=False).reset_index(drop=True)

# 🔍 Show table sorted by F2 Score
threshold_df


# Interpretation (Fire Risk Modeling by Location & Month):
# | Threshold | Precision | Recall | F2 Score  | Comment                                                     |
# | --------- | --------- | ------ | --------- | ----------------------------------------------------------- |
# | **0.50**  | 0.027     | 0.603  | **0.113** |  **Best F2 Score**: balances recall & tolerable precision |
# | 0.45      | 0.025     | 0.698  | 0.107     |  Higher recall but drops F2 slightly                      |
# | 0.40      | 0.023     | 0.774  | 0.102     |  More recall, still decent                                |
# | 0.35–0.20 | ↓         | ↑      | ↓         |  Recall increases but F2 & precision drop too much        |
# 
#     

# 🔍 Recommendation for Monthly Fire Risk by Location:
# Use threshold = 0.50 as your current working value:
# 
# It provides the highest F2 score, which prioritizes recall more than precision — ideal for risk modeling (catch more true fires).
# 
# Keeps false positives relatively lower than aggressive recall thresholds (like 0.2–0.3).
# 
# 

# In[24]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precision", marker='o')
plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall", marker='o')
plt.plot(threshold_df["Threshold"], threshold_df["F2 Score"], label="F2 Score", marker='o')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F2 Score vs Threshold")
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ⚙️ Apply threshold
threshold = 0.50
y_pred_50 = (y_probs >= threshold).astype(int)

# 📊 Compute confusion matrix
cm_50 = confusion_matrix(y_test, y_pred_50)
tn, fp, fn, tp = cm_50.ravel()

# 🖼️ Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm_50, annot=True, fmt='d', cmap='Blues', xticklabels=["No Fire", "Fire"], yticklabels=["No Fire", "Fire"])
plt.title(f"Confusion Matrix at Threshold = {threshold}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 📄 Classification report
print(f"\nClassification Report at Threshold = {threshold}:\n")
print(classification_report(y_test, y_pred_50, digits=3))

# 📌 Optional: print raw values
print(f"True Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")


# 📌 Suggested Next Steps:
# Try threshold = 0.45 for slightly higher recall (~70%) and more TP.
# 
# Investigate false positives:
# 
# Are they spatially or temporally concentrated?
# 
# Are they near real fires? → might still be valuable.
# 
# Try probabilistic risk scores instead of hard 0/1 labels for decision-making (e.g., ranking top 5% riskiest buildings each month).
# 
# Add contextual features:
# 
# Weather, previous nearby fires, building age, intervention history.
# 
# Consider ensemble models to better separate classes.

# Recommended Ensemble Approaches
# 1. XGBoost (Gradient Boosted Trees)
# Handles imbalance well with scale_pos_weight
# 
# Captures nonlinear relationships & feature interactions
# 
# Already in your codebase? → Try tuning thresholds further.
# 
# 
# 

# In[26]:


from xgboost import XGBClassifier

# Make sure these are dtype 'category'
categorical_cols = ["CATEGORIE_UEF", "NO_ARROND_ILE_CUM"]
for col in categorical_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# Create model with categorical support
model = XGBClassifier(
    enable_categorical=True,  # ✅ Tell XGBoost to handle 'category' dtypes
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])),
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train the model
model.fit(X_train, y_train)


# In[27]:


# Predict class probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Default threshold = 0.5
y_pred = (y_pred_proba >= 0.5).astype(int)


# In[28]:


#Evaluate the model:
from sklearn.metrics import classification_report, confusion_matrix

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))


# 🔍 Interpretation of Results
# Class	Precision	Recall	F1-score	Support
# No Fire	0.9903	0.7436	0.8494	3,674,405
# Fire	0.0243	0.4665	0.0461	50,239
# 
# ✅ High precision for "No Fire" class — the model is very confident when predicting a "0".
# 
# ⚠️ Very low precision for "Fire" class — many false positives.
# 
# ⚠️ Moderate recall for "Fire" class (~47%) — the model catches fewer than half of actual fire months.
# 
# ⚠️ Overall accuracy is misleading (73.99%) due to class imbalance.
# 
# ⚠️ F1-score for fires is very low (0.0461) — imbalance and prediction difficulty.

# ✅ Recommendations
# 1. Adjust the Threshold
# Your previous evaluations showed:
# 
# At threshold = 0.45, recall can go above 69% with slightly lower precision.
# 
# Use this if recall is more important than precision (which is likely for fire risk modeling).
# 
# 2. Try F2 Score Optimization
# Use F2 to focus more on recall than precision:

# In[29]:


from sklearn.metrics import fbeta_score

best_f2 = 0
best_t = 0.5
for t in np.arange(0.2, 0.6, 0.05):
    preds = (y_pred_proba >= t).astype(int)
    f2 = fbeta_score(y_test, preds, beta=2)
    if f2 > best_f2:
        best_f2 = f2
        best_t = t
print(f"✅ Best Threshold by F2: {best_t} with F2 Score = {best_f2:.4f}")


# ✅ Summary
# Best Threshold (by F2): 0.55
# 
# F2 Score: 0.1026
# 
# This means that at a threshold of 0.55:
# 
# You're catching more true fire cases (higher recall).
# 
# While precision remains low, this is acceptable in early warning systems.
# 
# 📌 What You Should Do Next
# 🔹 1. Recalculate Metrics at 0.55

# In[30]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred_055 = (y_pred_proba >= 0.55).astype(int)

print(confusion_matrix(y_test, y_pred_055))
print(classification_report(y_test, y_pred_055, digits=4))


# 🔹 Metrics
# Metric	Class 0 (No Fire)	Class 1 (Fire)
# Precision	0.9896	0.0262
# Recall	0.8081	0.3780
# F1-score	0.8897	0.0490
# 
# Accuracy: 80.2%
# 
# Macro Avg F1: 0.469
# 
# Weighted Avg F1: 0.878
# 
# F2 Score (used for tuning): ≈ 0.103
# 
# 🔍 Interpretation for Fire Risk Modeling
# Goal	Threshold = 0.55 Outcome
# Recall (catching fires)	✅ Improved: 37.8% fires detected
# False positives tolerated	⚠️ Over 700k buildings misclassified as fire
# Tradeoff	Better recall for fire cases at cost of precision and FP load
# 
# This threshold is reasonable if you're building a risk prioritization tool, not a strict classifier. You can use it to flag locations for inspection or prevention, not for punitive action.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




