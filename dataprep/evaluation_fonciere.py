import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ORIGINAL_FILE_NAME_EVAL = './datasets/raw/uniteevaluationfonciere.csv'
DESTINATION_FILE_NAME = './datasets/cleaned/eval_cleaned.csv'

df_eval = pd.read_csv(ORIGINAL_FILE_NAME_EVAL )

df_eval.head()

df_eval.info()

# Check distribution of ANNEE_CONSTRUCTION
year_series = df_eval["ANNEE_CONSTRUCTION"]

# Plot histogram to identify outliers visually
plt.figure(figsize=(10, 4))
plt.hist(year_series, bins=80, edgecolor='black')
plt.title("Distribution of ANNEE_CONSTRUCTION")
plt.xlabel("Year of Construction")
plt.ylabel("Number of Properties")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get descriptive statistics to help define outliers
year_series.describe()

# Mark unrealistic years as missing
mask = (df_eval["ANNEE_CONSTRUCTION"] < 1800) | (df_eval["ANNEE_CONSTRUCTION"] > 2025)
df_eval.loc[mask, "ANNEE_CONSTRUCTION"] = np.nan

# Summary
total_invalid = mask.sum()
print(f"Marked {total_invalid} entries ({total_invalid/len(df_eval)*100:.2f}%) as missing for ANNEE_CONSTRUCTION")
df_eval["ANNEE_CONSTRUCTION"].describe()


# Replace missing ANNEE_CONSTRUCTION with the label "unknown" temporarily for categorical handling
df_eval["ANNEE_CONSTRUCTION"] = df_eval["ANNEE_CONSTRUCTION"].fillna("unknown")

# Confirm replacement
df_eval["ANNEE_CONSTRUCTION"].value_counts(dropna=False).head()


# Filter entries where ANNEE_CONSTRUCTION is "unknown"
unknown_years = df_eval[df_eval["ANNEE_CONSTRUCTION"] == "unknown"]

# Count occurrences of LIBELLE_UTILISATION for these entries
unknown_usage_counts = unknown_years["LIBELLE_UTILISATION"].value_counts()

# Show top usage types for unknown construction years
unknown_usage_counts.head(10)


# Step 1: Convert ANNEE_CONSTRUCTION to numeric safely
df_eval["_ANNEE_CONSTRUCTION_NUM"] = pd.to_numeric(df_eval["ANNEE_CONSTRUCTION"], errors="coerce")

# Step 2: Identify missing construction years in logements/immeubles
mask_missing = df_eval["_ANNEE_CONSTRUCTION_NUM"].isna()
mask_buildings = df_eval["LIBELLE_UTILISATION"].str.contains("Logement|Immeuble", case=False, na=False)
mask_to_impute = mask_missing & mask_buildings

# Step 3: Compute median year by borough (from valid rows only)
median_years_by_borough = (
    df_eval.loc[~mask_missing]
    .groupby("NO_ARROND_ILE_CUM")["_ANNEE_CONSTRUCTION_NUM"]
    .median()
    .astype("Int64")
)

# Step 4: Impute missing years for buildings using borough medians
for borough, median_year in median_years_by_borough.items():
    idx = mask_to_impute & (df_eval["NO_ARROND_ILE_CUM"] == borough)
    df_eval.loc[idx, "_ANNEE_CONSTRUCTION_NUM"] = median_year

# Step 5: Finalize cleaned column
df_eval["ANNEE_CONSTRUCTION"] = df_eval["_ANNEE_CONSTRUCTION_NUM"].fillna("unknown").astype(str)
df_eval.drop(columns=["_ANNEE_CONSTRUCTION_NUM"], inplace=True)

# ✅ Done
print("Final cleaning complete:")
print("  - Missing ANNEE_CONSTRUCTION replaced with median or 'unknown'")
print("  - Column converted to string for modeling flexibility")
print(df_eval["ANNEE_CONSTRUCTION"].value_counts(dropna=False).head())

# Exclude 'unknown' for plotting and convert to numeric
construction_years = df_eval[df_eval["ANNEE_CONSTRUCTION"] != "unknown"]["ANNEE_CONSTRUCTION"].astype(float)

# Plot histogram with better x-axis labeling
plt.figure(figsize=(14, 6))
plt.hist(construction_years, bins=50, edgecolor='black')
plt.title("Distribution of ANNEE_CONSTRUCTION (excluding 'unknown')")
plt.xlabel("Year of Construction")
plt.ylabel("Number of Properties")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




# Define the columns to check for missing values
columns_to_check = ["NOMBRE_LOGEMENT", "ETAGE_HORS_SOL", "SUPERFICIE_BATIMENT"]

# Calculate missing count and percentage for each column
missing_summary = df_eval[columns_to_check].isna().sum().to_frame(name="Missing Count")
missing_summary["Missing %"] = 100 * missing_summary["Missing Count"] / len(df_eval)

# Display the result
print(missing_summary)


#  cleaning NOMBRE_LOGEMENT.  use a combination of borough + building type for the most contextual imputation.

import numpy as np

# Step 1: Create a median lookup table
median_units = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["NOMBRE_LOGEMENT"]
    .median()
    .dropna()
)

# Step 2: Define a function to apply median imputation
def impute_nombre_logement(row):
    if pd.isna(row["NOMBRE_LOGEMENT"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_units.get(key, np.nan)  # fallback to NaN if no match
    else:
        return row["NOMBRE_LOGEMENT"]

# Step 3: Apply imputation
df_eval["NOMBRE_LOGEMENT"] = df_eval.apply(impute_nombre_logement, axis=1)

# Step 4: Check remaining missing
missing_final = df_eval["NOMBRE_LOGEMENT"].isna().sum()
print(f"✅ Imputation complete. Remaining missing values: {missing_final}")


# Apply borough-level median fallback


# Step 1: Compute borough-level medians
borough_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["NOMBRE_LOGEMENT"].median()

# Step 2: Apply fallback imputation for remaining missing values
def fallback_impute_logement(row):
    if pd.isna(row["NOMBRE_LOGEMENT"]):
        return borough_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["NOMBRE_LOGEMENT"]

# Step 3: Apply the function
df_eval["NOMBRE_LOGEMENT"] = df_eval.apply(fallback_impute_logement, axis=1)

# Step 4: Final check
final_missing = df_eval["NOMBRE_LOGEMENT"].isna().sum()
print(f"✅ Borough-level fallback complete. Final missing: {final_missing}")


# Compute the median number of units per borough (NO_ARROND_ILE_CUM).
# 
# Apply it to any rows where NOMBRE_LOGEMENT is still missing.

# Step 1: Compute borough-level medians
borough_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["NOMBRE_LOGEMENT"].median()

# Step 2: Apply fallback imputation for remaining missing values
def fallback_impute_logement(row):
    if pd.isna(row["NOMBRE_LOGEMENT"]):
        return borough_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["NOMBRE_LOGEMENT"]

# Step 3: Apply the function
df_eval["NOMBRE_LOGEMENT"] = df_eval.apply(fallback_impute_logement, axis=1)

# Step 4: Final check
final_missing = df_eval["NOMBRE_LOGEMENT"].isna().sum()
print(f"✅ Borough-level fallback complete. Final missing: {final_missing}")


# Cleaning ETAGE_HORS_SOL: Impute with median by (borough, building type).
# 
# Fallback to borough-level median if group-specific median is unavailable.

# Step 1: Compute median ETAGE_HORS_SOL by (borough, building type)
median_etages = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["ETAGE_HORS_SOL"]
    .median()
    .dropna()
)

# Step 2: Apply contextual imputation function
def impute_etage(row):
    if pd.isna(row["ETAGE_HORS_SOL"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_etages.get(key, np.nan)
    return row["ETAGE_HORS_SOL"]

# Step 3: Apply contextual imputation
df_eval["ETAGE_HORS_SOL"] = df_eval.apply(impute_etage, axis=1)

# Step 4: Compute borough-level fallback medians
borough_etage_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["ETAGE_HORS_SOL"].median()

# Step 5: Apply fallback for remaining missing
def fallback_impute_etage(row):
    if pd.isna(row["ETAGE_HORS_SOL"]):
        return borough_etage_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["ETAGE_HORS_SOL"]

df_eval["ETAGE_HORS_SOL"] = df_eval.apply(fallback_impute_etage, axis=1)

# Step 6: Final check
final_missing_etage = df_eval["ETAGE_HORS_SOL"].isna().sum()
final_missing_etage


# Step 1: Compute median SUPERFICIE_BATIMENT by (borough, building type)
median_surface = (
    df_eval.groupby(["NO_ARROND_ILE_CUM", "LIBELLE_UTILISATION"])["SUPERFICIE_BATIMENT"]
    .median()
    .dropna()
)

# Step 2: Apply contextual imputation
def impute_surface(row):
    if pd.isna(row["SUPERFICIE_BATIMENT"]):
        key = (row["NO_ARROND_ILE_CUM"], row["LIBELLE_UTILISATION"])
        return median_surface.get(key, np.nan)
    return row["SUPERFICIE_BATIMENT"]

df_eval["SUPERFICIE_BATIMENT"] = df_eval.apply(impute_surface, axis=1)

# Step 3: Compute borough-level fallback medians
borough_surface_medians = df_eval.groupby("NO_ARROND_ILE_CUM")["SUPERFICIE_BATIMENT"].median()

# Step 4: Apply fallback
def fallback_impute_surface(row):
    if pd.isna(row["SUPERFICIE_BATIMENT"]):
        return borough_surface_medians.get(row["NO_ARROND_ILE_CUM"], np.nan)
    return row["SUPERFICIE_BATIMENT"]

df_eval["SUPERFICIE_BATIMENT"] = df_eval.apply(fallback_impute_surface, axis=1)

# Step 5: Final check
final_missing_surface = df_eval["SUPERFICIE_BATIMENT"].isna().sum()
final_missing_surface


# In[18]:


# Save the cleaned version of df_eval before feature engineering (excluding new derived columns)
columns_to_keep = [
    col for col in df_eval.columns
    if col not in ["AGE_BATIMENT", "RATIO_SURFACE", "DENSITE_LOGEMENT", "DENSITE_ETAGE", "IS_UNKNOWN_YEAR"]
]

df_eval_cleaned = df_eval[columns_to_keep]

# Save to CSV
df_eval_cleaned.to_csv(DESTINATION_FILE_NAME, index=False)


df_eval_cleaned=pd.read_csv(DESTINATION_FILE_NAME)
df_eval_cleaned.head()
df_eval_cleaned.info()
