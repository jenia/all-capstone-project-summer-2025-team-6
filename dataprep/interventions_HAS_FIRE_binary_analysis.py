#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
from typing import Union #allows to specific "or" conditions for argument types
from datetime import datetime


DESTINATION_FILE_NAME = './datasets/cleaned/interventions_cleaned_with_has_fire.csv'
ORIGINAL_FILE_NAME_2023_2025 = './datasets/raw/donneesouvertes-interventions-sim.csv'
ORIGINAL_FILE_NAME_2022_BEFORE = './datasets/raw/donneesouvertes-interventions-sim2020.csv'



# In[18]:


def is_date_format(string_input: str, date_format: str) -> bool:
    """
    Validates if a string matches a given date format.

    Args:
        string_input (str): The date string to validate
        date_format (str): The expected date format pattern

    Returns:
        bool: True if the string matches the format, False otherwise
    """
    try:
        # Attempt to parse the date string
        datetime.strptime(string_input, date_format)
        return True
    except ValueError:
        # If parsing fails, the format is invalid
        return False


# In[19]:


def convert_date_format(date_string: str) -> str:
    """
    Validates and converts date format from '%Y-%m-%d' to '%Y-%m-%dT%H:%M:%S'.

    Args:
        date_string (str): Input date string

    Returns:
        str: Date string in '%Y-%m-%dT%H:%M:%S' format if conversion was needed,
             original string if already in correct format or invalid
    """
    try:
        date_obj = datetime.strptime(date_string, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%dT%H:%M:%S')
    except ValueError:
        try:
            datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
            return date_string
        except ValueError:
            return date_string  # Return original string for invalid dates


# In[20]:


print("Loading data ...")
df = pd.read_csv(ORIGINAL_FILE_NAME_2023_2025)
df_old=pd.read_csv(ORIGINAL_FILE_NAME_2022_BEFORE)
df=pd.concat([df,df_old])


# In[21]:


# add a column to visualize dates with non-standard format
# df['IS_VALID_DATE'] = df['CREATION_DATE_TIME'].apply(
#     lambda x: is_date_format(x, '%Y-%m-%dT%H:%M:%S')
# )

# df[df['IS_VALID_DATE']==False]

#standardizes date format (we don't really care about the time component)
print("Fixing date times ...")
df['CREATION_DATE_TIME']=df['CREATION_DATE_TIME'].apply(convert_date_format)
df['CREATION_DATE_TIME']=df['CREATION_DATE_TIME'].apply(datetime.fromisoformat)


# In[22]:


# commented out to avoid rewriting over existing file
# Dropping MTM8_X and Y as they're internal coordinates and they're not described in the data dictionary so they're hard to interpret
# We already have the latitude/longitude for spatial coordinates
print("Dropping columns...")
df=df.drop(['MTM8_X','MTM8_Y'],axis=1)


# In[23]:


# 🔥 Define fire-related categories based on known labels
#fire_categories = ["Alarmes-incendies", "AUTREFEU", "INCENDIE"]#unrealistic number of houses with fire so we remove alarmes incendies
fire_categories = [ "AUTREFEU", "INCENDIE"]
print("Filter only fire incidents...")
# Filter only fire incidents
df = df[df["DESCRIPTION_GROUPE"].isin(fire_categories)]


# In[24]:


# Count total fire incidents
fire_incident_count = len(df)

# Display the result
print(f"🔥 Total fire incidents: {fire_incident_count:,}")


# In[25]:


# Count by DESCRIPTION_GROUPE
category_counts = df["DESCRIPTION_GROUPE"].value_counts()

print("🔥 Fire incident breakdown by type:")
print(category_counts)


# In[26]:


# 📋 Check missing values
missing_summary = df.isnull().sum()
missing_percentage = (df.isnull().mean() * 100).round(2)

# Combine into a single DataFrame for clarity
missing_report = pd.DataFrame({
    "Missing Count": missing_summary,
    "Missing %": missing_percentage
})

# Display only columns with missing values
missing_report = missing_report[missing_report["Missing Count"] > 0]

print("📉 Missing values summary:")
print(missing_report)


# In[27]:


# 🔍 Check missing values in NOMBRE_UNITES
missing_units = df["NOMBRE_UNITES"].isnull().sum()
total_rows = len(df)
missing_pct = (missing_units / total_rows) * 100

print(f"🔢 Missing NOMBRE_UNITES values: {missing_units:,} out of {total_rows:,} rows")
print(f"📉 Missing percentage: {missing_pct:.2f}%")


# In[28]:


df.head()


# In[29]:


# 💾 Save the filtered dataset with fire categories
output_path = "interventions_cleaned_with_has_fire.csv"
df.to_csv(output_path, index=False)

print(f"✅ Dataset saved as '{output_path}' with {len(df):,} fire incident records.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


import geopandas as gpd
from shapely.geometry import Point

# Convert to GeoDataFrame
fire_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]),
    crs="EPSG:4326"
)


# In[17]:


fire_gdf = fire_gdf.to_crs(epsg=32188)


# In[18]:


fire_gdf["buffer"] = fire_gdf.geometry.buffer(100)
buffer_gdf = fire_gdf.set_geometry("buffer")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




