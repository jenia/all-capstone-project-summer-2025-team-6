#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
from typing import Union #allows to specific "or" conditions for argument types
from datetime import datetime


# In[10]:


DIRECTORY = r'G:\.shortcut-targets-by-id\1uExmPmKnHKKlOfMdT70cXpwXvdf9aVEC\Capstone Project summer 2025- Team6\datasets'
ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED = 'interventions_cleaned.csv'
#ORIGINAL_FILE_NAME_2022_BEFORE = 'donneesouvertes-interventions-sim2020.csv'
DESTINATION_FILE_NAME = 'interventions_cleaned_with_has_fire.csv'


# In[11]:


df_interventions_cleaned=pd.read_csv(ORIGINAL_FILE_NAME_INTERVENTIONS_CLEANED)


# In[12]:


df_interventions_cleaned.info()


# In[13]:


# Check unique values in DESCRIPTION_GROUPE
description_group_counts = df_interventions_cleaned["DESCRIPTION_GROUPE"].value_counts().reset_index()
description_group_counts.columns = ["DESCRIPTION_GROUPE", "Count"]
description_group_counts


# In[14]:


# Add binary fire label: 1 for AUTREFEU and INCENDIE, 0 for Alarmes-incendies
#df_interventions_cleaned['HAS_FIRE'] = df_interventions_cleaned['DESCRIPTION_GROUPE'].apply(
#    lambda x: 1 if x in ['AUTREFEU', 'INCENDIE'] else 0
#)

# Check the distribution
#fire_counts = df_interventions_cleaned['HAS_FIRE'].value_counts().rename(index={0: 'No Fire', 1: 'Fire'})
#print("🔥 HAS_FIRE distribution:")
#print(fire_counts)


# In[15]:


# ✅ Label all 3 categories as fire-related
df_interventions_cleaned['HAS_FIRE'] = df_interventions_cleaned['DESCRIPTION_GROUPE'].apply(
    lambda x: 1 if x in ['AUTREFEU', 'INCENDIE', 'Alarmes-incendies'] else 0
)

# 📊 Check the distribution
fire_counts = df_interventions_cleaned['HAS_FIRE'].value_counts().rename(index={0: 'No Fire', 1: 'Fire'})
print("🔥 HAS_FIRE distribution:")
print(fire_counts)


# In[16]:


# Save to CSV
df_interventions_cleaned.to_csv("interventions_cleaned_with_has_fire.csv", index=False)

# Optional: Confirm save
print("✅ File saved as 'DESTINATION_FILE_NAME'")


# In[ ]:





# In[ ]:





# In[ ]:




