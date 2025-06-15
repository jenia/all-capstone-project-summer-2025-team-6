import pandas as pd
import numpy as np
import os
from typing import Union #allows to specific "or" conditions for argument types
from datetime import datetime

dirname = os.path.dirname(__file__)

ORIGINAL_FILE_NAME_2023_2025 = os.path.join(dirname,'..','datasets','raw','donneesouvertes-interventions-sim.csv')
ORIGINAL_FILE_NAME_2022_BEFORE = os.path.join(dirname,'..','datasets','raw','donneesouvertes-interventions-sim2020.csv')
DESTINATION_FILE_NAME = os.path.join(dirname,'..','datasets','cleaned','interventions_cleaned.csv')

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
print("Loading data ...")
df = pd.read_csv(ORIGINAL_FILE_NAME_2023_2025)
df_old=pd.read_csv(ORIGINAL_FILE_NAME_2022_BEFORE)
df=pd.concat([df,df_old])


#standardizes date format (we don't really care about the time component)
print("Fixing date times ...")
df['CREATION_DATE_TIME']=df['CREATION_DATE_TIME'].apply(convert_date_format)
df['CREATION_DATE_TIME']=df['CREATION_DATE_TIME'].apply(datetime.fromisoformat)

"""# Save dataset"""

# commented out to avoid rewriting over existing file
# Dropping MTM8_X and Y as they're internal coordinates and they're not described in the data dictionary so they're hard to interpret
# We already have the latitude/longitude for spatial coordinates
print("Dropping columns...")
df=df.drop(['MTM8_X','MTM8_Y','NOMBRE_UNITES'],axis=1)
print("Dropping rows not related to fire or with null DESCRIPTION_GROUPE ...")
df = df[~df['DESCRIPTION_GROUPE'].isnull() & ~df['DESCRIPTION_GROUPE'].isin(['SANS FEU','NOUVEAU','1-REPOND','FAU-ALER'])]
print(f"Saving as {DESTINATION_FILE_NAME}")
df.to_csv(os.path.join(DESTINATION_FILE_NAME),index=False)

print(df.info())
print(df.shape)

