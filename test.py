import pandas as pd

df = pd.read_csv('C:/Users/user/PycharmProjectshelloworld/SOIL MODEL 1/crop_yield_with_soil.csv')
print("Original columns:", df.columns.tolist())

# Clean column names
df.columns = [col.strip().replace(' ', '_').replace(':', '_') for col in df.columns]
print("\nCleaned columns:", df.columns.tolist())

required = ['Crop_Year', 'Season', 'State', 'Annual_Rainfall',
           'R_Fertilizer', 'Pesticide', 'Soil_Type']
missing = [col for col in required if col not in df.columns]
print("\nMissing required columns:", missing if missing else "None")