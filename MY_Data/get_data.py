import pandas as pd
import os

# Path to your downloaded CSV
file_path = r'C:\Users\FARIDA\Downloads\archive (1)\heart.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Ensure data folder exists
os.makedirs('data', exist_ok=True)

# Save a copy locally
df.to_csv('data/heart_raw.csv', index=False)

# Check the data
print(df.head())
