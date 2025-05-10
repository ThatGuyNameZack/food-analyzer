import pandas as pd
import os

# Load the original CSV
df = pd.read_csv('food_data.csv')

# Drop any rows with missing critical info
df.dropna(subset=['name', 'image', 'calories', 'proteins'], inplace=True)

# Build a new column with full image path
df['image_path'] = df['image'].apply(lambda x: os.path.join('images', x.strip()))

# Rename `name` to `label` to make it training-ready
df = df.rename(columns={'name': 'label'})

# Optional: remove unnecessary columns like 'id' if not needed
df = df[['image_path', 'label', 'calories', 'proteins', 'fat', 'carbohydrate']]

# Save cleaned version
df.to_csv('cleaned_food_data.csv', index=False)

print("✅ Cleaned CSV saved as 'cleaned_food_data.csv'")
