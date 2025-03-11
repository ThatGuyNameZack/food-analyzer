import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Load dataset
# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__)) #added his so it can detect in the same folder 

# Construct the relative path
file_path = os.path.join(script_dir, "food_data.csv")

# Load dataset
df = pd.read_csv(file_path)

# added to food_images folder.
save_dir = "food_images"
os.makedirs(save_dir, exist_ok=True)

# Function to download and save an image
def download_image(url, food_name, image_id):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            food_folder = os.path.join(save_dir, food_name.replace(" ", "_").lower())
            os.makedirs(food_folder, exist_ok=True)
            image_path = os.path.join(food_folder, f"{image_id}.jpg")
            image.save(image_path)
            print(f"Downloaded: {image_path}")
        else:
            print(f"Failed to download {url}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Download images
for _, row in df.iterrows():
    download_image(row['image'], row['name'], row['id']) #for the program to read the csv to diffrentiate images.

print("Image download complete!")
