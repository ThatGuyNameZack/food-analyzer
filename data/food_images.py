import pandas as pd
import requests
import os

def download_images(csv_path, output_dir):
    data = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    for i, row in data.iterrows():
        image_url = row['image_url']
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(f'{output_dir}/image_{i}.jpg', 'wb') as f:
                for chunk in response:
                    f.write(chunk)
        else: 
            print(f"Failed to download image {i} from {image_url}") #checck if theres an image error