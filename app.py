from flask import Flask, request, render_template
import torch
from PIL import Image
import pandas as pd
from model import load_model  # Import the load_model function from model.py
from torchvision import transforms

# Initialize Flask app
app = Flask(__name__)

# Load the model from the model.py file
model = load_model()

# Load the food data (assuming CSV has columns like 'id', 'calories', 'proteins', 'fat', 'carbohydrate', 'name', 'image', 'label')
df = pd.read_csv('food_data.csv')

# Load the food names from the CSV (the 'name' column has the food names)
labels = df['name'].tolist()  # Assuming 'name' is the column with food labels
print(f"Loaded {len(labels)} food labels.")

# Image transformation (based on how your model was trained)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize if needed
])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    nutrition_info = None  # Ensure nutrition_info is always defined

    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        img = Image.open(file.stream)  # Open the image from the uploaded file
        img_tensor = transform(img).unsqueeze(0)  # Apply transforms and add batch dimension

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)  # Get output from the model
            _, predicted = torch.max(output, 1)  # Get the class with the highest probability
            predicted_idx = predicted.item()

            # Handle out of range indices
            if predicted_idx < len(labels):
                predicted_class = labels[predicted_idx]  # Get the class label
                # Lookup nutritional info from the CSV based on 'name'
                nutrition = df[df['name'] == predicted_class].iloc[0]
                # You can extract specific nutritional info like this
                nutrition_info = {
                    'calories': nutrition['calories'],
                    'proteins': nutrition['proteins'],
                    'fat': nutrition['fat'],
                    'carbohydrates': nutrition['carbohydrate']
                }
            else:
                predicted_class = "Unknown"
                nutrition_info = None

        prediction = predicted_class

    # Return the rendered template with the results
    return render_template('index.html', prediction=prediction, nutrition=nutrition_info)

if __name__ == '__main__':
    app.run(debug=True)
