from flask import Flask, request, render_template
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from nutrition import nutrition_data
from label import labels  


# Initialize Flask app
app = Flask(__name__)

# Classes 
# labels = []

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define and load the model
def load_model(model_path='indonesia_food_resnet18.pth', num_classes=13):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Load the model once
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    nutrition_info = None

    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predicted_idx = predicted.item()

            if predicted_idx < len(labels):
                prediction = labels[predicted_idx]
                nutrition_info = nutrition_data.get(prediction, None)
            else:
                prediction = "Unknown"
                nutrition_info = None

    return render_template('index.html', prediction=prediction, nutrition=nutrition_info)


if __name__ == '__main__':
    app.run(debug=True)
