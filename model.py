import torch
import torch.nn as nn
from torchvision import models

# Constants
NUM_CLASSES = 13 #the number of classes in the dataset

def load_model():
    # Load the pretrained ResNet18 and modify it
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    # Load trained weights
    model.load_state_dict(torch.load("indonesia_food_resnet18.pth", map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model
