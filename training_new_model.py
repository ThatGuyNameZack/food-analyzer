import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Load cleaned CSV
df = pd.read_csv('cleaned_food_data.csv')

# Create label mapping
labels = df['label'].unique()
label_to_idx = {label: idx for idx, label in enumerate(labels)}
df['label_idx'] = df['label'].map(label_to_idx)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'], random_state=42)

# Dataset class
class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = row['label_idx']
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# DataLoaders
train_dataset = FoodDataset(train_df, transform=transform)
val_dataset = FoodDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(5):  # adjust as needed
    model.train()
    running_loss = 0.0
    for images, labels_batch in train_loader:
        images, labels_batch = images.to(device), labels_batch.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save model and label map
torch.save(model.state_dict(), 'food_classifier_new.pth')
torch.save(label_to_idx, 'label_map.pth')

print("✅ Model saved to food_classifier_new.pth")
