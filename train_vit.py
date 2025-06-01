import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 13
LEARNING_RATE = 0.0001  # Lower learning rate works better for ViT
DATA_DIR = './dataset_gambar'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use cuDNN autotuner for optimal settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Transform (ViT requires 224x224 just like ResNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Datasets and Loaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Load ViT base model
model = models.vit_b_16(weights='IMAGENET1K_V1')  # Pretrained on ImageNet
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Function
def train():
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "indonesia_food_vit.pth")
    print("Training complete. Model saved.")

if __name__ == '__main__':
    train()
