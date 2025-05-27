import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

# Custom Dataset to read images + labels from annotations json
class FoodDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Parse annotations, images, categories
        annotations = data['annotations']
        images = {img['id']: img['file_name'] for img in data['images']}
        categories = {cat['id']: cat['name'] for cat in data['categories']}

        # Build a list of (image_path, label_id)
        self.samples = []
        self.class_to_idx = {}
        idx = 0

        for ann in annotations:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            img_name = images[img_id]

            if cat_id not in self.class_to_idx:
                self.class_to_idx[cat_id] = idx
                idx += 1
            label_idx = self.class_to_idx[cat_id]

            self.samples.append((img_name, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# Simple CNN Model (similar to your TF model)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (32, 150, 150)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (32, 75, 75)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (64, 37, 37)

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # (128, 18, 18)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

from tqdm import tqdm  # Make sure you have tqdm imported

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=running_loss/total, accuracy=correct/total)

    return running_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss/total, accuracy=correct/total)

    return running_loss / total, correct / total

def main():
    # Paths (update as needed)
    train_images_dir = r"D:\GitHub\food-analyzer\raw_data\public_training_set_release_2.0\images"
    train_annotation_file = r"D:\GitHub\food-analyzer\raw_data\public_training_set_release_2.0\annotations.json"

    val_images_dir = r"D:\GitHub\food-analyzer\raw_data\public_validation_set_2.0\images"
    val_annotation_file = r"D:\GitHub\food-analyzer\raw_data\public_validation_set_2.0\annotations.json"

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = FoodDataset(train_images_dir, train_annotation_file, transform=train_transform)
    val_dataset = FoodDataset(val_images_dir, val_annotation_file, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = SimpleCNN(num_classes=len(train_dataset.class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print("Training complete. Best validation accuracy:", best_val_acc)


if __name__ == '__main__':
    main()