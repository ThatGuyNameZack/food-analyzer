import torch
from torch.utils.data import DataLoader
from torch import optim
from models.food_cnn import FoodCNN
from utils.dataset import FoodDataset, transform
from utils.download_images import download_images

# Download images from data folder
download_images('data/food_data.csv', 'data/food_images')

# Prepare dataset
image_paths = [f'data/food_images/image_{i}.jpg' for i in range(100)]  # Adjust range as needed
labels = ...  # Load labels from CSV file
dataset = FoodDataset(image_paths, labels, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = FoodCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model the model
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
scripted_model = torch.jit.script(model)
scripted_model.save('food_analyzer.pt')