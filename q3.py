import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Import the pre-trained MNIST network architecture from Task 1
from mnist1 import MyNetwork

# Define the transform for the Greek dataset
greek_transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to MNIST image size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

# Load the Greek training dataset
greek_train_dataset = datasets.ImageFolder(root='/home/haard/Desktop/PRCV/P5/greek_train', transform=greek_transform)

# Define data loader for Greek training dataset
greek_train_loader = DataLoader(greek_train_dataset, batch_size=64, shuffle=True)

# Load the pre-trained MNIST model
model = MyNetwork()

# Freeze the network weights
for param in model.parameters():
    param.requires_grad = False

# Replace the last layer with a new Linear layer with three nodes for Greek letters
model.fc2 = nn.Linear(50, 3)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model on the Greek training dataset
def train(model, train_loader, criterion, optimizer, num_epochs=15):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), 'fine_tuned_model.pth')


# Train the model
train(model, greek_train_loader, criterion, optimizer, num_epochs=10)

