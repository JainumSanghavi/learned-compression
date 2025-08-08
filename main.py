import torch
import torch.nn as nn
import torch.optim as optim
from model import ConvAutoencoder
from utils import get_dataloaders, show_reconstruction, save_reconstruction, calculate_psnr,get_celeba_loaders
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 1e-3
save_dir = "outputs"

# Load data
train_loader, test_loader = get_celeba_loaders(batch_size=batch_size)

# Initialize model, loss, optimizer
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")