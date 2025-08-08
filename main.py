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