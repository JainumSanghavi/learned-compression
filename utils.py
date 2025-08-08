import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.datasets import CelebA
from torchvision import transforms


#high level design
def get_celeba_loaders(batch_size=128, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = CelebA(root='./data', split='train', download=True, transform=transform)
    test_dataset  = CelebA(root='./data', split='test',  download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




def get_dataloaders(batch_size=128):
    """
    Returns PyTorch DataLoader for MNIST dataset (train and test).
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def show_reconstruction(original, reconstructed, n=6):
    pass

def save_reconstruction(original, reconstructed, path, n=6):

    pass
def calculate_psnr(mse, max_pixel=1.0):
    pass
