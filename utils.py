import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.datasets import CelebA
from torchvision import transforms


#high level design
# def get_celeba_loaders(batch_size=128, image_size=64):
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#     ])
#     train_dataset = CelebA(root='./data', split='train', download=False, transform=transform)
#     test_dataset  = CelebA(root='./data', split='test',  download=False, transform=transform)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader

def get_celeba_loaders(batch_size=128, image_size=64, max_samples=40000):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    full_train = CelebA(root='./data', split='train', download=False, transform=transform)
    full_test = CelebA(root='./data', split='test', download=False, transform=transform)
    
    # Use at most max_samples
    train_dataset = torch.utils.data.Subset(full_train, range(min(max_samples, len(full_train))))
    test_dataset = torch.utils.data.Subset(full_test, range(min(max_samples//5, len(full_test))))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader




def get_dataloaders(batch_size=128):
    """
    Returns PyTorch DataLoader for MNIST dataset (train and test).
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def show_reconstruction(original, reconstructed, n=6):
    """
    Plots original and reconstructed images side by side.
    """
    plt.figure(figsize=(12, 4))
    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)

        orig_img = original[i].permute(1, 2, 0).numpy()
        plt.imshow(orig_img)

        plt.axis('off')
        if i == 0:
            plt.title("Original")

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)

        recon_img = reconstructed[i].permute(1, 2, 0).numpy()
        plt.imshow(recon_img)
        plt.axis('off')
        if i == 0:
            plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()

def save_reconstruction(original, reconstructed, path, n=6):

    """
    Saves original and reconstructed images as a single comparison plot.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i].permute(1, 2, 0).numpy())       
        plt.axis('off')

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].permute(1, 2, 0).numpy())        
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def calculate_psnr(mse, max_pixel=1.0):
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio) from MSE.
    """
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))
