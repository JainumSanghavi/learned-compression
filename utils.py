import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from torchvision.datasets import CelebA
from torchvision import transforms


#high level design
def get_celeba_loaders(batch_size=128, image_size=64):
    pass



def get_dataloaders(batch_size=128):
    pass

def show_reconstruction(original, reconstructed, n=6):
    pass

def save_reconstruction(original, reconstructed, path, n=6):

    pass
def calculate_psnr(mse, max_pixel=1.0):
    pass
