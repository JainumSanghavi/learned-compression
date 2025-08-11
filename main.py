
'''
This main training script trains and evaluates a Convolutional Autoencoder on the CelebA 
dataset for image compression and reconstruction. It begins by automatically selecting the
 best available compute device (CUDA GPU, Apple MPS, or CPU) and setting core hyperparameters 
 such as learning rate, batch size, and number of epochs. It loads preprocessed CelebA data, 
 initializes the autoencoder model, mean squared error loss function, Adam optimizer, and a
   learning rate scheduler that reduces the LR when loss plateaus. The script then runs a multi-epoch 
   training loop where the model learns to reconstruct input images, periodically evaluating on a 
   validation set to monitor performance. After training, it tests the model on unseen data, computes 
   reconstruction quality using PSNR, and visualizes results by saving side-by-side original and 
   reconstructed images. Finally, it saves the trained model weights and output images to disk,
     making the pipeline reproducible and ready for further fine-tuning or deployment.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from model import ConvAutoencoder
from utils import get_dataloaders, show_reconstruction, save_reconstruction, calculate_psnr,get_celeba_loaders
import os
from datetime import datetime
# Set device
# Set device (automatically handles CUDA/MPS/CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")

#currently using MPS
elif torch.backends.mps.is_available():  # macOS Metal Performance Shaders
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameter
num_epochs = 50        # Number of times we iterate over the training set
batch_size = 256       # Number of images per mini-batch
learning_rate = 1e-3   # Initial learning rate for optimizer
save_dir = "outputs"   # Directory to store model outputs & logs
image_size = 64        # Resize CelebA images to 64x64 for training

# Setup TensorBoard
#log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))



# Load CelebA dataset
# Returns DataLoader objects for training and testing
train_loader, test_loader = get_celeba_loaders(batch_size=batch_size, image_size=image_size)

# Initialize model, loss, optimizer
model = ConvAutoencoder(input_channels=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
#adding a scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


# Training loop - completely updated for tensorbaord.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, _) in enumerate(train_loader):
        print(batch_idx)
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
    scheduler.step(avg_loss)

    # Validation
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, images).item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
# Evaluation
model.eval()
test_images, _ = next(iter(test_loader))
test_images = test_images.to(device)
with torch.no_grad():
    reconstructed = model(test_images)

# Compute PSNR for first batch
mse = nn.functional.mse_loss(reconstructed, test_images)
psnr = calculate_psnr(mse)
print(f"Test MSE: {mse.item():.4f}, PSNR: {psnr:.2f} dB")

# Visualize results
show_reconstruction(test_images.cpu(), reconstructed.cpu(), n=6)

# Save reconstruction comparison
save_reconstruction(test_images.cpu(), reconstructed.cpu(), os.path.join(save_dir, "reconstruction.png"))

# Save model
os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "checkpoints", "cae_mnist.pth"))
