import torch
import torch.nn as nn
import torch.optim as optim
from model import ConvAutoencoder
from utils import get_dataloaders, show_reconstruction, save_reconstruction, calculate_psnr,get_celeba_loaders
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 128
learning_rate = 1e-3
save_dir = "outputs"
image_size = 64

# Setup TensorBoard
log_dir = os.path.join(save_dir, "logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
writer = SummaryWriter(log_dir)

# Load data
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
        images = images.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log batch loss
        if batch_idx % 50 == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    writer.add_scalar('Loss/train', avg_loss, epoch)
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
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

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
writer.close()