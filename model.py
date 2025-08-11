import torch
import torch.nn as nn
import torch.nn.functional as F

# class ConvAutoencoder(nn.Module):
#     def __init__(self, input_channels = 3):
#         super(ConvAutoencoder, self).__init__()

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
#             nn.ReLU(),
#             nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
#             nn.ReLU()
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
#              nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


"""
    Convolutional Autoencoder with a linear bottleneck.
    
    **Architecture Overview**
    Input:  RGB image of shape [B, 3, 64, 64]
    Output: Reconstructed RGB image of shape [B, 3, 64, 64]
    
    1. **Encoder (Convolutional)**
        - Conv2d:    [B, 3, 64, 64]   → [B, 64, 32, 32]
        - ReLU
        - Conv2d:    [B, 64, 32, 32]  → [B, 128, 16, 16]
        - ReLU
        - Conv2d:    [B, 128, 16, 16] → [B, 256, 8, 8]
        - ReLU
    
    2. **Bottleneck (Fully Connected)**
        - Flatten:   [B, 256, 8, 8]   → [B, 16384]
        - LinearEnc: [B, 16384]       → [B, 512]    # Compressed latent representation
        - LinearDec: [B, 512]         → [B, 16384]
        - Reshape:   [B, 16384]       → [B, 256, 8, 8]
    
    3. **Decoder (Deconvolutional)**
        - ConvTranspose2d: [B, 256, 8, 8]  → [B, 128, 16, 16]
        - ReLU
        - ConvTranspose2d: [B, 128, 16, 16] → [B, 64, 32, 32]
        - ReLU
        - ConvTranspose2d: [B, 64, 32, 32]  → [B, 3, 64, 64]
        - Sigmoid (to normalize outputs in [0, 1])
"""

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # Encoder (conv layers)
        #  Reduce spatial size and increase channel depth
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Linear bottleneck (e.g., 256x8x8 -> 512 values)
        self.flatten = nn.Flatten()
        self.linear_enc = nn.Linear(256*8*8, 512)  # Adjust 512 to your target size
        self.linear_dec = nn.Linear(512, 256*8*8)
        
        # Decoder (conv transpose)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_encoder(x)          # [B, 256, 8, 8]
        x = self.flatten(x)               # [B, 256*8*8]
        encoded = self.linear_enc(x)      # [B, 512] (compressed!)
        x = self.linear_dec(encoded)      # [B, 256*8*8]
        x = x.view(-1, 256, 8, 8)         # Reshape
        decoded = self.decoder(x)         # [B, 3, 64, 64]
        return decoded