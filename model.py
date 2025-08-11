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


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        # Encoder (conv layers)
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