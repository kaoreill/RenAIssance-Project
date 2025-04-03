import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================================
# Dataset Description:
#
# The dataset consists of 6 scanned early modern printed sources.
# Each source is provided as a separate PDF file.
# (For processing, convert these PDFs to JPEG images.)
#
# Characteristics:
# - The images are already in grayscale.
# - They exhibit limitations due to the original OCR (e.g., missed letters, incorrect words).
# - The main text is the focus; marginalia and other embellishments should be ignored.
#
# Transcriptions:
# - The first three pages of each source have been transcribed.
# - These transcriptions are provided as reference files to guide training.
# - They include notes on issues like:
#   • u and v being used interchangeably
#   • Two types of lowercase “s” (both ‘s’ and ‘ſ’) should be transcribed as ‘s’
#   • Accents (except ñ) should be ignored
#   • Some letters have macrons (¯) indicating that an ‘n’ follows, or ‘ue’ after a capital Q
#   • Some line end hyphens may leave words split (handle later)
#   • Old spelling “ç” should always be interpreted as modern “z”
#
# Folder Structure Assumption:
# data/
#    ├── images/         -> Contains JPEG images (grayscale) converted from PDFs
#    ├── masks/          -> Contains corresponding segmentation masks (binary images; text=1, background=0)
#    └── transcriptions/ -> Contains transcription text files for the first 3 pages of each source (optional use)
# ==========================================================

# ========================
# Dataset Definition
# ========================
class LayoutDataset(Dataset):
    """
    Custom dataset for document layout segmentation.
    Expects a folder structure:
      - root/images: document images (grayscale JPEGs converted from PDFs)
      - root/masks: corresponding segmentation masks (binary images; text=1, background=0)
      - (optional) root/transcriptions: reference transcriptions for the first 3 pages of each source.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transcription_dir = os.path.join(root_dir, "transcriptions")
        self.transform = transform
        self.image_names = sorted(os.listdir(self.image_dir))
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_filename = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)  # assuming same filename for mask

        # Since images are already grayscale, we use "L"
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        # If transcriptions are available and you wish to use them, add additional processing here.
        # Example: transcription_path = os.path.join(self.transcription_dir, img_filename.replace('.jpg', '.txt'))
        # with open(transcription_path, 'r', encoding='utf-8') as f:
        #    transcription = f.read()
        # Optionally apply normalization rules for transcription text here.
        # e.g., replace 'ſ' with 's', ignore accents (except ñ), handle macrons, etc.

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # Ensure mask is binary (0,1)
            mask = (mask > 0.5).float()

        return image, mask

# ========================
# Transformer Block Definition
# ========================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, x):
        # x shape: [B, C, H, W] --> flatten spatial dimensions: [B, C, H*W]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(2, 0, 1)  # shape: [H*W, B, C]
        x = self.transformer_encoder(x)         # shape: [H*W, B, C]
        x = x.permute(1, 2, 0).view(B, C, H, W)    # reshape back to [B, C, H, W]
        return x

# ========================
# U-Net with Transformer Encoder at Bottleneck
# ========================
class UNetTransformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], nhead=8):
        """
        in_channels set to 1 for grayscale input.
        """
        super(UNetTransformer, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Downsampling (Encoder)
        prev_channels = in_channels
        for feature in features:
            self.downs.append(self.conv_block(prev_channels, feature))
            prev_channels = feature

        # Bottleneck with additional feature expansion and Transformer block
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)
        self.transformer = TransformerBlock(d_model=features[-1]*2, nhead=nhead)

        # Upsampling (Decoder)
        rev_features = features[::-1]
        current_channels = features[-1]*2
        for feature in rev_features:
            self.ups.append(
                nn.ConvTranspose2d(current_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.conv_block(feature * 2, feature))
            current_channels = feature

        self.final_conv = nn.Conv2d(rev_features[-1], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        # Apply transformer block for global context capture
        x = self.transformer(x)
        
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            # If dimensions differ, use interpolation
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

# ========================
# Loss and Evaluation Metrics
# ========================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + smooth)
    return 1 - dice.mean()

def iou_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# ========================
# Training Loop
# ========================
def train_model(model, dataloader, optimizer, num_epochs=10, device="cuda"):
    model.train()
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for segmentation
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        # Print epoch loss and IoU as evaluation metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, IoU: {iou_score(outputs, masks):.4f}")

# ========================
# Main Execution
# ========================
def main():
    # Hyperparameters and configuration
    root_dir = "./data"  # Path to dataset folder with images, masks, and (optionally) transcriptions
    batch_size = 4
    num_epochs = 20
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define transformations: resize to 256x256 and convert to tensor.
    # Note: For grayscale images, ToTensor() produces a tensor of shape [1, H, W]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader
    dataset = LayoutDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model and optimizer (set in_channels to 1 for grayscale)
    model = UNetTransformer(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, dataloader, optimizer, num_epochs=num_epochs, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "unet_transformer_layout_model.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
