import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class LayoutDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.transcription_dir = os.path.join(root_dir, "transcriptions")
        self.transform = transform
        self.image_names = sorted(os.listdir(self.image_dir))  # keep files in consistent order
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # load image and corresponding mask
        img_filename = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)

        # open images
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        #print(f"Loaded image {img_filename} with shape {image.size}, mask with shape {mask.size}")


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # ensure mask is strictly binary
            mask = (mask > 0.5).float()
            #print(f"Transformed image shape: {image.shape}, mask shape: {mask.shape}")  # check shape after transform


        return image, mask

# transformer module
class TransformerBlock(nn.Module):
    """
    Adds self-attention capability to capture long-range dependencies
    in the document layout.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # standard transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, x):
        # reshape for transformer: [B,C,H,W] -> [H*W,B,C]
        B, C, H, W = x.shape
        x = x.view(B, C, H*W).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        # reshape back to original dimensions
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x

# U-Net with transformer
class UNetTransformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], nhead=8):
        super(UNetTransformer, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # downsample path (encoder)
        prev_channels = in_channels
        for feature in features:
            self.downs.append(self.conv_block(prev_channels, feature))
            prev_channels = feature

        # bottleneck with expanded features and transformer
        self.bottleneck = self.conv_block(features[-1], features[-1]*2)
        self.transformer = TransformerBlock(d_model=features[-1]*2, nhead=nhead)

        # upsample path (decoder)
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
        
        # encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # bottleneck with transformer
        x = self.bottleneck(x)
        x = self.transformer(x)
        
        # decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)

# evaluation Metrics
def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for better segmentation performance"""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + smooth)
    return 1 - dice.mean()

def iou_score(pred, target, smooth=1e-6):
    """Intersection over Union metric"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# training function
def train_model(model, dataloader, optimizer, num_epochs=10, device="cuda"):
    model.train()
    criterion = nn.BCEWithLogitsLoss() 
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # combine BCE and dice loss for better performance
            loss = criterion(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, IoU: {iou_score(outputs, masks):.4f}")

# Main Execution
def main():
    # configuration settings
    root_dir = "./data"  
    batch_size = 4      
    num_epochs = 15 # training iterations
    learning_rate = 1e-4 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),    # Convert to PyTorch tensor
    ])

    dataset = LayoutDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = UNetTransformer(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, optimizer, num_epochs=num_epochs, device=device)

    # save trained model
    torch.save(model.state_dict(), "unet_transformer_layout_model.pth")
    print("training is complete.")

if __name__ == "__main__":
    main()
