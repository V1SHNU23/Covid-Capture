import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time  # Import time module to track the duration

# U-Net model with Dropout for regularization
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024 + 512, 512)  # Adjust channels after concatenation
        self.decoder3 = self.conv_block(512 + 256, 256)   # Adjust channels after concatenation
        self.decoder2 = self.conv_block(256 + 128, 128)   # Adjust channels after concatenation
        self.decoder1 = self.conv_block(128 + 64, 64)     # Adjust channels after concatenation

        # Output layer
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))

        # Decoder
        d4 = self.decoder4(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.decoder3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))

        # Output
        return torch.sigmoid(self.out_conv(d1))

# Load and preprocess data
def load_data(image_dir, mask_dir, img_height=256, img_width=256):
    images, masks, filenames = [], [], []

    for img_filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue  # Skip if image/mask is missing or corrupted

        img = cv2.resize(img, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))

        images.append(np.expand_dims(img, axis=0))  # Add channel dimension
        masks.append(np.expand_dims(mask, axis=0))  # Add channel dimension

        filenames.append(img_filename)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0

    return torch.tensor(images), torch.tensor(masks), filenames

# Train the model
def train_model(model, x_train, y_train, epochs=70, batch_size=12, validation_split=0.1, model_save_path='unet_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # Prepare dataset
    dataset_size = len(x_train)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset = torch.utils.data.TensorDataset(x_train[:train_size], y_train[:train_size])
    val_dataset = torch.utils.data.TensorDataset(x_train[train_size:], y_train[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    total_start_time = time.time()  # Start total training time

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start epoch timer

        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)
                val_outputs = model(val_images)
                val_loss += criterion(val_outputs, val_masks).item()

        epoch_duration = time.time() - epoch_start_time  # Calculate epoch duration
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Time: {epoch_duration:.2f} seconds")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    total_duration = time.time() - total_start_time  # Calculate total training time
    print(f"Total training time: {total_duration:.2f} seconds")

# Visualize predictions
def visualize_predictions(model, x_data, y_true, filenames):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    idx = np.random.randint(0, len(x_data))
    x_sample = x_data[idx].unsqueeze(0).to(device)  # Add batch dimension and move to device
    y_true_sample = y_true[idx].cpu().numpy()  # Ground truth mask

    with torch.no_grad():
        y_pred = model(x_sample).cpu().numpy()[0, 0]  # Get predicted mask

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Input X-ray: {filenames[idx]}")
    plt.imshow(x_sample.cpu().squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth Mask: {filenames[idx]}")
    plt.imshow(y_true_sample.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(y_pred.squeeze(), cmap='jet')

    plt.show()

# Paths to your data directories
image_dir = 'img 256'
mask_dir = 'labels 256'

# Load data
x_train, y_train, filenames = load_data(image_dir, mask_dir)

# Initialize the model
model = UNet()

# Train the model and save the best version with timing
train_model(model, x_train, y_train, epochs=70, batch_size=12)

# Visualize predictions
visualize_predictions(model, x_train, y_train, filenames)
