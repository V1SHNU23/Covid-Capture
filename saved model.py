import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# U-Net model class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.encoder3(F.max_pool2d(e2, kernel_size=2))
        e4 = self.encoder4(F.max_pool2d(e3, kernel_size=2))

        b = self.bottleneck(F.max_pool2d(e4, kernel_size=2))

        d4 = self.decoder4(torch.cat([F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True), e4], dim=1))
        d3 = self.decoder3(torch.cat([F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))

# Load and preprocess a single image and its mask
def load_image_and_mask(image_path, mask_dir, img_height=256, img_width=256):
    img_filename = os.path.basename(image_path)
    mask_path = os.path.join(mask_dir, img_filename)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Error: Image or mask not found for {img_filename}.")
        return None, None

    img = cv2.resize(img, (img_width, img_height))
    mask = cv2.resize(mask, (img_width, img_height))

    img = np.expand_dims(img, axis=0)  # Add channel dimension
    mask = np.expand_dims(mask, axis=0)  # Add channel dimension

    img_tensor = torch.tensor(img, dtype=torch.float32) / 255.0
    mask_tensor = torch.tensor(mask, dtype=torch.float32) / 255.0

    return img_tensor, mask_tensor

# Load the trained model
def load_trained_model(model_path='unet_model.pth'):
    model = UNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred_binary):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred_binary.flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat)
    recall = recall_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat)
    dice = (2 * np.sum(y_true_flat * y_pred_flat)) / (np.sum(y_true_flat) + np.sum(y_pred_flat))
    iou = jaccard_score(y_true_flat, y_pred_flat)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    print(f"IoU: {iou:.4f}")

# Visualize predictions and calculate metrics
def visualize_predictions(model, x_sample, y_true, image_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x_sample = x_sample.unsqueeze(0).to(device)  # Add batch dimension and move to device
    y_true_sample = (y_true.squeeze(0) > 0.5).cpu().numpy()  # Ensure y_true is binary

    with torch.no_grad():
        y_pred = model(x_sample).cpu().numpy()[0, 0]
        y_pred_binary = (y_pred > 0.5).astype(np.uint8)  # Binary threshold for mask

    # Visualize input, ground truth, and prediction
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Input X-ray: {image_name}")
    plt.imshow(x_sample.cpu().squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth Mask: {image_name}")
    plt.imshow(y_true_sample.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(y_pred_binary.squeeze(), cmap='jet')

    plt.show()

    # Calculate and print evaluation metrics
    calculate_metrics(y_true_sample, y_pred_binary)

# Open a file dialog to select a file
def select_file():
    Tk().withdraw()  # Hide the root window
    return askopenfilename(title="Select X-ray Image")

# Main inference script
if __name__ == "__main__":
    # Open file explorer to select the input image
    image_path = select_file()

    # Specify the directory where the ground truth masks are located
    mask_dir = 'labels 256'  # Correct path to your ground truth masks

    # Check if the selected path is valid
    if not image_path:
        print("Error: You must select an input image.")
    else:
        # Load the selected image and its corresponding mask
        x_sample, y_true = load_image_and_mask(image_path, mask_dir)

        if x_sample is None or y_true is None:
            print("Error: Could not load the selected image or corresponding mask.")
        else:
            # Load the trained model
            model_path = 'unet_model.pth'
            trained_model = load_trained_model(model_path)

            # Visualize predictions and calculate metrics
            visualize_predictions(trained_model, x_sample, y_true, os.path.basename(image_path))
