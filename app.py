import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# Define the Flask app
app = Flask(__name__)

# Folder to store uploaded images and prediction masks
UPLOAD_FOLDER = 'static/uploads'
PREDICTION_FOLDER = 'static/predictions'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER

# Ensure the uploads and predictions folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

# Check if the file is allowed (only .png, .jpg, .jpeg)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

# Load the trained model
def load_trained_model(model_path='unet_model.pth'):
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Load and preprocess image for model prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # Add batch dimension
    return img_tensor

# Predict and return result
def predict(model, image_tensor, filename):
    with torch.no_grad():
        output = model(image_tensor)
        output = output.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
        prediction_mask = (output > 0.5).astype(np.uint8)  # Threshold prediction (0.5 for binary segmentation)

        # Save the prediction mask inside 'static/predictions'
        pred_filename = f"pred_{filename}"
        pred_filepath = os.path.join(app.config['PREDICTION_FOLDER'], pred_filename)

        # Save mask as a binary image (0 or 255)
        cv2.imwrite(pred_filepath, prediction_mask * 255)

        return prediction_mask, pred_filename  # Return the mask and filename for display

# Route to upload the image
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If no file is selected
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('predict_result', filename=filename))
    return render_template('upload.html')

# Route to show prediction result
@app.route('/predict/<filename>')
def predict_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load model
    model = load_trained_model()

    # Preprocess the image
    image_tensor = preprocess_image(filepath)

    # Get the prediction and save the mask
    prediction, pred_filename = predict(model, image_tensor, filename)

    # Show the results
    if np.sum(prediction) > 0:
        result = "Signs of COVID-19 detected in the X-ray."
    else:
        result = "No signs of COVID-19 detected in the X-ray."

    return render_template('result.html', filename=filename, pred_filename=pred_filename, result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
