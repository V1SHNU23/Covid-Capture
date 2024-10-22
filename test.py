import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# U-Net model with Dropout for regularization
def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    def encoder_block(inputs, num_filters, dropout_rate):
        x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        p = MaxPooling2D(pool_size=(2, 2))(x)
        return x, p

    # Decoder
    def decoder_block(inputs, skip_features, num_filters, dropout_rate):
        x = UpSampling2D(size=(2, 2))(inputs)
        x = concatenate([x, skip_features])
        x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    # Encoder
    c1, p1 = encoder_block(inputs, 64, 0.1)
    c2, p2 = encoder_block(p1, 128, 0.1)
    c3, p3 = encoder_block(p2, 256, 0.2)
    c4, p4 = encoder_block(p3, 512, 0.2)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)

    # Decoder
    c6 = decoder_block(c5, c4, 512, 0.2)
    c7 = decoder_block(c6, c3, 256, 0.2)
    c8 = decoder_block(c7, c2, 128, 0.1)
    c9 = decoder_block(c8, c1, 64, 0.1)

    outputs = Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Local Paths to your data directories
image_dir = 'img 256'
mask_dir = 'labels 256'

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Load and preprocess data
def load_data(image_dir, mask_dir):
    images, masks, filenames = [], [], []

    for img_filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue  # Skip if image/mask is missing or corrupted

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))

        images.append(np.expand_dims(img, axis=-1))  # Add channel dimension
        masks.append(np.expand_dims(mask, axis=-1))  # Add channel dimension

        filenames.append(img_filename)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0

    return images, masks, filenames

# Load data
x_train, y_train, filenames = load_data(image_dir, mask_dir)

# Initialize the model
model = unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 1))

# Set up callbacks for early stopping and model checkpointing
callbacks = [
    EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint('unet_best_model.keras', save_best_only=True, verbose=1)
]

# Train the model
history = model.fit(x_train, y_train, epochs=70, batch_size=10, validation_split=0.1, callbacks=callbacks)

# Save the final model
model.save('unet_covid19_xray_final.keras')

# Visualize predictions
def visualize_predictions(model, x_data, y_true, filenames):
    idx = np.random.randint(0, len(x_data))
    x_sample = x_data[idx]
    y_true_sample = y_true[idx]
    file_name = filenames[idx]

    y_pred = model.predict(np.expand_dims(x_sample, axis=0))[0]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Input X-ray: {file_name}")
    plt.imshow(x_sample.squeeze(), cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title(f"Ground Truth Mask: {file_name}")
    plt.imshow(y_true_sample.squeeze(), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(y_pred.squeeze(), cmap='gray')

    plt.show()

# Visualize predictions
visualize_predictions(model, x_train, y_train, filenames)