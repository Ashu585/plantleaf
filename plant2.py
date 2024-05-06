import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the path to your dataset directories
healthy_leaves_dir = r"C:\Users\HP\Desktop\hlthy"
infected_leaves_dir = r"C:\Users\HP\Desktop\infec"

# Step 1: Data Preprocessing
# Define image size and batch size
image_size = (128, 128)
batch_size = 32

# Create data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation sets
)

train_generator = train_datagen.flow_from_directory(
    healthy_leaves_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',  # Use this subset for training
    shuffle=True  # Shuffle the data for better training
)

validation_generator = train_datagen.flow_from_directory(
    healthy_leaves_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',  # Use this subset for validation
    shuffle=False  # Do not shuffle the validation data
)

# Calculate the number of steps per epoch based on the number of training samples and batch size
steps_per_epoch = train_generator.samples // batch_size

# Increase the number of epochs since we have more data
epochs = 30

# Step 2: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification, so using sigmoid activation
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch  # Specify the number of batches per epoch
)

# Step 4: Predict using the Trained Model
# Load an image to predict (replace with the actual path to the image)
leaf_image_path = r"C:\Users\HP\Desktop\infec\acb4845c-bee1-478c-b11c-ade0ae397c51___JR_FrgE.S 8818.JPG"

# Preprocess the image
img = tf.keras.preprocessing.image.load_img(leaf_image_path, target_size=image_size)
img = tf.keras.preprocessing.image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Make a prediction
predictions = model.predict(img)

# Interpret the prediction (0: Healthy, 1: Infected)
if predictions[0] < 0.5:
    print("The leaf is healthy.")
else:
    print("The leaf is infected.")
