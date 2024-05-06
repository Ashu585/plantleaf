import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification: Healthy or Infected
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create an ImageDataGenerator to load and preprocess the data
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to the range [0, 1]
    validation_split=0.2  # Split the dataset into training and validation sets
)

# Load and preprocess healthy data
healthy_train_generator = datagen.flow_from_directory(
    "C:\Users\lap2\Desktop\Apple___healthy",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

healthy_validation_generator = datagen.flow_from_directory(
    "C:\Users\lap2\Desktop\Apple___healthy",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load and preprocess infected data
infected_train_generator = datagen.flow_from_directory(
    "C:\Users\lap2\Desktop\Apple___Black_rot",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

infected_validation_generator = datagen.flow_from_directory(
    "C:\Users\lap2\Desktop\Apple___Black_rot",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Train the model using the generators
epochs = 10  

# Train the model using the healthy dataset
history_healthy = model.fit(
    healthy_train_generator,
    epochs=epochs,
    validation_data=healthy_validation_generator
)

# Train the model using the infected dataset
history_infected = model.fit(
    infected_train_generator,
    epochs=epochs,
    validation_data=infected_validation_generator
)

# Save the trained model to an .h5 file
model.save("train.h5")
