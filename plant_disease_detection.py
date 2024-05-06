import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
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
    "C:/Users/HP/Desktop/hlthy1",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

healthy_validation_generator = datagen.flow_from_directory(
    "C:/Users/HP/Desktop/hlthy1",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load and preprocess infected data
infected_train_generator = datagen.flow_from_directory(
    "C:/Users/HP/Desktop/infec1",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

infected_validation_generator = datagen.flow_from_directory(
    "C:/Users/HP/Desktop/infec1",
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
model.save("plant_disease_model.h5")



# Load the trained model
model = load_model("plant_disease_model.h5")

# Define a function to detect disease
def detect_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img)

    # Interpret the predictions (you can modify this part)
    if predictions[0] < 0.5:
        return "Healthy"
    else:
        return "Infected"

