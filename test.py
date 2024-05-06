import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("train.h5")

# Define a function to detect disease
def detect_disease(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img)
    print(predictions)
    # Interpret the predictions
    if predictions[0] < 0.5:
        return "Healthy"
    else:
        return "Infected"

