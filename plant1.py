import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import io, transform
from skimage.color import rgb2gray

# Set the paths to your dataset directories
healthy_leaves_dir = "C:\Users\HP\Downloads\healthy\healthy\healthy"
infected_leaves_dir = "C:\Users\HP\Downloads\infected"

# Define image size (you can adjust this as needed)
image_size = (128, 128)

# Load and preprocess the dataset
def load_and_preprocess_data(directory, label):
    X = []
    y = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = io.imread(os.path.join(directory, filename))
            img = rgb2gray(img)  # Convert to grayscale
            img = transform.resize(img, image_size)
            
            X.append(img.flatten())  # Flatten the image
            y.append(label)
    
    return X, y

# Load and preprocess healthy leaves data
X_healthy, y_healthy = load_and_preprocess_data(healthy_leaves_dir, label=0)

# Load and preprocess infected leaves data
X_infected, y_infected = load_and_preprocess_data(infected_leaves_dir, label=1)

# Combine the datasets
X = np.vstack((X_healthy, X_infected))
y = np.hstack((y_healthy, y_infected))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Make predictions on new data
def classify_leaf(image_path):
    img = io.imread(image_path)
    img = rgb2gray(img)
    img = transform.resize(img, image_size)
    img = img.flatten()
    prediction = clf.predict([img])
    return "Healthy" if prediction == 0 else "Infected"

# Test the classifier on a new image
new_leaf_image_path = r"C:\Users\HP\Downloads\infected\infected\Black_rot\fcee4ece-01ce-45aa-a0f1-f20aa2ec8efb___FREC_Scab 3325.JPG"  # Replace with the path to your test image
result = classify_leaf(new_leaf_image_path)
print(f"The leaf is classified as: {result}")
