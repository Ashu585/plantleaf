import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Set app title and page icon
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
)

# Define the URL of your background image
background_image_url = "https://cordis.europa.eu/docs/results/images/2021-11/435268.jpg"

# Custom CSS for background image and title overlay
st.markdown(
    f"""
    <style>
    .main-container {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: absolute;
        padding: 40px;
        border-radius: 10px;
        height: 500px;
        width: 800px;
    }}
    .main-title {{
        font-size: 36px;
        padding: 20px;
        color: #000000;
        position: absolute;
        top: 1px;
        left: 180px;
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 10px;
        padding: 10px;
    }}
    .content {{
        color: #FFFFFF;
        z-index: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app title (displayed on the image)
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Plant Disease Detection App</div>", unsafe_allow_html=True)

# Load the trained model for disease detection
model = tf.keras.models.load_model("plant.py")  # Replace with your model path

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform disease detection
def detect_disease(uploaded_image):
    image = Image.open(uploaded_image)
    image = np.array(image)
    image = preprocess_image(image)
    
    # Perform prediction using the model
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    
    # Replace with your own class labels
    class_labels = ["Healthy", "Infected"]
    
    result = {
        "Class": class_labels[class_index],
        "Confidence": prediction[0][class_index]
    }
    
    return result

# Add a file uploader widget for image upload
uploaded_image = st.file_uploader("Upload an image of a plant", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Perform disease detection on the uploaded image
    result = detect_disease(uploaded_image)
    
    # Display the detection result
    st.write("Disease Detection Result:")
    st.write(f"Class: {result['Class']}")
    st.write(f"Confidence: {result['Confidence']:.2f}")

# Rest of your Streamlit app code
# ...
