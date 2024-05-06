import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model(r'C:/Users/HP/Desktop/Project/plant.py')
model.save(r'C:/Users/HP/Desktop/Project/plant.h5')


# Set app title and page icon
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        padding: 20px;
        color: #1E7F2B;
    }
    .file-uploader {
        padding: 20px;
        background-color: #F5F5F5;
        border-radius: 10px;
    }
    .image-container {
        padding: 20px;
        background-color: #FFFFFF;
        border-radius: 10px;
    }
    .result {
        font-size: 24px;
        padding: 20px;
        color: #1E7F2B;
    }
    .sidebar {
        background-color: #1E7F2B;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app title
st.title('Plant Disease Detection App')
st.markdown("<hr class='main-title' />", unsafe_allow_html=True)

# Upload an image
st.sidebar.header("About")
st.sidebar.markdown("This app is for plant disease detection.")
st.sidebar.markdown("Upload an image of a plant, and the app will attempt to detect any disease present.")

uploaded_image = st.file_uploader("Upload an image of a plant", type=["jpg", "png", "jpeg"], key="file_upload")

if uploaded_image is not None:
    # Display the uploaded image with custom styling
    st.subheader("Uploaded Image")
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess the image (you may need to adjust this based on your model)
    # For example, resizing the image to match the model's input size
    image = image.resize((224, 224))  # Adjust the size as needed
    image = np.asarray(image)
    image = image / 255.0  # Normalize the image

    # Make a prediction using the loaded model
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Get the class label (you may need a mapping from class index to label)
    class_label = "Disease" if prediction[0][0] > 0.5 else "Healthy"

    # Display the result with custom styling
    st.markdown("<hr class='result' />", unsafe_allow_html=True)
    st.subheader("Prediction")
    st.markdown("<div class='result'>", unsafe_allow_html=True)
    if class_label == "Disease":
        st.markdown("Disease Detected", unsafe_allow_html=True)
    else:
        st.markdown("Healthy", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
