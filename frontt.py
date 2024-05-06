import streamlit as st
from PIL import Image

# Streamlit app title
st.title('Plant Disease Detection App')

# Upload an image
uploaded_image = st.file_uploader("Upload an image of a plant", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Placeholder for the plant disease detection code
    # You'll need to integrate a machine learning model here
    # For demonstration, we assume a simple example

    # Example result:
    result = "Healthy"  # Replace with actual detection result

    # Display the result
    st.write(f"Prediction: {result}")

# Optional: Add a sidebar for additional controls or information
st.sidebar.header("About")
st.sidebar.markdown("This app is for plant disease detection.")
st.sidebar.markdown("Upload an image of a plant, and the app will attempt to detect any disease present.")

