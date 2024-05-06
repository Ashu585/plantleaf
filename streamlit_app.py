import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
)

background_image_url = "https://img.freepik.com/free-vector/hand-painted-watercolor-nature-background_23-2148934719.jpg"

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
        height: 500px; /* Adjust the height as needed */
        width:800px
    }}
    .main-title {{
        font-size: 36px;
        padding: 20px;
        color: #000000; /* Black text color */
        position: absolute;
        top: 1px; /* Adjust the top position to move the title up */
        left: 180px;
        background-color: rgba(255, 255, 255, 0.5); /* White background with opacity */
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

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Plant Disease Detection App</div>", unsafe_allow_html=True)

# Load the backend code for disease detection
import predit

st.markdown("<div class='content'>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    
    st.image(uploaded_image, use_column_width=True)

    result = predit.detect_disease(uploaded_image)


    st.write(f"Predicted Result: {result}")


