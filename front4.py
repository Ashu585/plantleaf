import streamlit as st
from PIL import Image

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

# Streamlit app title (displayed on the image)
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='main-title'>Plant Disease Detection App</div>", unsafe_allow_html=True)

# Rest of your Streamlit app code
# ...
