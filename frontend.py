import streamlit as st
import requests
from PIL import Image
import time
import io
import os
from main import RGB_DIR, GALAXY_DIR, L_CHANNEL_DIR  # API endpoints

# API endpoints
GALAXY_GENERATE_API = "http://127.0.0.1:8000/generate/"  # Endpoint for generating galaxy images
GALAXY_COLOURISE_API = "http://127.0.0.1:8000/colourise/"  # Endpoint for colourising images

# Set up the Streamlit page layout
st.set_page_config(page_title="AI Galaxy Generator & Colouriser", layout="wide")

# Add custom CSS for the background image
background_image_url = "https://plus.unsplash.com/premium_photo-1669839137069-4166d6ea11f4?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Z2FsYXh5fGVufDB8MHwwfHx8MA%3D%3D"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title for the page
st.markdown(
    """
    <h1 style="color: #f8f8f8; text-align: center; font-family: 'Trebuchet MS', sans-serif; letter-spacing: 4px; text-transform: uppercase; background: linear-gradient(to right, #8a2be2, #9370db, #dda0dd); -webkit-background-clip: text; color: transparent; padding: 10px 0; font-size: 38px;">
        AI Galaxy Generator & Colouriser
    </h1>
    """,
    unsafe_allow_html=True
)

# Buttons for page navigation
col2, col3, col4 = st.columns(3)
with col2:
    if st.button("Galaxy Generator"):
        st.session_state.page = "Galaxy Generator"  # Switch to Galaxy Generator page
with col3:
    if st.button("Galaxy Colouriser"):
        st.session_state.page = "Galaxy Colouriser"  # Switch to Galaxy Colouriser page
with col4:
    if st.button("About This App"):
        st.session_state.page = "About the App"  # Switch to About page

# Initialize page variable in session state
if 'page' not in st.session_state:
    st.session_state.page = "Galaxy Generator"  # Default page when first loaded

# Page 1: Generate and Colour the Random Galaxy Images
if st.session_state.page == "Galaxy Generator":
    """Page for generating a random galaxy image using AI. It also allows colourising the generated galaxy image.
    Displays the generated galaxy image and provides an option to colourise it."""

    st.markdown(
        """
        <h1 class="header1">
            Refresh to Generate a New Galaxy Image
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.text(" ")

    def load_recent_galaxy_img():
        """Loads the most recent galaxy image from the GALAXY_DIR directory.
        Returns: Image: The most recent galaxy image from the directory."""
        latest_image_path = sorted(os.listdir(GALAXY_DIR))[-1]  # Get the most recent file
        return Image.open(os.path.join(GALAXY_DIR, latest_image_path))  # Open and return the image file

    # Create columns for layout
    col1, col2 = st.columns(2)  # Create two columns for displaying images

    # Trigger image generation on first load
    if 'generated_galaxy_image' not in st.session_state:
        with st.spinner("Generating your galaxy image..."):  # Show spinner while generating
            response = requests.post(GALAXY_GENERATE_API)  # POST request to generate galaxy image
            if response.status_code == 200:  # Check if the request was successful
                st.session_state.generated_galaxy_image = load_recent_galaxy_img()  # Load the generated image
                st.session_state.image_displayed = True  # Track if an image has been displayed

    # Display the generated image if it exists
    if st.session_state.get('image_displayed', False):
        with col1:
            st.image(st.session_state.generated_galaxy_image, caption="AI Generated Galaxy Image", use_container_width=True)  # Display generated image

            # Button to colourise the currently displayed image
            if st.button("Colourise it"):
                img_byte_arr = io.BytesIO()  # Create a byte stream for the image
                st.session_state.generated_galaxy_image.save(img_byte_arr, format='PNG')  # Save the image to the byte stream
                img_byte_arr.seek(0)  # Seek to the beginning of the stream
                files = {'file': img_byte_arr.getvalue()}  # Prepare file for POST request

                with col2:
                    with st.spinner("AI is colourising the generated Galaxy..."):  # Show spinner while colourising
                        colourise_response = requests.post(GALAXY_COLOURISE_API, files=files)  # POST request to colourise the image

                        if colourise_response.status_code == 200:  # Check if colourisation was successful
                            # Load and display the latest RGB image
                            rgb_image_path = sorted(os.listdir(RGB_DIR))[-1]  # Get the most recent colourised image
                            colourised_image = Image.open(os.path.join(RGB_DIR, rgb_image_path))  # Open the image file

                            # Display colourised image in the second column
                            st.image(colourised_image, caption="AI Colourised Image", use_container_width=True)
    else:
        st.info("Generating your galaxy image...")  # Inform user that the galaxy image is being generated

# Page 2: Colourise an Image of Galaxy (Galaxy Zoo Format)
if st.session_state.page == "Galaxy Colouriser":
    """Page that allows users to upload their own galaxy image and colourises it using the AI model.
    Users can upload an image, and the app will colourise it and display the result."""

    st.markdown(
        """
        <h1 class="header2">
            Let us Colourise the Galaxy!
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.text(" ")

    # Reset the upload counter when the file uploader is activated
    uploaded_galaxy_image = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"],
        on_change=lambda: setattr(st.session_state, 'upload_counter', 0)  # Reset counter on upload
    )

    if uploaded_galaxy_image is not None:  # Check if a file has been uploaded
        # Prepare to colourise the uploaded image
        img_byte_arr = io.BytesIO()  # Create a byte stream
        uploaded_image = Image.open(uploaded_galaxy_image)  # Open the uploaded image file
        uploaded_image.save(img_byte_arr, format='PNG')  # Save to byte stream
        img_byte_arr.seek(0)  # Reset byte stream position
        files = {'file': img_byte_arr.getvalue()}  # Prepare the file data for request

        with st.spinner("Colourising your galaxy image..."):  # Show spinner
            colourise_response = requests.post(GALAXY_COLOURISE_API, files=files)  # Send to colourisation API

            if colourise_response.status_code == 200:  # If colourisation is successful
                colourised_image = Image.open(io.BytesIO(colourise_response.content))  # Load the colourised image
                st.image(colourised_image, caption="AI Colourised Galaxy Image", use_container_width=True)  # Display the image
    else:
        st.warning("Please upload a galaxy image to colourise.")  # Prompt user to upload an image

# Page 3: About the Application
if st.session_state.page == "About the App":

    st.markdown(
        """
        <h1 class="header2">
            AI Cosmic Colouriser
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Welcome to AI Galaxy Generator & Colouriser! This application is designed to generate and colourise galaxy images using AI.
        <br><br>

        **Developer**:
        - **Name**: Rajoshi Pahari
        - **Background**: I am a space enthusiast into Machine Learning, currently working on the **ML for Astronomy** project at Spartifical. With a deep interest in artificial intelligence, astrophysics, and their intersection, I have developed this application to demonstrate the power of AI in creating and colourising galaxy images. My passion lies in developing innovative machine learning models for solving complex problems, such as the classification of star sizes, star types and the generation of astronomical images using GANs (Generative Adversarial Networks).
        <br><br>

        **About the Application**:
        - The **AI Galaxy Generator & Colouriser** is designed to generate and colourise galaxy images using state-of-the-art deep learning techniques. It provides two key functionalities:
            1. **Galaxy Generator**: This feature generates realistic galaxy images based on random noise inputs using a Generative Adversarial Network (GAN).
            2. **Galaxy Colouriser**: This feature allows users to upload black and white galaxy images, which are then colourised using a pre-trained model based on the Unet architecture.
        <br><br>

        **How It Works**:
        - The app utilizes **GANs** (Generative Adversarial Networks) for generating galaxy images and a **deep neural network** (Unet) for colourising grayscale galaxy images. These models are trained using large datasets of galaxy images and are capable of producing realistic, high-quality results.
        - The **Galaxy Generator** uses a GAN to generate a random galaxy from noise, while the **Galaxy Colouriser** uses an image-to-image translation model (Unet) to predict the colour channels of a grayscale image, effectively 'colourising' it.
        <br><br>

        **Tech Stack**:
        - **Frontend**: Streamlit, which provides a simple yet powerful interface for building interactive web applications. It allows users to interact with the models and visualize the generated or colourised galaxy images.
        - **Backend**: FastAPI, a modern, fast web framework for building APIs with Python. FastAPI ensures quick responses and is highly compatible with asynchronous requests, making it ideal for machine learning applications.
        - **AI Models**: The models used in this application are based on **Generative Adversarial Networks (GANs)** for image generation and **Unet** for colourisation. These models are trained on large astronomical datasets and perform tasks such as image generation, style transfer, and colour prediction.
        - **Other Tools**:
            - **PyTorch** and **TensorFlow** for deep learning and model inference.
            - **PIL (Pillow)** for image processing and manipulation.
            - **skimage** for colour space transformations (Lab to RGB).
        <br><br>
        """,
        unsafe_allow_html=True
    )