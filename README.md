# Galaxy Generator & Colouriser

This repository hosts a machine learning project for generating and colourising galaxy images using deep learning models. The app leverages FastAPI for the backend and TensorFlow/PyTorch models for image generation and colourisation. It includes a web interface for users to upload grayscale images or generate galaxy images randomly.

## Features

- **Galaxy Generation:** Generate galaxy images using a pre-trained Generative Adversarial Network (GAN).
- **Image Colourisation:** Colourise grayscale images of galaxies using a pre-trained U-Net model.
- **Web Interface:** Upload grayscale images, and receive colourised images as a downloadable PNG.

## Prerequisites

Before running this project, ensure you have the following tools installed:

- Python 3.7 or higher
- Git
- TensorFlow
- PyTorch
- FastAPI
- Pillow
- Matplotlib
- scikit-image

## Installation

### Clone the repository
First, clone the repository to your local machine:



```
git clone https://github.com/rajoshi-pahari/Galaxy-Generator-and-Colouriser.git
cd Project4_Galaxy_Generator_-_Colorizer
```

## Install dependencies
Install the required Python dependencies:
```
pip install -r requirements.txt
```
## Git Large File Storage (LFS)
This project includes large model files, which are stored using Git LFS. Make sure to install Git LFS:
```
git lfs install
```

## Running the App
1. Start the FastAPI server
run the FastAPI app using `uvicorn`:
```
uvicorn main:app --reload
```
This will start a local server at `http://127.0.0.1:8000.`

Using the Web Interface

Navigate to `http://127.0.0.1:8000/docs` to interact with the API. You can upload grayscale images to be colorized, or generate new galaxy images.

Endpoints

* `POST /colorize/:` Upload a grayscale image (in PNG format) to be colorized.
Request: Upload an image file.
Response: A colorized galaxy image in PNG format.
* `POST /generate/`: Generate a random galaxy image using the pre-trained GAN model.
Request: No parameters required.
Response: A randomly generated galaxy image in PNG format
Using the Web Interface

Navigate to http://127.0.0.1:8000/docs to interact with the API. You can upload grayscale images to be colorized, or generate new galaxy images.

* To run a streamlit web app `streamlit run frontend.py`
## How to Use the app
* To use the app, start by refreshing the main page to continuously generate galaxy images. You’ll find a button that allows you to colorize the generated image using our pre-trained Pix2Pix Generator.
* navigate to the second page, where you can manually upload images. These images will be colorized by AI

## Database
* Generated_Galaxy_Images will store the image generated by AI of the galaxy whenever we refresh the app
* L_channel_Images will store the black-white images of the galaxy uploaded in page 2
* RGB_Generated_Images will store the RGB output generated by the app

## Model Details
* Galaxy Colorization Model: A U-Net model trained to colorize grayscale galaxy images. The model is loaded from the file galaxy_colorization.pth.
* Galaxy Generation Model: A pre-trained GAN model for generating galaxy images, loaded from generator_wgan_gp.h5.
