# Human Emotion Detection

Welcome to the **Human Emotion Detection** project! This project demonstrates how to classify human emotions based on facial images using a TensorFlow Lite model. The model is built using EfficientNetB4, modified by adding custom dense layers, and deployed through a Streamlit web interface. OpenCV is used for image preprocessing, making the application simple and interactive.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Technologies Used](#technologies-used)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Result](#result)

## Project Overview

This project provides a web application that allows users to upload an image and receive a prediction of the person's emotion. The application uses a pre-trained EfficientNetB4-based TensorFlow Lite model, with custom dense layers added for emotion classification. It classifies emotions into categories such as Angry, Happy, and Sad. The app is built with Streamlit for an easy-to-use interface and real-time interaction.

## Technologies Used

-   **TensorFlow Lite**: For model inference.
-   **EfficientNetB4**: As the base model for emotion classification, with additional dense layers added.
-   **OpenCV**: For image resizing and preprocessing.
-   **Streamlit**: For building the interactive web interface.
-   **Google Colab**: For training and converting the TensorFlow model into TensorFlow Lite format.

## Installation

To get started with this project, follow these steps:

1.  **Clone the Repository**
    
    `git clone https://github.com/jahirnstu14/human-emotion-detection.git` 
    
2.  **Set Up a Virtual Environment**
    
    `python -m venv venv
    venv/Scripts/activate  # On Windows
    
    
3.  **Install Dependencies**
 
    
    `pip install streamlit
    
    opencv-python
    
    tensorflow
    
    numpy
    
    pillow` 
    
4.  **Download the Model**
    
    Make sure to download the TensorFlow Lite model file (`human_emotion_detection.tflite`) and place it in the project directory.
    

## Usage

1.  **Run the Streamlit App**
    
  
    
    `streamlit run app.py` 
    
2.  **Open Your Browser**
    
    Once the app is running, open your web browser and go to `http://localhost:8501` to access the application.
    
3.  **Upload an Image**
    
    -   Use the file uploader to upload an image in JPG, JPEG, or PNG format.
    -   Click on the **"Predict"** button to get the predicted emotion and its probability.

## Result

Once an image is uploaded and analyzed, the app displays the predicted emotion and the probabilities for each emotion.

![enter image description here](https://github.com/jahirnstu14/human-emotion-detection/blob/main/Screenshot%202024-09-23%20100716.jpg)
