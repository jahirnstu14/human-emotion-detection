import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="human_emotion_detection.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of human emotions (Make sure they correspond to the classes your model was trained on)
emotion_labels = ["Angry","Happy", "Sad"]


# Function to preprocess the image
def preprocess_image(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (256, 256))  # Resize to match model input
    img_resized = img_resized.reshape(1, 256, 256, 3)  # Add batch dimension
    img_resized = img_resized.astype('float32') / 255.0  # Normalize
    return img_resized


# Function to predict human emotion
def predict(image):
    preprocessed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Use softmax output to get probabilities
    emotion_probabilities = tf.nn.softmax(output_data[0]).numpy()

    # Get the index of the highest probability
    predicted_emotion_index = np.argmax(emotion_probabilities)

    # Get the corresponding emotion label
    predicted_emotion = emotion_labels[predicted_emotion_index]

    return predicted_emotion, emotion_probabilities


# Streamlit interface
st.title("Human Emotion Detection")

# Set custom background color using Streamlit's markdown and CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #FF7F50, #1E90FF, #32CD32);
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        st.write("Detecting emotion...")
        predicted_emotion, probabilities = predict(image)

        # Show the predicted emotion and probabilities for all emotions
        st.write(f"Predicted Emotion: {predicted_emotion}")
