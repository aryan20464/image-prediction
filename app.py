import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Class labels for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def preprocess_image(image):
    """
    Preprocess the uploaded image to match the input format of the model.
    """
    image = image.resize((28, 28))  # Resize to 28x28
    image = image.convert("L")  # Convert to grayscale
    image_array = np.array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape to match model input
    return image_array

def predict_image(image):
    """
    Predict the class of the uploaded image.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_names[predicted_class], confidence

# Streamlit application
st.title("Fashion MNIST Image Classifier")
st.write("Upload an image to classify it into one of the Fashion MNIST categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        label, confidence = predict_image(image)
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
