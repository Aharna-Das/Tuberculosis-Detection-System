import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image

# Load the pre-trained model (replace 'model.h5' with your actual model path)
model = load_model('model.h5')

# Title of the app
st.title("Tuberculosis Classification App")

# Subtitle
st.write("Upload an image, and the app will classify it using the imported model!")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    # Convert to RGB (if it's grayscale or single channel)
    img = img.convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for model input
    img = img.resize((256, 256))  # Modify based on your model's input requirements
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    print(predictions.shape)

    
    # If your model output isn't compatible with `decode_predictions`, modify this part
    #decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display predictions
    
    # Define a threshold to classify as positive or negative (e.g., 0.5)
    predicted_probability = predictions[0][0]
    if predicted_probability > 0.5:
        st.write(f"Predicted Class: Tuberculosis, Confidence: {predicted_probability * 100:.2f}%")
    else:
        st.write(f"Predicted Class: No Tuberculosis, Confidence: {(1 - predicted_probability) * 100:.2f}%")