import cv2
import numpy as np
import tensorflow as tf
import pytesseract
import streamlit as st

# Paths
IMAGE_SIZE = (128, 128)  # Resize all images to this size

# Load the pre-trained model
model = tf.keras.models.load_model("license_plate_model.h5")

# Preprocessing function for image
def preprocess_image(image):
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_array = np.expand_dims(image_resized / 255.0, axis=0)  # Normalize
    return image_resized, image_array

# License Plate Detection
def detect_license_plate(image):
    _, preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    if prediction > 0.5:  # Assuming threshold of 0.5
        return True
    else:
        return False

# Plate Recognition using Tesseract
def recognize_plate(image):
    text = pytesseract.image_to_string(image, config='--psm 7')
    return text.strip()

# Streamlit interface
st.title("License Plate Character Recognition")

# File uploader for image
image_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Read and display the uploaded image
    image = np.array(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Detect license plate and recognize characters
    if detect_license_plate(image):
        st.success("License Plate Detected!")
        plate_text = recognize_plate(image)
        st.subheader("Recognized Plate:")
        st.write(plate_text)
    else:
        st.error("No License Plate Detected.")
