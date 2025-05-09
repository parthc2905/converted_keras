import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
import tempfile

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def preprocess_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape(1, 224, 224, 3)

def predict_image(image_array):
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    label = class_names[index].strip()[2:]  # Remove index number
    result = "Good" if "good" in label.lower() else "Bad"
    return result, confidence_score, label

def webcam_prediction():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB")

        img_array = (img_rgb.astype(np.float32) / 127.5) - 1
        prediction = model.predict(img_array.reshape(1, 224, 224, 3))
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]
        label = class_names[index].strip()[2:]
        result = "Good" if "good" in label.lower() else "Bad"

        st.markdown(f"### Class: {label}")
        st.markdown(f"### Prediction: {result}")
        st.markdown(f"### Confidence: {round(confidence_score * 100, 2)}%")

        if st.button("Stop Webcam"):
            break

    cap.release()

# Streamlit App
st.title("Good vs Bad Classifier")
option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        data = preprocess_image(image)
        result, score, label = predict_image(data)
        st.markdown(f"### Class: {label}")
        st.markdown(f"### Prediction: {result}")
        st.markdown(f"### Confidence: {round(score * 100, 2)}%")

elif option == "Use Webcam":
    st.markdown("### Allow webcam access and click below to start.")
    if st.button("Start Webcam"):
        webcam_prediction()
