from model_handler import SignLanguageModel
import streamlit as st
import cv2
import numpy as np
from PIL import Image

model = SignLanguageModel(model_path="./model/checkpoint-3806")

st.title("Sign Language Recognition")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_cv = np.array(img)
    result = model.analyze_sign(img_cv)
    st.subheader(result)

camera_img = st.camera_input("Take a picture")

if camera_img is not None:
    img = Image.open(camera_img)
    st.image(img, caption="Camera Image", use_column_width=True)

    img_cv = np.array(img)
    result = model.analyze_sign(img_cv)
    st.subheader(result)
