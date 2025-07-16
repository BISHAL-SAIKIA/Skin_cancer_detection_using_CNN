import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model=load_model("skin_cancer_cnn_new.keras")

def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    class_label = "Malignant" if prediction > 0.5 else "Benign"

    return class_label, img

#UI APP
st.title("Skin Cancer Detection")
st.markdown("""
This is an skin cancer detection app, UPLOAD IMAGE""")

uploaded_image= st.file_uploader("choose and image...", type=['jpg','png','jpeg'])

if uploaded_image is not None:
    class_label, img=predict_skin_cancer(uploaded_image,model)

    st.image(uploaded_image, caption='Uploaded_image', width=500)
    st.write(f"**Prediction: {class_label}**")
