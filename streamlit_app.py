import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Leaf Disease Classifier", layout="centered")
st.title("Potato & Tomato Leaf Disease Classifier 🌱")
st.write("Upload a leaf image to predict disease type.")

# Crop selection
crop = st.selectbox("Select crop", ["Potato", "Tomato"])

@st.cache_resource
def load_potato_model():
    return tf.keras.models.load_model("./saved_models/potato_model")

@st.cache_resource
def load_tomato_model():
    return tf.keras.models.load_model("./saved_models/tomato_model")

potato_classes = ["Potato_Early Blight", "Potato_Late Blight", "Potato_Healthy"]
tomato_classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(data)
    image = image.resize((256, 256))  # Resize if your model expects a specific size
    return np.array(image)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = read_file_as_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            img_batch = np.expand_dims(image, 0)
            if crop == "Potato":
                model = load_potato_model()
                class_names = potato_classes
            else:
                model = load_tomato_model()
                class_names = tomato_classes

            predictions = model.predict(img_batch)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))

            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: {confidence:.2f}")