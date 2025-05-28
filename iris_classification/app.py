import streamlit as st
import numpy as np
import joblib
from PIL import Image
import os

base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "iris_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))


species_names = ["Setosa", "Versicolor", "Virginica"]
image_paths = {
    "Setosa": "setosa.jpeg",
    "Versicolor": "versicolor.jpg",
    "Virginica": "virginica.jpg"
}

st.set_page_config(page_title="Iris Classifier 🌸", layout="wide")

st.markdown("<h1 style='text-align: center; color: #7b1fa2;'>🌸 Iris Flower Species Classifier</h1>",unsafe_allow_html=True)

st.markdown("<p style='text-align: center;'>Enter measurements below to predict the species of an Iris flower</p>",unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([2, 1])

with left:
    st.subheader("Input Flower Measurements")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, step=0.1, format="%.1f")
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1, format="%.1f")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3, step=0.1, format="%.1f")
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, step=0.1, format="%.1f")

    if st.button("Predict"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        predicted_species = species_names[prediction]

        with right:
            st.markdown("<h3 style='text-align: center;'>Predicted Iris Species</h3>",unsafe_allow_html=True)
            image_path = os.path.join(os.path.dirname(__file__), image_paths[predicted_species])
            image = Image.open(image_path)
            st.image(image,use_container_width=True)
            st.markdown(f"<h3 style='color: green; text-align: center;'>{predicted_species}</h3>", unsafe_allow_html=True)
