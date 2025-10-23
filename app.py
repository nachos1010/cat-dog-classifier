import streamlit as st
from model import CatDogClassifier
from PIL import Image

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and let AI decide if it's a cat or a dog!")

# Инициализация модели
@st.cache_resource
def load_model():
    return CatDogClassifier()

classifier = load_model()

# Загрузка изображения
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # Сохранение временного файла для предсказания
    temp_path = "temp_image.jpg"
    image.save(temp_path)
    # Предсказание
    with st.spinner('Analyzing...'):
        label, confidence = classifier.predict(temp_path)
    st.success(f"**Prediction:** {label}")
    st.info(f"**Confidence:** {confidence:.2%}")
