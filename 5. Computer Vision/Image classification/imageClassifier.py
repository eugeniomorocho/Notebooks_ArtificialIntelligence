import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cargar el modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('/Users/eugenio/Documents/Notebooks_ArtificialIntelligence/5. Computer Vision/Image classification/modelo_clasificacion_perros_gatos.h5')

model = load_model()

# Tamaño esperado por el modelo
IMG_SIZE = (224, 224)  # Ajustar según tu modelo

# Clases del modelo
CLASSES = ['Gato', 'Perro']  # Cambia según tu modelo

st.title("🧠 Clasificador de Imágenes con Transfer Learning")
st.write("🐱🐶 Sube una imagen de un gato o un perro y el modelo te dirá cuál es.")

# Cargar imagen
uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Preprocesar imagen
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Para batch

    # Inferencia
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]

    st.write(f"**Predicción:** {predicted_class}")
