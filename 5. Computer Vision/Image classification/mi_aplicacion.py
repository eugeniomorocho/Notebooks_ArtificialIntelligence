import streamlit as st

st.title("💻 Prueba de App con Streamlit")
st.write("Nombre del Estudiante: **Jorge Luis**")

file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

# Mostrar la imagen subida
if file is not None:
    st.image(file, caption="Imagen subida", use_column_width=True)
    st.write("Etiqueta de la clase")

