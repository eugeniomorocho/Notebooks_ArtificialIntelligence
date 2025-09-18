import streamlit as st
import numpy as np
import pandas as pd

st.title("💻 Gráfico de una función cuadrática")

# Slider
val = st.slider("Número", -10, 10, 0)

# Mostrar resultado
st.write("Resultado:", val**2)

# Datos para la curva
x = np.linspace(-50, 50, 200)
y = x**2

# Mostrar curva con Streamlit
data = pd.DataFrame({"x": x, "y": y})
st.line_chart(data.set_index("x"))

# Marcar el punto elegido
st.write("Punto seleccionado en la curva:", (val, val**2))

# HOW TO RUN THE APP
# Open terminal
# Go to the folder where the app_square.py file is located
# Run the following command: streamlit run app_square.py