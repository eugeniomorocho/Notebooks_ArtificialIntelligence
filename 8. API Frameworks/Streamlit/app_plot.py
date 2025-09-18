import streamlit as st
import numpy as np
import pandas as pd

st.title("Función cuadrática interactiva")

# Slider
val = st.slider("Número", -50, 50, 0)

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