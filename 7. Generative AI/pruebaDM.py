import streamlit as st

st.title('Hola mundo! 👋🏻')
st.caption('🚀 Esto es una prueba de Streamlit')
st.write('Esto es una prueba de Streamlit')

number = st.slider("Escoje un número", 0, 25)
st.write(f"El número seleccionado es: {number}")

# Grafica y = number^2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

chart_data = pd.DataFrame(np.random.randn(number, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

# Para correr este script, ejecuta en tu terminal:
# streamlit run pruebaDM.py