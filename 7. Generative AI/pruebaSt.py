# pip install streamlit

import streamlit as st

st.title("📊 Simple Streamlit App")
st.caption("This is a simple Streamlit app that allows you to pick a number and see its sine wave.")

# Pick a number using a slider
number = st.slider("Elige un número", 0, 100)

# Plotear el sin(number)
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
y = np.sin(x + number)
plt.plot(x, y)
st.pyplot(plt)

# ls -> enlista documentos
# cd -> cambia de directorio

# stremlit run pruebaSt.py