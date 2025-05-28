import streamlit as st

st.title("📊 Simple Streamlit App")
st.caption("This is a simple Streamlit app that allows you to pick a number and see its sine wave.")

number = st.slider("Pick a number", 0, 100)

# Plot sin(number)
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 100)
y = np.sin(x + number)
plt.plot(x, y)
st.pyplot(plt)



# HOW TO RUN THE APP
# Open terminal (Open in integrated terminal)
# Go to the folder where the my_chatbot.py file is located
# Run the following command: streamlit run my_chatbot.py