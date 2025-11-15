# ...existing code...
import streamlit as st
import numpy as np
import pandas as pd

st.title("Sine Wave Frequency Explorer")

# Slider: frequency in Hz (float, sensible range)
freq = st.slider("Frequency (Hz)", 0.1, 10.0, 1.0, step=0.1)

st.write(f"Selected frequency: {freq:.1f} Hz")

# Time range (1 second) and resolution
x = np.linspace(0, 1, 1000)  # seconds
y = np.sin(2 * np.pi * freq * x)

# Display the waveform
data = pd.DataFrame({"time_s": x, "amplitude": y})
st.line_chart(data.set_index("time_s"))

# Show sample point at t = 0.0
st.write("Sample at t=0 s:", float(y[0]))

# HOW TO RUN THE APP
# Open terminal
# Go to the folder where the app_square.py file is located
# Run the following command: streamlit run app_square.py