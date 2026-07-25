# Importa Streamlit como st
import streamlit as st

# App title shown at the top of the Streamlit app
st.title("💻 Frequency Plot from a slider")

# Slider widget to control frequency
frequency = st.slider("Select Frequency", 1, 100, 50)

# Display the selected frequency
st.write(f"Selected Frequency: {frequency} Hz")

# Compute a simple sine wave based on the selected frequency
import numpy as np  
import pandas as pd

t = np.linspace(0, 1, 500)
y = np.sin(2 * np.pi * frequency * t)
df = pd.DataFrame({"time_s": t, "amplitude": y})
# Display a line chart using the DataFrame, indexed by time_s
st.line_chart(df.set_index("time_s"))   



# Run the app with:
# Step 1: Open your terminal
# Step 2: Install Streamlit if you haven't already:
#    pip install streamlit
# Step 3: Navigate to the directory containing this app.py file
# Step 4: Run the app with the command:
# streamlit run app.py