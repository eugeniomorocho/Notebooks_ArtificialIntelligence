import streamlit as st                 # Import Streamlit for building the web app UI
import numpy as np                    # Import NumPy for numerical operations
import pandas as pd                   # Import pandas for tabular data handling

# App title shown at the top of the Streamlit app
st.title("Streamlit — Intro for Students")  

# Introductory markdown text shown below the title
st.markdown(
    """
    Welcome! This simple app shows a few basic Streamlit widgets and how to display data and plots.
    Try entering your name, choose a demo, adjust the slider and press "Generate".
    """
)

# Text input widget: allows the user to enter their name
name = st.text_input("Your name", "")

# If the user provided a name, show a small info box greeting them
if name:
    st.info(f"Hello, {name}! 👋")

# Selectbox widget to choose between two demos: "Sine wave" or "Random numbers"
demo = st.selectbox("Choose a demo", ["Sine wave", "Random numbers"])

# Slider widget to control the number of samples used in the demo (resolution)
samples = st.slider("Number of samples", 50, 2000, 200, step=50)

# Checkbox to toggle whether to show the raw data table
show_table = st.checkbox("Show raw data", value=False)

# Button widget that triggers generation of the demo when clicked
if st.button("Generate"):
    # Create an array of time values from 0 to 1 (inclusive) with 'samples' points
    t = np.linspace(0, 1, samples)
    if demo == "Sine wave":
        # Slider to control frequency for the sine wave (shown only when generating)
        freq = st.slider("Frequency (Hz)", 1.0, 10.0, 2.0, step=0.5)
        # Compute sine wave values at times t
        y = np.sin(2 * np.pi * freq * t)
        # Put results into a DataFrame with columns 'time_s' and 'amplitude'
        df = pd.DataFrame({"time_s": t, "amplitude": y})
        # Display a line chart using the DataFrame, indexed by time_s
        st.line_chart(df.set_index("time_s"))
    else:
        # Generate a cumulative sum of random numbers for the "Random numbers" demo
        data = np.random.randn(samples).cumsum()
        # Create a DataFrame with index and value columns
        df = pd.DataFrame({"index": np.arange(samples), "value": data})
        # Display a line chart using the DataFrame, indexed by 'index'
        st.line_chart(df.set_index("index"))

    # If the checkbox was checked, display the raw DataFrame as a table
    if show_table:
        st.dataframe(df)

# Small caption showing how to run the app from the terminal
st.caption("Run: streamlit run '1. Streamlit Intro.py'")

# ----------------------------
# How to run (step-by-step)
# ----------------------------
# 1) Open a terminal and activate your project virtual environment:
#    cd /Users/eugenio/Documents/Notebooks_ArtificialIntelligence
#    source .venv/bin/activate
#
# 2) Install Streamlit if not already installed:
#    pip install streamlit
#
# 3) Run the app from the project folder:
#    streamlit run "1. Streamlit Intro.py"
#
# 4) A browser window should open automatically; if not, copy the local URL
#    shown in the terminal (http://localhost:8501) into your browser.
#
# 5) Interact with the app: enter a name, choose a demo, set samples, and press "Generate".
#
# 6) To stop the app, press Ctrl+C in the terminal or close the terminal window.
#
# Note: If you run into permission or environment issues, make sure the virtualenv
#       is active and that Streamlit is installed in the same environment.