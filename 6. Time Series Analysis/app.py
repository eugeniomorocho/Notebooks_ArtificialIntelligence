import streamlit as st
import pandas as pd

st.write("""
        # Sales model
        Below are our sales predictions for this customer: 
        """)

df = pd.read_csv("/Users/eugenio/Documents/Notebooks_ArtificialIntelligence/Datasets/air_passengers.csv")
st.write(df)

# Plot the time series using Streamlit line_chart
# Assuming 'Month' is the date column and '#Passengers' is the value column
st.line_chart(df.set_index('Month')['#Passengers'])

#window = st.slider("Forecast")
######st.write(m.run(window=15))
#st.write(m.run(window=window))

number = st.slider('Pick a number', 0, 100)  # 👈 this is a widget
st.write(number, 'squared is', number * number)

file = st.file_uploader("Pick a file")

color = st.color_picker("Pick a color")

#st.altair_chart(my_chart)

pets = ["cat", "dog", "bird"]
pet = st.radio("Pick a pet", pets)

date = st.date_input("Pick a date")

# HOW TO RUN THE APP
# Open terminal
# Go to the folder where the app.py file is located
# Run the following command: streamlit run app.py