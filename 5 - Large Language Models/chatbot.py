import streamlit as st
from openai import OpenAI
openai = OpenAI()

# Title of the app
st.title("Chatbot")
# Subtitle of the app
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# chat_message: Display a welcome message when the app is first loaded
with st.chat_message("user"):
    st.write("Hello! 👋 I'm a chatbot. How can I help you today?")

# chat_input: Create a text input field for the user to enter their message
user_message = st.text_input("Type your message here")

# Check if the user has entered a message
if user_message:
    # Display the user message in the chat window
    with st.chat_message("user"):
        st.write(user_message)

    # Get the response from the chatbot
    bot_message = openai.get_response(user_message)

    # Display the chatbot response in the chat window
    with st.chat_message("bot"):
        st.write(bot_message)

# HOW TO RUN THE APP
# Open terminal
# Go to the folder where the my_chatbot.py file is located
# Run the following command: streamlit run my_chatbot.py