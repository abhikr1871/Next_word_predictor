import streamlit as st
import requests

st.title("Next Word Predictor")
st.write("Enter a sentence and predict the next word.")

# User input
text = st.text_input("Enter some text")

if st.button("Predict"):
    if text:
        # Send POST request to backend
        response = requests.post("http://127.0.0.1:5000/predict", json={"text": text})
        
        if response.status_code == 200:
            result = response.json().get('next_word', 'No word predicted')
            st.success(f"Next Word: {result}")
        else:
            st.error("Failed to get prediction from server.")
    else:
        st.warning("Please enter some text to predict.")
