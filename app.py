import streamlit as st
import pickle

with open('vectorizer.sav', 'rb') as f:  
    vectorizer = pickle.load(f)

with open('finalized_model.sav', 'rb') as f2:  
    model = pickle.load(f2)

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input:
        # Preprocess the input
        cleaned_text = user_input

        # Vectorize the text
        text_vec = vectorizer.transform([cleaned_text])

        # Make the prediction
        prediction = model.predict(text_vec)[0]

        st.write(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")