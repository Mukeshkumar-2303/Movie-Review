import streamlit as st 
import pandas as pd
import numpy as np  
from tensorflow.keras.datasets import imdb 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle

#load the imbd dataset 
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

reverse_word_index = {value: key for key, value in word_index.items()}


model = load_model('imdb_rnn_model.h5')

def decode_review(text):# function to decode reviews 
    return ' '.join([reverse_word_index.get(i-3,'?') for i in text]) 

def preprocess_technique(text):
    words=text.lower().split()
    encoded_review= [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


#prediction function 
def predict_sentiment(review):
    processed_review=preprocess_technique(review)
    prediction=model.predict(processed_review)
    sentiment='positive' if prediction[0][0]> 0.5 else 'negative'
    return sentiment,prediction[0][0]   


# streamlit interface
st.title("IMDB Sentiment Analysis")
st.write("Enter a movie review to classify its sentiment:")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input:
        
        preprocess_input=preprocess_technique(user_input)

        prediction = model.predict(preprocess_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        st.write(f"Predicted Sentiment: {sentiment}")
        st.write(f"Prediction Score: {prediction[0][0]:.4f}")

else:
    st.write("Please enter a movie review and click 'Classify' to see the sentiment prediction.")   