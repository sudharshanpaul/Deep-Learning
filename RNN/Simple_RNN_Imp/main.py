import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU Activation
model = load_model('movie_review_analysis_simple_rnn.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


### Creating our prediction function
def prediction_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]


st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie Review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Analyze'):

    sentiment,prediction_score = prediction_sentiment(user_input)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction_score}')
else:
    st.write('Please Enter a movie Review')

