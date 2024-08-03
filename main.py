import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
vocab_size = 10000

# Load the model
model = load_model('rnn_model.h5')

# Decode Function
def decode_review(review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])

# Function to preprocess user input
def preprocess_input(user_input):
    words = user_input.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Clip the indices to be within the vocab_size - 1
    encoded_review = [min(index, vocab_size - 1) for index in encoded_review]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_input(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit App
st.set_page_config(
    page_title="IMDB Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        background-color: #333;
        color: #fff;
    }
    .main {
        background-color: #444;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTextInput, .stTextInput input {
        color: #fff;
        background-color: #555;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton button {
        background-color: #007BFF;
        color: #fff;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #0056b3;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review to know the sentiment')

# User Input
user_input = st.text_input('Enter your review')

if st.button('Classify'):
    if user_input:
        sentiment, confidence = predict_sentiment(user_input)
        if sentiment == 'Positive':
            st.success(f'The sentiment of the review is {sentiment} with a confidence of {confidence:.2f}')
        else:
            st.error(f'The sentiment of the review is {sentiment} with a confidence of {confidence:.2f}')
    else:
        st.warning('Please enter a review to classify')

# Sidebar
st.sidebar.header('About the App')
st.sidebar.write("""
This app uses a pre-trained RNN model to analyze the sentiment of movie reviews from the IMDB dataset.
Simply enter a movie review and click the 'Classify' button to see if the review is positive or negative.
""")
