import numpy as np
import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index={value: key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])
def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,2)+3 for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review

def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)
  prediction=model.predict(preprocessed_input)
  sentiment = 'positive' if prediction[0][0]>0.5 else 'negative'
  return sentiment, prediction[0][0]

import streamlit as st

st.title('IMDB Movie review sentiment analysis')
st.write('Enter a movie review to classify')

user_input=st.text_area('Movie Review')

if st.button('Classify'):
  preprocessed_input= preprocess_text(user_input)
  prediction=model.predict(preprocessed_input)
  sentiment = 'positive' if prediction[0][0]>0.5 else 'negative'
  st.write('sentiment',sentiment)
  st.write('preiction score',prediction[0][0])
else:
  st.write('Please enter a movie review')  