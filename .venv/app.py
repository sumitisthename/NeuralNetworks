import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load LSTM model
model = load_model('model.h5')

#load tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_next_word = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_next_word:
            return word
    return None

#streamlit app
st.title('Next Word Prediction With LSTM and Early STOP')
input_text = st.text_input('Enter the sequence of words')
if st.button('Predict'):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"Output: {next_word}")









