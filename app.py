import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import joblib

# Function to download necessary NLTK resources
def download_nltk_resources():
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')

    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)

    # Download necessary resources to the specified path
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)

    # Set the NLTK data path to ensure the correct directory is used
    nltk.data.path.append(nltk_data_path)

# Call the function to download the resources
download_nltk_resources()

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the TF-IDF vectorizer and the model using joblib
tfidf = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # Preprocess the input text
        transformed_sms = transform_text(input_sms)
        # Vectorize the input text
        vector_input = tfidf.transform([transformed_sms])
        # Predict using the model
        result = model.predict(vector_input)[0]
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"Error during prediction: {e}")