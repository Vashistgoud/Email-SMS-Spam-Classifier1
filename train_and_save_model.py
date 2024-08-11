import string
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

# Sample training data
texts = [
    "Free money now!!!",
    "Hi, how are you?",
    "Get rich quick with this investment opportunity",
    "Meeting tomorrow at 10 AM",
    "Congratulations, you have won a prize!"
]
labels = [1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam

# Preprocess the texts
texts = [preprocess_text(text) for text in texts]

# Create and fit the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(texts)  # Fit and transform the data

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Save the fitted vectorizer and model
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Optionally, save metadata for future reference
metadata = {
    'vectorizer_params': vectorizer.get_params(),
    'model_params': model.get_params(),
    'training_data': texts,
    'labels': labels
}

with open('metadata.pkl', 'wb') as file:
    pickle.dump(metadata, file)

print("Vectorizer, model, and metadata have been saved.")

# Test loading the vectorizer and model
with open('vectorizer.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test prediction
sample_text = "You have won a prize!"
transformed_sample = loaded_vectorizer.transform([preprocess_text(sample_text)])
prediction = loaded_model.predict(transformed_sample)
print("Sample prediction:", "Spam" if prediction[0] == 1 else "Not Spam")
