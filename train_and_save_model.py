from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample training data
texts = ["Free money now!!!", "Hi, how are you?", "Get rich quick with this investment opportunity", "Meeting tomorrow at 10 AM", "Congratulations, you have won a prize!"]
labels = [1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam

# Create and fit the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)  # Fit and transform the data

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Save the fitted vectorizer and model
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Vectorizer and model have been saved.")
