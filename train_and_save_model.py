import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Sample training data
texts = ["Free money now!!!", "Hi, how are you?", "Get rich quick with this investment opportunity", "Meeting tomorrow at 10 AM", "Congratulations, you have won a prize!"]
labels = [1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam

# Create a pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
text_clf.fit(texts, labels)

# Save the model
joblib.dump(text_clf, 'text_clf_model.joblib')

print("Model has been saved as 'text_clf_model.joblib'.")
