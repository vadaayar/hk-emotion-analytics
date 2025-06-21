# train_model.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
texts = [
    "I am very happy today!",
    "This is so sad and depressing.",
    "I love this new phone!",
    "I hate getting up early.",
    "What a surprise!",
    "I'm afraid of failing.",
    "This makes me so angry!",
    "Feeling grateful and blessed.",
    "That was really disappointing.",
    "Wow, I can't believe it!"
]

labels = [
    "joy",
    "sadness",
    "joy",
    "anger",
    "surprise",
    "fear",
    "anger",
    "joy",
    "sadness",
    "surprise"
]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Model
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
with open("emotion_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved as 'emotion_classifier.pkl' and 'vectorizer.pkl'")
