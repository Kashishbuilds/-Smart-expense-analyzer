# models/categorizer.py
# Trains and uses a Naive Bayes text classifier to predict expense categories

import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Path where the trained model is saved so we don't retrain every time
MODEL_PATH = "models/category_model.pkl"

# ------------------------------------------------------------------
# Sample training data
# Each tuple is (description text, category label)
# The more examples you add, the better the model becomes
# ------------------------------------------------------------------
TRAINING_DATA = [
    # Food
    ("lunch at restaurant", "Food"),
    ("groceries from supermarket", "Food"),
    ("coffee and snacks", "Food"),
    ("dinner with family", "Food"),
    ("pizza order online", "Food"),
    ("vegetables and fruits", "Food"),
    ("milk bread eggs", "Food"),
    ("biryani from swiggy", "Food"),
    ("breakfast at cafe", "Food"),
    ("ice cream and dessert", "Food"),
    ("zomato food order", "Food"),
    ("fast food burger", "Food"),
    ("tea and biscuits", "Food"),
    ("cooking ingredients", "Food"),
    ("restaurant bill", "Food"),

    # Travel
    ("uber cab ride", "Travel"),
    ("bus ticket booking", "Travel"),
    ("train ticket", "Travel"),
    ("flight booking", "Travel"),
    ("fuel petrol car", "Travel"),
    ("ola auto ride", "Travel"),
    ("metro card recharge", "Travel"),
    ("toll charge highway", "Travel"),
    ("taxi airport", "Travel"),
    ("bike service", "Travel"),
    ("rapido ride", "Travel"),
    ("hotel booking trip", "Travel"),
    ("parking fee", "Travel"),
    ("rental car", "Travel"),
    ("travel expense", "Travel"),

    # Bills
    ("electricity bill payment", "Bills"),
    ("internet wifi recharge", "Bills"),
    ("mobile recharge plan", "Bills"),
    ("water bill", "Bills"),
    ("rent payment", "Bills"),
    ("gas cylinder booking", "Bills"),
    ("insurance premium", "Bills"),
    ("DTH recharge", "Bills"),
    ("credit card bill", "Bills"),
    ("loan EMI payment", "Bills"),
    ("broadband bill", "Bills"),
    ("netflix subscription", "Bills"),
    ("spotify premium", "Bills"),
    ("utility payment", "Bills"),
    ("phone bill", "Bills"),

    # Shopping
    ("amazon online order", "Shopping"),
    ("flipkart purchase", "Shopping"),
    ("clothes from mall", "Shopping"),
    ("shoes purchase", "Shopping"),
    ("electronics gadgets", "Shopping"),
    ("books stationery", "Shopping"),
    ("mobile accessories", "Shopping"),
    ("home decor items", "Shopping"),
    ("cosmetics beauty products", "Shopping"),
    ("sports equipment", "Shopping"),
    ("furniture purchase", "Shopping"),
    ("gift items", "Shopping"),
    ("watch jewelry", "Shopping"),
    ("toys games", "Shopping"),
    ("medicine pharmacy", "Shopping"),
]


def train_model():
    """
    Train a Naive Bayes text classifier on sample expense descriptions.
    Saves the trained model to disk using joblib.
    """
    # Separate descriptions and labels
    descriptions = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    # Build a pipeline: TF-IDF vectorizer → Naive Bayes classifier
    # TF-IDF converts text into numeric features
    # MultinomialNB is great for text classification
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", MultinomialNB())
    ])

    pipeline.fit(descriptions, labels)

    # Save the trained model so we don't retrain on every page load
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model():
    """Load the saved model. Train it first if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        return train_model()
    return joblib.load(MODEL_PATH)


def predict_category(description: str) -> str:
    """
    Predict the category for a given expense description.

    Parameters:
        description - text like "coffee at starbucks"

    Returns:
        One of: Food, Travel, Bills, Shopping
    """
    model = load_model()
    prediction = model.predict([description])
    return prediction[0]


# Run this file directly to retrain the model
if __name__ == "__main__":
    m = train_model()
    print("Model trained and saved!")
    # Quick test
    tests = ["uber ride", "pizza delivery", "electricity bill", "amazon shopping"]
    for t in tests:
        print(f"  '{t}' → {m.predict([t])[0]}")
