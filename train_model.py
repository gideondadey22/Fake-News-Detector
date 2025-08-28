# train_model.py
# Training script to create consistent model artifacts in models/
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = Path("datasets/scraped.csv")
if not DATA_PATH.exists():
    raise SystemExit(f"Data file not found: {DATA_PATH}. Place your scraped.csv in datasets/")

# Load data
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "label"])
X = df["text"].astype("U")
y_raw = df["label"].astype("U")

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y_raw)  # e.g. ['Real','Fake'] -> [0,1]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, lowercase=True, ngram_range=(1,2), min_df=3)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Classifier
clf = PassiveAggressiveClassifier(max_iter=1000, warm_start=True, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:\n", classification_report(y_test, y_pred))
X_all_tfidf = vectorizer.transform(X)
cv_scores = cross_val_score(clf, X_all_tfidf, y, cv=5)
print(f"5-fold CV average accuracy: {cv_scores.mean()*100:.2f}%")

# Save artifacts
joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
joblib.dump(clf, MODELS_DIR / "pac.joblib")
joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

print("Saved models to:", MODELS_DIR)