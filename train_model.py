import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load data
df = pd.read_csv("processed_dataset.csv")

# 2. Preprocessing
df = df.dropna(subset=["cleaned_text"])

X = df["cleaned_text"]
y = df["bias_rating"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Vectorize text with improvements
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',         # remove common stopwords
    ngram_range=(1, 2)             # unigrams + bigrams
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train model
model = LogisticRegression(
    max_iter=1000,
    C=0.5,
    solver='liblinear'
)
model.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Train/Test split accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
joblib.dump(model, "trained_logreg_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("\nModel and vectorizer saved.")
