import pandas as pd
import joblib
from sklearn.metrics import classification_report

# Load model and vectorizer
model = joblib.load('trained_logreg_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load processed dataset
df = pd.read_csv('processed_dataset.csv')

df = df.dropna(subset=['cleaned_text'])

# Vectorize the cleaned text
X = vectorizer.transform(df['cleaned_text'])
y_true = df['bias_rating']

# Predict
y_pred = model.predict(X)

# Evaluate
print("Classification Report on Full Dataset:")
print(classification_report(y_true, y_pred))

# Predict new article
def predict_new_article(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction
