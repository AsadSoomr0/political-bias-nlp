import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (only needed the first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = joblib.load('trained_logreg_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Preprocessing function (same as training)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction loop
while True:
    sample_text = input("\nEnter a political article snippet (or type 'exit' to quit):\n> ")
    if sample_text.lower() == 'exit':
        break

    processed_text = preprocess_text(sample_text)
    text_vec = vectorizer.transform([processed_text])

    predicted_label = model.predict(text_vec)[0]
    predicted_probs = model.predict_proba(text_vec)[0]

    label_to_prob = dict(zip(model.classes_, predicted_probs))

    print(f"\nPredicted Label: {predicted_label}")
    print(f"Confidence ({predicted_label}): {label_to_prob[predicted_label]*100:.2f}%")

    print("\nProbabilities:")
    for label in sorted(model.classes_):
        print(f"  {label}: {label_to_prob[label]*100:.2f}%")
