
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib

# 1. Load processed dataset
df = pd.read_csv('processed_dataset.csv')

# Only keep articles that are labeled as left or right
df = df[df['bias_rating'].isin(['left', 'right'])]
df = df.dropna(subset=['cleaned_text'])

X_text = df['cleaned_text'].tolist()
y = df['bias_rating'].tolist()

# 2. Load Sentence-BERT model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Convert articles to embeddings
print("Encoding articles into embeddings...")
X_embeddings = embedder.encode(X_text, batch_size=32, show_progress_bar=True)

# 4. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train a logistic regression classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 6. Evaluate
y_pred = classifier.predict(X_test)
print("Classification Report (Embeddings Model):")
print(classification_report(y_test, y_pred))

# 7. Save model and embedder
joblib.dump(classifier, 'trained_logreg_embeddings_model.joblib')
embedder.save('sentence_transformer_model')

print("Embedding-based model and encoder saved.")
