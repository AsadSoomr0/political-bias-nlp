import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv("allsides_balanced_news_headlines-texts.csv", usecols=["text", "bias_rating"])

df["text"] = df["text"].fillna("")

# Remove line breaks in text
df["text"] = df["text"].astype(str).replace("\n", " ", regex=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words).strip()

df["cleaned_text"] = df["text"].apply(preprocess_text)

df.to_csv("processed_dataset.csv", index=False)

print("Preprocessing Complete.")
