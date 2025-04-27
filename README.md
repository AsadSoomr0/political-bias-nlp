# Political Bias Detection Using NLP

## Overview
This project applies Natural Language Processing (NLP) techniques to classify political news articles as **left-leaning**, **right-leaning**, or **center**.  
We use the [QBias dataset](https://github.com/irgroup/Qbias) from [AllSides](https://www.allsides.com/headline-roundups) to train and evaluate a text classification model.

We experimented with multiple approaches and ultimately developed a **TF-IDF + Logistic Regression** model for final classification.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AsadSoomr0/political-bias-nlp.git
   cd political-bias-nlp
   ```

2. **Create and activate a virtual environment:**
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On Mac/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Access the Dataset:**
   - Download the dataset from:  
     [AllSides Balanced News Headline Roundups](https://www.allsides.com/headline-roundups)  
   - Alternatively, use the [QBias GitHub repository](https://github.com/irgroup/Qbias).
   - Place the CSV (`allsides_balanced_news_headlines-texts.csv`) into the project directory.

---

## Running the Project

### 1. Preprocess the Dataset
This script cleans and prepares the text data for model training.

```bash
python preprocess.py
```

### 2. Train the Model
This script trains a Logistic Regression model on the preprocessed text.

```bash
python train_model.py
```

The trained model and TF-IDF vectorizer will be saved locally.

### 3. Evaluate the Model
This script evaluates the trained model on the full dataset.

```bash
python predict_model.py
```

You will see a classification report showing precision, recall, and F1 scores for left, right, and center articles.

### 4. Predict Custom Text
You can test the model by entering your own article snippets.

```bash
python predict_custom.py
```

Example:

```
Enter a political article snippet (or type 'exit' to quit):
> The government must expand social welfare programs to help families.
Predicted Label: left
Confidence: 68.4%
```

---

## Notes
- Our final model uses **TF-IDF embeddings** (unigrams + bigrams) with a **Logistic Regression** classifier.
- The model classifies text into **left**, **right**, or **center** categories.
- Preprocessing includes lowercasing, punctuation removal, stopword removal, tokenization, and lemmatization.

---

## Repository Structure

```
├── baseline_model.py          # (old baseline, not primary focus)
├── baseline_model_metal.py     # (MacOS version of baseline)
├── explore_dataset.py          # Exploratory data analysis
├── preprocess.py               # Dataset cleaning and preprocessing
├── train_model.py              # TF-IDF + Logistic Regression model training
├── predict_model.py            # Evaluate the trained model
├── predict_custom.py           # Predict on user-input text
├── requirements.txt
├── README.md
└── processed_dataset.csv       # (after you run preprocess.py)
```

---

## Acknowledgments
- [QBias Dataset](https://github.com/irgroup/Qbias)
- [AllSides Balanced News Headlines](https://www.allsides.com/headline-roundups)
