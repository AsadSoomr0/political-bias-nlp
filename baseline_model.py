import torch
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

classifier = pipeline("text-classification", model="premsa/political-bias-prediction-allsides-DeBERTa", device=device)

# Load preprocessed dataset
df = pd.read_csv("processed_dataset.csv")

if "cleaned_text" not in df.columns:
    raise ValueError("Error: 'cleaned_text' column not found.")

df["cleaned_text"] = df["cleaned_text"].astype(str).fillna("")

batch_size = 16
texts = df["cleaned_text"].tolist()

# Ensure all inputs are strings before processing
valid_texts = [text if isinstance(text, str) else "" for text in texts]

# Run inference in batches
predictions = classifier(valid_texts, truncation=True, max_length=512, batch_size=batch_size)

# Print some predictions to check model output labels
for i in range(5):
    print(f"Text: {df['cleaned_text'].iloc[i]}")
    print(f"Model Output: {predictions[i]}")
    print("-" * 50)

# Mapping from model labels to expected labels
label_mapping = {
    "LABEL_0": "left",
    "LABEL_1": "center",
    "LABEL_2": "right"
}

# Apply label mapping
df["model_prediction"] = [label_mapping[pred["label"]] for pred in predictions]

# Evaluate performance
accuracy = accuracy_score(df["bias_rating"], df["model_prediction"])
print(f"Baseline Accuracy: {accuracy:.4f}")

# Print detailed classification performance
print(classification_report(df["bias_rating"], df["model_prediction"]))

# Save results to a file
df.to_csv("baseline_predictions.csv", index=False)
print("Baseline results saved as 'baseline_predictions.csv'.")
