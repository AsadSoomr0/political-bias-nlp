import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("allsides_balanced_news_headlines-texts.csv")

# Display first few rows
print("\nSample Data:")
print(df.head())

print("\nClass Distribution:")
print(df['bias_rating'].value_counts())

plt.figure(figsize=(6, 4))

sns.countplot(x=df['bias_rating'], palette="viridis")
plt.title("Distribution of Political Bias Labels")
plt.xlabel("Bias Category")
plt.ylabel("Count")
plt.show()
