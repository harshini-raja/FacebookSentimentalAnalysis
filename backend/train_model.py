import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import preprocess_text

# Load and clean the fb_sentiment.csv dataset
fb_df = pd.read_csv("fb_sentiment.csv")
fb_df = fb_df[["FBPost", "Label"]].rename(columns={"FBPost": "text", "Label": "label"})
label_map = {"o": "Neutral", "n": "Negative", "p": "Positive"}
fb_df["label"] = fb_df["label"].str.lower().str.strip().map(label_map)
fb_df = fb_df[fb_df["label"].notnull()]
fb_df["text"] = fb_df["text"].apply(preprocess_text)

# Load and format the Excel word list
word_data = pd.read_excel("Positive and Negative Word List.xlsx")
positive_words = word_data['Positive Sense Word List'].dropna().str.lower().str.strip()
negative_words = word_data['Negative Sense Word List'].dropna().str.lower().str.strip()

pos_df = pd.DataFrame({"text": positive_words, "label": "Positive"})
neg_df = pd.DataFrame({"text": negative_words, "label": "Negative"})

# Combine both datasets
combined_df = pd.concat([fb_df, pos_df, neg_df], ignore_index=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(combined_df["text"], combined_df["label"], test_size=0.2, random_state=42)

# Build model pipeline with class weights
model = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
])
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")