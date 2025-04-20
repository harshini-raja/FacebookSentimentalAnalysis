import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load word list from Excel
word_data = pd.read_excel("Positive and Negative Word List.xlsx")
positive_words = set(word_data['Positive Sense Word List'].dropna().str.lower().str.strip())
negative_words = set(word_data['Negative Sense Word List'].dropna().str.lower().str.strip())

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    words = text.split()
    essential_words = {"not", "so", "very", "never", "no"}
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words or word in essential_words]
    return ' '.join(words).strip()


