import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


def preprocess_data():

    df = pd.read_csv(RAW_DATA_PATH)

    df.dropna(inplace=True)

    df["text"] = df["text"].apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df