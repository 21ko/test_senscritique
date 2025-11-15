import pandas as pd
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Clean:
    def __init__(self):
       
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.stopwords = set(stopwords.words('french'))
        self.lemmatizer = WordNetLemmatizer()
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "["  
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(r"", text)

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = self.remove_emojis(text)
        text = text.lower()
        text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w not in self.stopwords]

        words = [self.lemmatizer.lemmatize(w) for w in words]

        return " ".join(words)

    def clean_dataframe(self, df, column_name):
        df[column_name + "_clean"] = df[column_name].astype(str).apply(self.clean_text)
        return df

    def embed_comments(self, comments):
        return self.model.encode(comments, convert_to_tensor=True)
