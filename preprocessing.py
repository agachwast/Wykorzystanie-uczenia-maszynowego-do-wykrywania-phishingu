import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocessTextForTfidf(text):
    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


def preprocessText(row):
    text = str(row['TEXT'])

    if row['URL'] == 1:
        url_pattern = r'(https?://\S+|www\.\S+)'
        text = re.sub(url_pattern, ' <URL> ', text)

    if row['EMAIL'] == 1:
        email_pattern = r'\b[\w.-]+?@\w+?\.\w+?\b'
        text = re.sub(email_pattern, ' <EMAIL> ', text)

    if row['PHONE'] == 1:
        phone_pattern = r'\b\d{6,15}\b'
        text = re.sub(phone_pattern, ' <PHONENUMBER> ', text)

    money_pattern = r'(?:ÂŁ|\£|\$)\s*\d+(?:[\.,]\d+)?|\b\d+p\b'
    text = re.sub(money_pattern, ' <MONEY> ', text, flags=re.IGNORECASE)

    text = text.lower()

    text = re.sub(r'[^a-z0-9<> ]', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def prepareDataset(file_path, sep=","):

    df = pd.read_csv(file_path, sep=sep)

    for col in ['EMAIL', 'URL', 'PHONE']:
        if col in df.columns:
            df[col] = df[col].str.lower()
            df[col] = df[col].map({'yes': 1, 'no': 0})

    df['LABEL'] = df['LABEL'].str.lower()
    label_map = {'ham': 0, 'spam': 1, 'smishing': 2}
    df['LABEL_NUM'] = df['LABEL'].map(label_map).astype(int)

    df = df.drop(columns=['LABEL'])

    ham_df = df[df['LABEL_NUM'] == 0].sample(n=1000, random_state=42)
    other_df = df[df['LABEL_NUM'] != 0]

    df = pd.concat([ham_df, other_df]).sample(frac=1, random_state=42)

    df['TEXT'] = df.apply(preprocessText, axis=1)
    df['TOKENS'] = df['TEXT'].str.split()
    df['TFIDF_TEXT'] = df['TEXT'].apply(preprocessTextForTfidf)
    df['TFIDF_TEXT_LIGHT'] = df['TEXT']

    return df
