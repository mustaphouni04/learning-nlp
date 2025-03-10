import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.linalg import svd
import numpy as np

df = pd.read_csv("IMDB_Dataset.csv")

sentences = list(df["review"])
print(sentences[0])

train_size = int(0.8 * len(sentences))
test_size = int(0.2 * len(sentences))

train_sentences = sentences[:train_size]
test_sentences = sentences[test_size:]

def clean_sentences(sentences):
    to_remove = "<br />."

    for sentence in sentences:
        sentence = re.sub(to_remove, "", sentence)

    return sentences

train_sentences = clean_sentences(train_sentences)
test_sentences = clean_sentences(test_sentences)

vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(train_sentences)
print(X.shape)

print(X.toarray()[0:5])

