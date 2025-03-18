import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("IMDB_Dataset.csv")

sentences = list(df["review"])
print(sentences[0])

train_size = int(0.8 * len(sentences))

train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]

def clean_sentences(sentences):
    to_remove = "<br />."

    for sentence in sentences:
        sentence = re.sub(to_remove, "", sentence)

    return sentences

train_sentences = clean_sentences(train_sentences)
test_sentences = clean_sentences(test_sentences)

vectorizer = TfidfVectorizer(ngram_range=(1,2),lowercase=True, max_features=5000)
X = vectorizer.fit_transform(train_sentences)
X_test = vectorizer.transform(test_sentences)
y_train = list(df['sentiment'][:train_size])
y_train = np.array(y_train)
y_test = list(df['sentiment'][train_size:])
y_test = np.array(y_test)
print(y_train.shape)

print(X.shape)

print(X.toarray()[0:5])

print(len(df))

clf = LogisticRegression(random_state=0).fit(X,y_train)

def evaluate(clf, test_sentences, target):
    score = 0
    total = test_sentences.shape[0]
    for idx, vector in enumerate(test_sentences):
        pred = clf.predict(vector)
        if str(pred[0]) == str(target[idx]):
            score += 1
    accuracy = score / total

    return accuracy
accuracy = evaluate(clf, X_test, y_test)

print("The accuracy for Tfidf is: ", accuracy)

     
