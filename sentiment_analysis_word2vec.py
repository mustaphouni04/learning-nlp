import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
import spacy
import re
from tqdm import tqdm

df = pd.read_csv("IMDB_Dataset.csv")

sentences = list(df["review"])
print(sentences[0])

train_size = int(0.8 * len(sentences)) 

train_sentences = sentences[:train_size//10]
test_sentences = sentences[train_size//10:(train_size//10)+(train_size//20)]
print(len(test_sentences))

def clean_sentences(sentences):
    to_remove = "<br />."

    for sentence in sentences:
        sentence = re.sub(to_remove, "", sentence)

    return sentences

train_sentences = clean_sentences(train_sentences)
test_sentences = clean_sentences(test_sentences)

nlp = spacy.load("en_core_web_md")

X_train = [nlp(str(sentence)).vector for sentence in tqdm(train_sentences)]

X_train_arr = np.zeros((len(train_sentences),300))
for idx,i in enumerate(X_train):
    X_train_arr[idx] = X_train[idx]

X_test = [nlp(str(sentence)).vector for sentence in tqdm(test_sentences)]

X_test_arr = np.zeros((len(test_sentences),300))
for idx,i in enumerate(X_test):
    X_test_arr[idx] = X_test[idx]

print(X_test_arr.shape)


y_train = list(df['sentiment'][:train_size//10])
y_train = np.array(y_train)
y_test = list(df['sentiment'][train_size//10:(train_size//10)+(train_size//20)])
y_test = np.array(y_test)

clf = LogisticRegression(random_state=0).fit(X_train_arr, y_train)


def evaluate(clf, test_array, target):
    score = 0
    total = test_array.shape[0]
    for idx, vector in enumerate(test_array):
        pred = clf.predict(vector.reshape(1,-1))
        if str(pred[0]) == str(target[idx]):
            score += 1
    accuracy = score / total

    return accuracy
accuracy = evaluate(clf, X_test_arr, y_test)

print("The accuracy for Word2Vec is: ", accuracy)

