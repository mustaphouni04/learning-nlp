import nltk
from nltk import NLTKWordTokenizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np

with open("inaugural_address_corpus.txt", "r") as f:
    txt = f.read()

tokens = NLTKWordTokenizer().tokenize(txt)
fdist = FreqDist()

for word in tokens:
    fdist[word.lower()] += 1

fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}
print(list(fdist.items())[:30])

top_30 = list(fdist.items())[:30]

plt.hist(list(fdist.values())[:29],cumulative=True)
plt.hist(list(fdist.values())[:29])
plt.show()

num_tokens = len(tokens)
vocab = set(tokens)
print(num_tokens, len(vocab))

parsing = np.arange(1000, 150000, 1000)
print(parsing[0:5])






