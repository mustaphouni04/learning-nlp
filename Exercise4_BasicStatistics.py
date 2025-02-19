import nltk
from nltk import NLTKWordTokenizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

with open("inaugural_address_corpus.txt", "r") as f:
    txt = f.read()

tokens = NLTKWordTokenizer().tokenize(txt)
fdist = FreqDist()

for word in tokens:
    fdist[word.lower()] += 1

fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}
print(list(fdist.items())[:29])

top_30 = list(fdist.items())[:29]

fig, ax = plt.subplots()
words = [tup[0] for tup in top_30]
counts = [tup[1] for tup in top_30]
ax.bar(words,counts)
ax.set_ylabel('frequency')
ax.set_title('top 30 most frequent words')
plt.xticks(rotation=45, ha="right")
plt.show()

cum_counts = np.cumsum(counts)
fig, ax = plt.subplots()
ax.bar(words, cum_counts)
ax.set_ylabel('frequency')
ax.set_title('top 30 most frequent words(cumulative)')
plt.xticks(rotation=45, ha="right")
plt.show()

num_tokens = len(tokens)
vocab = set(tokens)
print(num_tokens, len(vocab))

parsing = np.arange(1000, 146972, 1000).reshape(-1,1)
print(parsing.shape)
tokens_parsed = [tokens[:i[0]] for i in parsing]
vocabs = [len(set(toks)) for toks in tokens_parsed]
vocabs = np.array(vocabs)

reg = LinearRegression().fit(parsing, vocabs)
print("Vocabulary for 150000 tokens is ", reg.predict(np.array([[150000]])))

print("Weights (Coefficients):", reg.coef_)
print("Bias (Intercept):", reg.intercept_)

print("---------------------------------------------------------------")
print("MOBY DICK")

with open("moby_dick.txt", "r") as f:
    t = f.read()

tok = nltk.word_tokenize(t)

fdist = FreqDist()

for word in tok:
    fdist[word.lower()] += 1

fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=False)}
print(list(fdist.items())[:49])

top_50 = list(fdist.items())[:49]

fig, ax = plt.subplots()
words = [tup[0] for tup in top_50]
counts = [tup[1] for tup in top_50]
ax.bar(words,counts)
ax.set_ylabel('frequency')
ax.set_title('50 hapaxes')
plt.xticks(rotation=45, ha="right")
plt.show()

fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}
print(list(fdist.items())[:49])

top_50 = list(fdist.items())[:49]

fig, ax = plt.subplots()
words = [tup[0] for tup in top_50]
counts = [tup[1] for tup in top_50]
ax.bar(words,counts)
ax.set_ylabel('frequency')
ax.set_title('Top 50 most frequent')
plt.xticks(rotation=45, ha="right")
plt.show()

