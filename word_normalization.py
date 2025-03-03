import nltk
from nltk.stem.porter import *
import io
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

with open("Firefox.txt", "r", encoding='latin-1') as f:
    txt = f.read()

tokens = nltk.word_tokenize(txt)

vocabulary = set(tokens)
print(len(vocabulary))
stemmer = PorterStemmer()

singles1 = [stemmer.stem(token) for token in tokens]

print(singles1[:5])
vocabulary1 = set(singles1)
print(len(vocabulary1))

stemmer1 = SnowballStemmer("english")

singles2 = [stemmer1.stem(token) for token in tokens]
vocabulary2 = set(singles2)
print(singles2[:5])

print(len(vocabulary2))

stopwords = stopwords.words('english')
singles1_s = [word for word in singles1 if word not in stopwords]
singles2_s = [word for word in singles2 if word not in stopwords]

print(len(set(singles1_s)))
print(len(set(singles2_s)))

fdist = FreqDist()

for word in singles1:
    fdist[word.lower()] += 1

fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}

top_30 = list(fdist.items())[:29]

fig, ax = plt.subplots()
words = [tup[0] for tup in top_30]
counts = [tup[1] for tup in top_30]
ax.bar(words,counts)
ax.set_ylabel('frequency')
ax.set_title('top 30 most frequent words')
plt.xticks(rotation=45, ha="right")
plt.show()

fdist = FreqDist()

for word in singles1_s:
    fdist[word.lower()] += 1

fdist2 = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}
del fdist2['doe']

top_30 = list(fdist2.items())[:29]
print(top_30)
fig, ax = plt.subplots()
words = [tup[0] for tup in top_30]
counts = [tup[1] for tup in top_30]
ax.bar(words,counts)
ax.set_ylabel('frequency')
ax.set_title('top 30 most frequent words')
plt.xticks(rotation=45, ha="right")
plt.show()


cst_stp = 'doe'
stopwords.append(cst_stp)

print(len(set(singles1_s)))
singles1_s = [word for word in singles1_s if word != 'doe']
wordcloud = WordCloud(stopwords=stopwords, background_color='black', max_words=300).generate_from_frequencies(fdist2)

plt.figure(figsize=(15,10))
plt.clf()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



