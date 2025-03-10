import spacy
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

nlp = spacy.load("en_core_web_md")
word = nlp("football")
word_2 = nlp("frankfurteria")

print(word.vector)
print(word_2.vector)
print(len(word.vector))

word_3 = nlp("flowers")
print(word_3.vector)

sent = nlp("I love playing football with my friends")
print(sent.vector)
print(len(sent.vector))


utt1 = "I visited Scotland"
utt2 = "I went to Edinburgh"

A, B = nlp(utt1).vector, nlp(utt2).vector
cosine = np.dot(A,B)/(norm(A)*norm(B))
print("Cosine Similarity:", cosine)

utt1 = "I did it all by myself"
utt2 = "I don't need no help"

print("similar utterances:", utt1, utt2)

A, B = nlp(utt1).vector, nlp(utt2).vector
cosine = np.dot(A,B)/(norm(A)*norm(B))
print("Cosine Similarity:", cosine)

utt1 = "I spoke Russian with my homie"
utt2 = "He hates salad"
print("different utterances:", utt1, utt2)

A, B = nlp(utt1).vector, nlp(utt2).vector
cosine = np.dot(A,B)/(norm(A)*norm(B))
print("Cosine Similarity:", cosine)

words = ["cat", "dog", "tiger", "elephant", "bird", "monkey", "lion", "cheetah", "burger", "pizza", "food", "cheese", "wine", "salad", "noodles", "fruit", "vegetables"]

vecs = [nlp(word).vector for word in words]
ft = np.zeros((len(vecs),vecs[0].shape[0]))

for i in range(len(vecs)):
	ft[i] = vecs[i]

pca = PCA(n_components=2)
pcas = pca.fit_transform(ft)


fig, ax = plt.subplots()

ax.scatter(pcas[:,0], pcas[:,1])

for i in range(len(vecs)):
	plt.text(pcas[i,0],
			pcas[i,1],
			words[i],
			)



plt.show()

sentences = ["I purchased a science fiction book last week.", "I loved this fragance: light, floral and feminine.", "I purchased a bottle of wine"]

keyword = "perfume"

ft2 = [nlp(sent).vector for sent in sentences]
keyword_vector = nlp(keyword).vector

cosines = []
for sent in ft2:
	cosine = np.dot(sent,keyword_vector)/(norm(sent)*norm(keyword_vector))
	cosines.append(cosine)
	print("Cosine Similarity:", cosine)
print("We can filter out sentence:", np.argmin(cosines))


words = ["laptop", "computer", "windows", "door", "table", "legs", "arms", "eyes", "armpit", "torso", "linux", "admin","furnace", "television", "carpet", "parquet", "limb", "hair", "watch", "chair"]
vecs = [nlp(word).vector for word in words]
ft = np.zeros((len(vecs),vecs[0].shape[0]))

for i in range(len(vecs)):
	ft[i] = vecs[i]

pca = PCA(n_components=2)
pcas = pca.fit_transform(ft)


fig, ax = plt.subplots()

ax.scatter(pcas[:,0], pcas[:,1])

for i in range(len(vecs)):
	plt.text(pcas[i,0],
			pcas[i,1],
			words[i],
			)



plt.show()

alexa_reviews = pd.read_csv('amazon_alexa.tsv', sep='\t')
print(alexa_reviews)
print(alexa_reviews['verified_reviews'][0])
reviews = list(alexa_reviews['verified_reviews'])

music = nlp("music").vector
stack = []
for idx, sent in enumerate(reviews):
    cosine = np.dot(nlp(str(sent)).vector,music)/(norm(nlp(str(sent)).vector)*norm(music))
    stack.append((idx,cosine))

sorted_stack = [(idx,value) for idx,value in sorted(stack, key= lambda x: x[1], reverse=True)]
print(sorted_stack[0:5])
threshold = 0.25

filtered_reviews = [idx for idx,value in sorted_stack if value >= threshold]
print(len(filtered_reviews))

for i, idx in enumerate(filtered_reviews):
    sent = alexa_reviews['verified_reviews'][idx]
    print(sent)

    if i == 5:
        break
    
