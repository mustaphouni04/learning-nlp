import nltk
from nltk import NLTKWordTokenizer, Text
from nltk.util import ngrams
import re
from nltk.probability import FreqDist
import random
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import everygrams


sent = """
To Sherlock Holmes she is always 'The Woman'. 
I have seldom heard him mention her under any other name
"""

tokens = nltk.word_tokenize(sent)

print("2-gram")
print(list(ngrams(tokens, 2, pad_right=True)))
print("\n")
print("3-gram")
print(list(ngrams(tokens, 3, pad_right=True)))

two_grams = ngrams(tokens, 2)
clean = re.sub(r'[^a-zA-Z0-9 ]', '', sent)
print(clean)
two_grams = ngrams(nltk.word_tokenize(clean), 2)

fdist = FreqDist()

for gram in two_grams:
    fdist[(gram[0], gram[1].lower())] += 1

print("frequencies")
print({k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)})

with open("moby_dick.txt", "r") as f:
    txt = f.read()
txt = re.sub(r'[^a-zA-Z0-9 ]', '', txt)
tokens = nltk.word_tokenize(txt)
grams = [2,3,4]
print("\n")
for g in grams:
    gramos = ngrams(tokens, g)

    fdist = FreqDist()

    for gram in gramos:
        fdist[gram] += 1
 
    print(f"for {g}-gram: ", list({k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}.items())[:9])

    print("\n\n")

paragraphs = []
paragraph = []
with open("Adventures_Holmes.txt", "r") as f:
    for line in f.readlines():
        if line == "\n":
            paragraphs.append(paragraph)
            paragraph = []
        else:
            paragraph.append([line]) 

test_paragraph = paragraphs[random.randint(0,len(paragraphs))]
test_paragraph2 = [nltk.word_tokenize(line[0]) for line in test_paragraph]
print("GROUND TRUTH\n")
test_paragraph2 = [word for line in test_paragraph2 for word in line]
print(" ".join(test_paragraph2))
train_paragraphs = [nltk.word_tokenize(line[0]) for paragraph in paragraphs for line in paragraph if paragraph != []]
lm = MLE(2)

train, vocab = padded_everygram_pipeline(2, train_paragraphs)
lm.fit(train, vocab)


print("SEED")
print(" ".join(test_paragraph2[:5]))
print("GENERATION")
count = 0
r = 0
temp_ls = []
for idx,word in enumerate(test_paragraph2):
    count += 1
    temp_ls.append(word)
    word_gen = lm.generate(1,text_seed=temp_ls)
    if idx == len(test_paragraph2) - 1:
        break
    else:
        gt = test_paragraph2[idx+1]
        if word_gen == gt:
            r += 1
print("WAR")
print(r/len(test_paragraph2))
print(" ".join(lm.generate(len(test_paragraph2), text_seed=test_paragraph2[:5])))



