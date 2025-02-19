import nltk
from nltk import NLTKWordTokenizer, Text
from nltk.util import ngrams
import re
from nltk.probability import FreqDist

sent = """
To Sherlock Holmes she is always 'The Woman'. 
I have seldom heard him mention her under any other name
"""

tokens = nltk.word_tokenize(sent)

print(list(ngrams(tokens, 2, pad_right=True)))
print(list(ngrams(tokens, 3, pad_right=True)))

two_grams = ngrams(tokens, 2)
clean = re.sub(r'[^a-zA-Z0-9 ]', '', sent)
print(clean)
two_grams = ngrams(nltk.word_tokenize(clean), 2)

fdist = FreqDist()

for gram in two_grams:
    fdist[(gram[0].lower(), gram[1].lower())] += 1

print(fdist.items())

with open("Adventures_Holmes.txt", "r") as f:
    txt = f.read()


