from nltk.book import *
from nltk.text import Text
from nltk.tokenize import NLTKWordTokenizer

with open("Adventures_Holmes.txt", "r") as f:
    a = f.read()
    tokens = NLTKWordTokenizer().tokenize(a) 
    tokens = Text(tokens)

print(tokens.concordance("Sherlock"))
print(tokens.concordance("extreme"))
print(tokens.similar("extreme", num=3))
print(tokens.common_contexts(("extreme", "gathering")))
print(tokens.collocations())
