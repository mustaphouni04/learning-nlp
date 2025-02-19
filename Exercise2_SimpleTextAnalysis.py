from nltk.text import Text
from nltk.tokenize import NLTKWordTokenizer

print("-----------------------------------------------------------\n")
print("SHERLOCK HOLMES")
with open("Adventures_Holmes.txt", "r") as f:
    a = f.read()
    tokens = NLTKWordTokenizer().tokenize(a) 
    tokens = Text(tokens)

print(tokens.concordance("Sherlock"))
print(tokens.concordance("extreme"))
print(tokens.similar("extreme", num=3))
print(tokens.common_contexts(("extreme", "gathering")))
print(tokens.collocations())

print("-----------------------------------------------------------\n")
print("ALICE")

with open("alice.txt", "r") as f:
    a = f.read()
    tokens = NLTKWordTokenizer().tokenize(a) 
    tokens = Text(tokens)

print(tokens.concordance("Alice"))
print(tokens.concordance("magic"))
print(tokens.similar("bottle", num=3))
print(tokens.common_contexts(("there", "bottle")))
print(tokens.collocations())

print("-----------------------------------------------------------\n")
print("MOBY DICK")

with open("moby_dick.txt", "r") as f:
    a = f.read()
    tokens = NLTKWordTokenizer().tokenize(a) 
    tokens = Text(tokens)

print(tokens.concordance("Moby"))
print(tokens.concordance("magic"))
print(tokens.similar("bottle", num=3))
print(tokens.common_contexts(("long", "bottle")))
print(tokens.collocations())
