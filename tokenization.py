import nltk
from nltk.tokenize import NLTKWordTokenizer

utt = "Hello world. How are you?"
print(NLTKWordTokenizer().tokenize(utt))
print("The number of tokens is ", len(NLTKWordTokenizer().tokenize(utt)))

count = 0
vocabulary = set()
ls = []
with open("alice.txt", "r") as f:
    for line in f.readlines():
        a = NLTKWordTokenizer().tokenize(line)
        ls.append(a)
        vocab = {s for s in a}
        print(vocab)
        count += len(a)

for sent in ls:
    for word in sent:
        vocabulary.add(word)

print(len(vocabulary))

lexical_diversity = count/len(vocabulary)
print(lexical_diversity)
