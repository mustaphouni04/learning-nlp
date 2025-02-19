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
        count += len(a)

for sent in ls:
    for word in sent:
        vocabulary.add(word)

print("Number of tokens in Alice is ", count)
print("Vocab length is ",len(vocabulary))

lexical_diversity = count/len(vocabulary)
print("Lexical diversity is ", lexical_diversity)

utt2 = "Jane lent $100 to Peter early this morning."
auto = nltk.word_tokenize(utt2)
manual = utt2.split(" ")
print("tokenize function: ", auto)
print("manual splits: ", manual)



