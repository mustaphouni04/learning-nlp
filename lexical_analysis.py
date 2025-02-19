import matplotlib.pyplot as plt
import nltk
from nltk import Text, NLTKWordTokenizer

with open("moby_dick.txt", "r") as f:
    txt = f.read()
print(len(txt))
tokens = NLTKWordTokenizer().tokenize(txt)

axes = nltk.draw.dispersion_plot(tokens, ["good", 
                                      "happy",
                                      "strong",
                                      "bad",
                                      "sad",
                                      "weak"])
plt.show()



