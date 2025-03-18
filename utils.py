import re

def clean_sentences(sentences):
    to_remove = "<br />."
    new_set = set()
    for sentence in sentences:
        sentence = re.sub(to_remove, "", sentence)
        new_set.add(sentence)

    return new_set 
