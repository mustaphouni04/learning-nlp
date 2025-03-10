import nltk
from nltk.corpus import treebank
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

nltk.download('treebank')
nltk.download('universal_tagset')

print(treebank.parsed_sents('wsj_0003.mrg')[0])

def ufeatures(utt, idx):
    ftdist = {}
    ftdist['word'] = utt[idx]
    ftdist['dist_from_first'] = idx - 0
    ftdist['dist_from_last'] = len(utt) - idx
    ftdist['capitalized'] = utt[idx][0].upper() == utt[idx][0]
    ftdist['prefix1'] = utt[idx][0]
    ftdist['prefix2'] = utt[idx][:2]
    ftdist['prefix3'] = utt[idx][:3]
    ftdist['suffix1'] = utt[idx][-1]
    ftdist['suffix2'] = utt[idx][-2:]
    ftdist['suffix3'] = utt[idx][-3:]
    ftdist['prev_word'] = '' if idx==0 else utt[idx-1]
    ftdist['next_word'] = '' if idx==(len(utt)-1) else utt[idx+1]
    ftdist['numeric'] = utt[idx].isdigit()
    return ftdist

tagged_sentences = treebank.tagged_sents(tagset='universal')

X = []
y = [] 

for sentence in tqdm(tagged_sentences):
    words = [word for word, tag in sentence]
    tags = [tag for word, tag in sentence]
    
    for i in range(len(words)):
        X.append(ufeatures(words, i))
        y.append(tags[i])

vectorizer = DictVectorizer(sparse=True)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"accuracy: {accuracy:.4f}")
print("\nclassification report:")
print(report)

feature_names = vectorizer.get_feature_names_out()
feature_importances = clf.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

print("\nTop 20 features:")
for idx in sorted_idx[:20]:
    print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")

