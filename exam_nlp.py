# EXAM
# Import necessary libraries
import pandas as pd 
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Union, Tuple, List, Optional, Dict
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import spacy
from tqdm import tqdm
import numpy as np

# Load dataset
df = pd.read_excel("ExercisesTest_filtered.xlsx")

# Scan the dataset, analyze it
print(df.head())
print(df.columns)
print(df["summary"].values[0:10])
print(df["category"].values[0:10])

# Number of categories is 2
print(df["category"].unique())

# Compute the lexical diversity of the dataset
def return_lexical_diversity(df: pd.DataFrame) -> Tuple[float, List[str]]:
    summaries = df["summary"].values
    sentences = [word_tokenize(str(text)) for text in summaries]
    all_words = [word for sent in sentences for word in sent]
    unique = set(all_words)

    return len(all_words) / len(unique), all_words 


lexical_diversity, all_words = return_lexical_diversity(df)
print(f"The lexical diversity of the dataset is: {lexical_diversity}")

"""The dataset has a lexical diversity of 25.39 which suggests the vocabulary is not that rich"""

# Use the stopwords module to filter bad words
stopwords = stopwords.words('english')
#print(stopwords)

# Function used to plot frequency distribution and word cloud of clean words
def word_analysis(all_words, stopwords: List[str]):
    # Preprocess words prior to calculating frequency distribution and word cloud
    specials = [".", ",", "...", "!", "-", "n't", "n's", "'s"]
    # First filtering stage
    all_words = [word.lower() for word in all_words if word.lower() not in stopwords]
    # Second filtering stage, normalize the words
    stemmer = SnowballStemmer("english")
    all_words = [stemmer.stem(token) for token in all_words]

    # Calculate word frequencies
    fdist = FreqDist()
    
    for word in all_words:
        fdist[word.lower()] += 1
    
    for sp in specials:
        del fdist[sp]

    # Sort them from most frequent to less frequent
    fdist = {k:v for k,v in sorted(fdist.items(), key= lambda x:x[1], reverse=True)}
   
    # Get top 30
    top_30 = list(fdist.items())[:29]
    
    # Plot bar chart
    _, ax = plt.subplots()
    words = [tup[0] for tup in top_30]
    counts = [tup[1] for tup in top_30]
    ax.bar(words,counts)
    ax.set_ylabel('frequency')
    ax.set_title('top 30 most frequent words')
    plt.xticks(rotation=45, ha="right")
    plt.show()

    # Plot word cloud
    wordcloud = WordCloud(stopwords=stopwords, background_color='white', max_words=300).generate_from_frequencies(fdist)

    plt.figure(figsize=(15,10))
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# Perform word analysis
word_analysis(all_words,stopwords)

# Task -> Text categorization
"""
The goal is to see if a given text comes from the 'Home' category of reviews or 'Automotive' category.
"""

# Perform train test split

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

"""
Preprocess the text, we use SnowballStemmer and stopwords to clean all the junk from the dataset.
"""
def preprocess_text(df: pd.DataFrame) -> List[str]:
    # Get summaries and categories
    summaries = df["summary"].values
    categories = df["category"].values
    # Turn categories into numbers
    categories = [0 if cat == 'Home' else 1 for cat in categories]
    # Tokenize
    sentences = [word_tokenize(str(text)) for text in summaries]
    filtered_sentences = []
    stemmer = SnowballStemmer("english")
    
    # Filtering all sentences
    specials = [".", ",", "...", "!", "-", "n't", "n's", "'s"]
    for sentence in sentences:
        tokens = [tok for tok in sentence if tok not in specials and tok not in stopwords] 
        tokens = [stemmer.stem(token) for token in tokens]
        sentence = " ".join(tokens)
        filtered_sentences.append(sentence)

    return filtered_sentences, categories
        
# Preprocess the sentences
train_sentences, train_categories = preprocess_text(train_df)
test_sentences, test_categories = preprocess_text(test_df)

# Use nlp module to calculate word embeddings
nlp = spacy.load("en_core_web_md")

# Get vectorial representations to fit then for model
X_train = [nlp(str(sentence)).vector for sentence in tqdm(train_sentences)]
    
X_train_arr = np.zeros((len(train_sentences),300)) # Embedding dim is 300

for idx,i in enumerate(X_train):
    X_train_arr[idx] = X_train[idx]

X_test = [nlp(str(sentence)).vector for sentence in tqdm(test_sentences)]

X_test_arr = np.zeros((len(test_sentences),300)) # Embedding dim is 300

for idx,i in enumerate(X_test):
    X_test_arr[idx] = X_test[idx]

print(X_test_arr.shape)
# Get vectors for the categories as well
y_train = np.array(train_categories)
y_test = np.array(test_categories)

clf = LogisticRegression(random_state=0).fit(X_train_arr, y_train)

# Used to evaluate the accuracy of classification
def evaluate(clf, test_array, target):
    score = 0
    total = test_array.shape[0]
    for idx, vector in enumerate(test_array):
        pred = clf.predict(vector.reshape(1,-1)) # Predict the class
        if str(pred[0]) == str(target[idx]):
            score += 1
    accuracy = score / total

    return accuracy

accuracy = evaluate(clf, X_test_arr, y_test)

print("The accuracy for Word2Vec is: ", accuracy)


"""
Summary of the task: The NLP based task performed is a text classification task. 
We compute word embeddings using Word2Vec to get dense representations of summaries that capture meaning.
In this case, the accuracy is not that high ~0.65, what could we do to improve it?
First off:
- We use SnowballStemmer which is kind of aggressive and can suppress meaning from the words.
- Logisitic Regression doesn't take order into account.

Steps to improve:
- Use a sequence-based model like RNNs, GRU or Transformer.
- Preprocess the text in a different way.
- Use a tokenizer, like the one from GPT-2.

Overall, the task is to predict the kind of review (summary) from the texts.
"""







