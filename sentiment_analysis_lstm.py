import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nltk
from tqdm import tqdm
from utils import clean_sentences
from torch import nn
from collections import Counter
from nltk.corpus import stopwords
nltk.download('stopwords')

MEAN_CONTEXT_LENGTH = 237
NUM_EPOCHS = 10 

# Load the Dataset
df = pd.read_csv("IMDB_Dataset.csv")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
print(df.head())
print(df.columns) # review, sentiment

WORD = re.compile(r'\w+')

def regTokenize(text):
    stop_words = set(stopwords.words('english'))
    words = WORD.findall(text)
    negation_words = {'not', 'no', 'nor', 'neither', 'never'}
    words = [w.lower() for w in words if not w.lower() in stop_words or w.lower() in negation_words]

    return words

class ReviewDataset(Dataset):
    def __init__(self, df, ref_dataset=None):
        self.df = df
        if ref_dataset is not None:
            self.idx2word = ref_dataset.idx2word
            self.word2idx = ref_dataset.word2idx
            self.unk_idx = ref_dataset.unk_idx
        else:
            self.idx2word = dict() 
            self.word2idx = dict()
            self.unk_idx = None
            self.build_vocab()
        self.dataset, self.targets = self.build_dataset()
    def build_vocab(self):
        special_tokens = ("<PAD>", "<SOR>", "<EOR>", "<UNK>")
        reviews = self.df['review'].tolist()
        reviews = clean_sentences(reviews)
        tokens = []
        for review in tqdm(reviews):
            tokens.extend(self.tokenize(review))
        word_counts = Counter(tokens)
        sorted_words = sorted(word_counts.items(), key= lambda x:x[1], reverse=True)

        self.idx2word = {i: token for i, token in enumerate(special_tokens)}
        self.word2idx = {token: i for i, token in enumerate(special_tokens)}

        current_idx = len(special_tokens)
        for word, count in sorted_words:
            if word not in special_tokens:
                self.idx2word[current_idx] = word
                self.word2idx[word] = current_idx
                current_idx += 1
        self.unk_idx = self.word2idx["<UNK>"]

    def tokenize(self, review):
        tokens = regTokenize(review)
        return tokens

    def build_dataset(self):

        reviews = list(self.df['review'])
        targets = list(self.df['sentiment'])
        reviews = clean_sentences(reviews)
        reviews = [self.tokenize(review) for review in tqdm(reviews)]
        
        starter = np.zeros((len(reviews),MEAN_CONTEXT_LENGTH)) # account for the <SOR> and <EOR>

        for idx,review in tqdm(enumerate(reviews)):
            review = ["<SOR>"] + review + ["<EOR>"]
            if len(review) >= MEAN_CONTEXT_LENGTH:
                review = review[:MEAN_CONTEXT_LENGTH-1] + ["<EOR>"] # we're deleing eor here
            else:
                k = MEAN_CONTEXT_LENGTH - len(review)
                review = review + ["<PAD>"]*k

            indexed_review = [self.word2idx.get(word, self.unk_idx) for word in review]
            starter[idx] = np.array(indexed_review)
        
        dataset = torch.from_numpy(starter).long()
        targets = [1 if t=="positive" else 0 for t in targets]
        targets = torch.FloatTensor(targets)
        targets = targets.unsqueeze(1) # shape (batch_size,1)
        
        return dataset, targets

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]

train_review_class = ReviewDataset(train_df)
test_review_class = ReviewDataset(test_df, ref_dataset=train_review_class)
print("train_vocab_size:", len(train_review_class.word2idx))
print("test_vocab_size:", len(test_review_class.word2idx))

train_loader = DataLoader(train_review_class, batch_size = 32, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_review_class, batch_size = 32, shuffle=False, num_workers = 4)

class LSTMSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx, num_layers=2, dropout=0.2):
        super(LSTMSentimentAnalysis, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.decode = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        x = self.embedding(x)

        out, (h_n,c_n) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]),dim=1)
        return self.decode(h_n)

vocab_size = len(train_review_class.word2idx)
padding_idx = train_review_class.word2idx.get("<PAD>")
print("The padding index is:", padding_idx)
embedding_dim = 512
lstm = LSTMSentimentAnalysis(vocab_size,embedding_dim, hidden_dim=128, padding_idx=padding_idx).to("cuda")
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=lstm.parameters())

for epoch in tqdm(range(NUM_EPOCHS)):
    for _,(input, target) in enumerate(train_loader):
        total_loss = 0
        input = input.type(torch.long).to("cuda") # (32,237)
        target = target.to("cuda")
        optimizer.zero_grad()

        outputs = lstm(input) # (32,1)

        output_loss = loss(outputs,target)
        total_loss += output_loss.item()
    
        output_loss.backward()
        for name, param in lstm.named_parameters():
            if param.grad is not None:
                pass
                #print(name, "gradient mean: ", param.grad.mean())
        

        optimizer.step()
 
    count = 0 
    total = len(test_df)
    print("loss is: ", total_loss/len(train_loader)) 
    with torch.no_grad():
        for idx, (input, targ) in enumerate(test_loader):
            input = input.to("cuda")
            targ = targ.to("cuda")
            out = lstm(input.type(torch.long))
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            count += (preds == targ).sum().item()
             
        
        epoch_accuracy = count / total
        print("Accuracy for epoch is: ", epoch_accuracy)


        

        


