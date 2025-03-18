import pandas as pd
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import nltk
from tqdm import tqdm
from utils import clean_sentences
from torch import nn

MEAN_CONTEXT_LENGTH = 40
NUM_EPOCHS = 10 

# Load the Dataset
df = pd.read_csv("IMDB_Dataset.csv")
print(df.head())
print(df.columns) # review, sentiment

WORD = re.compile(r'\w+')

def regTokenize(text):
    words = WORD.findall(text)
    return words

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.idx2word = dict() 
        self.word2idx = dict()
        self.dataset = None
        self.dataset, self.targets = self.build_dataset()

    def build_vocab(self):
        special_tokens = ("<PAD>", "<SOR>", "<EOR>", "<UNK>")
        reviews = set(self.df['review'])
        reviews = clean_sentences(reviews)
        tokens = {word for review in tqdm(reviews) for word in self.tokenize(review)}
        for token in special_tokens: tokens.add(token)
        self.idx2word = {k:v for k,v in enumerate(tokens)}
        self.word2idx = {v:k for k,v in self.idx2word.items()}
        
    def tokenize(self, review):
        tokens = regTokenize(review)
        return tokens

    def build_dataset(self):

        self.build_vocab()
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

            indexed_review = [self.word2idx[word] for word in review]
            starter[idx] = np.array(indexed_review)
        
        dataset = torch.from_numpy(starter).long()
        targets = list(map(lambda x: x.replace('negative', '0'), targets))
        targets = list(map(lambda x: x.replace('positive', '1'), targets))
        targets = [int(target) for target in targets]
        targets = torch.FloatTensor(targets)
        targets = targets.unsqueeze(1) # shape (batch_size,1)
        
        return dataset, targets

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]

review_class = ReviewDataset(df)

train_size = int(0.8*len(review_class))
test_size = len(review_class) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(review_class, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, num_workers = 4)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle=False, num_workers = 4)

class LSTMSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx, num_layers=2):
        super(LSTMSentimentAnalysis, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decode = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)

        out, _ = self.lstm(x)
        decoded = self.decode(out[:,-1,:]) # (32, 1)
        
        return decoded 

vocab_size = len(review_class.word2idx)
padding_idx = review_class.word2idx.get("<PAD>")
embedding_dim = 128
lstm = LSTMSentimentAnalysis(vocab_size,embedding_dim, hidden_dim=128, padding_idx=padding_idx).to("cuda")
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=lstm.parameters())

for epoch in tqdm(range(NUM_EPOCHS)):
    for _,(input, target) in enumerate(train_loader):
        input = input.type(torch.long).to("cuda") # (32,237)
        target = target.to("cuda")
        optimizer.zero_grad()

        outputs = lstm(input) # (32,1)

        output_loss = loss(outputs,target)
    
        output_loss.backward()

        optimizer.step()
 
    count = 0 
    total = test_size
    
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


        

        


