import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,classification_report
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple
import numpy.typing as npt
import tiktoken
import random
from tqdm import tqdm

encoding = tiktoken.get_encoding("gpt2")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open("deu.txt", "r") as f:
    txt = f.readlines()

def preprocess_corpus(lines: List[str]) -> pd.DataFrame:
    data = dict()
    english_sentences = []
    german_sentences = []
    for line in lines:
        en,de = line.split("\t")[:-1]
        english_sentences.append(en) 
        german_sentences.append(de)
        
    data['english'] = english_sentences 
    data['german'] = german_sentences
    df = pd.DataFrame(data)

    return df

def tokenize(text:str) -> List[str]:
    tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
    return tokens

def decode_tokens(tokens:List[int]) -> List[str]:
    decoded = [encoding.decode_single_token_bytes(token) for token in tokens]
    decoded = [word.decode("utf-8", errors="ignore") for word in decoded] 
    return decoded

def build_sequences(df: pd.DataFrame, max_len:int = 75, targ:bool = True) -> torch.Tensor:
    eos = '<|endoftext|>'
    pad = {'<|pad|>': 50257}
    if targ:
        target = df["german"].values
        target_sequences = [tokenize(eos)+tokenize(sent)+tokenize(eos) for sent in target]

    else:
        target = df["english"].values
        target_sequences = [tokenize(sent) for sent in target]
    
    for idx,target in enumerate(target_sequences):
        length = len(target)
        if length < max_len:
            padding_number = max_len-length
            target_sequences[idx] = target_sequences[idx] + [pad['<|pad|>']]*padding_number
        else:
            target_sequences[idx] = target_sequences[idx][:max_len]
    target_sequences = torch.LongTensor(target_sequences)
    
    return target_sequences 

example = tokenize("Hello World!")
print(example)

text = decode_tokens([0])
print(text)

df = preprocess_corpus(txt)
print(df.head(10))
print(df["english"].values)


class GermanEnglish(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.sequences = build_sequences(df, targ=False)
        self.target = build_sequences(df)
        
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.sequences[idx],self.target[idx]        

train, test = train_test_split(df, test_size=0.2)
eng_ger_train = GermanEnglish(train)
eng_ger_test = GermanEnglish(test)

train_dataloader = DataLoader(eng_ger_train, batch_size = 64, shuffle=True)
test_dataloader = DataLoader(eng_ger_train, batch_size = 64)

for elem in train_dataloader:
    print(elem)
    print(elem[0].shape)
    break

class NeuralMachineTranslation(nn.Module):
    def __init__(self, vocab_size = 50258, 
                 hidden_dim = 256, 
                 embedding_dim = 256,
                 output_dim = 50258,
                 drop = 0.5,
                 padding_idx=50257,
                 p_tf = 0.5
                 ):
        super().__init__()
        self.padding_idx = padding_idx
        self.p_tf = p_tf
        self.embed = nn.Embedding(vocab_size, 
                                  embedding_dim, 
                                  padding_idx= padding_idx)
        self.dropout = nn.Dropout(drop)

        self.encoder = nn.LSTM(embedding_dim, 
                               hidden_dim, 
                               num_layers=2,
                               dropout=drop, 
                               batch_first=True)

        self.decoder = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=2,
                               dropout=drop,
                               batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, trg):
        batch_size, trg_len = trg.shape

        trg_vocab_size = self.fc.out_features
        # Encoder
        embedded_src = self.dropout(self.embed(src))

        _, (hidden, cell) = self.encoder(embedded_src)
        
        outputs = torch.zeros(batch_size,trg_len,trg_vocab_size).to(trg.device)
        input_token = trg[:,0] # (64,)
        
        # autoregressive decoding with teacher forcing
        for t in range(1, trg_len):
            embedded = self.dropout(self.embed(input_token.unsqueeze(1))) # (64, 1, 256)
            output, (hidden,cell) = self.decoder(embedded, (hidden,cell)) # (64, 1, 256)

            pred_token = self.fc(output.squeeze(1)) # (64, 50258)

            outputs[:,t,:] = pred_token # store predicted token

            use_teacher_forcing = random.random() < self.p_tf
            input_token = trg[:,t] if use_teacher_forcing else pred_token.argmax(1)
        
        return outputs 
        
nmt_model = NeuralMachineTranslation()
nmt_model = nmt_model.to(device)
print(nmt_model)

optim = torch.optim.Adam(nmt_model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=50257)

num_epochs = 10

def train(model, iterator, optimizer, criterion):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    epoch_loss = 0 
    
    for batch in tqdm(iterator):
        source, target = batch
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            predictions = model(source,target)
            loss = criterion(predictions.view(-1, model.fc.out_features),target.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

train = False

if train:
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch+1}/{num_epochs}")
        train_loss = train(nmt_model,train_dataloader,optim,criterion)

        print(f"\tTrain Loss: {train_loss:.3f}")
        torch.save(nmt_model.state_dict(), 'lstm-machine-translation.pt')
        print(f"Model saved!")


def translate_sentence(model, text):
    model.eval()
    
    tokens = tokenize(text)
    
    if len(tokens) < 75:
        length = len(tokens)
        to_pad = 75 - length

        tokens = tokens + [50257]*to_pad
    else:
        tokens = tokens[:75]

    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device) # (1, 75)

    with torch.no_grad():
        embedded_src = model.dropout(model.embed(tokens))
        _, (hidden, cell) = model.encoder(embedded_src)
        
        outputs = torch.zeros(1,75,50258).to(device)
        result = []
        input_token = torch.LongTensor([50256]).to(device)
        print(input_token.shape)

        for t in range(1, 75):
            embedded = model.dropout(model.embed(input_token.unsqueeze(1))) # (1, 1, 256)
            output, (hidden,cell) = model.decoder(embedded, (hidden,cell)) # (1, 1, 256)

            pred_token = model.fc(output.squeeze(1)) # (1, 50258)

            outputs[:,t,:] = pred_token # store predicted token

            input_token = pred_token.argmax(1)
            result.append(input_token.item())
    return result

nmt_model.load_state_dict(torch.load('lstm-machine-translation.pt'))
sentences = ["What is your favorite song?", "Do you know where the nearest restaurant is?", "Go pick up something for me."]

for sentence in sentences:
    print("Source: ", sentence)
    translation = translate_sentence(nmt_model, sentence)
    eos = translation.index(50256)

    translation = decode_tokens(translation[:eos])
    print("Translation: ", "".join(translation))








