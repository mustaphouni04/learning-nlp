import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from collections import Counter
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data():
    df = pd.read_csv('IMDB_Dataset.csv')
    
    df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    return df

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    text = text.lower() 
    return text

def create_vocabulary(texts, max_words=10000):
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(max_words - 2)]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word_to_idx

def texts_to_sequences(texts, word_to_idx, max_len=200):
    sequences = []
    for text in texts:
        words = text.split()[:max_len]  
        sequence = [word_to_idx.get(word, 1) for word in words]
        if len(sequence) < max_len:
            sequence = sequence + [0] * (max_len - len(sequence))  
        sequences.append(sequence)
    return np.array(sequences)

class IMDBDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        
        output, (hidden, cell) = self.lstm(embedded)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(sequences).squeeze(1)
        loss = criterion(predictions, labels)
        
        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in iterator:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            
            predictions = model(sequences).squeeze(1)
            loss = criterion(predictions, labels)
            
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_sentiment(model, text, word_to_idx, max_len=200):
    model.eval()
    
    processed_text = preprocess_text(text)
    
    words = processed_text.split()[:max_len]
    sequence = [word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
    
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))  # 0 is <PAD>
    
    tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor).squeeze(1))
    
    sentiment = "positive" if prediction.item() > 0.5 else "negative"
    confidence = prediction.item() if prediction.item() > 0.5 else 1 - prediction.item()
    
    return sentiment, confidence

def main():
    print("Loading and preprocessing data...")
    df = load_data()
    df['review'] = df['review'].apply(preprocess_text)
    
    print("Creating vocabulary...")
    vocab, word_to_idx = create_vocabulary(df['review'].values)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    print("Converting texts to sequences...")
    sequences = texts_to_sequences(df['review'].values, word_to_idx)
    labels = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    
    print("Creating DataLoaders...")
    train_dataset = IMDBDataset(X_train, y_train)
    test_dataset = IMDBDataset(X_test, y_test)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("Initializing model...")
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    dropout = 0.5
    
    model = LSTMSentimentAnalysis(
        vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    n_epochs = 10
    best_valid_loss = float('inf')
    
    print("Training model...")
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch+1}/{n_epochs}")
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'lstm-sentiment-model.pt')
            print(f"\tModel saved!")
    
    model.load_state_dict(torch.load('lstm-sentiment-model.pt'))
    
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
    
    print("\nTesting with some sample reviews:")
    sample_reviews = [
        "This movie was fantastic! I really enjoyed the plot and acting.",
        "Terrible movie, waste of time and money. The acting was horrible.",
        "It was okay, nothing special but not bad either."
    ]
    
    for review in sample_reviews:
        sentiment, confidence = predict_sentiment(model, review, word_to_idx)
        print(f"Review: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})\n")
    
    epochs = range(1, n_epochs + 1)
    
if __name__ == "__main__":
    main()
