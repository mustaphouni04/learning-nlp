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

MEAN_CONTEXT_LENGTH = 150  # Reduced from 237 to focus on more relevant text
NUM_EPOCHS = 15  # Increased epochs to allow more training time
BATCH_SIZE = 64  # Increased batch size for better gradient estimates
EMBEDDING_DIM = 256  # Reduced from 512 to prevent overfitting
HIDDEN_DIM = 128
MAX_VOCAB_SIZE = 25000  # Limit vocabulary size to most common words

# Load the Dataset
df = pd.read_csv("IMDB_Dataset.csv")
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
print(df.head())
print(df.columns)

WORD = re.compile(r'\w+')

def regTokenize(text):
    # Be more selective about which stopwords to remove
    stop_words = set(stopwords.words('english'))
    words = WORD.findall(text)
    
    # Keep sentiment-relevant words even if they're stopwords
    sentiment_words = {'not', 'no', 'nor', 'neither', 'never', 'very', 'too', 'only', 'but', 'and', 'or'}
    words = [w.lower() for w in words if w.lower() not in stop_words or w.lower() in sentiment_words]
    
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
        sorted_words = sorted(word_counts.items(), key=lambda x:x[1], reverse=True)
        
        # Limit vocabulary size
        sorted_words = sorted_words[:MAX_VOCAB_SIZE-len(special_tokens)]

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
        
        starter = np.zeros((len(reviews), MEAN_CONTEXT_LENGTH))

        for idx, review in tqdm(enumerate(reviews)):
            # For shorter reviews, keep the full text
            # For longer reviews, keep both beginning and end parts
            if len(review) >= MEAN_CONTEXT_LENGTH - 2:  # -2 for <SOR> and <EOR>
                # Keep first half and last half of the review
                half_length = (MEAN_CONTEXT_LENGTH - 2) // 2
                review = review[:half_length] + review[-half_length:]
            
            review = ["<SOR>"] + review + ["<EOR>"]
            
            if len(review) >= MEAN_CONTEXT_LENGTH:
                review = review[:MEAN_CONTEXT_LENGTH-1] + ["<EOR>"]
            else:
                k = MEAN_CONTEXT_LENGTH - len(review)
                review = review + ["<PAD>"]*k

            indexed_review = [self.word2idx.get(word, self.unk_idx) for word in review]
            starter[idx] = np.array(indexed_review)
        
        dataset = torch.from_numpy(starter).long()
        targets = [1 if t=="positive" else 0 for t in targets]
        targets = torch.FloatTensor(targets)
        targets = targets.unsqueeze(1)
        
        return dataset, targets

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], self.targets[idx]

train_review_class = ReviewDataset(train_df)
test_review_class = ReviewDataset(test_df, ref_dataset=train_review_class)
print("train_vocab_size:", len(train_review_class.word2idx))
print("test_vocab_size:", len(test_review_class.word2idx))

train_loader = DataLoader(train_review_class, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_review_class, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

class LSTMSentimentAnalysis(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx, num_layers=2, dropout=0.3):
        super(LSTMSentimentAnalysis, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Add embedding dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Add attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Output layers with dropout
        self.dropout = nn.Dropout(0.4)
        self.decode = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        
        # Get LSTM outputs
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply layer normalization and dropout
        context_vector = self.layer_norm(context_vector)
        context_vector = self.dropout(context_vector)
        
        # Decode
        output = self.decode(context_vector)
        
        return output

vocab_size = len(train_review_class.word2idx)
padding_idx = train_review_class.word2idx.get("<PAD>")
print("The padding index is:", padding_idx)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentimentAnalysis(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, padding_idx=padding_idx).to(device)

# Loss and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training and evaluation
best_accuracy = 0.0
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
        for batch_idx, (input_data, target) in enumerate(pbar):
            input_data = input_data.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_data)
            
            # Calculate loss
            batch_loss = criterion(outputs, target)
            total_loss += batch_loss.item()
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    # Validate the model
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    
    with torch.no_grad():
        for input_data, target in test_loader:
            input_data = input_data.to(device)
            target = target.to(device)
            
            outputs = model(input_data)
            batch_loss = criterion(outputs, target)
            val_loss += batch_loss.item()
            
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct += (predictions == target).sum().item()
            total += target.size(0)
    
    accuracy = correct / total
    avg_val_loss = val_loss / len(test_loader)
    
    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_sentiment_model.pth")
        print(f"Best model saved with accuracy: {best_accuracy:.4f}")

print(f"Training complete. Best accuracy: {best_accuracy:.4f}")
