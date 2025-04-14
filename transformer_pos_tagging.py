import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Tuple, Dict


writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads to avoid fractioning"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Input is (B, T, d_model)
        # After splitting into heads is (B, T, num_heads, dim_k)
        # After rearranging properly we have (B, num_heads, T, dim_k)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dim_k) # (B, num_heads, T, dim_k) * (B, num_heads, dim_k, T) == (B, num_heads, T, T)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -float('inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1) # (B, num_heads, T, T)
        output = torch.matmul(attn_probs, V) # (B, num_heads, T, dim_k)

        return output
    
    def split_heads(self, x):
        B, T, d_model = x.size()
        return x.view(B, T, self.num_heads, self.dim_k).transpose(1,2)

    def combine_heads(self, x):
        B, _, T, dim_k = x.size()
        return x.transpose(1,2).contiguous().view(B, T, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # (max_seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        # add batch dim, ensure pe is part of model, it's moved to device and it's saved with the model
        self.register_buffer('pe', pe.unsqueeze(0)) # buffer since it's not trainable 
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Pre-norm
        attn_output = self.self_attn(self.norm1(x),self.norm1(x),self.norm1(x),mask)
        x = x + self.dropout(attn_output)
        # Pre-norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, num_pos_tags,
                 d_model, num_heads,
                 num_layers, d_ff,
                 max_seq_length, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = 0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_pos_tags)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, enc_input, pad_token_id):
        # enc_input = (B,T) 
        # output mask = (B,1,1,T)
        mask = (enc_input != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, x, pad_token_id = 0):
        assert torch.all(x >= 0) and torch.all(x < self.vocab_size), "Invalid input token IDs!"
        pad_mask = self.generate_mask(x,pad_token_id)
        embedded = self.dropout(self.positional_encoding(self.embedding(x)))

        enc_output = embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, pad_mask)
        
        output = self.fc(enc_output)

        return output


dataset = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)
# xpos (actual tag), tokens,


def index_analysis(split:str ="train") -> Tuple[Tuple[int,int], Dict[str,int], Dict[str,int]]: 
    # Get available 
    xpos = dataset[split]["xpos"]
    sentences = dataset[split]["tokens"]

    unique_words = { word.lower() for sentence in sentences for word in sentence if word is not None }
    specials = ["<PAD>", "<UNK>"]
    unique_words -= set(specials)
    sorted_rest = sorted(unique_words)
    vocabulary = specials + sorted_rest

    word2idx = {word:idx for idx, word in enumerate(vocabulary)}
    print(word2idx["<UNK>"])
    print(word2idx["<PAD>"])
    pos_tags = set(tag for pos in xpos for tag in pos if tag is not None)
    pos_tags = sorted(list(pos_tags))
    pos2idx = {tag:idx for idx, tag in enumerate(pos_tags)}
    print(pos2idx["XX"])

    print("Vocabulary size:", len(word2idx))
    print("Unique POS tags:", len(pos2idx))

    return ((len(word2idx), len(pos2idx)), (word2idx, pos2idx))

(vocab_size, pos_tags), (vocab, pos2idx) = index_analysis(split="train")


class PosTagDataset(Dataset):
    def __init__(self, split:str = "train", max_length: int = 50):
        self.split = split
        (vocab_size, pos_tags), (vocab, pos2idx) = index_analysis(split="train")
        
        sequences = dataset[split]["tokens"]
        targets = dataset[split]["xpos"]

        padded_sequences = self._process_sequences(sequences, vocab, max_length)
        padded_targets = self._process_targets(targets,pos2idx,max_length)

        self.sequences = torch.LongTensor(padded_sequences)
        self.targets = torch.LongTensor(padded_targets)
        self.vocab_size = vocab_size
        self.pos_tags = pos_tags
        self.vocab = vocab
        self.pos2idx = pos2idx

    def _process_sequences(self, sequences, vocab, max_length):
        padded_sequences = []
        for seq in sequences:
            idx_seq = [vocab.get(word.lower(),vocab["<UNK>"]) for word in seq]
            padded = (idx_seq + [vocab["<PAD>"]] * max_length)[:max_length]
            padded_sequences.append(padded)
        return padded_sequences

    def _process_targets(self, targets, pos2idx, max_length, pad_value = -100):
        padded_targets = []
        for tgt in targets:
            idx_tgt = [pos2idx.get(tag,pos2idx["XX"]) for tag in tgt]
            padded = (idx_tgt + [pad_value] * max_length)[:max_length]
            padded_targets.append(padded)
        return padded_targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

train_dataset = PosTagDataset()
test_dataset = PosTagDataset(split="test")
valid_dataset = PosTagDataset(split="validation")

print(len(train_dataset.pos2idx.keys()))
print(len(train_dataset))
print("vocab size is", train_dataset.vocab_size)
print(train_dataset[0])
print(len(test_dataset))
print(test_dataset[0])

print(dataset["train"]["xpos"][0:5])
print(dataset["train"]["tokens"][0:5])

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

encoder = EncoderTransformer(train_dataset.vocab_size, 
                             train_dataset.pos_tags,
                             d_model = 256,
                             num_heads = 4,
                             num_layers = 3,
                             d_ff = 512,
                             max_seq_length=50,
                             dropout=0.5).to(device)

print(encoder.parameters)
print(dataset)

loss = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.AdamW(encoder.parameters())

def train(model, iterator, valid_iterator, criterion, optimizer):
    model.train()

    padding_val = -100
    for idx, batch in enumerate(tqdm(iterator)):
        sequence, targets = batch
        sequence, targets = sequence.to(device), targets.to(device)
        assert sequence.max().item() < model.vocab_size, "Token ID exceeds vocabulary size!"
        optimizer.zero_grad()
        predictions = model(sequence)  # (B, T, pos_tags)
        
        num_pos_tags = train_dataset.pos_tags  
        flat_preds = predictions.view(-1, num_pos_tags)
        flat_targets = targets.view(-1)
        
        loss = criterion(flat_preds, flat_targets)
        
        mask = flat_targets != padding_val
        
        predicted_classes = flat_preds.argmax(dim=-1)
        correct = (predicted_classes[mask] == flat_targets[mask]).float()
        acc = correct.sum() / len(correct)

        writer.add_scalar("Loss/train", loss.item(), idx)
        writer.add_scalar("Acc/train", acc.item(), idx)
        loss.backward() 
        optimizer.step()
    
    epoch_loss = 0 
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_iterator)):
            sequence, targets = batch
            sequence, targets = sequence.to(device), targets.to(device)
            predictions = model(sequence)  # (B, T, pos_tags)
            
            flat_preds = predictions.view(-1, num_pos_tags)
            flat_targets = targets.view(-1)
            loss = criterion(flat_preds, flat_targets)

            mask = flat_targets != padding_val

            predicted_classes = flat_preds.argmax(dim=-1)
            correct = (predicted_classes[mask] == flat_targets[mask]).float()
            acc = correct.sum() / len(correct)
            
            epoch_acc += acc.item()
            epoch_loss += loss.item()

    writer.add_scalar("Loss/valid", epoch_loss / len(valid_iterator))
    writer.add_scalar("Acc/valid", epoch_acc / len(valid_iterator))

num_epochs = 5

for epoch in tqdm(range(num_epochs)):
    train(encoder, train_loader, valid_loader, loss, optimizer) 
    torch.save(encoder.state_dict(), f"model_checkpoint/epoch_{epoch}.pt")

writer.flush()
writer.close()

                
        

    


