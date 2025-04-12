import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
from torch.utils.data import DataLoader,Dataset
import copy
import nltk
import numpy as np 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Tuple, Dict
from collections import Counter
import itertools

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
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1) # (B, num_heads, T, T)
        output = torch.matmul(attn_probs, V) # (B, num_heads, T, dim_k)

        return output
    
    def split_heads(self, x):
        B, T, d_model = x.size()
        return x.view(B, T, self.num_heads, self.dim_k).transpose(1,2)

    def combine_heads(self, x):
        B, _, T, dim_k = x.size()
        return x.transpose(1,2).contiguous().view(B, T, self.d_model)

    def forward(self, Q, K, V):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V)
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
        attn_output = self.self_attn(x,x,x,mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size, num_pos_tags,
                 d_model, num_heads,
                 num_layers, d_ff,
                 max_seq_length, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_pos_tags)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, enc_input, pad_token_id):
        # enc_input = (B,T) 
        # output mask = (B,1,1,T)
        mask = (enc_input != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask
    
    def forward(self, x, pad_token_id):
        pad_mask = self.generate_mask(x,pad_token_id)
        embedded = self.dropout(self.positional_encoding(self.embedding(x)))

        enc_output = embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, pad_mask)
        
        output = self.fc(enc_output)

        return output


dataset = load_dataset("universal_dependencies", "en_ewt", trust_remote_code=True)
# upos (indices of tags), xpos (actual tag), tokens,


def index_analysis(split:str ="train") -> Tuple[Tuple[int,int], Dict[str,int]]: 
    # Get available 
    upos = dataset[split]["upos"]
    xpos = dataset[split]["xpos"]
    sentences = dataset[split]["tokens"]

    vocab = set()
    pos_tags = set()

    for sentence in sentences:
        for word in sentence:
            vocab.add(word.lower())
    vocab.add("<PAD>")
    vocab.add("<UNK>")
    for pos in upos:
        for tag in pos:
            pos_tags.add(tag)
    
    # Build the vocabulary (use the whole entire thing)
    all_words = list(itertools.chain.from_iterable(sentences))
    counter = Counter(all_words)
    special_tokens = ["<PAD>", "<UNK>"]

    # keep order based on count
    vocabulary = special_tokens + [word.lower() for word,_ in counter.most_common(len(vocab)+10000)] 
    word2idx = {word:idx for idx, word in enumerate(vocabulary)}    

    return ((len(vocab), len(pos_tags)), word2idx)

    
(vocab_size, pos_tags), vocab = index_analysis(split="train")
print(len(vocab.keys()))
print(vocab_size, pos_tags, vocab["hello"])

print(dataset["train"]["upos"][0:5])
print(dataset["train"]["xpos"][0:5])
print(dataset["train"]["tokens"][0:5])





    


