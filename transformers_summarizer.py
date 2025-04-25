from os import wait
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple, Dict
from einops import einsum, rearrange
import pandas as pd
import logging
import tiktoken
import random
import torch.nn.functional as F

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads_kv: int, num_heads_q: int):
        super().__init__()
        assert (d_model % num_heads_kv == 0) & (d_model % num_heads_q == 0), "d_model must be divisible by number of heads to avoid fractioning"
        self.d_model = d_model
        self.num_heads_kv = num_heads_kv
        self.num_heads_q = num_heads_q
        self.dim_q = d_model // num_heads_q # not really used
        self.dim_kv = d_model // num_heads_kv # not really used
        if self.dim_kv != self.dim_q:
            self.value_proj = nn.Linear(self.dim_kv, self.dim_q)
            self.match_proj = nn.Linear(self.dim_kv, self.dim_q)
        else:
            self.value_proj = None
            self.match_proj = None
        self.num_groups = num_heads_q // num_heads_kv

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def grouped_scaled_dot_product_attention(self, Q, K, V, mask=None, is_causal: bool = False):
        # b: batch size, h: num heads, g: num groups, t: seq length of queries, n: seq length of keys/values, d: dimensionality of head
        # Input is (B, t, d_model) or (B, n, d_model)
        # After splitting into heads is (B, T, num_heads_kv, dim_head) or (B, T, num_heads_q, dim_head)
        Q = rearrange(Q, "b t h d -> b h t d")
        bq, hq, tq, dq = Q.shape
        K = rearrange(K, "b n h d -> b h n d") 
        if self.match_proj is not None:
            K = self.match_proj(K)
        bk, hk, tk, dk = K.shape
        V = rearrange(V, "b n h d -> b h n d") # different notation for sequence length
        
        Q = Q / (Q.size(-1) ** 0.5)
        # After rearranging properly we have (B, num_heads_q, t, dim_head) or (B, num_heads_kv, n, dim_head)
        # Group query with num_groups
        Q = rearrange(Q, "b (g h) t d -> b g h t d", g=self.num_groups)
        attn_scores = einsum(Q, K, "b g h t d, b h n d -> b g h t n")

        if is_causal:
            causal_mask = torch.ones((bq, tq, tk), device=Q.device, dtype=torch.bool).tril() # need to reshape to (1, 1 tq, tk) later
            causal_mask = causal_mask[:, None, None, :, :] # shape (bq, 1, 1, tq, tk)
            if mask is not None:
                if mask.ndim == 2:
                    mask = mask[:, None, None, None, :] # (bq, 1, 1, 1, tk)
                elif mask.ndim == 3:
                    mask = mask[:, None, None, :, :] # (bq, 1, 1, tq, tk)
                mask = mask & causal_mask # combine pad mask with causal mask
            else:
                mask = causal_mask
        
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, None, :] # (bk, 1, 1, 1, tk)
            elif mask.ndim == 3:
                mask = mask[:, None, None, :, :] # (bk, 1, 1, tq, tk)

            attn_scores.masked_fill_(~mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1) # (b, groups, num_heads, tq, tn)
        output = einsum(attn_probs, V, "b g h t n, b h n d -> b g h t d") 
        output = rearrange(output, "b g h t d -> b t (g h) d")
        output = output.view(bq, tq, self.d_model)

        return output
    
    def split_heads(self, x, query:bool = False):
        if query:
            B, T, d_model = x.size()
            return x.view(B, T, self.num_heads_q, self.dim_q)
        else:
            B, T, d_model = x.size()
            return x.view(B, T, self.num_heads_kv, self.dim_kv)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None, is_causal: bool = False):
        Q = self.split_heads(self.W_q(Q), query=True)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        if self.value_proj is not None:
            V = self.value_proj(V)

        attn_output = self.grouped_scaled_dot_product_attention(Q, K, V, mask, is_causal)
        output = self.W_o(attn_output)

        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2) # activation expects doubled
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swish = nn.SiLU()
    def forward(self, x):
        x = self.fc1(x)
        a, b = x.chunk(2, dim=-1)
        x = self.swish(b)*a
        x = self.fc2(x)
        return x

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
    def __init__(self, d_model, num_heads_kv, num_heads_q, d_ff, dropout):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads_kv, num_heads_q)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, is_causal=False):
        # Pre-norm
        attn_output = self.self_attn(self.norm1(x),self.norm1(x),self.norm1(x),mask,is_causal)
        x = x + self.dropout(attn_output)
        # Pre-norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads_kv, num_heads_q, d_ff, dropout):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, num_heads_kv, num_heads_q)
        self.cross_attn = GroupedQueryAttention(d_model, num_heads_kv, num_heads_q)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, is_causal=True):
        # Pre-norm
        masked_attn_output = self.self_attn(self.norm1(x),self.norm1(x),self.norm1(x), mask=None, is_causal=is_causal)
        x = x + self.dropout(masked_attn_output)
        # Pre-norm
        attn_output = self.cross_attn(self.norm2(x), self.norm2(enc_output), self.norm2(enc_output), src_mask, is_causal=False)
        x = x + self.dropout(attn_output)
        # Pre-norm
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_output)
        return x

class TextSummarizer(nn.Module):
    def __init__(self, vocab_size, d_model, 
                 num_heads_kv, num_heads_q, 
                 num_layers, d_ff,
                 max_seq_length_enc,
                 max_seq_length_dec, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = 50257) 
        self.positional_encoding_enc = PositionalEncoding(d_model, max_seq_length_enc)
        self.positional_encoding_dec = PositionalEncoding(d_model, max_seq_length_dec)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads_kv, num_heads_q, d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads_kv, num_heads_q, d_ff,dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, enc_input, pad_token_id):
        # enc_input = (B,T) 
        pad_mask = (enc_input != pad_token_id) # (B, T)
        # mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2) # (B, T, T) pairwise attention mask
        return pad_mask
    
    def forward(self, x, tgt, pad_token_id = 50257):
        assert torch.all(x >= 0) and torch.all(x < self.vocab_size), "Invalid input token IDs!"
        src_pad_mask = self.generate_mask(x,pad_token_id)
        tgt_pad_mask = self.generate_mask(tgt,pad_token_id)
        src_embedded = self.dropout(self.positional_encoding_enc(self.embedding(x)))
        tgt_embedded = self.dropout(self.positional_encoding_dec(self.embedding(tgt)))

        enc_output = src_embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_pad_mask)

        dec_output = tgt_embedded 
        
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_pad_mask, is_causal=True)

        output = self.fc(dec_output)

        return output


train_df = pd.read_csv("../CNN_dataset/train.csv")
valid_df = pd.read_csv("../CNN_dataset/validation.csv")
test_df = pd.read_csv("../CNN_dataset/test.csv")

# article, highlights, id
logger = logging.getLogger("summarizer")
logging.basicConfig(level=logging.INFO)
logger.info("The length of the train set is %i", len(train_df["article"].values))

encoding = tiktoken.get_encoding("gpt2")

def tokenize(text:str) -> List[str]:
    tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
    return tokens

def decode_tokens(tokens:List[int]) -> List[str]:
    decoded = [encoding.decode_single_token_bytes(token) for token in tokens if token != 50257]
    decoded = [word.decode("utf-8", errors="ignore") for word in decoded] 
    return decoded

def build_sequences(df: pd.DataFrame, max_len_art:int = 512, max_len_targ:int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    eos = '<|endoftext|>'
    pad = {'<|pad|>': 50257}
    targets = df["highlights"].values
    target_sequences = [tokenize(eos)+tokenize(sent)+tokenize(eos) for sent in tqdm(targets, desc="Tokenizing summaries...may take a while")]

    article = df["article"].values
    article_sequences = [tokenize(sent) for sent in tqdm(article, desc="Tokenizing articles...may take a while")]
    
    padded_targ = []
    padded_arts = []

    for targ in target_sequences:
        padded = (targ + [pad['<|pad|>']] * max_len_targ)[:max_len_targ]
        padded_targ.append(padded)
    
    for art in article_sequences:
        padded = (art + [pad['<|pad|>']] * max_len_art)[:max_len_art]
        padded_arts.append(padded)

    article_sequences = torch.LongTensor(padded_arts)
    target_sequences = torch.LongTensor(padded_targ)
    
    return article_sequences, target_sequences 

"""
article_sequences, target_sequences = build_sequences(train_df)

print(article_sequences[0])
print(decode_tokens(article_sequences[0]))
print(target_sequences[0])
print(decode_tokens(target_sequences[0]))
"""

class CNN_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len_art = 512, max_len_targ = 128):
        article_sequences, target_sequences = build_sequences(df, max_len_art, max_len_targ)
        self.article_sequences = article_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return self.article_sequences.size(0)

    def __getitem__(self, idx):
        return self.article_sequences[idx], self.target_sequences[idx]

batch_size = 32
vocab_size = 50258
d_model = 512 
num_heads_kv = 4
num_heads_q = 8
num_layers = 3
d_ff = 2048
max_seq_length_enc = 512
max_seq_length_dec = 128
dropout = 0.1
n_epochs = 10
process = False


summarizer = TextSummarizer(vocab_size, d_model,
                            num_heads_kv, num_heads_q,
                            num_layers, d_ff,
                            max_seq_length_enc,
                            max_seq_length_dec,
                            dropout).to(device)

checkpoint = torch.load("model_checkpoint/epoch_summarizer_24.pt")
summarizer.load_state_dict(checkpoint["model_state_dict"]) # load from epoch 25

if process:
    train_dataset = CNN_Dataset(train_df)
    valid_dataset = CNN_Dataset(valid_df)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size)

    """
vocab_size, d_model, 
                 num_heads_kv, num_heads_q, 
                 num_layers, d_ff,
                 max_seq_length_enc,
                 max_seq_length_dec, dropout
    """


    optimizer = optim.AdamW(summarizer.parameters(), lr=3e-4)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss = nn.CrossEntropyLoss(ignore_index=50257, label_smoothing=0.05)
    accum_steps = 2
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
#total_steps = (steps_per_epoch * n_epochs)

"""
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-3,
    pct_start=0.2,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs,
    anneal_strategy='cos',
    final_div_factor=1e2
)
"""
#scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

def train(model, iterator, criterion, optimizer):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    padding_val = 50257
    optimizer.zero_grad()

    for idx, batch in enumerate(tqdm(iterator)):
        article, targets = batch # (B, max_seq_length_enc) & (B, max_seq_length_dec)
        article, targets = article.to(device), targets.to(device)
        # assert article.max().item() < model.vocab_size, "Token ID exceeds vocabulary size!"
        # optimizer.zero_grad()
        predictions = model(article, targets[:, :-1], pad_token_id=padding_val)  # (B, max_seq_length_dec, vocab_size)
        
        flat_preds = predictions.view(-1, vocab_size)
        flat_targets = targets[:, 1:].reshape(-1)
        
        loss = criterion(flat_preds, flat_targets)
        loss = loss / accum_steps
        
        mask = flat_targets != padding_val
        
        predicted_words = flat_preds.argmax(dim=-1)
        correct = (predicted_words[mask] == flat_targets[mask]).float()
        war = correct.sum() / len(correct)

        writer.add_scalar("Loss/Train", loss.item(), idx)
        writer.add_scalar("WAR/Train", war.item(), idx)
        epoch_loss += loss.item()
        epoch_acc += war.item()
        loss.backward() 

        if (idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            #scheduler.step()

    writer.add_scalar("EpochLoss/Train", epoch_loss / len(iterator))
    writer.add_scalar("EpochWAR/Train", epoch_acc / len(iterator))
    

def inference(model, iterator, criterion):
    model.eval()
    padding_val = 50257
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(iterator)):
            article, targets = batch # (B, max_seq_length_enc) & (B, max_seq_length_dec)
            article, targets = article.to(device), targets.to(device)
            predictions = model(article, targets[:, :-1], pad_token_id=padding_val)

            flat_preds = predictions.view(-1,vocab_size)
            flat_targets = targets[:,1:].reshape(-1)

            loss = criterion(flat_preds, flat_targets)
        
            mask = flat_targets != padding_val
        
            predicted_words = flat_preds.argmax(dim=-1)
            correct = (predicted_words[mask] == flat_targets[mask]).float()
            war = correct.sum() / len(correct)

            writer.add_scalar("Loss/Valid", loss.item(), idx)
            writer.add_scalar("WAR/Valid", war.item(), idx)
            epoch_loss += loss.item()
            epoch_acc += war.item()

    writer.add_scalar("EpochLoss/Valid", epoch_loss / len(iterator))
    writer.add_scalar("EpochWAR/Valid", epoch_acc / len(iterator))

train = False

if train:
    for epoch in tqdm(range(n_epochs)):
        train(summarizer, train_loader, loss, optimizer) 
        torch.save({'epoch': epoch+15,
            'model_state_dict': summarizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f"model_checkpoint/epoch_summarizer_{epoch+15}.pt")

        inference(summarizer, valid_loader, loss)

    writer.flush()
    writer.close()

# lil skeleton for inference

def generation_inference(model, sentences, top_p, top_k, temp):
    model.eval()
    padding_val = 50257 
    
    lengths = [len(tokenize(sentence)) for sentence in sentences]
    tokenized = [(tokenize(sentence)+[padding_val]*max_seq_length_enc)[:max_seq_length_enc] for sentence in sentences]
    tokenized = torch.LongTensor(tokenized).unsqueeze(0).to(device)
    count = 0 

    with torch.no_grad():
        for idx, sentence in enumerate(tokenized):
            generated = torch.LongTensor(tokenize("<|endoftext|>")).unsqueeze(0).to(device)
            
            for _ in range(1, max_seq_length_dec):
                pred_logits = model(sentence, generated, pad_token_id = padding_val)
                pred_logits = pred_logits[:,-1,:] / temp

                if top_k > 0:
                    topk_vals, topk_idx = torch.topk(pred_logits, top_k)
                    logits_filtered = torch.full_like(pred_logits, float('-inf'))
                    logits_filtered.scatter_(1, topk_idx, topk_vals)
                else:
                    logits_filtered = pred_logits

                sorted_logits, sorted_indices = torch.sort(logits_filtered, descending=True) # from big to small
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits_filtered[indices_to_remove] = float('-inf')

                probs = F.softmax(logits_filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                count +=1

                generated = torch.cat((generated, next_token), dim=1)

                if next_token == 50256 and count > 1:
                    break

            return generated.squeeze()

generated = generation_inference(summarizer, ["""Fatou, the world’s oldest gorilla, has just celebrated her 68th birthday at the Berlin Zoo. She is a Western Lowland Gorilla, a species native to parts of Central Africa and classified as critically endangered due to habitat loss.

Fatou arrived at the zoo when she was about two years old and has lived there for 66 years. In the wild, gorillas usually live only 30 to 40 years, so her age is remarkable. Despite her age, Fatou is doing well. She has stiff joints and can’t fully straighten her arms or legs due to arthrosis, a common condition in elderly gorillas, but she’s still in good spirits. Her movements are slower and a bit hunched, but she doesn’t need much medical care.

Zoo veterinarians help her by adjusting her diet since she no longer has teeth. She eats cooked vegetables and gets fruit as a treat. Fatou is watched closely by the staff to make sure she stays healthy. She may be older, but she’s still strong and beloved at the zoo."""], top_p=0.8, top_k=20, temp = 0.7)

generated = generated.tolist()
result = " ".join(decode_tokens(generated)[1:-1])

print(result)













