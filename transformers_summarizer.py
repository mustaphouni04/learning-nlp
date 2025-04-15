import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader,Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple, Dict
from einops import einsum, rearrange

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
        self.dim_head = d_model // num_heads_q
        self.dim_q = self.dim_head * num_heads_q
        self.dim_kv = self.dim_head * num_heads_kv
        self.num_groups = num_heads_q // num_heads_kv

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def grouped_scaled_dot_product_attention(self, Q, K, V, mask=None, is_causal: bool):
        # b: batch size, h: num heads, g: num groups, t: seq length of queries, n: seq length of keys/values, d: dimensionality of head
        # Input is (B, t, d_model) or (B, n, d_model)
        # After splitting into heads is (B, T, num_heads_kv, dim_head) or (B, T, num_heads_q, dim_head)
        Q = rearrange(Q, "b t h d -> b h t d")
        bq, hq, tq, dq = Q.shape
        K = rearrange(K, "b n h d -> b h n d") 
        bk, hk, tk, dk = K.shape
        V = rearrange(V, "b n h d -> b h n d") # different notation for sequence length
        
        Q = Q / (Q.size(-1) ** 0.5)
        # After rearranging properly we have (B, num_heads_q, t, dim_head) or (B, num_heads_kv, n, dim_head)
        # Group query with num_groups
        Q = rearrange(Q, "b (g h) t d -> b g h t d", g=self.num_groups)
        attn_scores = einsum(Q, K, "b g h t d, b h n d -> b g h t n")

        if is_causal:
            mask = torch.ones((tq, tk), device=Q.device, dtype=torch.bool).tril() # need to reshape to (1, 1 tq, tk) later
            mask = mask[None, None, None, :, :] # shape (1, 1, 1, tq, tk)
        
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, None, :, :] # (1, 1, 1, tq, tk)
            elif mask.ndim == 3:
                mask = mask[:, None, None, :, :]

            attn_scores.masked_fill_(~mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = torch.softmax(attn_scores, dim=-1) # (B, num_heads, T, N)
        output = einsum(attn_probs, V, "b g h t n, b h n d -> b g h t d") 
        output = rearrange(output, "b g h t d -> b t (g h) d")
        output = output.view(bq, tq, self.d_model)

        return output
    
    def split_heads(self, x, query:bool = False):
        if query:
            B, T, d_model = x.size()
            return x.view(B, T, self.num_heads_q, self.dim_head)
        else:
            B, T, d_model = x.size()
            return x.view(B, T, self.num_heads_kv, self.dim_head)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None, is_causal: bool = False):
        Q = self.split_heads(self.W_q(Q), query=True)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.grouped_scaled_dot_product_attention(Q, K, V, mask, is_causal)
        output = self.W_o(attn_output)

        return output











