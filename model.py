#@title GPT-3

import torch
from torch import nn, optim
import functorch
import math
import random
import datasets
import spacy
import gc
import re
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import tiktoken  # GPT tokenizer
from torch import optim
from tqdm import tqdm
from collections import Counter
from itertools import islice, chain # Mix two languages
from torch.amp import autocast, GradScaler
from typing import List

class Attention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask = None): # mask is optional
    b, s, _ = x.size() # b = batch_size, s = seq_len
    # (bs, seq, num_heads, d_k) → (bs, num_heads, seq, d_k)
    q = self.w_q(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(x).view(b, s, self.num_heads, self.d_k).transpose(1,2)

    # scaling dividing by math.sqrt(self.d_k)
    attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      # mask changes pos 0 to -inf so that makes prob to 0 after softmax
      attn_out = attn_out.masked_fill(mask == 0, float('-inf'))
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, s, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.d_k = d_model // num_heads

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask = None):
    """
    q: (b, q_len, d_model)
    k,v: (b, kv_len, d_model)
    """
    b, q_len, _ = q.size() # b = batch_size, s = seq_len
    b, kv_len, _ = k.size()
    # (bs, seq, num_heads, d_k) → (bs, num_heads, seq, d_k)
    q = self.w_q(q).view(b, q_len, self.num_heads, self.d_k).transpose(1,2)
    k = self.w_k(k).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)
    v = self.w_v(v).view(b, kv_len, self.num_heads, self.d_k).transpose(1,2)

    # scaling dividing by math.sqrt(self.d_k)
    attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if mask is not None:
      # mask changes pos 0 to -inf so that makes prob to 0 after softmax
      attn_out = attn_out.masked_fill(mask == 0, float('-inf'))
    attn_out = torch.softmax(attn_out, dim=-1)
    attn_out = self.dropout(attn_out)
    attn_out = torch.matmul(attn_out, v)
    attn_out = attn_out.transpose(1,2).reshape(b, q_len, self.d_model)
    attn_out = self.w_o(attn_out)

    return attn_out

def look_ahead_mask_(q_len, k_len=None, device=None):
    """
    Improved causal mask:
      - supports q_len != k_len (useful when using cached past key/values)
      - returns a boolean mask of shape (1, 1, q_len, k_len) where True = allowed, False = masked
    """
    if k_len is None:
        k_len = q_len
    device = device if device is not None else torch.device('cpu')

    q_idx = torch.arange(q_len, device=device).unsqueeze(1)   # (q_len, 1)
    k_idx = torch.arange(k_len, device=device).unsqueeze(0)   # (1, k_len)
    offset = k_len - q_len
    mask = (k_idx <= (q_idx + offset))                        # (q_len, k_len)
    return mask.unsqueeze(0).unsqueeze(0)                     # (1, 1, q_len, k_len)

class Decoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super().__init__()

    self.self_attention = MultiHeadAttention(d_model, num_heads, dropout) #Masked MHA
    self.dropout1 = nn.Dropout(dropout)
    self.layer_norm1 = nn.LayerNorm(d_model)

    self.ffn = FeedForward(d_model, d_ff, dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.layer_norm2 = nn.LayerNorm(d_model)

  def forward(self, x, look_ahead_mask_ = None):
    attention_out = self.self_attention(x, x, x, look_ahead_mask_)
    x = x + self.dropout1(attention_out)
    x = self.layer_norm1(x)

    ffn_out = self.ffn(x)
    x = x + self.dropout2(ffn_out)
    x = self.layer_norm2(x)

    return x

class DecoderStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            Decoder(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, look_ahead_mask_=None):
        for layer in self.layers:
            x = layer(x, look_ahead_mask_)
        return x


class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()

    self.linear1 = nn.Linear(d_model, d_ff)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model)

  def forward(self, x):
    x = self.linear1(x)
    x = self.gelu(x)
    x = self.dropout(x)
    x = self.linear2(x)

    return x

class Embedding(nn.Module):
  def __init__(self, vocab_size, d_model): # vocab_size is the total number of words/tokens
    super().__init__()
    self.d_model = d_model
    self.emb = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len, dropout): # vocab_size is the total number of words/tokens
    super().__init__()
    self.d_model = d_model
    self.dropout = nn.Dropout(dropout)
    self.max_len = max_len

    # learned positional embeddings
    self.pos_emb = nn.Embedding(self.max_len, d_model)
    # initialize similar to transformer practice
    nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

  def forward(self, x):
    # x: (B, L, d_model)
    b, l, _ = x.size()
    positions = torch.arange(l, device=x.device).unsqueeze(0)  # (1, L)
    pos = self.pos_emb(positions)                             # (1, L, d_model)
    x = x + pos
    return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()

        # single token embedding (use this for input tokens)
        self.token_embedding = Embedding(vocab_size, d_model)
        # keep positional encoding (we will change to learned in a later step)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # decoder-only stack
        self.decoder = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, dropout)  # Pass d_ff to each Decoder
            for _ in range(num_layers)
        ])

        # language modeling head
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, tgt_mask=None):
        x = self.token_embedding(input_ids)       # (B, L, d_model)
        x = self.pos_encoding(x)                  # (B, L, d_model)

        for layer in self.decoder:
            x = layer(x, tgt_mask)

        logits = self.fc_out(x)                  # (B, L, vocab_size)
        return logits