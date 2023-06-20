import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class FunnelUpsample(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.upsampler = nn.Linear(config.n_embd, config.n_embd * config.kernel_size, bias=config.bias)

    def forward(self, x, pad_mask=None, causal=True):
        B, T, C = x.size()

        out = self.upsampler(x)
        out = out.reshape(B, -1, C) # B, kT, C
        attn = out @ x.transpose(1, 2) # B, kT, T
        out = out + attn @ x

        return out

class FunnelAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pooler = nn.AvgPool1d(config.kernel_size, stride=config.kernel_size)

        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, pad_mask=None, causal=True):

        out = self.pooler(x.transpose(1, 2)).transpose(1, 2) # B, T/k, C
        attn = out @ x.transpose(1, 2) # B, T/k, T
        out = out + attn @ x # B, T/k, C

        return out

class PatchAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.ntokens, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        #self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.ntokens = config.ntokens

    def forward(self, x, pad_mask=None, causal=True):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # query and values are the same
        qv = x # (B, T, D)

        k = self.c_attn(x) # (B, T, ntok)
        k = self.attn_dropout(k)
        y = k.transpose(1, 2) @ qv # (B, ntok, D)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask=None, causal=True):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1) # (B, nh, T, T)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # default to causal if attn_mask not provided
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=(attn_mask is None or causal))
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is None or causal:
                attn_mask = self.bias[:,:,:T,:T]
            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


