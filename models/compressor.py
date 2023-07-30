import os
import math
import inspect
import torch
from typing import Tuple, Optional
import torch.nn as nn
from torch.nn import functional as F

from modules import CompressedAttention, TransformerBlock, LayerNorm, Transpose
from models import GPT, GPTConfig

class Compressor(nn.Module):
    def __init__(self, config, meta_specials=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.meta_specials = meta_specials

        self.base_transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.compress = self._build_compressor(config)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _build_compressor(self, config):
        raise NotImplementedError

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx,
        attn_mask=None,
    ):
        stats = {}

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        pos_emb = self.base_transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        tok_emb = self.base_transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.base_transformer.drop(tok_emb + pos_emb)


        # full size transformer
        for block in self.base_transformer.h:
            x = block(x, attn_mask=attn_mask, causal=False)
        x = self.base_transformer.ln_f(x)

        # compress to token length
        if self.compress is not None:
            x = self.compress(x)

        return x

class GLOMCompressor(nn.Module):
    def __init__(self, config, meta_specials=None):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.meta_specials = meta_specials

        self.n_layer = config.n_layer
        self.pos_embs = [nn.Embedding(config.block_size, config.n_embd) for _ in range(self.n_layer)]

        self.top_downs = [nn.ModuleDict(
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f1 = LayerNorm(config.n_embd, bias=config.bias),
            attn = FunnelUpsample(config),
            ln_f2 = LayerNorm(config.n_embd, bias=config.bias),
        ) for _ in range(self.n_layer - 1)]

        self.bottom_ups = [nn.ModuleDict(
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
            attn = FunnelAttention(config),
            ln_f2 = LayerNorm(config.n_embd, bias=config.bias),
        ) for _ in range(self.n_layer - 1)]

    def apply_layer(self, emb, pos_layer, layer):
        b, t, d = emb.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        pos_emb = pos_layer(pos)
        emb += pos_emb

        for block in layer.h:
            emb = block(emb, causal=False)
        emb = layer.ln_f1(emb)

        # contract embeddings
        emb = layer.attn(emb)
        emb = layer.ln_f2(emb)

        return emb


    def forward(x):

        enc_embs = [x]

        # bottom up encoding
        for layer in range(self.n_layer - 1):
            td_layer = self.top_downs[layer]
            pos_layer = self.pos_embs[layer]
            emb = enc_embs[-1]
            new_emb = self.apply_layer(emb, pos_layer, td_layer)
            enc_embs.append(new_emb)

        dec_embs = [enc_embs[-1]]
        # top down decoding
        for layer in range(self.n_layer - 2, -1, -1):
            bu_layer = self.bottom_ups[layer]
            pos_layer = self.pos_embs[layer]
            emb = dec_embs[-1]
            new_emb = self.apply_layer(emb, pos_layer, bu_layer)
            dec_embs.append(new_emb)

        dec_embs = dec_embs[::-1]

        loss = [F.mse_loss(d_emb, e_emb) for d_emb, e_emb in zip(dec_embs[:-1], e_embs[:-1])]

        loss = torch.sum(loss)

        return enc_embs[1:], loss


class PatchCompressor(Compressor):
    def _build_compressor(self, config):
        return CompressedAttention(config)


class ConvCompressor(Compressor):
    def _build_compressor(self, config):
        return nn.Sequential(
            Transpose(),
            nn.Conv1d(config.n_embd, config.n_embd, kernel_size=config.kernel_size, stride=config.kernel_size),
            Transpose(),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
        )

class GatedCompressor(Compressor):
    def _build_compressor(self, config):
        raise NotImplementedError


