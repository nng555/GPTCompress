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


class GPTCompress(nn.Module):

    def __init__(self, config, meta_specials):
        assert meta_specials is not None, "Need special tokens for compression"
        assert 'doc' in meta_specials, "Require special doc token"
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.meta_specials = meta_specials
        self.register_buffer('doc_ids', torch.tensor([v for k, v in self.meta_specials.items() if 'doc' in k]))

        self.topk = config.topk
        self.temperature = config.temperature

        self.hard_loss_weight = config.hard_loss_weight
        self.soft_loss_weight = config.soft_loss_weight

        # load pretrained lm
        ckpt_path = os.path.join(config.out_dir, 'ckpt_lm.pt')
        assert os.path.exists(ckpt_path), "No pretrained LM found"
        checkpoint = torch.load(ckpt_path)
        lmconf = GPTConfig(**checkpoint['model_args'])
        self.lm = GPT(lmconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.lm.load_state_dict(state_dict)

        self.encoder = GLOMCompressor(config)
        #self.encoder.base_transformer.wte = self.lm.transformer.wte
        #for param in self.encoder.base_transformer.wte.parameters():
        #    param.requires_grad = False

        # set to eval and freeze parameters
        self.lm.eval()
        for param in self.lm.parameters():
            param.requires_grad = False

        # TODO: always load in half?
        self.lm = self.lm.half()

        # report number of parameters
        print("number of frozen lm parameters: %.2fM" % (self.lm.get_num_params() / 1e6))

    def length_transformation(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        deltas: torch.Tensor,
        sigma: torch.Tensor,
        arange: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Functional form of the length transformation mechanism

        Parameters
        ----------
        x: torch.Tensor
            Input representation
        x_mask: torch.Tensor
            Padding mask
        deltas: torch.Tensor
            Length delta changes
        sigma: torch.Tensor
            Bandwidth parameter (could be a nn.Parameter or torch.Tensor)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Output representation, output padding mask, and output weight
        """
        x_mask = x_mask.bool()
        batch_size, max_len_in, d_model = x.size()
        deltas = deltas.reshape(-1)
        # compute input lengths

        encoder_mask = ~x_mask
        lens = encoder_mask.sum(-1).float()
        changed_lens = (lens + deltas.reshape(-1)).float()

        # max length
        max_len_out = int(torch.max(changed_lens))

        # outward proxy
        # tile [0, ..., Lmax] batch_size times
        if arange is None:
            arange_l = torch.arange(0, max_len_out, dtype=torch.float).tile(batch_size, 1).type_as(x)
            arange_z = (
                torch.arange(0, max_len_in, dtype=torch.float)
                .tile(batch_size, max_len_out, 1)
                .type_as(x)
            )
        else:
            arange_l = arange[:max_len_out].tile(batch_size, 1)
            arange_z = arange[:max_len_in].tile(batch_size, max_len_out, 1)

        # (batch_size, Lmax)
        mu = arange_l * lens[:, None] / changed_lens[:, None]

        # KC: in this way, we don’t get nan when learning sigma with the perfectly correct attention
        z_prime_mask = arange_l < changed_lens[:, None]  # new antimask
        padding_mask = ~z_prime_mask

        logits = -1.0 * torch.pow(arange_z - mu[:, :, None], 2) / (2.0 * torch.exp(sigma))

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(x_mask.unsqueeze(1), 0.0)
        exp_logits = exp_logits.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        denom = exp_logits.sum(dim=2, keepdim=True)
        exp_logits = exp_logits.contiguous()
        denom = denom.masked_fill(denom == 0.0, 1.0)
        weight = exp_logits / denom

        z_prime = torch.matmul(weight, x)
        return z_prime, padding_mask, weight

    def forward(self, idx, lm_idx, enc_idx, targets, teacher_mask, enc_attn_mask=None, lm_attn_mask=None, full_mask=None, tau=1):

        stats = {}

        # get context embedding
        # TODO: add linear layer to project through bottleneck
        ctx_embs, encoder_loss = self.encoder(enc_idx, attn_mask=enc_attn_mask)

        for ctx_emb in ctx_embs:

            ctx_emb = ctx_emb.reshape(-1, ctx_emb.shape[-1])

            tok_emb = self.lm.transformer.wte(lm_idx)
            tok_emb[lm_idx == self.meta_specials['doc']] = ctx_emb.half()

            logits, hard_loss, lm_stats = self.lm(lm_idx, tok_emb=tok_emb, targets=targets, attn_mask=lm_attn_mask, causal=True)

            stats['hard_loss'] = hard_loss

            for k in lm_stats:
                stats[k] = lm_stats[k]

            if self.soft_loss_weight > 0:
                # get soft labels
                with torch.no_grad():
                    teacher_logits = self.lm(idx, return_logits=True, attn_mask=full_mask, causal=True)[teacher_mask.bool()]

                logits = logits[(targets != -1) & (targets != self.meta_specials['doc']) & (targets != self.meta_specials['pad'])]

                # reduce to topk logits
                if self.topk != -1:
                    topk_idxs = torch.topk(teacher_logits, self.topk, dim=-1, sorted=False).indices
                    topk_teacher_logits = torch.take(teacher_logits, topk_idxs)
                    topk_logits = torch.take(logits, topk_idxs)
                else:
                    topk_teacher_logits = teacher_logits
                    topk_logits = logits

                soft_loss = F.kl_div(
                    F.log_softmax(topk_logits / self.temperature, -1),
                    F.softmax(topk_teacher_logits / self.temperature, -1),
                    reduction='batchmean',
                ) * self.temperature**2
                loss = soft_loss * self.soft_loss_weight + \
                       hard_loss * self.hard_loss_weight

                stats['soft_loss'] = float(soft_loss)
                stats['hard_loss'] = float(hard_loss)
                stats['cross_ppl'] = 2 ** (F.kl_div(
                    F.log_softmax(logits, -1),
                    F.softmax(teacher_logits, -1), reduction='batchmean') / math.log(2))

            else:
                loss = hard_loss
        return logits, loss, stats

    # TODO: should we inherit this?
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters

        #use_fused = fused_available and device_type == 'cuda'
        # NOTE: disable fused to get float16 to work
        use_fused = False

        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):


        for _ in (ens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, return_logits=True)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None:
                if idx_next == eos_id:
                    break



