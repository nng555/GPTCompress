import os
import math
import inspect
import torch
from typing import Tuple, Optional
import torch.nn as nn
from torch.nn import functional as F

from modules import CompressedAttention, TransformerBlock, LayerNorm, Transpose
from models import GPT, GPTConfig

class Memento(nn.Module):

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
        self.comp_loss_weight = config.comp_loss_weight

        # MAYBE load pretrained lm
        ckpt_path = os.path.join(config.out_dir, 'ckpt_lm.pt')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            lmconf = GPTConfig(**checkpoint['model_args'])
            self.lm = GPT(lmconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            self.lm.load_state_dict(state_dict)
        else:
            self.lm = GPT(self.config)
        assert os.path.exists(ckpt_path), "No pretrained LM found"

        self.encoder = GLOMCompressor(self.config)
        #self.encoder.base_transformer.wte = self.lm.transformer.wte
        #for param in self.encoder.base_transformer.wte.parameters():
        #    param.requires_grad = False

        # set to eval and freeze parameters
        #self.lm.eval()
        #for param in self.lm.parameters():
        #    param.requires_grad = False

        # TODO: always load in half?
        self.lm = self.lm.half()

        # report number of parameters
        print("number of frozen lm parameters: %.2fM" % (self.lm.get_num_params() / 1e6))

    def forward(self, idx, lm_idx, enc_idx, targets, teacher_mask, enc_attn_mask=None, lm_attn_mask=None, full_mask=None, tau=1):

        stats = {}

        # get context embedding
        # TODO: add linear layer to project through bottleneck
        ctx_embs, encoder_loss = self.encoder(enc_idx, attn_mask=enc_attn_mask)
        stats['encoder_loss'] = encoder_loss
        loss = encoder_loss * self.encoder_loss_weight

        doc_len = (tok_emb[0] == self.meta_specials['doc']).sum()
        tok_emb = self.lm.transformer.wte(lm_idx)

        if self.soft_loss_weight > 0:
            # get soft labels
            with torch.no_grad():
                teacher_logits = self.lm(idx, return_logits=True, attn_mask=full_mask, causal=True)[teacher_mask.bool()]

        for ctx_emb in ctx_embs:

            B, T, D = ctx_emb.shape

            cur_tok_emb = tok_emb[doc_len - T:]
            cur_tok_emb[:T] = ctx_emb
            cur_targets = targets[doc_len - T:]

            logits, hard_loss, lm_stats = self.lm(lm_idx, tok_emb=tok_emb, targets=targets, attn_mask=lm_attn_mask, causal=True)

            loss += hard_loss * self.hard_loss_weight

            stats['hard_loss'] = hard_loss

            for k in lm_stats:
                stats[k] = lm_stats[k]

            if self.soft_loss_weight > 0:
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

                loss += soft_loss * self.soft_loss_weight

        return ctx_embs, loss, stats

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




