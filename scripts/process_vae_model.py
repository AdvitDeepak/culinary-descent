#!/usr/bin/env python3
"""
Per-step Contextual VQ-VAE — learns ideal cooking operation vocabulary from data.

Goal:
Learn a discrete codebook where each entry = one learned cooking operation type
(like bake, sauté, mix — but discovered from data, not hand-designed).
Trained on 1M recipes via embedding reconstruction.

Architecture:
Encoder (per step t, causal):
  h[t] = step_emb[t]                          ← MiniLM 384-dim
       + causal_self_attn(h[t], h[0..t-1])   ← what happened before
       + cross_attn(h[t], ing_embs)           ← which ingredients am I acting on
  z[t] = EMA-VQ( linear(h[t]) )              ← discrete code = learned op type

Decoder (per step t, causal):
  d[t] = code_emb[t]
       + causal_self_attn(d[t], d[0..t-1])   ← prior codes give process context
       + cross_attn(d[t], ing_embs)           ← ingredient re-grounding
  recon[t] = linear(d[t]) → 384-dim          ← reconstructed step embedding

Loss = cosine_recon + vq_commitment + entropy_uniform_codes

Atomicity:
~9.3% of recipe steps are compound ("cook; drain" or "mix. Chop.").
Use split_atomic() before MiniLM embedding to get clean operation-level steps.
The training script has an --atomic flag that re-splits and re-embeds on the fly.
Without --atomic, trains on the existing cached embeddings (faster start,
slightly noisier codes for the compound 9%).

Pipeline stages:
Stage 1 (this file): train VQ-VAE on 1M MiniLM step embeddings.
  Output: frozen codebook = learned operation vocabulary.
  Evaluate: do the codes recover bake/sauté/mix? what new types emerge?

Stage 2 (process_vae_stage2.py — TBD): train Qwen decoder conditioned on
  discrete DAG codes. Input: DAG = {code sequence + adjacency}. Output: recipe text.
  Uses frozen Stage 1 codebook as prefix tokens for Qwen.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Atomicity ──────────────────────────────────────────────────────────────────

# Patterns that indicate a step contains multiple distinct operations
_SPLIT_RE = re.compile(
    r'\s*\.\s+(?=[A-Z])'                   # "Mix well. Chop finely."
    r'|\s*;\s+'                             # "Cook pasta; drain well."
    r'|\s*,?\s+(?:and\s+)?then\b'          # "mix, then chop" / "and then"
    r'|\s+(?:next|after\s+that),?\s+'      # "next, ..." / "after that, ..."
    r'|\s+(?:afterward[s]?),?\s+',         # "afterwards, ..."
    re.IGNORECASE
)

# Heuristics for non-operational steps (notes, tips, serving suggestions)
_NOTE_RE = re.compile(
    r'^\s*\('                              # starts with parenthesis = clarification
    r'|(?:can\s+be|may\s+be)\s+(?:made|prepared|stored|frozen|refrigerated)'
    r'|(?:note|tip|optional|serve|serving|yield|makes\s+about|store\s+in)'
    r'|(?:this\s+recipe|this\s+dish|this\s+sauce)',
    re.IGNORECASE
)


def split_atomic(step_text: str) -> list:
    """
    Split a potentially compound step into atomic sub-steps.
    Returns list of atomic step strings (may be length 1 if already atomic).
    Filters out non-operational steps (notes, tips, etc.).
    """
    if _NOTE_RE.search(step_text):
        return []
    parts = _SPLIT_RE.split(step_text.strip())
    result = []
    for p in parts:
        p = p.strip()
        if len(p) < 8:          # too short to be a real step
            continue
        if _NOTE_RE.search(p):  # filter sub-parts that are notes
            continue
        result.append(p)
    return result if result else [step_text.strip()]


# ── EMA Vector Quantizer ───────────────────────────────────────────────────────

class EMAVectorQuantizer(nn.Module):
    """
    EMA-updated discrete codebook.
    Straight-through estimator: gradients flow through as if z_q == z.
    """
    def __init__(self, n_codes=256, d_code=64, decay=0.99, eps=1e-5,
                 commitment_cost=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.d_code = d_code
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        embed = torch.randn(n_codes, d_code)
        nn.init.uniform_(embed, -1 / n_codes, 1 / n_codes)
        self.register_buffer("embed",        embed)
        self.register_buffer("cluster_size", torch.ones(n_codes))
        self.register_buffer("embed_avg",    embed.clone())

    def forward(self, z):
        """z: (N, d_code) — flattened batch×steps"""
        flat = z.detach()
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        if self.training:
            one_hot = F.one_hot(idx, self.n_codes).float()
            self.cluster_size.mul_(self.decay).add_(one_hot.sum(0) * (1 - self.decay))
            dw = one_hot.t() @ flat
            self.embed_avg.mul_(self.decay).add_(dw * (1 - self.decay))
            n  = self.cluster_size.sum()
            cs = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
            self.embed.data.copy_(self.embed_avg / cs.unsqueeze(1))
        z_q = self.embed[idx]
        commit_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z)
        # Straight-through: forward = z_q, backward = z
        z_q_st = z + (z_q - z).detach()
        return z_q_st, idx, commit_loss

    def entropy_loss(self, indices):
        """Penalise skewed usage — push toward uniform codebook utilisation."""
        counts = torch.bincount(indices.reshape(-1),
                                minlength=self.n_codes).float()
        probs  = counts / (counts.sum() + 1e-8)
        H      = -(probs * (probs + 1e-8).log()).sum()
        return torch.tensor(self.n_codes, device=indices.device).float().log() - H

    def usage_fraction(self, indices):
        return indices.reshape(-1).unique().numel() / self.n_codes


# ── Model ──────────────────────────────────────────────────────────────────────

class ProcessVAE(nn.Module):
    """
    Per-step Contextual VQ-VAE.

    Parameters
    ----------
    n_codes : int   — codebook size (learned operation vocabulary)
    d_code  : int   — code vector dimension
    d_model : int   — internal transformer dimension
    d_input : int   — MiniLM embedding dimension (384)
    n_heads : int   — attention heads
    """
    def __init__(self, n_codes=256, d_code=64, d_model=256, d_input=384,
                 n_heads=4, dropout=0.1, entropy_weight=0.05,
                 commitment_cost=0.25, d_meta=0):
        super().__init__()
        self.d_input        = d_input
        self.d_meta         = d_meta   # 0 = no meta features; 2 = temp + duration
        self.d_model        = d_model
        self.entropy_weight = entropy_weight

        # ── Encoder ──────────────────────────────────────────────────────────
        self.step_proj  = nn.Linear(d_input + d_meta, d_model)
        self.ing_proj   = nn.Linear(d_input, d_model)

        self.enc_self   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.enc_norm1  = nn.LayerNorm(d_model)
        self.enc_cross  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.enc_norm2  = nn.LayerNorm(d_model)
        self.enc_ff     = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.enc_norm3  = nn.LayerNorm(d_model)

        self.pre_vq     = nn.Linear(d_model, d_code)
        self.vq         = EMAVectorQuantizer(n_codes, d_code,
                                             commitment_cost=commitment_cost)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.code_up    = nn.Linear(d_code, d_model)

        self.dec_self   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.dec_norm1  = nn.LayerNorm(d_model)
        self.dec_cross  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.dec_norm2  = nn.LayerNorm(d_model)
        self.dec_ff     = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.dec_norm3  = nn.LayerNorm(d_model)
        self.out_proj   = nn.Linear(d_model, d_input)

    @staticmethod
    def _causal_mask(T, device):
        """Upper-triangular -inf mask: position t cannot attend to t+1..T-1."""
        mask = torch.full((T, T), float("-inf"), device=device)
        mask.triu_(1)
        return mask

    def encode_dag(self, step_embs, ing_embs, ing_mask):
        """
        Like encode(), but returns attention weights for explicit DAG extraction.

        Returns
        -------
        z_q        : (B, T, d_code)  quantised codes
        indices    : (B, T)          code indices
        self_attn  : (B, T, T)       causal step→step attention (averaged over heads)
                                     self_attn[b, t, s] = how much step t attends to step s
        cross_attn : (B, T, N)       step→ingredient attention
                                     cross_attn[b, t, n] = how much step t uses ingredient n
        """
        B, T, _ = step_embs.shape
        x = self.step_proj(step_embs)
        g = self.ing_proj(ing_embs)

        causal = self._causal_mask(T, x.device)
        # need_weights=True, is_causal=False (we pass explicit mask) to get attn back
        h, s_attn = self.enc_self(x, x, x, attn_mask=causal,
                                   need_weights=True, average_attn_weights=True)
        x = self.enc_norm1(x + h)

        h, c_attn = self.enc_cross(x, g, g, key_padding_mask=ing_mask,
                                    need_weights=True, average_attn_weights=True)
        x = self.enc_norm2(x + h)
        x = self.enc_norm3(x + self.enc_ff(x))

        z = self.pre_vq(x)
        z_q, idx, _ = self.vq(z.reshape(B * T, -1))
        return (z_q.reshape(B, T, -1), idx.reshape(B, T),
                s_attn,   # (B, T, T)
                c_attn)   # (B, T, N)

    def encode(self, step_embs, ing_embs, ing_mask, use_vq=True, meta=None):
        """
        step_embs : (B, T, 384)
        ing_embs  : (B, N, 384)
        ing_mask  : (B, N) bool  — True = padding
        meta      : (B, T, d_meta) float — optional temp/duration features
        Returns z_q (B,T,d_code), indices (B,T)|None, commit_loss scalar
        """
        B, T, _ = step_embs.shape
        inp = torch.cat([step_embs, meta], dim=-1) if meta is not None and self.d_meta > 0 else step_embs
        x = self.step_proj(inp)             # (B, T, d_model)
        g = self.ing_proj(ing_embs)         # (B, N, d_model)

        causal = self._causal_mask(T, x.device)
        h, _   = self.enc_self(x, x, x, attn_mask=causal)
        x      = self.enc_norm1(x + h)

        h, _   = self.enc_cross(x, g, g, key_padding_mask=ing_mask)
        x      = self.enc_norm2(x + h)
        x      = self.enc_norm3(x + self.enc_ff(x))

        z = self.pre_vq(x)                  # (B, T, d_code)
        if use_vq:
            z_q, idx, cl = self.vq(z.reshape(B * T, -1))
            return z_q.reshape(B, T, -1), idx.reshape(B, T), cl
        return z, None, torch.tensor(0.0, device=x.device)

    def decode(self, z_q, ing_embs, ing_mask):
        """z_q: (B, T, d_code)  →  (B, T, 384)"""
        B, T, _ = z_q.shape
        x = self.code_up(z_q)              # (B, T, d_model)
        g = self.ing_proj(ing_embs)        # (B, N, d_model)

        causal = self._causal_mask(T, x.device)
        h, _   = self.dec_self(x, x, x, attn_mask=causal)
        x      = self.dec_norm1(x + h)

        h, _   = self.dec_cross(x, g, g, key_padding_mask=ing_mask)
        x      = self.dec_norm2(x + h)
        x      = self.dec_norm3(x + self.dec_ff(x))

        return self.out_proj(x)            # (B, T, 384)

    def forward(self, step_embs, ing_embs, ing_mask, step_mask, use_vq=True, meta=None):
        """
        step_mask : (B, T) float — 1 for real steps, 0 for padding
        meta      : (B, T, d_meta) optional temp/duration features
        """
        z_q, idx, commit_loss = self.encode(step_embs, ing_embs, ing_mask, use_vq, meta=meta)
        recon = self.decode(z_q, ing_embs, ing_mask)

        rn = F.normalize(recon,     dim=-1)
        tn = F.normalize(step_embs, dim=-1)
        cos        = (rn * tn).sum(-1)                   # (B, T)
        recon_loss = ((1 - cos) * step_mask).sum() / (step_mask.sum() + 1e-8)

        if idx is not None:
            valid_idx = idx[step_mask.bool()]   # exclude padding positions
            entropy_loss = self.vq.entropy_loss(valid_idx)
        else:
            entropy_loss = torch.tensor(0.0, device=step_embs.device)

        total = recon_loss + commit_loss + self.entropy_weight * entropy_loss

        return {
            "total":   total,
            "recon":   recon_loss,
            "commit":  commit_loss,
            "entropy": entropy_loss,
            "indices": idx,
            "z_q":     z_q,
            "recon_embs": recon,
        }
