"""
SignalTransformerEncoder — NumPy multi-head self-attention encoder.

Architecture
------------
Input  : (n_signals, TOTAL=80) matrix — one row per signal provider
Output : (TOTAL,) fused probability distribution

Steps
1. Linear projection of each signal vector: 80 → d_model
2. Multi-head self-attention across signal tokens (n_signals heads → d_model)
3. Memory cross-attention: query from aggregated signal, key/value from MemoryBank state
4. Feed-forward per-number projection: d_model → 1 (scalar score per number)
5. Softmax normalisation → probability distribution

All weights are kept in plain numpy arrays and updated via a simple gradient-
free online learning rule (exponential moving average toward the best-performing
configuration seen so far in the walk-forward).  This keeps the implementation
dependency-free while giving the Transformer its structural benefits.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

TOTAL = 80


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    sigma = x.std(-1, keepdims=True)
    return (x - mu) / (sigma + eps)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


class MultiHeadSelfAttention:
    """
    Scaled dot-product multi-head attention (NumPy, inference + lightweight update).

    Tokens  : (seq_len, d_model)
    Returns : (seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, seed: int = 0) -> None:
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, d_model))
        self.Wv = rng.normal(0, scale, (d_model, d_model))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        seq_len, _ = x.shape
        Q = x @ self.Wq.T  # (seq, d)
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # Split into heads  (n_heads, seq, d_k)
        Q = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = _softmax(scores, axis=-1)
        out = attn @ V  # (n_heads, seq, d_k)
        out = out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        return out @ self.Wo.T

    def ema_update(self, other: "MultiHeadSelfAttention", alpha: float = 0.1) -> None:
        """Exponential moving average toward a better-performing copy."""
        self.Wq = (1 - alpha) * self.Wq + alpha * other.Wq
        self.Wk = (1 - alpha) * self.Wk + alpha * other.Wk
        self.Wv = (1 - alpha) * self.Wv + alpha * other.Wv
        self.Wo = (1 - alpha) * self.Wo + alpha * other.Wo


class CrossAttention:
    """
    Memory cross-attention: query from signal aggregate, key/value from memory state.

    query   : (1, d_model)
    kv_src  : (mem_dim,)  — flat memory state vector
    Returns : (d_model,)
    """

    def __init__(self, d_model: int, mem_dim: int, seed: int = 1) -> None:
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)
        self.Wq = rng.normal(0, scale, (d_model, d_model))
        self.Wk = rng.normal(0, scale, (d_model, mem_dim))
        self.Wv = rng.normal(0, scale, (d_model, mem_dim))
        self.Wo = rng.normal(0, scale, (d_model, d_model))

    def forward(self, query: np.ndarray, memory_state: np.ndarray) -> np.ndarray:
        # query: (d_model,)  memory_state: (mem_dim,)
        q = (self.Wq @ query).reshape(1, -1)
        k = (self.Wk @ memory_state).reshape(1, -1)
        v = (self.Wv @ memory_state).reshape(1, -1)
        score = (q @ k.T) / np.sqrt(self.d_model)
        attn = _softmax(score.flatten())
        out = attn[0] * v.flatten()
        return self.Wo @ out


class FeedForward:
    """Position-wise feed-forward: d_model → d_ff → d_model."""

    def __init__(self, d_model: int, d_ff: int, seed: int = 2) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = rng.normal(0, scale, (d_ff, d_model))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale, (d_model, d_ff))
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W2 @ _relu(self.W1 @ x + self.b1) + self.b2


class SignalTransformerEncoder:
    """
    Transformer encoder that fuses n_signals × (TOTAL,) score vectors into
    a single (TOTAL,) probability distribution, conditioned on MemoryBank state.

    Architecture per call
    ---------------------
    1. Project each of the n_signals score vectors: TOTAL → d_model
       (learned linear embeddings, one per signal)
    2. Multi-head self-attention across the n_signals tokens
    3. Residual + LayerNorm
    4. Per-token feed-forward
    5. Residual + LayerNorm
    6. Aggregate tokens (weighted mean by method_weights from MemoryBank)
    7. Cross-attention with flat MemoryBank state vector
    8. Residual: add cross-attention output to aggregated token
    9. Output projection: d_model → TOTAL  → softmax
    """

    def __init__(
        self,
        n_signals: int,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        seed: int = 42,
    ) -> None:
        self.n_signals = n_signals
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / TOTAL)

        # Input projections: one per signal provider
        self.input_proj = rng.normal(0, scale, (n_signals, d_model, TOTAL))
        self.input_bias = np.zeros((n_signals, d_model))

        # Transformer blocks
        self.attn = MultiHeadSelfAttention(d_model, n_heads, seed=seed)
        self.ff = FeedForward(d_model, d_ff, seed=seed + 1)

        # Cross-attention memory dim: 2*TOTAL + n_signals (att + pair_boost + method_weights)
        mem_dim = 2 * TOTAL + n_signals
        self.cross_attn = CrossAttention(d_model, mem_dim, seed=seed + 2)

        # Output head: d_model → TOTAL
        self.W_out = rng.normal(0, scale, (TOTAL, d_model))
        self.b_out = np.zeros(TOTAL)

        # Best-weights checkpoint for EMA update
        self._best_score: float = -np.inf
        self._best_state: Optional[dict] = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode(
        self,
        signal_scores: np.ndarray,
        method_weights: np.ndarray,
        memory_state: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        signal_scores  : (n_signals, TOTAL)  — normalised scores from each provider
        method_weights : (n_signals,)        — adaptive weights from MemoryBank
        memory_state   : (mem_dim,)          — flat MemoryBank state vector

        Returns
        -------
        (TOTAL,) probability distribution
        """
        # 1. Input projection → token matrix (n_signals, d_model)
        tokens = np.zeros((self.n_signals, self.d_model))
        for i in range(self.n_signals):
            tokens[i] = self.input_proj[i] @ signal_scores[i] + self.input_bias[i]

        tokens = _layer_norm(tokens)

        # 2. Self-attention + residual
        attn_out = self.attn.forward(tokens)
        tokens = _layer_norm(tokens + attn_out)

        # 3. Feed-forward + residual (applied per token)
        ff_out = np.array([self.ff.forward(tokens[i]) for i in range(self.n_signals)])
        tokens = _layer_norm(tokens + ff_out)

        # 4. Weighted aggregation using method_weights
        w = method_weights / (method_weights.sum() + 1e-8)
        aggregated = (tokens * w[:, None]).sum(0)  # (d_model,)

        # 5. Cross-attention with memory
        mem_out = self.cross_attn.forward(aggregated, memory_state)
        aggregated = _layer_norm(aggregated + mem_out)

        # 6. Output projection → (TOTAL,)
        logits = self.W_out @ aggregated + self.b_out
        return _softmax(logits)

    # ------------------------------------------------------------------
    # Online weight update (gradient-free EMA toward better checkpoint)
    # ------------------------------------------------------------------

    def checkpoint_if_better(self, score: float) -> None:
        """Store current weights if this is the best score seen so far."""
        if score > self._best_score:
            self._best_score = score
            self._best_state = {
                "input_proj": self.input_proj.copy(),
                "input_bias": self.input_bias.copy(),
                "W_out": self.W_out.copy(),
                "b_out": self.b_out.copy(),
            }

    def ema_toward_best(self, alpha: float = 0.05) -> None:
        """Blend current weights toward best-known checkpoint."""
        if self._best_state is None:
            return
        for key in ("input_proj", "input_bias", "W_out", "b_out"):
            cur = getattr(self, key)
            best = self._best_state[key]
            setattr(self, key, (1 - alpha) * cur + alpha * best)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            input_proj=self.input_proj,
            input_bias=self.input_bias,
            W_out=self.W_out,
            b_out=self.b_out,
            Wq=self.attn.Wq,
            Wk=self.attn.Wk,
            Wv=self.attn.Wv,
            Wo=self.attn.Wo,
            ff_W1=self.ff.W1,
            ff_b1=self.ff.b1,
            ff_W2=self.ff.W2,
            ff_b2=self.ff.b2,
            ca_Wq=self.cross_attn.Wq,
            ca_Wk=self.cross_attn.Wk,
            ca_Wv=self.cross_attn.Wv,
            ca_Wo=self.cross_attn.Wo,
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.input_proj = data["input_proj"]
        self.input_bias = data["input_bias"]
        self.W_out = data["W_out"]
        self.b_out = data["b_out"]
        # Infer d_model from saved weights and update sub-objects accordingly
        d_model_saved = int(self.input_proj.shape[1])
        n_heads_saved = self.attn.n_heads  # keep existing head count if compatible
        if d_model_saved % n_heads_saved != 0:
            n_heads_saved = 1  # fallback
        if d_model_saved != self.d_model:
            self.d_model = d_model_saved
            self.attn.d_model = d_model_saved
            self.attn.d_k = d_model_saved // n_heads_saved
            self.ff.b2 = np.zeros(d_model_saved)
        self.attn.Wq = data["Wq"]
        self.attn.Wk = data["Wk"]
        self.attn.Wv = data["Wv"]
        self.attn.Wo = data["Wo"]
        self.ff.W1 = data["ff_W1"]
        self.ff.b1 = data["ff_b1"]
        self.ff.W2 = data["ff_W2"]
        self.ff.b2 = data["ff_b2"]
        self.cross_attn.Wq = data["ca_Wq"]
        self.cross_attn.Wk = data["ca_Wk"]
        self.cross_attn.Wv = data["ca_Wv"]
        self.cross_attn.Wo = data["ca_Wo"]
        self.cross_attn.d_model = d_model_saved
        # Reset best-checkpoint so EMA doesn't mix old and new dims
        self._best_score = -np.inf
        self._best_state = None
