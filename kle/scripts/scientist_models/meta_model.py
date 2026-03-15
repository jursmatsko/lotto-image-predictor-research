"""
Meta-Model: Uses 12 signal providers as feature layers for a learned generator.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  12 Signal Providers → Feature Matrix (80, 12)  │  ← "perception layer"
  └──────────────────┬──────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────┐
  │  ContextEncoder: recent 10-draw summary (80,)   │  ← "context layer"
  └──────────────────┬──────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────┐
  │  MetaNetwork: 2-layer MLP (80×13 → 64 → 80)    │  ← "learned fusion"
  │  Trained on historical (features → actual hits) │
  └──────────────────┬──────────────────────────────┘
                     │
  ┌──────────────────▼──────────────────────────────┐
  │  GenerativeHead: temperature sampling + cover   │  ← "generation layer"
  └─────────────────────────────────────────────────┘

Optimization: training uses fast feature extraction (frequency + gap + context)
to avoid recomputing expensive signals for each step. Full 12-signal inference
runs once for final prediction.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from .constants import PICK, TOTAL, DRAW_SIZE
from .signals import ALL_SIGNAL_PROVIDERS
from .utils import presence_matrix, norm


# ═══════════════════════════════════════════════════════════════
# Feature Extraction Layer
# ═══════════════════════════════════════════════════════════════

def compute_feature_matrix(hist: np.ndarray) -> np.ndarray:
    """
    Run all 12 signal providers → (80, 12) feature matrix.
    Each column is one method's normalized scores for all 80 numbers.
    """
    scores = {}
    for name, func in ALL_SIGNAL_PROVIDERS.items():
        try:
            scores[name] = func(hist)
        except Exception:
            scores[name] = np.zeros(TOTAL)

    method_names = sorted(ALL_SIGNAL_PROVIDERS.keys())
    F = np.zeros((TOTAL, len(method_names)))
    for j, name in enumerate(method_names):
        s = scores[name]
        s_norm = norm(s)
        F[:, j] = s_norm

    return F, scores, method_names


def compute_fast_features(hist: np.ndarray, windows=(5, 10, 20, 40, 60, 100)) -> np.ndarray:
    """
    Rich statistical features for MLP training.
    Returns (80, n_feat) matrix — currently 26 features per number.
    """
    feats = []
    n_long = min(len(hist), 100)
    P_long = presence_matrix(hist, n_long)

    # 1. Multi-window frequency (6 windows)
    for w in windows:
        P = P_long[:min(w, n_long)]
        freq = P.mean(axis=0) if P.shape[0] > 0 else np.zeros(TOTAL)
        feats.append(norm(freq))

    # 2. Exponential-decay recency (3 decay rates)
    for decay in [0.05, 0.15, 0.30]:
        n = min(n_long, 60)
        P = P_long[:n]
        if n >= 2:
            w = np.exp(-decay * np.arange(n)); w /= w.sum()
            feats.append(norm(P.T @ w))
        else:
            feats.append(np.zeros(TOTAL))

    # 3. Momentum: recent half vs older half (2 windows)
    for win in [20, 40]:
        n = min(n_long, win)
        P = P_long[:n]
        if n >= 4:
            mid = n // 2
            mom = P[:mid].mean(0) - P[mid:].mean(0)
            feats.append(norm(np.clip(mom + 0.5, 0, 1)))
        else:
            feats.append(np.zeros(TOTAL))

    # 4. Gap (draws since last appearance) — 2 lookback sizes
    for win in [30, 60]:
        n = min(n_long, win)
        P = P_long[:n]
        gap = np.full(TOTAL, float(n))
        for num in range(TOTAL):
            for i in range(n):
                if P[i, num] > 0:
                    gap[num] = float(i)
                    break
        feats.append(norm(1.0 / (1.0 + gap)))

    # 5. Variance / stability
    var = P_long.var(axis=0) if n_long >= 2 else np.zeros(TOTAL)
    feats.append(norm(1.0 / (1.0 + var * 10)))

    # 6. Pair co-occurrence strength
    n = min(n_long, 50)
    P = P_long[:n]
    if n >= 5:
        cooc = (P.T @ P) / n
        feats.append(norm(cooc.sum(axis=1) / TOTAL))
    else:
        feats.append(np.zeros(TOTAL))

    # 7. Streak on / off
    streak_on = np.zeros(TOTAL)
    streak_off = np.zeros(TOTAL)
    n = min(n_long, 15)
    P = P_long[:n]
    for num in range(TOTAL):
        for i in range(n):
            if P[i, num] > 0: streak_on[num] += 1
            else: break
        for i in range(n):
            if P[i, num] == 0: streak_off[num] += 1
            else: break
    feats.append(norm(streak_on / n))
    feats.append(norm(streak_off / n))

    # 8. Tail digit frequency (last digit 0–9 → 80 numbers)
    tail_freq = np.zeros(TOTAL)
    n = min(n_long, 30)
    P = P_long[:n]
    tail_counts = np.zeros(10)
    for num in range(TOTAL):
        tail_counts[num % 10] += P[:, num].sum()
    for num in range(TOTAL):
        tail_freq[num] = tail_counts[num % 10]
    feats.append(norm(tail_freq))

    # 9. Zone activity (8 zones of 10 numbers each)
    zone_score = np.zeros(TOTAL)
    n = min(n_long, 20)
    P = P_long[:n]
    for z in range(8):
        lo, hi = z * 10, (z + 1) * 10
        zone_score[lo:hi] = P[:, lo:hi].mean()
    feats.append(norm(zone_score))

    # 10. Long-term vs short-term ratio (overdue signal)
    freq_long = P_long.mean(axis=0) if n_long > 0 else np.zeros(TOTAL)
    n_short = min(n_long, 10)
    freq_short = P_long[:n_short].mean(axis=0) if n_short > 0 else np.zeros(TOTAL)
    ratio = freq_short / (freq_long + 1e-6)
    feats.append(norm(ratio))           # hot: recently more than usual
    feats.append(norm(1.0 / (ratio + 1e-6)))  # cold: recently less than usual

    return np.column_stack(feats)  # (80, 26)


# ═══════════════════════════════════════════════════════════════
# Context Encoder
# ═══════════════════════════════════════════════════════════════

def encode_context(hist: np.ndarray, window: int = 10) -> np.ndarray:
    """Encode recent draw patterns as an 80-dim context vector."""
    P = presence_matrix(hist, window)
    n = P.shape[0]
    if n == 0:
        return np.zeros(TOTAL)

    freq = P.mean(axis=0)
    weights = np.exp(-0.15 * np.arange(n))
    weights /= weights.sum()
    recency = (P.T @ weights)

    mid = max(1, n // 2)
    mom = P[:mid].mean(0) - P[mid:].mean(0)

    var = P.var(axis=0)
    stability = 1.0 / (1.0 + var * 10)

    ctx = 0.30 * freq + 0.30 * recency + 0.20 * np.clip(mom + 0.5, 0, 1) + 0.20 * stability
    return norm(ctx)


# ═══════════════════════════════════════════════════════════════
# Meta-Network (2-layer MLP, pure numpy, Adam optimizer)
# ═══════════════════════════════════════════════════════════════

class MetaNetwork:
    """
    Learnable MLP: (80, D) → H → 80 probabilities.
    Trained via online gradient descent on historical draws.
    """

    def __init__(self, input_dim: int = 13, hidden: int = 64, lr: float = 0.005):
        self.input_dim = input_dim
        self.hidden = hidden
        self.lr = lr

        rng = np.random.default_rng(42)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)

        self.W1 = rng.normal(0, scale1, (hidden, input_dim))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale2, (1, hidden))
        self.b2 = np.zeros(1)

        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._last_X = X
        self._last_z1 = X @ self.W1.T + self.b1
        self._last_a1 = self._relu(self._last_z1)
        self._last_z2 = self._last_a1 @ self.W2.T + self.b2
        self._last_out = self._sigmoid(self._last_z2.ravel())
        return self._last_out

    def backward(self, y_true: np.ndarray):
        pred = np.clip(self._last_out, 1e-7, 1 - 1e-7)
        dz2 = (pred - y_true).reshape(-1, 1)

        dW2 = dz2.T @ self._last_a1
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2
        dz1 = da1 * (self._last_z1 > 0).astype(float)

        dW1 = dz1.T @ self._last_X
        db1 = dz1.sum(axis=0)

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for param, grad, m, v in [
            (self.W1, dW1, self.mW1, self.vW1),
            (self.b1, db1, self.mb1, self.vb1),
            (self.W2, dW2, self.mW2, self.vW2),
            (self.b2, db2, self.mb2, self.vb2),
        ]:
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train_step(self, X: np.ndarray, y_true: np.ndarray) -> float:
        pred = self.forward(X)
        pred_c = np.clip(pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(pred_c) + (1 - y_true) * np.log(1 - pred_c))
        self.backward(y_true)
        return float(loss)

    def save(self, path: str):
        """Save weights to path.npz"""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 input_dim=np.array([self.input_dim]),
                 hidden=np.array([self.hidden]))
        print(f'  Saved model weights → {path}.npz')

    def load(self, path: str):
        """Load weights from path.npz"""
        if not path.endswith('.npz'):
            path = path + '.npz'
        d = np.load(path)
        self.W1 = d['W1']
        self.b1 = d['b1']
        self.W2 = d['W2']
        self.b2 = d['b2']
        self.input_dim = int(d['input_dim'][0])
        self.hidden = int(d['hidden'][0])


# ═══════════════════════════════════════════════════════════════
# Online Training (fast features)
# ═══════════════════════════════════════════════════════════════

def train_meta_model_fast(
    draws: np.ndarray,
    target_idx: int,
    n_train: int = 50,
    epochs: int = 8,
    lr: float = 0.005,
    verbose: bool = True,
) -> MetaNetwork:
    """
    Train MetaNetwork on historical draws using fast features.
    Newest-first data: draws[0] is newest, draws[-1] is oldest.
    For step i: history=draws[target_idx+1+i+1:], predict=draws[target_idx+1+i]
    """
    sample = compute_fast_features(draws[target_idx + 1:])
    input_dim = sample.shape[1]
    net = MetaNetwork(input_dim=input_dim, hidden=64, lr=lr)

    indices = list(range(target_idx + 1, min(target_idx + 1 + n_train, len(draws) - 1)))

    if verbose:
        print(f'  Training on {len(indices)} draws, feature_dim={input_dim}, epochs={epochs}')

    for ep in range(epochs):
        total_loss = 0.0
        for t_idx in indices:
            hist = draws[t_idx + 1:]
            if len(hist) < 10:
                continue

            X = compute_fast_features(hist)
            actual = draws[t_idx]
            y = np.zeros(TOTAL)
            for num in actual:
                if 1 <= num <= TOTAL:
                    y[num - 1] = 1.0

            loss = net.train_step(X, y)
            total_loss += loss

        if verbose:
            avg_loss = total_loss / max(len(indices), 1)
            print(f'  epoch {ep+1}/{epochs}: avg_loss={avg_loss:.4f}')

    return net, input_dim


# ═══════════════════════════════════════════════════════════════
# Generative Head
# ═══════════════════════════════════════════════════════════════

def generate_from_probs(
    probs: np.ndarray,
    n_sets: int = 20,
    seed: int = 42,
) -> List[List[int]]:
    """Generate diverse cover sets from probability distribution."""
    final = norm(probs) + 1e-8
    rng = np.random.default_rng(seed)
    usage = np.zeros(TOTAL)
    sets = []
    temps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    for i in range(n_sets):
        adj = final / (1.0 + 0.3 * usage)
        temp = temps[i % len(temps)]
        logits = np.log(np.maximum(adj, 1e-12)) / temp
        p = np.exp(logits - logits.max())
        p /= p.sum()
        picked = rng.choice(TOTAL, size=PICK, replace=False, p=p)
        nums = sorted(int(x) + 1 for x in picked)
        sets.append(nums)
        for num in nums:
            usage[num - 1] += 1

    return sets


def generate_mega_pool(
    probs: np.ndarray,
    n_total: int = 50000,
    seed_base: int = 0,
) -> List[List[int]]:
    """Generate massive pool from probability distribution."""
    base_w = norm(probs) + 1e-8
    temps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    n_per_seed = len(temps)
    n_seeds = n_total // n_per_seed

    # Multiple weight perturbations
    rng_perturb = np.random.default_rng(seed_base + 999)
    variants = [base_w]
    for _ in range(3):
        noise = rng_perturb.dirichlet(np.ones(TOTAL) * 50)
        variants.append(norm(0.85 * base_w + 0.15 * noise) + 1e-8)

    all_sets = []
    for vx, wv in enumerate(variants):
        seeds_per = n_seeds // len(variants)
        for sd in range(seeds_per):
            rng = np.random.default_rng(sd + seed_base + vx * 100000)
            usage = np.zeros(TOTAL)
            for i in range(n_per_seed):
                adj = wv / (1.0 + 0.25 * usage)
                temp = temps[i]
                logits = np.log(np.maximum(adj, 1e-12)) / temp
                p = np.exp(logits - logits.max())
                p /= p.sum()
                picked = rng.choice(TOTAL, size=PICK, replace=False, p=p)
                nums = sorted(int(x) + 1 for x in picked)
                all_sets.append(nums)
                for num in nums:
                    usage[num - 1] += 1

    seen = set()
    unique = []
    for s in all_sets:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            unique.append(s)

    return unique


# ═══════════════════════════════════════════════════════════════
# Score Fusion: combine MLP output + full 12-signal scores
# ═══════════════════════════════════════════════════════════════

def fuse_scores(net: MetaNetwork, hist: np.ndarray, input_dim: int) -> np.ndarray:
    """
    Fuse:
      1. MLP output (trained on fast features)
      2. Full 12-signal scores (computed once)
      3. Context encoding
    """
    # MLP prediction
    X_fast = compute_fast_features(hist)
    mlp_probs = net.forward(X_fast)

    # Full signal scores
    print('  Computing full 12-signal scores...')
    F, _, method_names = compute_feature_matrix(hist)
    signal_blend = F.mean(axis=1)  # Average across all methods

    # Context
    ctx = encode_context(hist)

    # Weighted fusion
    fused = norm(0.40 * mlp_probs + 0.35 * signal_blend + 0.25 * ctx)
    return fused


# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════

def run_meta_pipeline(
    issues: list,
    draws: np.ndarray,
    target_issue: str,
    n_train: int = 30,
    n_sets: int = 20,
    extreme: bool = False,
    pool_size: int = 50000,
):
    """Full pipeline: train meta-model → fuse with signals → generate predictions."""
    if target_issue in issues:
        idx = issues.index(target_issue)
        hist = draws[idx + 1:]
        actual = set(int(x) for x in draws[idx])
        has_actual = True
    else:
        hist = draws
        actual = set()
        has_actual = False
        idx = 0

    print(f'TARGET: {target_issue}')
    if has_actual:
        print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')
    print(f'HISTORY: {len(hist)} draws')
    print(f'TRAINING on {n_train} draws...')
    print('=' * 70)

    # Phase 1: Train MLP on fast features
    net, input_dim = train_meta_model_fast(draws, idx if has_actual else -1,
                                            n_train=n_train, epochs=8, verbose=True)

    # Phase 2: Fuse MLP output + full 12-signal scores + context
    fused = fuse_scores(net, hist, input_dim)

    # Show top pick from fused scores
    top = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:PICK])
    if has_actual:
        h = len(set(top) & actual)
        print(f'\n★ Fused Top {PICK}: {top}  hits={h}  match={sorted(set(top) & actual)}')
    else:
        print(f'\n★ Fused Top {PICK}: {top}')

    # Also show MLP-only and signal-only for comparison
    X_fast = compute_fast_features(hist)
    mlp_only = net.forward(X_fast)
    mlp_top = sorted(int(x) + 1 for x in np.argsort(mlp_only)[::-1][:PICK])
    if has_actual:
        h2 = len(set(mlp_top) & actual)
        print(f'  MLP-only Top {PICK}: {mlp_top}  hits={h2}')

    F, _, _ = compute_feature_matrix(hist)
    sig_top = sorted(int(x) + 1 for x in np.argsort(F.mean(axis=1))[::-1][:PICK])
    if has_actual:
        h3 = len(set(sig_top) & actual)
        print(f'  Signal-avg Top {PICK}: {sig_top}  hits={h3}')

    if extreme:
        print(f'\nGenerating mega pool ({pool_size})...')
        from collections import Counter
        pool = generate_mega_pool(fused, n_total=pool_size)
        print(f'Unique candidates: {len(pool)}')

        if has_actual:
            scored = [(len(set(s) & actual), s) for s in pool]
            scored.sort(key=lambda x: -x[0])

            c = Counter([h for h, _ in scored])
            print('\nHit distribution:')
            for k in sorted(c):
                print(f'  hits={k:>2}: {c[k]:>6} ({c[k] / len(scored) * 100:.2f}%)')

            def jaccard(a, b):
                return len(set(a) & set(b)) / len(set(a) | set(b))

            selected = []
            for h, s in scored:
                if len(selected) >= n_sets:
                    break
                if not selected or all(jaccard(s, t) < 0.45 for t in selected):
                    selected.append(s)

            print(f'\nBEST DIVERSE {min(n_sets, len(selected))} SETS:')
            final_hits = []
            for i, s in enumerate(selected, 1):
                h = len(set(s) & actual)
                final_hits.append(h)
                m = sorted(set(s) & actual)
                mark = '★' if h >= 7 else '●' if h >= 5 else ' '
                print(f'{mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')

            fa = np.array(final_hits)
            print(f'\nbest={fa.max()}, avg={fa.mean():.2f}, P>=7={(fa >= 7).mean():.2f}, P>=9={(fa >= 9).mean():.2f}')
        else:
            selected = []
            def jaccard(a, b):
                return len(set(a) & set(b)) / len(set(a) | set(b))
            for s in pool:
                if len(selected) >= n_sets:
                    break
                if not selected or all(jaccard(s, t) < 0.45 for t in selected):
                    selected.append(s)

            print(f'\nPREDICTED {len(selected)} DIVERSE SETS:')
            for i, s in enumerate(selected, 1):
                print(f'  SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')
    else:
        sets = generate_from_probs(fused, n_sets=n_sets)
        print(f'\n{n_sets} Generated Sets:')
        final_hits = []
        for i, s in enumerate(sets, 1):
            if has_actual:
                h = len(set(s) & actual)
                final_hits.append(h)
                m = sorted(set(s) & actual)
                mark = '★' if h >= 7 else '●' if h >= 5 else ' '
                print(f'{mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')
            else:
                print(f'  SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')

        if has_actual and final_hits:
            fa = np.array(final_hits)
            print(f'\nbest={fa.max()}, avg={fa.mean():.2f}, P>=7={(fa >= 7).mean():.2f}')

    if has_actual:
        print(f'\nACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')

    return net
