"""
Generative Memory-Augmented Predictor
=====================================
Architecture:
  - 12 signal providers (methods) feed scores into a MemoryBank
  - MemoryBank tracks per-method performance and adapts weights online
  - ExperienceReplay stores successful number patterns and reinforces them
  - GenerativeModel uses attention-weighted fusion + temperature sampling
  - Walk-forward: each draw updates memory, next draw uses updated memory

Pick 11 numbers from 1-80 to match 20 drawn numbers.
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

PICK = 11
TOTAL = 80
DRAW_SIZE = 20


# ═══════════════════════════════════════════════════════════════
# SIGNAL PROVIDERS (12 methods, compact)
# ═══════════════════════════════════════════════════════════════

def _presence(hist, window=None):
    h = hist[:window] if window else hist
    P = np.zeros((len(h), TOTAL), dtype=int)
    for t, row in enumerate(h):
        for num in row:
            if 1 <= num <= TOTAL:
                P[t, num - 1] = 1
    return P


def sig_info_theory(hist, window=80):
    P = _presence(hist, window)
    n = P.shape[0]
    mp = np.clip(P.mean(0), 1e-6, 1 - 1e-6)
    scores = np.zeros(TOTAL)
    for i in range(TOTAL):
        ent = -mp[i] * np.log2(mp[i]) - (1 - mp[i]) * np.log2(1 - mp[i])
        w1, w2, w3 = P[:n // 4, i].mean(), P[n // 4:n // 2, i].mean(), P[n // 2:, i].mean()
        mom = 2 * w1 + w2 - 0.5 * w3
        mi = 0.0
        for j in range(TOTAL):
            if j == i:
                continue
            p_ij = np.mean(P[:, i] * P[:, j])
            if p_ij > 1e-8:
                mi += p_ij * np.log2(p_ij / (mp[i] * mp[j] + 1e-12) + 1e-12)
        scores[i] = 0.3 * ent + 0.4 * mom + 0.3 * mi / TOTAL
    return scores


def sig_markov(hist, order=3):
    P = _presence(hist)[::-1]
    n = P.shape[0]
    tc = np.zeros((TOTAL, 2 ** order, 2))
    for i in range(TOTAL):
        for t in range(order, n):
            s = sum(P[t - order + k, i] * (2 ** (order - 1 - k)) for k in range(order))
            tc[i, int(s), P[t, i]] += 1
    Pr = _presence(hist)
    pred = np.zeros(TOTAL)
    for i in range(TOTAL):
        s = sum(Pr[k, i] * (2 ** (order - 1 - k)) for k in range(min(order, Pr.shape[0])))
        tot = tc[i, int(s), 0] + tc[i, int(s), 1]
        pred[i] = tc[i, int(s), 1] / tot if tot > 0 else 0.25
    return pred


def sig_graph(hist, window=100):
    P = _presence(hist, window)
    nf = P.mean(0)
    adj = np.zeros((TOTAL, TOTAL))
    for row in hist[:window]:
        ns = [x - 1 for x in row if 1 <= x <= TOTAL]
        for a in ns:
            for b in ns:
                if a != b:
                    adj[a, b] += 1
    adj /= max(window, 1)
    deg = adj.sum(1)
    cent = deg / (deg.max() + 1e-8)
    ra = np.zeros((TOTAL, TOTAL))
    for t, row in enumerate(hist[:30]):
        w = np.exp(-0.05 * t)
        ns = [x - 1 for x in row if 1 <= x <= TOTAL]
        for a in ns:
            for b in ns:
                if a != b:
                    ra[a, b] += w
    rs = ra.sum(1)
    rs /= (rs.max() + 1e-8)
    return 0.35 * nf + 0.30 * cent + 0.35 * rs


def sig_bayesian(hist, decay=0.995):
    a = np.ones(TOTAL)
    b = np.ones(TOTAL)
    for t in range(len(hist) - 1, -1, -1):
        ap = set(int(x) for x in hist[t] if 1 <= x <= TOTAL)
        a = 1 + decay * (a - 1)
        b = 1 + decay * (b - 1)
        for i in range(TOTAL):
            if (i + 1) in ap:
                a[i] += 1
            else:
                b[i] += 1
    pm = a / (a + b)
    pv = (a * b) / ((a + b) ** 2 * (a + b + 1))
    rng = np.random.default_rng(42)
    ts = rng.beta(a, b)
    return 0.4 * pm + 0.2 * (1 - pv / (pv.max() + 1e-8)) + 0.4 * ts


def sig_poisson(hist):
    scores = np.zeros(TOTAL)
    n = len(hist)
    for i in range(TOTAL):
        num = i + 1
        gaps = []
        last = -1
        for t in range(n - 1, -1, -1):
            if num in hist[t]:
                if last >= 0:
                    gaps.append(last - t)
                last = t
        if len(gaps) < 3:
            scores[i] = 0.25
            continue
        lam = np.mean(gaps)
        cur_gap = 0
        for t in range(len(hist)):
            if num in hist[t]:
                break
            cur_gap += 1
        scores[i] = 0.6 * (1 - np.exp(-cur_gap / max(lam, 1e-6))) + 0.4 / (1 + np.std(gaps) / (lam + 1e-6))
    return scores


def sig_svd(hist, window=200, rank=15):
    P = _presence(hist, window).astype(float)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    r = min(rank, len(S))
    recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    w = np.exp(-0.02 * np.arange(min(30, recon.shape[0])))
    w /= w.sum()
    pred = sum(w[t] * recon[t] for t in range(min(30, recon.shape[0])))
    return np.clip(pred, 0, 1)


def sig_ucb(hist):
    n = len(hist)
    rewards = np.zeros(TOTAL)
    counts = np.zeros(TOTAL) + 1e-6
    for t in range(n - 1, -1, -1):
        ap = set(int(x) for x in hist[t] if 1 <= x <= TOTAL)
        rw = np.exp(-0.005 * t)
        for i in range(TOTAL):
            counts[i] += rw
            if (i + 1) in ap:
                rewards[i] += rw
    avg = rewards / counts
    return avg + 0.5 * np.sqrt(2 * np.log(counts.sum()) / counts)


def sig_fourier(hist, window=500):
    P = _presence(hist, window)[::-1].astype(float)
    scores = np.zeros(TOTAL)
    for i in range(TOTAL):
        sig = P[:, i]
        n = len(sig)
        if n < 20:
            scores[i] = 0.25
            continue
        fft = np.fft.rfft(sig - sig.mean())
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)
        if len(power) > 1:
            di = np.argmax(power[1:]) + 1
            phase = np.angle(fft[di])
            pred = np.cos(2 * np.pi * freqs[di] * n + phase)
            sr = power[di] / (power[1:].sum() + 1e-8)
            scores[i] = 0.3 * sig[-10:].mean() + 0.3 * max(0, pred) + 0.4 * sr
        else:
            scores[i] = sig.mean()
    return scores


def sig_knn(hist, k=7, fw=10):
    n = len(hist)

    def ctx(h, s):
        rc = h[s:s + fw]
        f = np.zeros(TOTAL)
        for r in rc:
            for num in r:
                if 1 <= num <= TOTAL:
                    f[num - 1] += 1
        return f / max(len(rc), 1)

    cur = ctx(hist, 0)
    sims = []
    for t in range(1, n - fw - 1):
        c = ctx(hist, t)
        dot = np.dot(cur, c)
        nm = np.linalg.norm(cur) * np.linalg.norm(c)
        sim = dot / (nm + 1e-8)
        nd = hist[t - 1] if t > 0 else hist[0]
        sims.append((sim, nd))
    sims.sort(key=lambda x: -x[0])
    scores = np.zeros(TOTAL)
    for sim, draw in sims[:k]:
        for num in draw:
            if 1 <= num <= TOTAL:
                scores[num - 1] += sim
    return scores


def sig_recurrent(hist, hidden=32, window=120, lr=0.01, epochs=12):
    P = _presence(hist, window)[::-1].astype(float)
    n = P.shape[0]
    rng = np.random.default_rng(77)
    Wh = rng.normal(0, 0.1, (hidden, hidden))
    Wx = rng.normal(0, 0.1, (hidden, TOTAL))
    Wy = rng.normal(0, 0.1, (TOTAL, hidden))
    bh, by = np.zeros(hidden), np.zeros(TOTAL)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def tanh(x):
        return np.tanh(np.clip(x, -10, 10))

    for _ in range(epochs):
        h = np.zeros(hidden)
        for t in range(n - 1):
            h = tanh(Wh @ h + Wx @ P[t] + bh)
            y = sigmoid(Wy @ h + by)
            err = y - P[t + 1]
            Wy -= lr * np.outer(err, h)
            by -= lr * err
            dh = (Wy.T @ err) * (1 - h ** 2)
            Wx -= lr * np.outer(dh, P[t])
            bh -= lr * dh

    h = np.zeros(hidden)
    Pc = _presence(hist, window)[::-1].astype(float)
    for t in range(Pc.shape[0]):
        h = tanh(Wh @ h + Wx @ Pc[t] + bh)
    return sigmoid(Wy @ h + by)


def sig_genetic_scores(hist, pop_sz=150, gens=100, window=60):
    """Returns score vector (not just top-N) based on GA frequency."""
    rng = np.random.default_rng(2024)
    recent = hist[:window]

    def fit(c):
        s = set(c)
        return sum(np.exp(-0.03 * t) * len(s & set(int(x) for x in r)) for t, r in enumerate(recent))

    pop = [sorted(rng.choice(np.arange(1, 81), PICK, replace=False)) for _ in range(pop_sz)]
    for _ in range(gens):
        fs = np.array([fit(c) for c in pop])
        new = [pop[i][:] for i in np.argsort(fs)[::-1][:5]]
        while len(new) < pop_sz:
            i1, i2 = rng.choice(len(pop), 2, replace=False)
            p1 = pop[i1] if fs[i1] > fs[i2] else pop[i2]
            i3, i4 = rng.choice(len(pop), 2, replace=False)
            p2 = pop[i3] if fs[i3] > fs[i4] else pop[i4]
            an = sorted(set(p1) | set(p2))
            child = sorted(rng.choice(an, PICK, replace=False)) if len(an) >= PICK else sorted(
                an + list(set(range(1, 81)) - set(an))[:PICK - len(an)])
            if rng.random() < 0.15:
                im = rng.integers(0, PICK)
                cs = list(set(range(1, 81)) - set(child))
                if cs:
                    child[im] = rng.choice(cs)
                    child = sorted(child)
            new.append(child)
        pop = new

    # Convert top population to score vector
    fs = np.array([fit(c) for c in pop])
    top_pop = [pop[i] for i in np.argsort(fs)[::-1][:30]]
    scores = np.zeros(TOTAL)
    for c in top_pop:
        for num in c:
            scores[num - 1] += 1
    return scores


ALL_SIGNAL_PROVIDERS = {
    'info_theory': sig_info_theory,
    'markov3': lambda h: sig_markov(h, 3),
    'markov5': lambda h: sig_markov(h, 5),
    'graph': sig_graph,
    'bayesian': sig_bayesian,
    'poisson': sig_poisson,
    'svd': sig_svd,
    'ucb': sig_ucb,
    'fourier': sig_fourier,
    'knn': sig_knn,
    'recurrent': sig_recurrent,
    'genetic': sig_genetic_scores,
}


# ═══════════════════════════════════════════════════════════════
# MEMORY BANK
# ═══════════════════════════════════════════════════════════════

class MemoryBank:
    """
    Stores:
      - Per-method adaptive weights (updated after each draw)
      - Per-number "attention" scores (reinforced by hits)
      - Method correlation matrix (to diversify)
    """

    def __init__(self, method_names: List[str], lr: float = 0.15, decay: float = 0.92):
        self.method_names = list(method_names)
        self.n_methods = len(method_names)
        self.lr = lr
        self.decay = decay

        # Adaptive method weights (start uniform)
        self.method_weights = np.ones(self.n_methods) / self.n_methods

        # Per-number attention memory (80,) — reinforced by hits
        self.number_attention = np.ones(TOTAL) * 0.5

        # Per-method hit history for online weight update
        self.method_hit_history: Dict[str, List[float]] = {m: [] for m in method_names}

        # Experience replay buffer: stores (predicted_set, actual_set, hits)
        self.replay_buffer: List[Tuple[List[int], set, int]] = []
        self.max_replay = 200

        # Number co-occurrence memory: which pairs succeed together
        self.pair_success = np.zeros((TOTAL, TOTAL), dtype=float)

    def update_after_draw(
        self,
        method_scores: Dict[str, np.ndarray],
        actual: set,
        predicted_sets: Optional[List[List[int]]] = None,
    ):
        """Update memory after observing a draw result."""
        actual_vec = np.zeros(TOTAL)
        for num in actual:
            if 1 <= num <= TOTAL:
                actual_vec[num - 1] = 1

        # 1) Update method weights based on how well each method's scores
        #    correlated with actual outcome
        method_perfs = []
        for i, name in enumerate(self.method_names):
            if name not in method_scores:
                method_perfs.append(0.0)
                continue
            sc = method_scores[name]
            sc_norm = sc - sc.min()
            mx = sc_norm.max()
            if mx > 0:
                sc_norm /= mx
            # Rank correlation: how many of the method's top-20 were in actual?
            top20 = set(int(x) + 1 for x in np.argsort(sc)[::-1][:DRAW_SIZE])
            overlap = len(top20 & actual) / DRAW_SIZE
            method_perfs.append(overlap)
            self.method_hit_history[name].append(overlap)

        perfs = np.array(method_perfs)
        if perfs.max() > 0:
            # Softmax update
            exp_p = np.exp(3.0 * (perfs - perfs.mean()))
            target_w = exp_p / exp_p.sum()
            self.method_weights = (1 - self.lr) * self.method_weights + self.lr * target_w

        # Normalize
        self.method_weights = np.maximum(self.method_weights, 0.02)
        self.method_weights /= self.method_weights.sum()

        # 2) Update number attention
        self.number_attention *= self.decay
        for num in actual:
            if 1 <= num <= TOTAL:
                self.number_attention[num - 1] += 1.0

        # 3) Update pair success memory
        actual_list = sorted(actual)
        for a in actual_list:
            for b in actual_list:
                if a != b and 1 <= a <= TOTAL and 1 <= b <= TOTAL:
                    self.pair_success[a - 1, b - 1] += 0.1
        self.pair_success *= 0.98

        # 4) Experience replay
        if predicted_sets:
            for pset in predicted_sets:
                h = len(set(pset) & actual)
                self.replay_buffer.append((pset, actual, h))
            if len(self.replay_buffer) > self.max_replay:
                self.replay_buffer = self.replay_buffer[-self.max_replay:]

    def get_method_weights(self) -> np.ndarray:
        return self.method_weights.copy()

    def get_number_attention(self) -> np.ndarray:
        att = self.number_attention.copy()
        att = att - att.min()
        mx = att.max()
        if mx > 0:
            att /= mx
        return att

    def get_pair_boost(self) -> np.ndarray:
        """Per-number boost from pair co-success."""
        boost = self.pair_success.sum(axis=1)
        mx = boost.max()
        if mx > 0:
            boost /= mx
        return boost


# ═══════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY
# ═══════════════════════════════════════════════════════════════

class ExperienceReplay:
    """Extracts patterns from successful predictions in memory."""

    @staticmethod
    def extract_success_pattern(memory: MemoryBank, min_hits: int = 4) -> np.ndarray:
        """
        From replay buffer, extract numbers that appeared frequently
        in high-hit predictions.
        """
        scores = np.zeros(TOTAL)
        if not memory.replay_buffer:
            return scores

        for pset, actual, hits in memory.replay_buffer:
            if hits >= min_hits:
                weight = (hits - min_hits + 1) ** 2
                matched = set(pset) & actual
                for num in matched:
                    if 1 <= num <= TOTAL:
                        scores[num - 1] += weight
                # Also boost numbers that were predicted AND hit
                for num in pset:
                    if num in actual and 1 <= num <= TOTAL:
                        scores[num - 1] += weight * 0.5

        mx = scores.max()
        if mx > 0:
            scores /= mx
        return scores


# ═══════════════════════════════════════════════════════════════
# GENERATIVE MODEL
# ═══════════════════════════════════════════════════════════════

class GenerativeModel:
    """
    Generates number sets by:
    1. Fusing all method scores with memory-adaptive weights
    2. Adding number attention and pair boost from memory
    3. Adding experience replay signal
    4. Temperature-controlled sampling with diversity enforcement
    """

    def __init__(self, memory: MemoryBank):
        self.memory = memory

    def compute_fused_distribution(
        self,
        method_scores: Dict[str, np.ndarray],
        hist: np.ndarray,
    ) -> np.ndarray:
        """Compute the final generative distribution over 80 numbers."""
        weights = self.memory.get_method_weights()
        attention = self.memory.get_number_attention()
        pair_boost = self.memory.get_pair_boost()
        replay_signal = ExperienceReplay.extract_success_pattern(self.memory)

        # Normalize each method's scores to [0,1]
        normed = {}
        for name, sc in method_scores.items():
            s = sc - sc.min()
            mx = s.max()
            normed[name] = s / mx if mx > 0 else np.zeros(TOTAL)

        # Weighted fusion
        fused = np.zeros(TOTAL)
        for i, name in enumerate(self.memory.method_names):
            if name in normed:
                fused += weights[i] * normed[name]

        # Add memory signals
        fused = (
            0.45 * fused
            + 0.20 * attention
            + 0.15 * pair_boost
            + 0.20 * replay_signal
        )

        # Final normalization
        fused = np.maximum(fused, 1e-8)
        return fused

    def generate_sets(
        self,
        fused_dist: np.ndarray,
        n_sets: int = 20,
        seed: int = 42,
    ) -> List[List[int]]:
        """Generate diverse cover sets from the fused distribution."""
        rng = np.random.default_rng(seed)
        usage = np.zeros(TOTAL)
        sets = []
        temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

        for i in range(n_sets):
            # Reduce probability for already-used numbers (diversity)
            adj = fused_dist / (1.0 + 0.35 * usage)
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


# ═══════════════════════════════════════════════════════════════
# MAIN: Walk-forward with memory
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if c.startswith('红球')]
    issues = df['期数'].astype(str).tolist()
    draws = df[num_cols].to_numpy(dtype=int)
    return issues, draws


def run_walk_forward_with_memory(
    issues, draws, target_issue, n_eval=20, n_warmup=10, n_cover=20,
):
    """
    Walk-forward evaluation with memory:
    1. Warm up memory on n_warmup draws before evaluation window
    2. Evaluate on n_eval draws, updating memory after each
    """
    idx = issues.index(target_issue)
    total_window = n_warmup + n_eval

    if idx + total_window >= len(draws):
        total_window = len(draws) - idx - 30
        n_eval = min(n_eval, total_window - n_warmup)

    method_names = list(ALL_SIGNAL_PROVIDERS.keys())
    memory = MemoryBank(method_names)
    gen = GenerativeModel(memory)

    print(f'TARGET: {target_issue}')
    print(f'WARMUP: {n_warmup} draws, EVAL: {n_eval} draws')
    print(f'COVER: {n_cover} sets per draw')
    print('=' * 80)

    # Phase 1: Warmup (update memory without evaluation)
    print('\n--- WARMUP PHASE ---')
    for step in range(n_warmup):
        t_idx = idx + n_eval + step  # older draws
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])

        scores = {}
        for name, func in ALL_SIGNAL_PROVIDERS.items():
            scores[name] = func(t_hist)

        memory.update_after_draw(scores, t_actual)
        t_issue = issues[t_idx]
        print(f'  warmup {step + 1:>2}/{n_warmup}: {t_issue}')

    print(f'\nMemory weights after warmup:')
    for i, name in enumerate(method_names):
        print(f'  {name:<15} {memory.method_weights[i]:.4f}')

    # Phase 2: Evaluation with memory
    print('\n--- EVALUATION PHASE (with memory) ---')
    eval_results = {'single': [], 'cover_best': [], 'cover_all': []}
    random_results = []
    rng = np.random.default_rng(42)

    for step in range(n_eval):
        t_idx = idx + n_eval - 1 - step  # newest to oldest within eval window
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        t_issue = issues[t_idx]

        # Get all method scores
        scores = {}
        for name, func in ALL_SIGNAL_PROVIDERS.items():
            scores[name] = func(t_hist)

        # Generate fused distribution
        fused = gen.compute_fused_distribution(scores, t_hist)

        # Single best set (top-PICK)
        single = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:PICK])
        single_hits = len(set(single) & t_actual)

        # Cover sets
        cover = gen.generate_sets(fused, n_sets=n_cover, seed=42 + t_idx)
        cover_hits = [len(set(s) & t_actual) for s in cover]
        best_cover_h = max(cover_hits)

        eval_results['single'].append(single_hits)
        eval_results['cover_best'].append(best_cover_h)
        eval_results['cover_all'].append(cover_hits)

        # Random baseline (best of n_cover)
        rand_best = max(
            len(set(rng.choice(np.arange(1, 81), PICK, replace=False)) & t_actual)
            for _ in range(n_cover)
        )
        random_results.append(rand_best)

        # Update memory with this draw's results
        memory.update_after_draw(scores, t_actual, predicted_sets=cover)

        matched = sorted(set(single) & t_actual)
        best_set = cover[np.argmax(cover_hits)]
        best_matched = sorted(set(best_set) & t_actual)

        print(
            f'  {t_issue}: single={single_hits}/{PICK} '
            f'cover_best={best_cover_h}/{PICK} '
            f'rand_best={rand_best} '
            f'| single_match={matched}'
        )
        if best_cover_h >= 6:
            print(
                f'    ★ cover_best_set: {" ".join(f"{x:02d}" for x in sorted(best_set))} '
                f'match={best_matched}'
            )

    # Summary
    sa = np.array(eval_results['single'])
    ca = np.array(eval_results['cover_best'])
    ra = np.array(random_results)

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'{"Mode":<25} {"Avg":>6} {"Max":>4} {"P>=4":>6} {"P>=5":>6} {"P>=6":>6} {"P>=7":>6}')
    print('-' * 65)
    for label, arr in [('Single (memory)', sa), ('Cover best-of-20', ca), ('Random best-of-20', ra)]:
        print(
            f'{label:<25} {arr.mean():>6.2f} {arr.max():>4d} '
            f'{(arr >= 4).mean():>6.2f} {(arr >= 5).mean():>6.2f} '
            f'{(arr >= 6).mean():>6.2f} {(arr >= 7).mean():>6.2f}'
        )

    print(f'\nFinal memory weights:')
    for i, name in enumerate(method_names):
        w = memory.method_weights[i]
        bar = '█' * int(w * 80)
        print(f'  {name:<15} {w:.4f} {bar}')

    # Final prediction for target issue
    print('\n' + '=' * 80)
    print(f'FINAL PREDICTION FOR {target_issue}')
    print('=' * 80)

    final_hist = draws[idx + 1:]
    final_scores = {}
    for name, func in ALL_SIGNAL_PROVIDERS.items():
        final_scores[name] = func(final_hist)

    fused_final = gen.compute_fused_distribution(final_scores, final_hist)
    final_sets = gen.generate_sets(fused_final, n_sets=n_cover, seed=9999)
    actual_final = set(int(x) for x in draws[idx])

    print(f'\nTop {PICK} by fused score: {sorted(int(x)+1 for x in np.argsort(fused_final)[::-1][:PICK])}')
    print(f'\n{n_cover} Generated Sets:')
    final_hits = []
    for i, s in enumerate(final_sets, 1):
        h = len(set(s) & actual_final)
        final_hits.append(h)
        m = sorted(set(s) & actual_final)
        mark = '★' if h >= 6 else '●' if h >= 5 else ' '
        print(f'{mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in sorted(s))}  match={m}')

    fa = np.array(final_hits)
    print(f'\nbest={fa.max()}, avg={fa.mean():.2f}, P>=5={(fa >= 5).mean():.2f}, P>=6={(fa >= 6).mean():.2f}')
    print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual_final))}')

    return memory, gen


if __name__ == '__main__':
    csv_path = 'data/data.csv'
    target = '2026040'
    issues, draws = load_data(csv_path)
    run_walk_forward_with_memory(issues, draws, target, n_eval=20, n_warmup=15, n_cover=20)
