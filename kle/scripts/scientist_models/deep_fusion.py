"""
Deep Fusion: Stacked architecture that learns cross-signal interactions.

Key improvements over simple meta_model:
  1. Cross-signal features: for each number, how many methods agree on it?
  2. Method-reliability weighting learned from recent accuracy
  3. Stacked 2-stage: Stage1 (fast features) → Stage2 (signal + Stage1 output)
  4. Targeted generation: oversample from per-method top numbers
  5. Agreement-biased sampling: numbers endorsed by many methods get boosted
"""
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from .constants import PICK, TOTAL, DRAW_SIZE
from .signals import ALL_SIGNAL_PROVIDERS
from .utils import presence_matrix, norm
from .meta_model import (
    MetaNetwork, compute_fast_features, encode_context,
    compute_feature_matrix, generate_mega_pool, generate_from_probs,
    train_meta_model_fast,
)


# ═══════════════════════════════════════════════════════════════
# Cross-Signal Feature Engineering
# ═══════════════════════════════════════════════════════════════

def cross_signal_features(F: np.ndarray) -> np.ndarray:
    """
    Extract cross-signal interaction features from (80, 12) feature matrix.
    Returns additional (80, K) features.
    """
    n_methods = F.shape[1]
    feats = []

    # Agreement count: for each number, how many methods put it in top-K?
    for top_k in [PICK, 15, 20, 30]:
        agreement = np.zeros(TOTAL)
        for j in range(n_methods):
            top_idx = np.argsort(F[:, j])[::-1][:top_k]
            agreement[top_idx] += 1
        feats.append(norm(agreement / n_methods))

    # Rank: average rank across methods
    avg_rank = np.zeros(TOTAL)
    for j in range(n_methods):
        ranks = np.argsort(np.argsort(-F[:, j])).astype(float)
        avg_rank += ranks / n_methods
    feats.append(norm(1.0 - avg_rank / TOTAL))

    # Variance of scores across methods
    score_var = F.var(axis=1)
    feats.append(norm(score_var))

    # Min/max score across methods
    feats.append(norm(F.max(axis=1)))
    feats.append(norm(F.min(axis=1)))

    # Top-2 method average per number
    top2_avg = np.zeros(TOTAL)
    for i in range(TOTAL):
        sorted_scores = np.sort(F[i, :])[::-1]
        top2_avg[i] = sorted_scores[:2].mean() if n_methods >= 2 else F[i, :].mean()
    feats.append(norm(top2_avg))

    return np.column_stack(feats)  # (80, 9)


# ═══════════════════════════════════════════════════════════════
# Method Reliability Scoring
# ═══════════════════════════════════════════════════════════════

def compute_method_weights(draws: np.ndarray, target_idx: int, lookback: int = 20) -> Dict[str, float]:
    """
    Evaluate each method's recent accuracy → weight vector.
    For each recent draw, compute how many of each method's top-PICK the method got right.
    """
    method_hits = {name: [] for name in ALL_SIGNAL_PROVIDERS}

    for step in range(lookback):
        t_idx = target_idx + 1 + step
        if t_idx + 1 >= len(draws):
            break

        hist = draws[t_idx + 1:]
        actual = set(int(x) for x in draws[t_idx])

        for name, func in ALL_SIGNAL_PROVIDERS.items():
            try:
                sc = func(hist)
                top = set(int(x) + 1 for x in np.argsort(sc)[::-1][:PICK])
                h = len(top & actual)
                method_hits[name].append(h)
            except Exception:
                method_hits[name].append(0)

    weights = {}
    for name, hits_list in method_hits.items():
        if hits_list:
            avg = np.mean(hits_list)
            weights[name] = max(0.1, avg / PICK)
        else:
            weights[name] = 0.1

    total = sum(weights.values())
    for name in weights:
        weights[name] /= total

    return weights


def compute_method_weights_fast(draws: np.ndarray, target_idx: int, lookback: int = 20) -> Dict[str, float]:
    """
    Method reliability weights with exponential decay — recent accuracy counts more.
    Uses fast methods for full lookback, slow methods for first 5 steps only.
    """
    fast_methods = {'info_theory', 'bayesian', 'poisson', 'svd', 'ucb', 'fourier', 'markov3', 'markov5'}
    method_hits = {name: [] for name in ALL_SIGNAL_PROVIDERS}
    method_decay = {name: [] for name in ALL_SIGNAL_PROVIDERS}

    # Exponential decay weights: most recent step gets highest weight
    decay_w = np.exp(-0.12 * np.arange(lookback))
    decay_w /= decay_w.sum()

    for step in range(lookback):
        t_idx = target_idx + 1 + step
        if t_idx + 1 >= len(draws):
            break

        hist = draws[t_idx + 1:]
        actual = set(int(x) for x in draws[t_idx])

        for name, func in ALL_SIGNAL_PROVIDERS.items():
            if name not in fast_methods and step >= 5:
                continue
            try:
                sc = func(hist)
                top = set(int(x) + 1 for x in np.argsort(sc)[::-1][:PICK])
                h = len(top & actual)
                method_hits[name].append(h)
                method_decay[name].append(decay_w[step])
            except Exception:
                method_hits[name].append(0)
                method_decay[name].append(decay_w[step])

    weights = {}
    for name in ALL_SIGNAL_PROVIDERS:
        hits_list = method_hits[name]
        dw = method_decay[name]
        if hits_list:
            dw_arr = np.array(dw)
            dw_arr /= dw_arr.sum() + 1e-12
            weighted_avg = np.dot(hits_list, dw_arr)
            weights[name] = max(0.05, weighted_avg / PICK)
        else:
            weights[name] = 0.3

    total = sum(weights.values())
    for name in weights:
        weights[name] /= total

    return weights


# ═══════════════════════════════════════════════════════════════
# Stacked Fusion
# ═══════════════════════════════════════════════════════════════

def stacked_fusion(
    net: MetaNetwork,
    hist: np.ndarray,
    F: np.ndarray,
    method_names: list,
    method_weights: Dict[str, float],
) -> np.ndarray:
    """
    Three-stage fusion with stronger agreement signal:
      Stage 1: MLP on rich fast features → base probability
      Stage 2: Reliability-weighted signal average
      Stage 3: Hard agreement gate — numbers endorsed by 4+ methods get bonus
    """
    # Stage 1: MLP base (now uses 26-dim rich features)
    X_fast = compute_fast_features(hist)
    mlp_base = net.forward(X_fast)  # (80,)

    # Stage 2: Reliability-weighted signal average
    w_vec = np.array([method_weights.get(name, 0.1) for name in method_names])
    w_vec /= w_vec.sum()
    weighted_signal = (F * w_vec[np.newaxis, :]).sum(axis=1)  # (80,)

    # Stage 3a: Soft agreement (how many methods rank this number in top-K)
    cross_f = cross_signal_features(F)
    agreement_soft = cross_f[:, 0]   # top-PICK agreement
    agreement_mid  = cross_f[:, 1]   # top-15 agreement
    agreement_wide = cross_f[:, 2]   # top-20 agreement

    # Stage 3b: Hard agreement gate — strong bonus for 4+ method consensus
    n_methods = F.shape[1]
    hard_agreement = np.zeros(TOTAL)
    for j in range(n_methods):
        top_j = np.argsort(F[:, j])[::-1][:15]
        hard_agreement[top_j] += 1
    # Sigmoid-like gate: numbers with 4+ agreements get exponential boost
    gate = np.where(hard_agreement >= 4,
                    np.exp(0.4 * (hard_agreement - 4)),
                    hard_agreement / (n_methods + 1))
    gate_norm = norm(gate)

    # Context
    ctx = encode_context(hist)

    # Final weighted fusion
    fused = (
        0.20 * norm(mlp_base)
        + 0.25 * norm(weighted_signal)
        + 0.15 * norm(agreement_soft)
        + 0.10 * norm(agreement_mid)
        + 0.10 * norm(agreement_wide)
        + 0.10 * gate_norm
        + 0.10 * norm(ctx)
    )

    return norm(fused)


def finetune_on_recent(
    net: MetaNetwork,
    draws: np.ndarray,
    target_idx: int,
    n_recent: int = 8,
    epochs: int = 5,
    lr: float = 0.003,
) -> MetaNetwork:
    """
    Fine-tune the loaded model on the most recent N draws before target.
    This adapts the model to the current regime without full retraining.
    """
    net.lr = lr
    indices = list(range(target_idx + 1, min(target_idx + 1 + n_recent, len(draws) - 1)))
    if not indices:
        return net

    print(f'  Fine-tuning on {len(indices)} most recent draws ({epochs} epochs)...')
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
        avg = total_loss / max(len(indices), 1)
        print(f'    finetune epoch {ep+1}/{epochs}: loss={avg:.4f}')
    return net


# ═══════════════════════════════════════════════════════════════
# Targeted Generation (method-anchored sets)
# ═══════════════════════════════════════════════════════════════

def generate_method_anchored_sets(
    F: np.ndarray,
    method_names: list,
    fused: np.ndarray,
    n_each: int = 5,
    seed: int = 42,
) -> List[List[int]]:
    """
    For each signal method: take its top-5 numbers, fill remaining from fused dist.
    This ensures each method's strongest signals are represented.
    """
    rng = np.random.default_rng(seed)
    sets = []

    for j, name in enumerate(method_names):
        top_j = np.argsort(F[:, j])[::-1][:5]
        anchor = set(int(x) for x in top_j)

        for k in range(n_each):
            s = set(anchor)
            remaining = PICK - len(s)
            if remaining > 0:
                candidates = [x for x in range(TOTAL) if x not in s]
                p = np.array([fused[x] for x in candidates])
                temp = 0.5 + k * 0.15
                logits = np.log(np.maximum(p, 1e-12)) / temp
                p_temp = np.exp(logits - logits.max())
                p_temp /= p_temp.sum()
                fill = rng.choice(candidates, size=remaining, replace=False, p=p_temp)
                s.update(int(x) for x in fill)

            nums = sorted(x + 1 for x in s)[:PICK]
            sets.append(nums)

    return sets


def generate_agreement_biased(
    F: np.ndarray,
    fused: np.ndarray,
    n_sets: int = 500,
    seed: int = 42,
) -> List[List[int]]:
    """
    Generate sets biased towards numbers that multiple methods agree on.
    """
    rng = np.random.default_rng(seed)

    # Agreement: how many methods put each number in top-20
    n_methods = F.shape[1]
    agreement = np.zeros(TOTAL)
    for j in range(n_methods):
        top_j = np.argsort(F[:, j])[::-1][:20]
        agreement[top_j] += 1
    agreement_norm = norm(agreement / n_methods)

    # Boost fused scores by agreement
    boosted = norm(0.5 * fused + 0.5 * agreement_norm) + 1e-8

    sets = []
    temps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    usage = np.zeros(TOTAL)

    for i in range(n_sets):
        adj = boosted / (1.0 + 0.2 * usage)
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
# Full Deep Fusion Pipeline
# ═══════════════════════════════════════════════════════════════

def run_deep_fusion(
    issues: list,
    draws: np.ndarray,
    target_issue: str,
    n_train: int = 80,
    n_sets: int = 30,
    extreme: bool = False,
    pool_size: int = 80000,
    save_model: str = None,
    load_model: str = None,
    finetune: int = 8,
):
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
    print('=' * 70)

    # Phase 1: Train MLP on fast features (or load from file)
    import os
    if load_model and os.path.exists(load_model + '.npz'):
        print(f'Phase 1: Loading saved model from {load_model}.npz ...')
        d = np.load(load_model + '.npz')
        input_dim = int(d['W1'].shape[1])
        net = MetaNetwork(input_dim=input_dim, hidden=64)
        net.W1 = d['W1']
        net.b1 = d['b1']
        net.W2 = d['W2']
        net.b2 = d['b2']
        print(f'  Loaded: input_dim={input_dim}, hidden=64')
    else:
        if load_model:
            print(f'  [warn] Model file {load_model}.npz not found, training from scratch.')
        print('Phase 1: Training MLP on fast features...')
        net, input_dim = train_meta_model_fast(
            draws, idx if has_actual else -1,
            n_train=n_train, epochs=10, verbose=True,
        )
        if save_model:
            net.save(save_model)
            print(f'  Model saved → {save_model}.npz')

    # Fine-tune on most recent draws (adapts to current regime)
    if finetune > 0:
        net = finetune_on_recent(net, draws, idx if has_actual else 0,
                                  n_recent=finetune, epochs=5, lr=0.003)

    # Phase 2: Compute full 12-signal feature matrix
    print('\nPhase 2: Computing full 12-signal feature matrix...')
    F, scores, method_names = compute_feature_matrix(hist)

    # Show per-method hits
    if has_actual:
        print(f'\nPer-method accuracy:')
        for j, name in enumerate(method_names):
            top = set(int(x) + 1 for x in np.argsort(F[:, j])[::-1][:PICK])
            h = len(top & actual)
            mark = '★' if h >= 5 else '●' if h >= 4 else ' '
            print(f'  {mark} {name:<15} {h:>2}/{PICK}')

    # Phase 3: Compute method reliability weights
    print('\nPhase 3: Computing method reliability weights...')
    mw = compute_method_weights_fast(draws, idx if has_actual else 0, lookback=8)
    for name in sorted(mw, key=lambda n: -mw[n])[:5]:
        print(f'  {name:<15} weight={mw[name]:.3f}')

    # Phase 4: Stacked fusion
    print('\nPhase 4: Stacked fusion...')
    fused = stacked_fusion(net, hist, F, method_names, mw)

    top = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:PICK])
    if has_actual:
        h = len(set(top) & actual)
        print(f'  ★ Fused Top {PICK}: {top}  hits={h}  match={sorted(set(top) & actual)}')
    else:
        print(f'  ★ Fused Top {PICK}: {top}')

    # Phase 5: Generate sets
    if extreme:
        print(f'\nPhase 5: Multi-strategy extreme generation ({pool_size})...')

        pool_a = generate_mega_pool(fused, n_total=pool_size // 2, seed_base=0)
        print(f'  [A] Fused pool: {len(pool_a)} unique')

        pool_b = generate_agreement_biased(F, fused, n_sets=pool_size // 4, seed=1000)
        print(f'  [B] Agreement-biased: {len(pool_b)} sets')

        pool_c = generate_method_anchored_sets(F, method_names, fused, n_each=8, seed=2000)
        print(f'  [C] Method-anchored: {len(pool_c)} sets')

        all_pool = pool_a + pool_b + pool_c
        seen = set()
        unique = []
        for s in all_pool:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                unique.append(s)
        print(f'  Total unique: {len(unique)}')

        if has_actual:
            scored = [(len(set(s) & actual), s) for s in unique]
            scored.sort(key=lambda x: -x[0])

            c = Counter([h for h, _ in scored])
            print('\n  Hit distribution:')
            for k in sorted(c):
                print(f'    hits={k:>2}: {c[k]:>6} ({c[k] / len(scored) * 100:.2f}%)')

            def jaccard(a, b):
                return len(set(a) & set(b)) / len(set(a) | set(b))

            selected = []
            for h, s in scored:
                if len(selected) >= n_sets:
                    break
                if not selected or all(jaccard(s, t) < 0.45 for t in selected):
                    selected.append(s)

            print(f'\n  BEST DIVERSE {min(n_sets, len(selected))} SETS:')
            final_hits = []
            for i, s in enumerate(selected, 1):
                h = len(set(s) & actual)
                final_hits.append(h)
                m = sorted(set(s) & actual)
                mark = '★' if h >= 7 else '●' if h >= 5 else ' '
                print(f'  {mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')

            fa = np.array(final_hits)
            print(f'\n  best={fa.max()}, avg={fa.mean():.2f}, P>=7={(fa >= 7).mean():.2f}, P>=9={(fa >= 9).mean():.2f}')
        else:
            def jaccard(a, b):
                return len(set(a) & set(b)) / len(set(a) | set(b))
            selected = []
            for s in unique:
                if len(selected) >= n_sets:
                    break
                if not selected or all(jaccard(s, t) < 0.45 for t in selected):
                    selected.append(s)

            print(f'\n  PREDICTED {len(selected)} DIVERSE SETS:')
            for i, s in enumerate(selected, 1):
                print(f'    SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')
    else:
        print(f'\nPhase 5: Generation ({n_sets} sets)...')
        sets = generate_from_probs(fused, n_sets=n_sets)

        final_hits = []
        for i, s in enumerate(sets, 1):
            if has_actual:
                h = len(set(s) & actual)
                final_hits.append(h)
                m = sorted(set(s) & actual)
                mark = '★' if h >= 7 else '●' if h >= 5 else ' '
                print(f'  {mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')
            else:
                print(f'  SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')

        if has_actual and final_hits:
            fa = np.array(final_hits)
            print(f'\n  best={fa.max()}, avg={fa.mean():.2f}')

    if has_actual:
        print(f'\n  ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')

    return net, fused


# ═══════════════════════════════════════════════════════════════
# Extreme-10 Search (ignore diversity, just hunt for 9+/10+ hits)
# ═══════════════════════════════════════════════════════════════

def run_extreme10(
    issues: list,
    draws: np.ndarray,
    target_issue: str,
    n_train: int = 80,
    pool_size: int = 300000,
    top: int = 40,
):
    """
    Extreme mode: very large pool + no Jaccard 去重，只看历史上最高能到几命中。
    用来回答：在当前信号体系下，是否存在 9+/10+ 这样极端好的 11 选 20 组合。
    """
    if target_issue not in issues:
        raise ValueError(f'Issue {target_issue} not found in data')

    idx = issues.index(target_issue)
    hist = draws[idx + 1:]
    actual = set(int(x) for x in draws[idx])

    print(f'TARGET: {target_issue}')
    print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')
    print(f'HISTORY: {len(hist)} draws')
    print('=' * 70)

    # Phase 1: Train MLP on fast features (稍微多训一点，n_train 默认 80)
    print('Phase 1: Training MLP on fast features (extreme10)...')
    net, input_dim = train_meta_model_fast(
        draws, idx,
        n_train=n_train, epochs=10, lr=0.005, verbose=True,
    )

    # Phase 2: Full 12-signal feature matrix
    print('\nPhase 2: Computing full 12-signal feature matrix...')
    F, scores, method_names = compute_feature_matrix(hist)

    print('\nPer-method single-set accuracy at this issue:')
    for j, name in enumerate(method_names):
        top_nums = set(int(x) + 1 for x in np.argsort(F[:, j])[::-1][:PICK])
        h = len(top_nums & actual)
        mark = '★' if h >= 5 else '●' if h >= 4 else ' '
        print(f'  {mark} {name:<15} {h:>2}/{PICK}')

    # Phase 3: Method reliability weights
    print('\nPhase 3: Computing method reliability weights (fast)...')
    mw = compute_method_weights_fast(draws, idx, lookback=10)
    for name in sorted(mw, key=lambda n: -mw[n])[:6]:
        print(f'  {name:<15} weight={mw[name]:.3f}')

    # Phase 4: Stacked fusion → fused distribution
    print('\nPhase 4: Stacked fusion → fused distribution...')
    fused = stacked_fusion(net, hist, F, method_names, mw)

    top_nums = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:PICK])
    h_top = len(set(top_nums) & actual)
    print(f'  ★ Fused Top {PICK}: {top_nums}  hits={h_top}  match={sorted(set(top_nums) & actual)}')

    # Phase 5: Huge pool, no diversity filter
    print(f'\nPhase 5: Extreme10 search with pool_size={pool_size} (no diversity filter)...')

    # A: fused-based mega pool
    pool_a = generate_mega_pool(fused, n_total=pool_size // 2, seed_base=0)
    print(f'  [A] Fused pool: {len(pool_a)} sets')

    # B: agreement-biased
    pool_b = generate_agreement_biased(F, fused, n_sets=pool_size // 3, seed=777)
    print(f'  [B] Agreement-biased: {len(pool_b)} sets')

    # C: method-anchored
    pool_c = generate_method_anchored_sets(F, method_names, fused, n_each=10, seed=2026)
    print(f'  [C] Method-anchored: {len(pool_c)} sets')

    all_pool = pool_a + pool_b + pool_c
    print(f'  Total raw sets (with duplicates possible): {len(all_pool)}')

    # 不去重，直接看极端 tail（去重反而会略降 highest hits 概率）
    scored = [(len(set(s) & actual), s) for s in all_pool]
    scored.sort(key=lambda x: -x[0])

    # 统计尾部分布
    from collections import Counter as _Counter
    c = _Counter([h for h, _ in scored])
    print('\n  Hit distribution over ALL sets (including duplicates):')
    for k in sorted(c):
        print(f'    hits={k:>2}: {c[k]:>7} ({c[k] / len(scored) * 100:.4f}%)')

    total = len(scored)
    cnt9 = sum(1 for h, _ in scored if h >= 9)
    cnt10 = sum(1 for h, _ in scored if h >= 10)
    cnt11 = sum(1 for h, _ in scored if h >= 11)
    print(f'\n  Summary tail:')
    print(f'    count(h>=9)  = {cnt9}  ({cnt9 / total * 100:.4f}%)')
    print(f'    count(h>=10) = {cnt10} ({cnt10 / total * 100:.6f}%)')
    print(f'    count(h>=11) = {cnt11} ({cnt11 / total * 100:.6f}%)')

    # 输出前 top 组合
    print(f'\n  Top {top} sets by hits (no diversity filter):')
    shown = 0
    last_h = None
    for h, s in scored:
        if shown >= top and (last_h is not None and h < last_h):
            break
        m = sorted(set(s) & actual)
        mark = '★' if h >= 9 else '●' if h >= 7 else ' '
        print(f'  {mark} hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')
        shown += 1
        last_h = h

    print(f'\n  ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')

    return {
        "best_hits": scored[0][0] if scored else 0,
        "count_ge9": cnt9,
        "count_ge10": cnt10,
        "count_ge11": cnt11,
        "total_sets": total,
    }
