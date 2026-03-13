"""
12 Signal Providers — each returns an 80-dim score vector.
"""
import numpy as np
from .constants import PICK, TOTAL, DRAW_SIZE
from .utils import presence_matrix


# ── M1: Information Theory ───────────────────────────────────

def sig_info_theory(hist, window=80):
    P = presence_matrix(hist, window)
    n = P.shape[0]
    mp = np.clip(P.mean(0), 1e-6, 1 - 1e-6)
    scores = np.zeros(TOTAL)
    for i in range(TOTAL):
        ent = -mp[i] * np.log2(mp[i]) - (1 - mp[i]) * np.log2(1 - mp[i])
        w1 = P[:n // 4, i].mean()
        w2 = P[n // 4:n // 2, i].mean()
        w3 = P[n // 2:, i].mean()
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


# ── M2: Markov Transition Matrix ────────────────────────────

def sig_markov(hist, order=3):
    P = presence_matrix(hist)[::-1]
    n = P.shape[0]
    tc = np.zeros((TOTAL, 2 ** order, 2))
    for i in range(TOTAL):
        for t in range(order, n):
            s = sum(P[t - order + k, i] * (2 ** (order - 1 - k)) for k in range(order))
            tc[i, int(s), P[t, i]] += 1
    Pr = presence_matrix(hist)
    pred = np.zeros(TOTAL)
    for i in range(TOTAL):
        s = sum(Pr[k, i] * (2 ** (order - 1 - k)) for k in range(min(order, Pr.shape[0])))
        tot = tc[i, int(s), 0] + tc[i, int(s), 1]
        pred[i] = tc[i, int(s), 1] / tot if tot > 0 else 0.25
    return pred


# ── M3: Graph Co-occurrence Network ─────────────────────────

def sig_graph(hist, window=100):
    P = presence_matrix(hist, window)
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


# ── M4: Bayesian Posterior ───────────────────────────────────

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


# ── M5: Poisson Arrival Process ──────────────────────────────

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
        scores[i] = (
            0.6 * (1 - np.exp(-cur_gap / max(lam, 1e-6)))
            + 0.4 / (1 + np.std(gaps) / (lam + 1e-6))
        )
    return scores


# ── M6: SVD Matrix Factorization ─────────────────────────────

def sig_svd(hist, window=200, rank=15):
    P = presence_matrix(hist, window).astype(float)
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    r = min(rank, len(S))
    recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
    w = np.exp(-0.02 * np.arange(min(30, recon.shape[0])))
    w /= w.sum()
    pred = sum(w[t] * recon[t] for t in range(min(30, recon.shape[0])))
    return np.clip(pred, 0, 1)


# ── M7: UCB Multi-Armed Bandit ───────────────────────────────

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


# ── M8: Fourier / Spectral Analysis ──────────────────────────

def sig_fourier(hist, window=500):
    P = presence_matrix(hist, window)[::-1].astype(float)
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


# ── M9: KNN Instance-Based ───────────────────────────────────

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
        sims.append((dot / (nm + 1e-8), hist[t - 1] if t > 0 else hist[0]))
    sims.sort(key=lambda x: -x[0])
    scores = np.zeros(TOTAL)
    for sim, draw in sims[:k]:
        for num in draw:
            if 1 <= num <= TOTAL:
                scores[num - 1] += sim
    return scores


# ── M10: Elman RNN (hand-coded) ──────────────────────────────

def sig_recurrent(hist, hidden=32, window=120, lr=0.01, epochs=12):
    P = presence_matrix(hist, window)[::-1].astype(float)
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
    Pc = presence_matrix(hist, window)[::-1].astype(float)
    for t in range(Pc.shape[0]):
        h = tanh(Wh @ h + Wx @ Pc[t] + bh)
    return sigmoid(Wy @ h + by)


# ── M11: Genetic Algorithm (score vector) ────────────────────

def sig_genetic(hist, pop_sz=150, gens=100, window=60):
    rng = np.random.default_rng(2024)
    recent = hist[:window]

    def fit(c):
        s = set(c)
        return sum(
            np.exp(-0.03 * t) * len(s & set(int(x) for x in r))
            for t, r in enumerate(recent)
        )

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
            child = (
                sorted(rng.choice(an, PICK, replace=False))
                if len(an) >= PICK
                else sorted(an + list(set(range(1, 81)) - set(an))[:PICK - len(an)])
            )
            if rng.random() < 0.15:
                im = rng.integers(0, PICK)
                cs = list(set(range(1, 81)) - set(child))
                if cs:
                    child[im] = rng.choice(cs)
                    child = sorted(child)
            new.append(child)
        pop = new

    fs = np.array([fit(c) for c in pop])
    top_pop = [pop[i] for i in np.argsort(fs)[::-1][:30]]
    scores = np.zeros(TOTAL)
    for c in top_pop:
        for num in c:
            scores[num - 1] += 1
    return scores


# ── Registry ─────────────────────────────────────────────────

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
    'genetic': sig_genetic,
}
