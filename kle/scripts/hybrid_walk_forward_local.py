#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid local walk-forward:
RandomForest + ExtraTrees + Statistical (freq/gap/momentum)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


TOTAL_NUMBERS = 80
DRAW_SIZE = 20
FEAT_WINDOWS = [5, 10, 20, 50]


def load_draws(data_csv: Path) -> tuple[list[str], list[list[int]]]:
    df = pd.read_csv(data_csv, encoding="utf-8")
    issue_col = "期数" if "期数" in df.columns else "期号"
    cols = [f"红球_{i}" for i in range(1, 21)]
    df = df.sort_values(issue_col, ascending=True).reset_index(drop=True)
    issues, draws = [], []
    for _, row in df.iterrows():
        nums = [int(row[c]) for c in cols if 1 <= int(row[c]) <= 80]
        if len(nums) == 20:
            issues.append(str(row[issue_col]))
            draws.append(nums)
    return issues, draws


def presence_matrix(draws: list[list[int]]) -> np.ndarray:
    p = np.zeros((len(draws), TOTAL_NUMBERS), dtype=np.float32)
    for i, d in enumerate(draws):
        for n in d:
            p[i, n - 1] = 1.0
    return p


def compute_features_at_t(presence: np.ndarray, t: int) -> np.ndarray:
    feats = []

    for w in FEAT_WINDOWS:
        s = max(0, t - w)
        feats.append(presence[s:t].sum(axis=0) / max(t - s, 1))

    max_gap = 120
    gaps = np.full(TOTAL_NUMBERS, min(t, max_gap), dtype=np.float32)
    found = np.zeros(TOTAL_NUMBERS, dtype=bool)
    for i in range(t - 1, max(-1, t - max_gap - 1), -1):
        hit = presence[i] > 0
        new_hit = hit & ~found
        gaps[new_hit] = t - 1 - i
        found |= hit
        if found.all():
            break
    feats.append(gaps / max_gap)

    hs = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    ms = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    ah = np.ones(TOTAL_NUMBERS, dtype=bool)
    am = np.ones(TOTAL_NUMBERS, dtype=bool)
    for i in range(t - 1, max(-1, t - 35), -1):
        ah &= presence[i] > 0
        am &= presence[i] == 0
        hs[ah] += 1
        ms[am] += 1
        if not ah.any() and not am.any():
            break
    feats.append(hs / 10.0)
    feats.append(ms / 10.0)

    s5 = max(0, t - 5)
    r5 = presence[s5:t]
    if len(r5) > 0:
        w = np.arange(1, len(r5) + 1, dtype=np.float32)
        mom = (r5 * w[:, None]).sum(axis=0) / w.sum()
    else:
        mom = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    feats.append(mom)

    # static
    idx = np.arange(1, TOTAL_NUMBERS + 1, dtype=np.float32)
    feats.append((idx % 2).astype(np.float32))
    feats.append((idx > 40).astype(np.float32))
    feats.append(((idx - 1) // 20) / 3.0)
    feats.append(idx / 80.0)

    return np.stack(feats, axis=1).astype(np.float32)  # (80, 12)


def build_xy(presence: np.ndarray, start_t: int, end_t: int) -> tuple[np.ndarray, np.ndarray]:
    min_t = max(start_t, max(FEAT_WINDOWS) + 1)
    n = end_t - min_t
    x = np.empty((n * TOTAL_NUMBERS, 12), dtype=np.float32)
    y = np.empty(n * TOTAL_NUMBERS, dtype=np.float32)
    for j, t in enumerate(range(min_t, end_t)):
        x[j * TOTAL_NUMBERS:(j + 1) * TOTAL_NUMBERS] = compute_features_at_t(presence, t)
        y[j * TOTAL_NUMBERS:(j + 1) * TOTAL_NUMBERS] = presence[t]
    return x, y


def stat_frequency(draws: list[list[int]], window: int) -> np.ndarray:
    recent = draws[-window:]
    cnt = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    for d in recent:
        for n in d:
            cnt[n - 1] += 1
    return cnt / max(cnt.max(), 1e-6)


def stat_gap(draws: list[list[int]], window: int) -> np.ndarray:
    recent = draws[-window:]
    last = np.full(TOTAL_NUMBERS, -1, dtype=np.float32)
    for i, d in enumerate(recent):
        for n in d:
            last[n - 1] = i
    total = len(recent)
    gaps = np.where(last >= 0, total - 1 - last, total).astype(np.float32)
    return gaps / max(gaps.max(), 1e-6)


def stat_momentum(draws: list[list[int]], window: int) -> np.ndarray:
    recent = draws[-window:]
    sc = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    for t, d in enumerate(recent):
        w = (t + 1) / max(len(recent), 1)
        for n in d:
            sc[n - 1] += w
    return sc / max(sc.max(), 1e-6)


def norm(v: np.ndarray) -> np.ndarray:
    vmax = float(v.max())
    return v / vmax if vmax > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid walk-forward local")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--train-window", type=int, default=320)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    issues, draws = load_draws(root / "data" / "data.csv")
    presence = presence_matrix(draws)

    eval_start = max(max(FEAT_WINDOWS) + 1, len(draws) - args.last)
    n_steps = len(draws) - eval_start
    hits_list: list[int] = []

    print(f"Total draws: {len(draws)} | Walk-forward steps: {n_steps}")
    print("=" * 60)

    for test_idx in range(eval_start, len(draws)):
        start_t = max(0, test_idx - args.train_window)
        x_train, y_train = build_xy(presence, start_t, test_idx)
        x_test = compute_features_at_t(presence, test_idx)

        rf = RandomForestClassifier(
            n_estimators=240,
            max_depth=9,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        et = ExtraTreesClassifier(
            n_estimators=320,
            max_depth=10,
            random_state=43,
            n_jobs=-1,
            class_weight="balanced",
        )
        rf.fit(x_train, y_train)
        et.fit(x_train, y_train)

        p_rf = rf.predict_proba(x_test)[:, 1]
        p_et = et.predict_proba(x_test)[:, 1]
        p_ml = 0.55 * norm(p_rf) + 0.45 * norm(p_et)

        history = draws[:test_idx]
        p_freq = stat_frequency(history, 30)
        p_gap = stat_gap(history, 120)
        p_mom = stat_momentum(history, 7)

        # Hybrid fusion: ML + robust stats
        combined = 0.42 * p_ml + 0.30 * p_freq + 0.18 * p_gap + 0.10 * p_mom

        pred = set(int(i + 1) for i in np.argsort(combined)[::-1][:20])
        actual = set(draws[test_idx])
        hits = len(pred & actual)
        hits_list.append(hits)
        avg = float(np.mean(hits_list))

        print(f"[{len(hits_list):>2}/{n_steps}] Draw {issues[test_idx]}: {hits:>2}/20 hits  avg={avg:.2f}")

    avg_hits = float(np.mean(hits_list))
    print("=" * 60)
    print(f"Average hits  : {avg_hits:.2f} / 20")
    print("Random baseline: 5.00 / 20")
    print(f"vs Random      : {avg_hits - 5:+.2f}")
    print(f"Draws > 5 hits : {sum(h > 5 for h in hits_list)} / {len(hits_list)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

