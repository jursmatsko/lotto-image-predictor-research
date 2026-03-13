#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local RandomForest walk-forward backtest for KLE.

Usage:
  python3 kle/scripts/rf_walk_forward_local.py --last 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


TOTAL_NUMBERS = 80
DRAW_SIZE = 20
FEAT_WINDOWS = [5, 10, 20, 50]


def load_draws(data_csv: Path) -> tuple[list[str], list[list[int]]]:
    df = pd.read_csv(data_csv, encoding="utf-8")
    issue_col = "期数" if "期数" in df.columns else "期号"
    number_cols = [f"红球_{i}" for i in range(1, 21)]
    df = df.sort_values(issue_col, ascending=True).reset_index(drop=True)

    issues: list[str] = []
    draws: list[list[int]] = []
    for _, row in df.iterrows():
        nums: list[int] = []
        for c in number_cols:
            v = int(row[c])
            if 1 <= v <= TOTAL_NUMBERS:
                nums.append(v)
        if len(nums) == DRAW_SIZE:
            issues.append(str(row[issue_col]))
            draws.append(nums)
    return issues, draws


def build_presence(draws: list[list[int]]) -> np.ndarray:
    p = np.zeros((len(draws), TOTAL_NUMBERS), dtype=np.float32)
    for i, d in enumerate(draws):
        for n in d:
            p[i, n - 1] = 1.0
    return p


def compute_features_at_t(presence: np.ndarray, t: int) -> np.ndarray:
    feats = []

    # 1-4: frequency in multi windows
    for w in FEAT_WINDOWS:
        s = max(0, t - w)
        feats.append(presence[s:t].sum(axis=0) / max(t - s, 1))

    # 5: gap
    max_gap = 100
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

    # 6-7: hit/miss streak
    hs = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    ms = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    active_h = np.ones(TOTAL_NUMBERS, dtype=bool)
    active_m = np.ones(TOTAL_NUMBERS, dtype=bool)
    for i in range(t - 1, max(-1, t - 30), -1):
        active_h &= presence[i] > 0
        active_m &= presence[i] == 0
        hs[active_h] += 1
        ms[active_m] += 1
        if not active_h.any() and not active_m.any():
            break
    feats.append(hs / 10.0)
    feats.append(ms / 10.0)

    # 8: short-term momentum (weighted recent presence)
    s5 = max(0, t - 5)
    r5 = presence[s5:t]
    if len(r5) > 0:
        w = np.arange(1, len(r5) + 1, dtype=np.float32)
        mom = (r5 * w[:, None]).sum(axis=0) / w.sum()
    else:
        mom = np.zeros(TOTAL_NUMBERS, dtype=np.float32)
    feats.append(mom)

    # 9-12: static
    idx = np.arange(1, TOTAL_NUMBERS + 1, dtype=np.float32)
    feats.append((idx % 2).astype(np.float32))              # odd
    feats.append((idx > 40).astype(np.float32))             # high
    feats.append(((idx - 1) // 20) / 3.0)                   # zone id
    feats.append(idx / 80.0)                                # normalized position

    return np.stack(feats, axis=1).astype(np.float32)       # (80, 12)


def build_training_data(
    presence: np.ndarray, start_t: int, end_t: int
) -> tuple[np.ndarray, np.ndarray]:
    min_t = max(start_t, max(FEAT_WINDOWS) + 1)
    n = end_t - min_t
    x = np.empty((n * TOTAL_NUMBERS, 12), dtype=np.float32)
    y = np.empty(n * TOTAL_NUMBERS, dtype=np.float32)
    for j, t in enumerate(range(min_t, end_t)):
        x[j * TOTAL_NUMBERS : (j + 1) * TOTAL_NUMBERS] = compute_features_at_t(presence, t)
        y[j * TOTAL_NUMBERS : (j + 1) * TOTAL_NUMBERS] = presence[t]
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RF walk-forward backtest")
    parser.add_argument("--last", type=int, default=20, help="Evaluate last N draws")
    parser.add_argument("--train-window", type=int, default=300, help="Use last N draws for training")
    parser.add_argument("--estimators", type=int, default=200, help="RF trees")
    parser.add_argument("--max-depth", type=int, default=8, help="RF max depth")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_csv = root / "data" / "data.csv"
    issues, draws = load_draws(data_csv)
    presence = build_presence(draws)

    eval_start = max(max(FEAT_WINDOWS) + 1, len(draws) - args.last)
    hits_list: list[int] = []
    print(f"Total draws: {len(draws)} | Walk-forward steps: {len(draws) - eval_start}")
    print("=" * 60)

    for test_idx in range(eval_start, len(draws)):
        start_t = max(0, test_idx - args.train_window) if args.train_window > 0 else 0
        x_train, y_train = build_training_data(presence, start_t, test_idx)

        rf = RandomForestClassifier(
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(x_train, y_train)

        x_test = compute_features_at_t(presence, test_idx)
        probs = rf.predict_proba(x_test)[:, 1]
        pred = set(int(i + 1) for i in np.argsort(probs)[::-1][:DRAW_SIZE])
        actual = set(draws[test_idx])
        hits = len(pred & actual)
        hits_list.append(hits)
        avg = float(np.mean(hits_list))
        print(f"[{len(hits_list):>2}/{len(draws)-eval_start}] Draw {issues[test_idx]}: {hits:>2}/20 hits  avg={avg:.2f}")

    avg_hits = float(np.mean(hits_list))
    print("=" * 60)
    print(f"Average hits  : {avg_hits:.2f} / 20")
    print("Random baseline: 5.00 / 20")
    print(f"vs Random      : {avg_hits - 5:+.2f}")
    print(f"Draws > 5 hits : {sum(h > 5 for h in hits_list)} / {len(hits_list)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

