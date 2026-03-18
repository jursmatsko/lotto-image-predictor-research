#!/usr/bin/env python3
"""
Inspect a saved memory .npz file: list arrays and show a short summary.

Usage (from kle/):
  python scripts/inspect_memory.py storage/memory.npz
  python scripts/inspect_memory.py storage/memory.npz --brief
"""
import argparse
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect memory .npz file")
    parser.add_argument("path", default="storage/memory.npz", nargs="?", help="Path to .npz file")
    parser.add_argument("--brief", action="store_true", help="Only list keys and shapes")
    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    print(f"Memory file: {path}")
    print(f"Keys: {keys}")
    print()

    for k in keys:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")

    if args.brief:
        return

    print()
    # Method weights
    if "method_names" in data and "method_weights" in data:
        names = list(data["method_names"])
        weights = data["method_weights"]
        if len(weights.shape) == 1 and len(weights) == len(names):
            print("Method weights (normalized):")
            for n, w in zip(names, weights):
                print(f"    {n}: {w:.4f}")
        else:
            print("method_weights:", weights)
    print()

    # Number attention (80 numbers)
    if "number_attention" in data:
        att = data["number_attention"]
        att = np.asarray(att).flatten()
        if att.size >= 80:
            top = np.argsort(att)[::-1][:10]
            print("Top 10 numbers by attention (1-based):", [int(x) + 1 for x in top])
            print("  attention min/max/mean:", att.min(), att.max(), att.mean())
    print()

    # Pair success (80x80)
    if "pair_success" in data:
        ps = data["pair_success"]
        print("pair_success: shape", ps.shape, "min/max/mean:", ps.min(), ps.max(), ps.mean())
    print()

    # Replay buffer
    if "replay_hits" in data:
        hits = data["replay_hits"]
        n = len(hits)
        print(f"Replay buffer: {n} entries")
        if n > 0:
            print("  hits: min/max/mean", int(hits.min()), int(hits.max()), float(hits.mean()))
            if "replay_sets" in data:
                print("  replay_sets shape:", data["replay_sets"].shape)
    print()

    if "lr" in data and "decay" in data:
        print("Hyperparams: lr =", float(data["lr"][0]), ", decay =", float(data["decay"][0]))


if __name__ == "__main__":
    main()
