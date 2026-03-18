#!/usr/bin/env python3
"""
Iterative prediction: run multiple batches of set generation with different seeds,
count hits vs a given actual draw, and report per-batch and overall stats (best hit, 5+/6+/7+/8+).

Usage (from kle):
  python scripts/iterative_predict.py --actual "7,13,16,21,25,31,32,38,45,49,57,58,59,61,63,67,69,72,75,77" --iterations 10 --per-iter 500 --memory storage/memory.npz --pick 10
"""
import argparse
import os
import sys

# Run from kle; add parent (kle) so we can import from scripts
_kle_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_kle_root, "scripts"))

import generative_memory_predictor as gp


def main():
    parser = argparse.ArgumentParser(description="Iterative prediction: multiple batches, hit stats vs actual draw")
    parser.add_argument("--memory", default="storage/memory.npz", help="Memory .npz path")
    parser.add_argument("--pick", type=int, default=10, help="Numbers per set (default 10)")
    parser.add_argument(
        "--actual",
        type=str,
        required=True,
        help='Actual draw 20 numbers, comma-separated, e.g. "7,13,16,21,25,31,32,38,45,49,57,58,59,61,63,67,69,72,75,77"',
    )
    parser.add_argument("--iterations", type=int, default=10, help="Number of batches (default 10)")
    parser.add_argument("--per-iter", type=int, default=500, help="Sets per batch (default 500)")
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed; batch i uses seed_base + i")
    parser.add_argument("--csv", default="data/data.csv", help="Data CSV path")
    parser.add_argument("--stake", type=float, default=2.0, help="Stake per ticket (default 2.0)")
    parser.add_argument("--prize5", type=float, default=3.0, help="Prize for 5 hits (default 3)")
    parser.add_argument("--prize6", type=float, default=5.0, help="Prize for 6 hits (default 5)")
    parser.add_argument("--prize7", type=float, default=80.0, help="Prize for 7 hits (default 80)")
    parser.add_argument("--prize8", type=float, default=720.0, help="Prize for 8 hits (default 720)")
    parser.add_argument("--prize9", type=float, default=8000.0, help="Prize for 9 hits (default 8000)")
    parser.add_argument(
        "--prize10",
        type=float,
        default=5000000.0,
        help="Prize for 10 hits / jackpot (default 5000000)",
    )
    args = parser.parse_args()

    actual = {int(x.strip()) for x in args.actual.split(",") if x.strip()}
    if len(actual) != 20:
        print(f"Expected 20 numbers; got {len(actual)}. Check --actual.")
        sys.exit(1)

    gp._set_pick(args.pick)

    csv_path = os.path.join(_kle_root, args.csv) if not os.path.isabs(args.csv) else args.csv
    memory_path = os.path.join(_kle_root, args.memory) if not os.path.isabs(args.memory) else args.memory

    issues, draws = gp.load_data(csv_path)
    hist = draws

    method_names = list(gp.ALL_SIGNAL_PROVIDERS.keys())
    if not os.path.isfile(memory_path):
        print(f"Memory not found: {memory_path}")
        sys.exit(1)
    memory = gp.MemoryBank.load(memory_path, method_names)

    att_w = 0.20
    pb_w = 0.15
    rw = 0.20
    fused_w = 1.0 - att_w - pb_w - rw
    fused_w = max(0.25, min(0.65, fused_w))
    total_w = fused_w + att_w + pb_w + rw
    fusion_weights = (fused_w / total_w, att_w / total_w, pb_w / total_w, rw / total_w)

    gen = gp.GenerativeModel(
        memory,
        fusion_weights=fusion_weights,
        temp_base=1.0,
        replay_min_hits=4,
    )

    scores = {}
    for name, func in gp.ALL_SIGNAL_PROVIDERS.items():
        scores[name] = func(hist)
    fused = gen.compute_fused_distribution(scores, hist)

    print(f"Iterative prediction: {args.iterations} x {args.per_iter} sets (PICK={args.pick})")
    print(f"Actual draw (20): {sorted(actual)}")
    print("=" * 72)
    print(f"{'iter':>4}  {'best':>4}  {'n_5+':>6}  {'n_6+':>6}  {'n_7+':>6}  {'n_8+':>6}")
    print("-" * 72)

    overall_best = 0
    iters_with_8 = 0
    all_best_sets = []
    # Global hit histogram over all tickets (0..10)
    hit_hist = [0] * 11

    for i in range(args.iterations):
        seed = args.seed_base + i
        sets = gen.generate_sets(fused, n_sets=args.per_iter, seed=seed)
        hits_list = []
        for s in sets:
            h = len(set(s) & actual)
            hits_list.append(h)
            if 0 <= h <= 10:
                hit_hist[h] += 1
        best = max(hits_list)
        n5 = sum(1 for h in hits_list if h >= 5)
        n6 = sum(1 for h in hits_list if h >= 6)
        n7 = sum(1 for h in hits_list if h >= 7)
        n8 = sum(1 for h in hits_list if h >= 8)
        if best >= 8:
            iters_with_8 += 1
        overall_best = max(overall_best, best)
        if best >= 7:
            idx = hits_list.index(best)
            all_best_sets.append((i + 1, best, sets[idx]))
        print(f"{i+1:4d}  {best:4d}  {n5:6d}  {n6:6d}  {n7:6d}  {n8:6d}")

    print("=" * 72)
    print(f"Overall best hit: {overall_best}")
    print(f"Iterations with at least one 8+ hit: {iters_with_8}/{args.iterations}")
    if all_best_sets:
        print("\nSets with 7+ hits (iter, best, numbers):")
        for it, b, s in sorted(all_best_sets, key=lambda x: -x[1])[:20]:
            match = sorted(set(s) & actual)
            print(f"  iter {it}  best={b}  {s}  -> match {match}")

    total_tickets = args.iterations * args.per_iter
    if sum(hit_hist) != total_tickets:
        print("\n[warn] Hit histogram does not sum to total tickets; investment/prize may be off.")

    # Prize calculation following 快乐8 选十: stake back on 0 hits, fixed prizes for 5–9, jackpot for 10.
    c0 = hit_hist[0]
    c5 = hit_hist[5]
    c6 = hit_hist[6]
    c7 = hit_hist[7]
    c8 = hit_hist[8]
    c9 = hit_hist[9]
    c10 = hit_hist[10]

    print("\nHit distribution over all tickets:")
    print(f"  total tickets: {total_tickets}")
    print(f"  0 hits : {c0}")
    print(f"  5 hits : {c5}")
    print(f"  6 hits : {c6}")
    print(f"  7 hits : {c7}")
    print(f"  8 hits : {c8}")
    print(f"  9 hits : {c9}")
    print(f"  10 hits: {c10}")

    # Assume 0-hit prize equals stake (refund).
    prize0 = args.stake
    total_prize = (
        c0 * prize0
        + c5 * args.prize5
        + c6 * args.prize6
        + c7 * args.prize7
        + c8 * args.prize8
        + c9 * args.prize9
        + c10 * args.prize10
    )
    investment = total_tickets * args.stake
    net = total_prize - investment

    print("\nPayout summary:")
    print(f"  stake per ticket       : {args.stake}")
    print(f"  total investment       : {investment}")
    print(f"  total prize (incl. 0-hit refunds and 10-hit jackpot): {total_prize}")
    print(f"  net profit (prize - investment)                     : {net}")
    if investment > 0:
        roi = net / investment * 100.0
        print(f"  ROI                                                        : {roi:.3f}%")


if __name__ == "__main__":
    main()
