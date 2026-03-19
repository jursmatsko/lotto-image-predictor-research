#!/usr/bin/env python3
"""Evaluate 2026046 prediction against actual winning numbers."""

import re

# Actual winning numbers for 2026046 (2026-02-25)
ACTUAL = {}

# kl8_pick10 payout (stake=2, refund0=True)
PRIZES = {0: 2.0, 5: 3.0, 6: 5.0, 7: 80.0, 8: 720.0, 9: 8000.0, 10: 5_000_000.0}
STAKE = 2.0

def parse_sets(terminal_path: str):
    """Parse SET lines from terminal output."""
    sets = []
    with open(terminal_path) as f:
        for line in f:
            m = re.match(r'\s*SET_(\d+):\s+([\d\s]+)\s+\|', line)
            if m:
                idx, nums_str = m.groups()
                nums = [int(x) for x in nums_str.split()]
                if len(nums) == 10:
                    sets.append((int(idx), set(nums)))
    return sets

def main():
    terminal_path = "/Users/huaduojiejia/.cursor/projects/Users-huaduojiejia-Projects-lotto-image-predictor-research/terminals/1.txt"
    sets = parse_sets(terminal_path)
    print(f"Parsed {len(sets)} sets")
    print(f"Actual winning numbers: {sorted(ACTUAL)}")
    print()

    hit_hist = [0] * 11
    results = []
    for idx, s in sets:
        hits = len(s & ACTUAL)
        hit_hist[hits] += 1
        prize = PRIZES.get(hits, 0)
        match_nums = sorted(s & ACTUAL)
        results.append((idx, hits, prize, match_nums, s))

    # Sort by hits desc, then by prize desc
    results.sort(key=lambda x: (-x[1], -x[2]))

    print("=" * 90)
    print("HITS AND PRICE PER SET (sorted by hits, then prize)")
    print("=" * 90)
    print(f"{'SET':>6}  {'Hits':>4}  {'Prize':>10}  {'Match numbers'}")
    print("-" * 90)

    for idx, hits, prize, match_nums, _ in results:
        match_str = " ".join(f"{n:02d}" for n in match_nums) if match_nums else "-"
        print(f"{idx:>6}  {hits:>4}  {prize:>10.2f}  {match_str}")

    print()
    print("=" * 90)
    print("HIT DISTRIBUTION")
    print("=" * 90)
    for h in range(10, -1, -1):
        c = hit_hist[h]
        if c > 0:
            p = PRIZES.get(h, 0)
            print(f"  {h} hits: {c:4d} sets  (prize each: {p:>10.2f})")

    total_investment = len(sets) * STAKE
    total_prize = sum(r[2] for r in results)
    net = total_prize - total_investment
    print()
    print("=" * 90)
    print("MONEY SUMMARY")
    print("=" * 90)
    print(f"  Total tickets: {len(sets)}")
    print(f"  Investment (stake=2): {total_investment:.2f}")
    print(f"  Total prize: {total_prize:.2f}")
    print(f"  Net: {net:.2f}")
    print(f"  ROI: {100*net/total_investment:.2f}%")

    # Best sets
    print()
    print("=" * 90)
    print("TOP 20 SETS BY HITS")
    print("=" * 90)
    for idx, hits, prize, match_nums, full_set in results[:20]:
        full_str = " ".join(f"{n:02d}" for n in sorted(full_set))
        match_str = " ".join(f"{n:02d}" for n in match_nums) if match_nums else "-"
        print(f"  SET_{idx:03d}  hits={hits}  prize={prize:.2f}  match=[{match_str}]")
        print(f"           full=[{full_str}]")

if __name__ == "__main__":
    main()
