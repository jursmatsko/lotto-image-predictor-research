#!/usr/bin/env python3
"""Evaluate 2026046 pick-11 prediction against actual winning numbers."""

import re

# Actual winning numbers for 2026046 (2026-02-25)
ACTUAL = {}

def parse_sets(terminal_path: str, pick: int = 11):
    """Parse SET lines from terminal output."""
    sets = []
    with open(terminal_path) as f:
        for line in f:
            m = re.match(r'\s*SET_(\d+):\s+([\d\s]+)\s+\|', line)
            if m:
                idx, nums_str = m.groups()
                nums = [int(x) for x in nums_str.split()]
                if len(nums) == pick:
                    sets.append((int(idx), set(nums)))
    return sets

def main():
    terminal_path = "/Users/huaduojiejia/.cursor/projects/Users-huaduojiejia-Projects-lotto-image-predictor-research/terminals/3.txt"
    sets = parse_sets(terminal_path, pick=11)
    print(f"Parsed {len(sets)} sets (pick 11)")
    print(f"Actual winning numbers: {sorted(ACTUAL)}")
    print()

    hit_hist = [0] * 12  # 0..11 hits
    results = []
    for idx, s in sets:
        hits = len(s & ACTUAL)
        hit_hist[hits] += 1
        match_nums = sorted(s & ACTUAL)
        results.append((idx, hits, match_nums, s))

    # Sort by hits desc
    results.sort(key=lambda x: -x[1])

    print("=" * 95)
    print("HITS PER SET (pick 11, sorted by hits)")
    print("=" * 95)
    print(f"{'SET':>6}  {'Hits':>4}  {'Match numbers'}")
    print("-" * 95)

    for idx, hits, match_nums, _ in results:
        match_str = " ".join(f"{n:02d}" for n in match_nums) if match_nums else "-"
        print(f"{idx:>6}  {hits:>4}  {match_str}")

    print()
    print("=" * 95)
    print("HIT DISTRIBUTION")
    print("=" * 95)
    for h in range(11, -1, -1):
        c = hit_hist[h]
        if c > 0:
            print(f"  {h} hits: {c:4d} sets")

    # Best sets
    print()
    print("=" * 95)
    print("TOP 20 SETS BY HITS")
    print("=" * 95)
    for idx, hits, match_nums, full_set in results[:20]:
        full_str = " ".join(f"{n:02d}" for n in sorted(full_set))
        match_str = " ".join(f"{n:02d}" for n in match_nums) if match_nums else "-"
        print(f"  SET_{idx:03d}  hits={hits}  match=[{match_str}]")
        print(f"           full=[{full_str}]")

if __name__ == "__main__":
    main()
