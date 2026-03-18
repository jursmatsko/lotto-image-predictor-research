"""
Signal quality analyzer for KL8 predictor.

Runs all signal providers against recent history, compares each signal's
top-N picks against actual draws, and ranks signals by hit rate.
Also shows which number features (hot/cold/gap/pair) correlate most with
actual draws.

Usage (from kle/ directory):
    python scripts/analyze_signals.py
    python scripts/analyze_signals.py --window 50    # last 50 draws only
    python scripts/analyze_signals.py --pick 10      # analyze pick=10 coverage
    python scripts/analyze_signals.py --draws 30     # evaluate on last 30 draws
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from predictor.signals import ALL_SIGNAL_PROVIDERS, TOTAL, DRAW_SIZE

# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path="data/data.csv"):
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if c.startswith("红球")]
    issues = df["期数"].astype(str).tolist()
    draws = df[num_cols].to_numpy(dtype=int)
    return issues, draws


def evaluate_signals(issues, draws, eval_draws=50, pick=10):
    """
    Walk-forward: for each of the last eval_draws draws, compute all signal
    scores using prior history, then measure how many of top-pick land in actual.

    Returns DataFrame with per-signal stats.
    """
    n = len(draws)
    results = {name: [] for name in ALL_SIGNAL_PROVIDERS}
    random_hits = []
    rng = np.random.default_rng(42)

    print(f"Evaluating {eval_draws} draws, pick={pick} ...")
    print("=" * 70)

    for step in range(eval_draws):
        i = step                    # newest=0, oldest=n-1
        hist = draws[i + 1:]        # all draws older than this one
        actual = set(int(x) for x in draws[i])
        issue  = issues[i]

        if len(hist) < 30:
            continue

        for name, func in ALL_SIGNAL_PROVIDERS.items():
            try:
                scores = func(hist)
                top_n = set(int(x) + 1 for x in np.argsort(scores)[::-1][:pick])
                hits = len(top_n & actual)
                results[name].append(hits)
            except Exception:
                results[name].append(0)

        rand_hits = len(set(rng.choice(np.arange(1, 81), pick, replace=False)) & actual)
        random_hits.append(rand_hits)

    # Build summary DataFrame
    rows = []
    for name, hit_list in results.items():
        if not hit_list:
            continue
        arr = np.array(hit_list)
        rows.append({
            "signal": name,
            "avg_hits": round(float(arr.mean()), 3),
            "max_hits": int(arr.max()),
            "p_ge_5":   round(float((arr >= 5).mean()), 3),
            "p_ge_6":   round(float((arr >= 6).mean()), 3),
            "p_ge_7":   round(float((arr >= 7).mean()), 3),
            "std":      round(float(arr.std()), 3),
        })
    df = pd.DataFrame(rows).sort_values("avg_hits", ascending=False).reset_index(drop=True)

    rand_arr = np.array(random_hits)
    rand_row = {
        "signal": "RANDOM (baseline)",
        "avg_hits": round(float(rand_arr.mean()), 3),
        "max_hits": int(rand_arr.max()),
        "p_ge_5":   round(float((rand_arr >= 5).mean()), 3),
        "p_ge_6":   round(float((rand_arr >= 6).mean()), 3),
        "p_ge_7":   round(float((rand_arr >= 7).mean()), 3),
        "std":      round(float(rand_arr.std()), 3),
    }
    df = pd.concat([df, pd.DataFrame([rand_row])], ignore_index=True)
    return df


def analyze_number_features(issues, draws, window=100):
    """
    For each number 1-80, compute features over recent history and
    rank by correlation with appearance in the NEXT draw.
    """
    n = len(draws)
    # Use window+1 draws: predict draw[0] using draws[1..window]
    eval_n = min(50, n - window - 1)

    feature_hits = {
        "frequency":  [],   # how often appeared in last window
        "gap":        [],   # draws since last appearance (cold = high gap)
        "streak":     [],   # consecutive draws appeared
        "pair_freq":  [],   # average co-occurrence with other numbers
    }

    for step in range(eval_n):
        hist = draws[step + 1 : step + 1 + window]
        actual = set(int(x) for x in draws[step])

        freq   = np.zeros(TOTAL)
        gap    = np.full(TOTAL, window)
        streak = np.zeros(TOTAL)
        pair   = np.zeros(TOTAL)

        for t, row in enumerate(hist):
            nums = [x - 1 for x in row if 1 <= x <= TOTAL]
            for ni in nums:
                freq[ni] += np.exp(-0.02 * t)  # recency-weighted frequency
                if gap[ni] == window:           # first seen → set gap
                    gap[ni] = t
                # pairs
                for nj in nums:
                    if ni != nj:
                        pair[ni] += np.exp(-0.02 * t)

        # Streak: how many of last 5 draws did number appear
        for t in range(min(5, len(hist))):
            for ni in [x - 1 for x in hist[t] if 1 <= x <= TOTAL]:
                streak[ni] += 1

        for feat_name, scores in [("frequency", freq), ("gap", -gap),
                                   ("streak", streak), ("pair_freq", pair)]:
            top20 = set(int(x) + 1 for x in np.argsort(scores)[::-1][:20])
            feature_hits[feat_name].append(len(top20 & actual))

    rows = []
    for feat, hits in feature_hits.items():
        arr = np.array(hits)
        rows.append({
            "feature": feat,
            "avg_hits_in_top20": round(float(arr.mean()), 3),
            "expected_random": round(20 * 20 / 80, 3),
            "lift_vs_random":  round(float(arr.mean()) / (20 * 20 / 80), 3),
        })
    return pd.DataFrame(rows).sort_values("lift_vs_random", ascending=False)


def analyze_best_sets(actual_str, sets_str, pick=10):
    """
    Given actual draw numbers and a list of (hits, set) from terminal output,
    find which number combinations appear most in high-hit sets.
    """
    actual = set(int(x) for x in actual_str.split())
    lines = [l.strip() for l in sets_str.strip().splitlines() if "hits=" in l]

    hit_sets = []
    for line in lines:
        try:
            h = int(line.split("hits=")[1].split("/")[0])
            nums_part = line.split("|")[2].split("match=")[0].strip()
            nums = set(int(x) for x in nums_part.split())
            hit_sets.append((h, nums))
        except Exception:
            continue

    if not hit_sets:
        return

    hit_sets.sort(key=lambda x: -x[0])
    top_sets = [s for h, s in hit_sets if h >= hit_sets[0][0] - 1]

    # Count how often each number appears in the top sets
    freq = {}
    for s in top_sets:
        for n in s:
            freq[n] = freq.get(n, 0) + 1

    print(f"\nTop sets had {hit_sets[0][0]} hits. Numbers appearing in ≥50% of top sets:")
    threshold = len(top_sets) * 0.5
    hot = sorted([(n, c) for n, c in freq.items() if c >= threshold], key=lambda x: -x[1])
    for num, cnt in hot:
        marker = " ✓ IN ACTUAL" if num in actual else ""
        print(f"  {num:>3}  (appears in {cnt}/{len(top_sets)} top sets){marker}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Signal quality analyzer for KL8 predictor")
    parser.add_argument("--draws",  type=int, default=50, help="Number of recent draws to evaluate (default: 50)")
    parser.add_argument("--pick",   type=int, default=10, help="Numbers picked per set (default: 10)")
    parser.add_argument("--window", type=int, default=100, help="History window for feature analysis (default: 100)")
    parser.add_argument("--fast",   action="store_true", help="Skip slow signals (genetic, recurrent)")
    args = parser.parse_args()

    issues, draws = load_data()
    print(f"\nData: {len(issues)} draws, latest={issues[0]}")

    # ── 1. Signal ranking ────────────────────────────────────────────────────
    providers = dict(ALL_SIGNAL_PROVIDERS)
    if args.fast:
        for slow in ("genetic", "recurrent", "svd", "fourier"):
            providers.pop(slow, None)
    original = ALL_SIGNAL_PROVIDERS.copy()
    ALL_SIGNAL_PROVIDERS.clear()
    ALL_SIGNAL_PROVIDERS.update(providers)

    print(f"\n{'─'*70}")
    print(f"SIGNAL RANKING  (pick={args.pick}, evaluated on last {args.draws} draws)")
    print(f"{'─'*70}")
    sig_df = evaluate_signals(issues, draws, eval_draws=args.draws, pick=args.pick)
    print(sig_df.to_string(index=True))

    ALL_SIGNAL_PROVIDERS.clear()
    ALL_SIGNAL_PROVIDERS.update(original)

    # ── 2. Number feature analysis ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"NUMBER FEATURE ANALYSIS  (window={args.window}, eval on last 50 draws)")
    print(f"  lift_vs_random > 1.0 means the feature beats random selection")
    print(f"{'─'*70}")
    feat_df = analyze_number_features(issues, draws, window=args.window)
    print(feat_df.to_string(index=False))

    # ── 3. Improvement suggestions ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("IMPROVEMENT SUGGESTIONS")
    print(f"{'─'*70}")
    best_sig = sig_df.iloc[0]["signal"]
    worst_sig = sig_df[sig_df["signal"] != "RANDOM (baseline)"].iloc[-1]["signal"]
    rand_avg  = sig_df[sig_df["signal"] == "RANDOM (baseline)"]["avg_hits"].values[0]
    best_avg  = sig_df.iloc[0]["avg_hits"]
    beat_random = sig_df[(sig_df["signal"] != "RANDOM (baseline)") & (sig_df["avg_hits"] > rand_avg)]

    print(f"  Best signal  : {best_sig} (avg {best_avg} hits)")
    print(f"  Worst signal : {worst_sig}")
    print(f"  Random baseline avg: {rand_avg} hits")
    print(f"  Signals beating random: {len(beat_random)}/{len(sig_df)-1}")
    print()
    print("  To boost accuracy:")
    if len(beat_random) > 0:
        top3 = ", ".join(beat_random.head(3)["signal"].tolist())
        print(f"  1. Up-weight these signals in MemoryBank: {top3}")
    print(f"  2. Use --epochs 3 to refine memory weights over same eval window")
    print(f"  3. Use --latest 200 --full-dataset for deeper training")
    print(f"  4. Increase --n-cover (more sets = better coverage probability)")
    print(f"  5. Consider ensemble: run pick=10 AND pick=8 and take union of best sets")
    print()


if __name__ == "__main__":
    main()
