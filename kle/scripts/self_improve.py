#!/usr/bin/env python3
"""
Self-improvement loop for the KLE predictor.

Two goals (modes):
  1) predictor: improve money EV/ROI by evolving predictor hyperparameters (walk-forward evaluation).
  2) estimation: improve "Top-K estimated hit numbers" by backtesting match extraction on history.

This is intentionally lightweight (no extra deps beyond numpy/pandas already used by the project).
It writes an "OpenSage-style" report (md + json) so improvements are persistent and auditable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Candidate:
    # Predictor hyperparams
    memory_lr: float
    memory_decay: float
    pair_boost_weight: float
    replay_weight: float
    temp_base: float
    replay_min_hits: int

    # Estimation params
    top_k: int
    windows: tuple[int, ...]  # history window lengths used for window-consensus voting

    # Score / metadata
    score: float = float("-inf")
    details: dict | None = None


SURROGATE_GE_DEFAULT = 7

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _mutate(c: Candidate, rng: random.Random) -> Candidate:
    # Gaussian-ish mutations with clipping.
    def n(mu: float, sigma: float) -> float:
        return rng.gauss(mu, sigma)

    memory_lr = _clip(n(c.memory_lr, 0.03), 0.03, 0.40)
    memory_decay = _clip(n(c.memory_decay, 0.01), 0.85, 0.99)
    pair_boost_weight = _clip(n(c.pair_boost_weight, 0.03), 0.0, 0.45)
    replay_weight = _clip(n(c.replay_weight, 0.03), 0.0, 0.45)
    temp_base = _clip(n(c.temp_base, 0.10), 0.6, 2.0)
    replay_min_hits = int(_clip(round(n(float(c.replay_min_hits), 1.0)), 2, 10))

    top_k = int(_clip(round(n(float(c.top_k), 3.0)), 10, 35))

    wins = list(c.windows)
    # Occasionally tweak one window or add/drop.
    if rng.random() < 0.6 and wins:
        j = rng.randrange(len(wins))
        wins[j] = int(_clip(round(n(float(wins[j]), max(5.0, wins[j] * 0.15))), 10, 600))
    if rng.random() < 0.25 and len(wins) < 6:
        wins.append(rng.choice([20, 30, 45, 60, 90, 120, 180, 240, 300]))
    if rng.random() < 0.15 and len(wins) > 1:
        wins.pop(rng.randrange(len(wins)))
    wins = tuple(sorted({int(w) for w in wins if w >= 5}))
    if not wins:
        wins = (30, 60, 120, 240)

    return Candidate(
        memory_lr=memory_lr,
        memory_decay=memory_decay,
        pair_boost_weight=pair_boost_weight,
        replay_weight=replay_weight,
        temp_base=temp_base,
        replay_min_hits=replay_min_hits,
        top_k=top_k,
        windows=wins,
    )


def _random_candidate(rng: random.Random) -> Candidate:
    wins = tuple(sorted({rng.choice([20, 30, 45, 60, 90, 120, 180, 240]) for _ in range(rng.randint(2, 4))}))
    return Candidate(
        memory_lr=rng.uniform(0.06, 0.25),
        memory_decay=rng.uniform(0.88, 0.97),
        pair_boost_weight=rng.uniform(0.05, 0.30),
        replay_weight=rng.uniform(0.05, 0.30),
        temp_base=rng.uniform(0.8, 1.4),
        replay_min_hits=rng.randint(3, 8),
        top_k=rng.randint(15, 28),
        windows=wins or (30, 60, 120),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_report(out_dir: str, title: str, best: Candidate, history: list[dict]) -> tuple[str, str]:
    _ensure_dir(out_dir)
    stamp = _now_stamp()
    md_path = os.path.join(out_dir, f"{stamp}_{title}.md")
    json_path = os.path.join(out_dir, f"{stamp}_{title}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"best": asdict(best), "history": history}, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"## {title} report")
    lines.append("")
    lines.append(f"- **best_score**: {best.score:.6f}")
    lines.append(f"- **best_config**: `{json.dumps({k:v for k,v in asdict(best).items() if k not in ('score','details')}, ensure_ascii=False)}`")
    if best.details:
        lines.append(f"- **best_details**: `{json.dumps(best.details, ensure_ascii=False)}`")
    lines.append("")
    lines.append("## Top improvements (latest)")
    lines.append("")
    for row in history[-10:]:
        lines.append(f"- score={row['score']:.6f} cfg={row['cfg']}")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return md_path, json_path


def _eval_predictor(
    candidate: Candidate,
    csv_path: str,
    target: str,
    n_warmup: int,
    n_eval: int,
    n_cover: int,
    epochs: int,
    surrogate_ge: int = SURROGATE_GE_DEFAULT,
) -> tuple[float, dict]:
    # Import here so running "estimation" mode doesn't need optuna installed.
    from optimize_predictor import run_one_trial

    metrics = run_one_trial(
        csv_path=csv_path,
        target_issue=target,
        pick=10,
        n_warmup=n_warmup,
        n_eval=n_eval,
        n_cover=n_cover,
        epochs=epochs,
        memory_lr=candidate.memory_lr,
        memory_decay=candidate.memory_decay,
        quiet=True,
        trial_number=None,
        status_stream=False,
        pair_boost_weight=candidate.pair_boost_weight,
        replay_weight=candidate.replay_weight,
        temp_base=candidate.temp_base,
        replay_min_hits=candidate.replay_min_hits,
        save_memory_path="",
        payout="kl8_pick10",
    )
    if not metrics:
        return float("-inf"), {"error": "trial_failed"}
    # Jackpot-focused scoring: use a dense surrogate (P>=surrogate_ge) for learning,
    # while still reporting P>=9 and P10 as tail metrics.
    hit_hist = metrics.get("hit_hist_0_pick")
    if isinstance(hit_hist, list) and hit_hist:
        total = float(sum(hit_hist))
        p10 = float(hit_hist[10]) / total if total > 0 and len(hit_hist) > 10 else 0.0
        pge9 = float(hit_hist[9] + hit_hist[10]) / total if total > 0 and len(hit_hist) > 10 else 0.0
        ge = int(max(5, min(surrogate_ge, 10)))
        pge_surr = float(sum(hit_hist[ge:])) / total if total > 0 and len(hit_hist) > 10 else 0.0
    else:
        p10 = 0.0
        pge9 = 0.0
        pge_surr = 0.0
    # Dense objective: P>=surrogate_ge dominates; tail metrics are tie-breakers.
    score = (pge_surr * 1_000_000.0) + (pge9 * 1_000.0) + (p10 * 1.0)
    details = {
        "p10": float(p10),
        "pge9": float(pge9),
        "surrogate_ge": int(max(5, min(surrogate_ge, 10))),
        "pge_surr": float(pge_surr),
        "ev_per_ticket": float(metrics.get("ev_per_ticket", 0.0)) if metrics.get("ev_per_ticket") is not None else None,
        "roi": float(metrics.get("roi", 0.0)) if metrics.get("roi") is not None else None,
        "cover_best_mean": float(metrics.get("cover_best_mean", 0.0)),
        "p_ge_8": float(metrics.get("p_ge_8", 0.0)),
        "total_tickets": int(metrics.get("total_tickets", 0)),
    }
    return score, details


def _eval_estimation(candidate: Candidate, csv_path: str, backtest_n: int, pick: int = 15) -> tuple[float, dict]:
    """
    Backtest "estimated hit numbers" extraction:
      - For each of last backtest_n draws i, compute fused distribution using history after i (t_hist),
        and create a Top-K estimate.
      - If windows are provided, do window-consensus voting across multiple history window sizes (on t_hist).
      - Score = average matches between estimate and actual (out of 20), across backtest draws.
    """
    from generative_memory_predictor import load_data, FAST_SIGNAL_PROVIDERS, MemoryBank, GenerativeModel, TOTAL

    issues, draws = load_data(csv_path)
    # draws[0] is latest. We backtest on a slice that has enough history after it.
    n = int(max(5, min(backtest_n, len(draws) - 10)))
    method_names = list(FAST_SIGNAL_PROVIDERS.keys())
    mem = MemoryBank(method_names)  # fresh memory; estimation here focuses on fused signals stability
    gen = GenerativeModel(mem)

    wins = candidate.windows or (30, 60, 120, 240)
    wins = tuple(int(w) for w in wins if w >= 5)
    k = int(max(5, min(candidate.top_k, 40)))

    matches = []
    for i in range(n):
        t_actual = set(int(x) for x in draws[i])
        t_hist_full = draws[i + 1:]

        counts = np.zeros(TOTAL, dtype=int)
        for w in wins:
            t_hist = t_hist_full[:w] if w < len(t_hist_full) else t_hist_full
            scores = {name: func(t_hist) for name, func in FAST_SIGNAL_PROVIDERS.items()}
            fused = gen.compute_fused_distribution(scores, t_hist)
            top = np.argsort(fused)[::-1][:k]
            counts[top] += 1

        est = {int(j) + 1 for j in np.argsort(counts)[::-1][:k]}
        matches.append(len(est & t_actual))

    arr = np.array(matches, dtype=float)
    score = float(arr.mean())
    details = {
        "avg_matches": float(arr.mean()),
        "std_matches": float(arr.std()),
        "min_matches": int(arr.min()) if len(arr) else 0,
        "max_matches": int(arr.max()) if len(arr) else 0,
        "k": k,
        "windows": list(wins),
        "backtest_n": n,
    }
    return score, details


def evolve(
    mode: str,
    csv_path: str,
    target: str,
    generations: int,
    pop_size: int,
    elite: int,
    seed: int,
    # predictor eval settings
    n_warmup: int,
    n_eval: int,
    n_cover: int,
    epochs: int,
    # estimation eval settings
    backtest_n: int,
    pick: int,
    surrogate_ge: int = SURROGATE_GE_DEFAULT,
) -> tuple[Candidate, list[dict]]:
    rng = random.Random(seed)
    pop = [_random_candidate(rng) for _ in range(pop_size)]

    history: list[dict] = []
    best = None

    for g in range(generations):
        scored: list[Candidate] = []
        for c in pop:
            if mode == "predictor":
                score, details = _eval_predictor(c, csv_path, target, n_warmup, n_eval, n_cover, epochs, surrogate_ge=surrogate_ge)
            else:
                score, details = _eval_estimation(c, csv_path, backtest_n=backtest_n, pick=pick)
            c.score = float(score)
            c.details = details
            scored.append(c)
            if mode == "predictor":
                # Minimal progress line (jackpot-focused)
                p10 = (details or {}).get("p10", 0.0)
                pge9 = (details or {}).get("pge9", 0.0)
                ge = (details or {}).get("surrogate_ge", SURROGATE_GE_DEFAULT)
                pge_surr = (details or {}).get("pge_surr", 0.0)
                print(f"[gen {g}] score={c.score:.2f} p>={ge}={pge_surr:.6f} pge9={pge9:.6f} p10={p10:.6f}", flush=True)
            history.append(
                {
                    "gen": g,
                    "score": float(score),
                    "cfg": {k: v for k, v in asdict(c).items() if k not in ("score", "details")},
                    "details": details,
                }
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        if best is None or scored[0].score > best.score:
            best = scored[0]

        # Next generation: keep elites, fill rest by mutating elites.
        elites = scored[: max(1, min(elite, len(scored)))]
        next_pop = [Candidate(**{k: v for k, v in asdict(e).items() if k not in ("score", "details")}) for e in elites]
        while len(next_pop) < pop_size:
            parent = rng.choice(elites)
            child = _mutate(parent, rng)
            next_pop.append(child)
        pop = next_pop

    assert best is not None
    return best, history


def main() -> None:
    p = argparse.ArgumentParser(description="Self-improve predictor / estimation (AlphaEvolve + OpenSage report).")
    p.add_argument("--mode", choices=["predictor", "estimation"], default="estimation")
    p.add_argument("--csv", default="data/data.csv", help="CSV path (auto-fallback exists in generative_memory_predictor).")
    p.add_argument("--target", default="2026065", help="Target issue for predictor mode.")
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--pop-size", type=int, default=10)
    p.add_argument("--elite", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)

    # Predictor eval settings
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-eval", type=int, default=15)
    p.add_argument("--n-cover", type=int, default=20)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--surrogate-ge", type=int, default=SURROGATE_GE_DEFAULT, help="Jackpot surrogate objective: optimize P(hits>=N) (default: 7)")

    # Estimation eval settings
    p.add_argument("--backtest-n", type=int, default=80)
    p.add_argument("--pick", type=int, default=15)

    p.add_argument("--out-dir", default="storage/sage_reports", help="Where to write reports (relative to kle/).")
    args = p.parse_args()

    # Make paths work when running from repo root or kle/
    if not os.path.isabs(args.csv) and not os.path.isfile(args.csv):
        here = os.path.dirname(os.path.abspath(__file__))  # .../kle/scripts
        candidate = os.path.normpath(os.path.join(here, "..", "data", "data.csv"))
        if os.path.isfile(candidate):
            args.csv = candidate

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        # place under kle/
        kle_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        out_dir = os.path.join(kle_dir, out_dir)

    best, history = evolve(
        mode=args.mode,
        csv_path=args.csv,
        target=args.target,
        generations=args.generations,
        pop_size=args.pop_size,
        elite=args.elite,
        seed=args.seed,
        n_warmup=args.n_warmup,
        n_eval=args.n_eval,
        n_cover=args.n_cover,
        epochs=args.epochs,
        backtest_n=args.backtest_n,
        pick=args.pick,
        surrogate_ge=args.surrogate_ge,
    )

    title = f"self_improve_{args.mode}"
    md_path, json_path = _write_report(out_dir, title, best, history)
    print("BEST SCORE:", f"{best.score:.6f}")
    print("BEST CONFIG:", json.dumps({k: v for k, v in asdict(best).items() if k not in ("score", "details")}, ensure_ascii=False))
    print("BEST DETAILS:", json.dumps(best.details or {}, ensure_ascii=False))
    print("REPORTS:", md_path, json_path)


if __name__ == "__main__":
    main()

