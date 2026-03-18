#!/usr/bin/env python3
"""
Auto-tune KLE generative memory predictor hyperparameters.

Inspired by Hyperparameter Hunter (https://github.com/HunterMcGushion/hyperparameter_hunter):
  - Automatically search over key parameters
  - Persist all trials (SQLite + optional CSV) so results are reused
  - No re-runs of the same configuration

Usage (from kle/):
  pip install optuna   # if not installed
  python scripts/optimize_predictor.py --target 2026060 --n-trials 20
  python scripts/optimize_predictor.py --target 2026060 --n-trials 50 --storage optuna_study.db

Best params are printed and can be copied into CLI or a config.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

# Add kle/scripts so we can import generative_memory_predictor
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

import numpy as np
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Install optuna: pip install optuna", file=sys.stderr)
    sys.exit(1)


# Default CSV path relative to kle/
DEFAULT_CSV = "data/data.csv"
DEFAULT_STORAGE = "storage/optuna_predictor_study.db"


def load_data(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if c.startswith("红球")]
    issues = df["期数"].astype(str).tolist()
    draws = df[num_cols].to_numpy(dtype=int)
    return issues, draws


def run_one_trial(
    csv_path: str,
    target_issue: str,
    n_warmup: int,
    n_eval: int,
    n_cover: int,
    epochs: int,
    memory_lr: float,
    memory_decay: float,
    quiet: bool = True,
    trial_number: int | None = None,
    status_stream: bool = True,
    pair_boost_weight: float = 0.15,
    replay_weight: float = 0.20,
    temp_base: float = 1.0,
    replay_min_hits: int = 4,
    save_memory_path: str = "",
    payout: str = "none",
    pick: int = 15,
) -> dict | None:
    """Run walk-forward once; return metrics dict or None on failure.
    If save_memory_path is set, memory is saved there at the end of the run."""
    from generative_memory_predictor import (
        run_walk_forward_with_memory,
        load_data as load_data_predictor,
        _set_pick,
    )

    _set_pick(int(pick))
    issues, draws = load_data_predictor(csv_path)
    if target_issue not in issues:
        return None

    def _progress(phase: str, current: int, total: int) -> None:
        if not status_stream or trial_number is None:
            return
        label = {"warmup": "warmup", "eval_epoch": "epoch", "eval_step": "eval"}[phase]
        if phase == "eval_epoch":
            msg = f"  [Trial {trial_number}] {label} {current}/{total}"
        else:
            msg = f"  [Trial {trial_number}] {label} {current}/{total}"
        print(msg, file=sys.stderr, flush=True)

    progress_cb = _progress if (status_stream and trial_number is not None) else None

    # Fusion: fused + attention(0.2) + pair_boost + replay = 1 (normalized)
    attention_weight = 0.20
    fused_weight = 1.0 - attention_weight - pair_boost_weight - replay_weight
    fused_weight = max(0.25, min(0.65, fused_weight))
    total = fused_weight + attention_weight + pair_boost_weight + replay_weight
    fusion_weights = (
        fused_weight / total,
        attention_weight / total,
        pair_boost_weight / total,
        replay_weight / total,
    )

    if status_stream and trial_number is not None:
        print(
            f"  [Trial {trial_number}] starting warmup={n_warmup} eval={n_eval} epochs={epochs} ...",
            file=sys.stderr,
            flush=True,
        )

    try:
        save_path = save_memory_path if save_memory_path else ""
        if quiet:
            f = io.StringIO()
            with redirect_stdout(f):
                _, _, metrics = run_walk_forward_with_memory(
                    issues,
                    draws,
                    target_issue,
                    payout=payout,
                    n_eval=n_eval,
                    n_warmup=n_warmup,
                    n_cover=n_cover,
                    epochs=epochs,
                    memory_lr=memory_lr,
                    memory_decay=memory_decay,
                    save_memory=save_path,
                    load_memory="",
                    progress_callback=progress_cb,
                    fusion_weights=fusion_weights,
                    temp_base=temp_base,
                    replay_min_hits=replay_min_hits,
                    fast_signals=True,
                )
        else:
            _, _, metrics = run_walk_forward_with_memory(
                issues,
                draws,
                target_issue,
                payout=payout,
                n_eval=n_eval,
                n_warmup=n_warmup,
                n_cover=n_cover,
                epochs=epochs,
                memory_lr=memory_lr,
                memory_decay=memory_decay,
                save_memory=save_path,
                load_memory="",
                progress_callback=progress_cb,
                fusion_weights=fusion_weights,
                temp_base=temp_base,
                replay_min_hits=replay_min_hits,
                fast_signals=True,
            )
        if status_stream and trial_number is not None and metrics is not None:
            print(
                f"  [Trial {trial_number}] done  cover_best_mean={metrics['cover_best_mean']:.3f}",
                file=sys.stderr,
                flush=True,
            )
        return metrics
    except Exception:
        return None


def create_study(storage: str, study_name: str = "kle_predictor", metric: str = "cover_best_mean"):
    """Create or load Optuna study with persistent storage (like HH's result saving)."""
    os.makedirs(os.path.dirname(storage) or ".", exist_ok=True)
    load_if_exists = os.path.isfile(storage)
    sampler = TPESampler(seed=42, n_startup_trials=5)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage}",
        load_if_exists=load_if_exists,
        direction="maximize",
        sampler=sampler,
    )
    return study


def objective(
    trial: optuna.Trial,
    csv_path: str,
    target_issue: str,
    fixed_n_cover: int | None,
    fixed_epochs: int | None,
    show_status: bool = True,
    quick: bool = False,
    pick: int = 15,
    payout: str = "none",
    objective_name: str = "ev",
) -> float:
    """Optuna objective: ev (money) or jackpot probabilities (p10/pge9/both)."""
    if quick:
        n_warmup = trial.suggest_int("n_warmup", 3, 10)
        n_eval = trial.suggest_int("n_eval", 6, 18)
    else:
        n_warmup = trial.suggest_int("n_warmup", 5, 25)
        n_eval = trial.suggest_int("n_eval", 10, 40)
    n_cover = fixed_n_cover if fixed_n_cover is not None else trial.suggest_int("n_cover", 10, 50)
    epochs = fixed_epochs if fixed_epochs is not None else trial.suggest_int("epochs", 1, 5)
    memory_lr = trial.suggest_float("memory_lr", 0.05, 0.35, log=False)
    memory_decay = trial.suggest_float("memory_decay", 0.85, 0.99, log=False)
    pair_boost_weight = trial.suggest_float("pair_boost_weight", 0.05, 0.40, log=False)
    replay_weight = trial.suggest_float("replay_weight", 0.10, 0.35, log=False)
    temp_base = trial.suggest_float("temp_base", 0.6, 1.6, log=False)
    replay_min_hits = trial.suggest_int("replay_min_hits", 3, 7)

    metrics = run_one_trial(
        csv_path=csv_path,
        target_issue=target_issue,
        n_warmup=n_warmup,
        n_eval=n_eval,
        n_cover=n_cover,
        epochs=epochs,
        memory_lr=memory_lr,
        memory_decay=memory_decay,
        quiet=True,
        trial_number=trial.number + 1,
        status_stream=show_status,
        pair_boost_weight=pair_boost_weight,
        replay_weight=replay_weight,
        temp_base=temp_base,
        replay_min_hits=replay_min_hits,
        payout=payout,
        pick=pick,
    )
    if metrics is None:
        return float("-inf")
    hit_hist = metrics.get("hit_hist_0_pick")
    if isinstance(hit_hist, list) and hit_hist:
        total = float(sum(hit_hist))
        if total > 0 and len(hit_hist) >= 11:
            p10 = float(hit_hist[10]) / total
            pge9 = float(hit_hist[9] + hit_hist[10]) / total
        else:
            p10 = 0.0
            pge9 = 0.0
    else:
        p10 = 0.0
        pge9 = 0.0

    if objective_name == "p10":
        return p10
    if objective_name == "pge9":
        return pge9
    if objective_name == "both":
        # prioritize jackpot, but reward 9-hit as a denser learning signal
        return (p10 * 1_000_000.0) + (pge9 * 1_000.0)

    # Default EV objective.
    if metrics.get("ev_per_ticket") is not None:
        return float(metrics["ev_per_ticket"])
    return float(metrics["cover_best_mean"])


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune KLE predictor hyperparameters (Optuna + persistent storage)."
    )
    parser.add_argument("--target", default="2026060", help="Target issue for walk-forward")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--storage",
        default=DEFAULT_STORAGE,
        help=f"SQLite path for study (default: {DEFAULT_STORAGE})",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to data.csv")
    parser.add_argument("--pick", type=int, default=15, help="Pick size (default: 15). Use 10 for kl8_pick10 jackpot.")
    parser.add_argument("--payout", type=str, default="none", help="Payout name passed to evaluator (default: none).")
    parser.add_argument(
        "--objective",
        type=str,
        default="ev",
        choices=["ev", "p10", "pge9", "both"],
        help="Optimization objective: ev (money), p10, pge9, or both (jackpot-focused).",
    )
    parser.add_argument(
        "--n-cover",
        type=int,
        default=None,
        help="Fix n_cover (else searched in [10,50])",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Fix epochs (else searched in [1,5])",
    )
    parser.add_argument(
        "--save-best",
        type=str,
        default="",
        help="Save best params to this JSON file",
    )
    parser.add_argument(
        "--save-best-memory",
        type=str,
        default="",
        metavar="PATH",
        help="After optimization, re-run the best trial and save memory to PATH (e.g. storage/memory_best.npz)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each trial (no redirect)")
    parser.add_argument(
        "--no-status",
        action="store_true",
        dest="no_status",
        help="Disable live status lines (warmup/eval progress) during trials",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller n_warmup/n_eval ranges so each trial is ~2–3x faster (less accurate proxy)",
    )
    parser.add_argument(
        "--predict-after",
        type=int,
        default=0,
        metavar="N",
        help="After saving best memory, run prediction for next draw and print N sets (0=skip). Requires --save-best-memory.",
    )
    parser.add_argument(
        "--actual",
        type=str,
        default="",
        help='If set, after prediction compute match (hits) and payout vs this draw. 20 comma-separated numbers, e.g. "7,13,16,21,25,31,32,38,45,49,57,58,59,61,63,67,69,72,75,77".',
    )
    args = parser.parse_args()
    args.show_status = not args.no_status

    # Run from kle/ so paths are correct
    kle_dir = os.path.dirname(_script_dir)
    os.chdir(kle_dir)
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(kle_dir, args.csv)
    storage_path = args.storage if os.path.isabs(args.storage) else os.path.join(kle_dir, args.storage)

    study = create_study(storage_path)
    print(f"Study: {study.study_name}, storage: {storage_path}")
    print(f"Target issue: {args.target}, n_trials: {args.n_trials}")
    if args.n_cover is not None:
        print(f"Fixed n_cover: {args.n_cover}")
    if args.epochs is not None:
        print(f"Fixed epochs: {args.epochs}")
    if args.quick:
        print("Quick mode: smaller warmup/eval (faster trials)")
    print(f"Optimizing: {args.objective} (maximize)\n")

    study.optimize(
        lambda t: objective(
            t, csv_path, args.target, args.n_cover, args.epochs,
            show_status=args.show_status,
            quick=args.quick,
            pick=args.pick,
            payout=args.payout,
            objective_name=args.objective,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    if study.best_trial is None:
        print("No successful trials.")
        return

    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST PARAMETERS (copy to CLI or config)")
    print("=" * 60)
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Best cover_best_mean: {best.value:.4f}")

    print("\nExample CLI (from kle/):")
    p = best.params
    n_cover = args.n_cover if args.n_cover is not None else p.get("n_cover", 20)
    epochs_val = args.epochs if args.epochs is not None else p.get("epochs", 1)
    cmd = (
        f"  python scripts/generative_memory_predictor.py --target {args.target} "
        f"--n-warmup {p['n_warmup']} --n-eval {p['n_eval']} --n-cover {n_cover} "
        f"--epochs {epochs_val} --memory-lr {p['memory_lr']:.4f} --memory-decay {p['memory_decay']:.4f} "
        f"--pair-boost-weight {p['pair_boost_weight']:.4f} --replay-weight {p['replay_weight']:.4f} "
        f"--temp-base {p['temp_base']:.4f} --replay-min-hits {p['replay_min_hits']} "
        f"--save-memory storage/memory_best.npz"
    )
    print(cmd)

    if args.save_best:
        out = {
            "target_issue": args.target,
            "cover_best_mean": best.value,
            "params": best.params,
        }
        with open(args.save_best, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nBest params saved to {args.save_best}")

    if args.save_best_memory:
        save_path = args.save_best_memory if os.path.isabs(args.save_best_memory) else os.path.join(kle_dir, args.save_best_memory)
        print(f"\nRe-running best trial to save memory to {save_path} ...")
        # (Trials don't persist memory; re-run once with best params so we can save the .npz.)
        p = best.params
        n_cover = args.n_cover if args.n_cover is not None else p.get("n_cover", 20)
        epochs_val = args.epochs if args.epochs is not None else p.get("epochs", 1)
        run_one_trial(
            csv_path=csv_path,
            target_issue=args.target,
            n_warmup=p["n_warmup"],
            n_eval=p["n_eval"],
            n_cover=n_cover,
            epochs=epochs_val,
            memory_lr=p["memory_lr"],
            memory_decay=p["memory_decay"],
            quiet=True,
            trial_number=None,
            status_stream=True,
            pair_boost_weight=p["pair_boost_weight"],
            replay_weight=p["replay_weight"],
            temp_base=p["temp_base"],
            replay_min_hits=p["replay_min_hits"],
            save_memory_path=save_path,
        )
        print(f"Best-trial memory saved to {save_path}")

    if args.predict_after > 0 and study.best_trial is not None:
        n_sets = args.predict_after
        if args.save_best_memory:
            memory_path = args.save_best_memory if os.path.isabs(args.save_best_memory) else os.path.join(kle_dir, args.save_best_memory)
        else:
            memory_path = os.path.join(kle_dir, "storage/memory_best.npz")
        if not os.path.isfile(memory_path):
            print("\n[--predict-after] No memory file found; run with --save-best-memory PATH first.")
        else:
            print(f"\n--- PREDICTION FOR NEXT DRAW (using best memory, {n_sets} sets) ---")
            from generative_memory_predictor import (
                load_data as load_data_predictor,
                MemoryBank,
                GenerativeModel,
                ALL_SIGNAL_PROVIDERS,
                _set_pick,
            )
            _set_pick(10)  # match typical pick-10 usage
            issues, draws = load_data_predictor(csv_path)
            hist = draws
            method_names = list(ALL_SIGNAL_PROVIDERS.keys())
            memory = MemoryBank.load(memory_path, method_names)
            p = best.params
            att_w = 0.20
            fused_w = 1.0 - att_w - p["pair_boost_weight"] - p["replay_weight"]
            fused_w = max(0.25, min(0.65, fused_w))
            total_w = fused_w + att_w + p["pair_boost_weight"] + p["replay_weight"]
            fusion_weights = (fused_w / total_w, att_w / total_w, p["pair_boost_weight"] / total_w, p["replay_weight"] / total_w)
            gen = GenerativeModel(memory, fusion_weights=fusion_weights, temp_base=p["temp_base"], replay_min_hits=p["replay_min_hits"])
            scores = {name: func(hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
            fused = gen.compute_fused_distribution(scores, hist)
            top10 = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:10])
            sets = gen.generate_sets(fused, n_sets=n_sets, seed=9999)
            print(f"Top 10 by fused score: {top10}")
            print(f"\n{n_sets} generated sets:")
            for i, s in enumerate(sets, 1):
                print(f"  SET_{i:02d}: {' '.join(f'{x:02d}' for x in sorted(s))}")

            if args.actual:
                actual = {int(x.strip()) for x in args.actual.split(",") if x.strip()}
                if len(actual) == 20:
                    stake, prize5, prize6, prize7, prize8, prize9, prize10 = 2.0, 3.0, 5.0, 80.0, 720.0, 8000.0, 5000000.0
                    hit_hist = [0] * 11
                    for s in sets:
                        h = len(set(s) & actual)
                        if 0 <= h <= 10:
                            hit_hist[h] += 1
                    c0, c5, c6, c7, c8, c9, c10 = hit_hist[0], hit_hist[5], hit_hist[6], hit_hist[7], hit_hist[8], hit_hist[9], hit_hist[10]
                    total_prize = c0 * stake + c5 * prize5 + c6 * prize6 + c7 * prize7 + c8 * prize8 + c9 * prize9 + c10 * prize10
                    investment = n_sets * stake
                    net = total_prize - investment
                    roi = (net / investment * 100) if investment else 0
                    print(f"\nMatch vs actual (选十): hits 0/5/6/7/8/9/10: {c0}, {c5}, {c6}, {c7}, {c8}, {c9}, {c10}")
                    print(f"  Investment: {investment:.0f}   Total prize: {total_prize:.0f}   Net: {net:.0f}   ROI: {roi:.2f}%")
                else:
                    print(f"\n[--actual] Expected 20 numbers, got {len(actual)}; skipping match/payout.")
            print("---")


if __name__ == "__main__":
    main()
