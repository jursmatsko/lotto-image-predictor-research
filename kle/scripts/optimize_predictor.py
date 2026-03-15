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
) -> dict | None:
    """Run walk-forward once; return metrics dict or None on failure."""
    from generative_memory_predictor import (
        run_walk_forward_with_memory,
        load_data as load_data_predictor,
    )

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

    if status_stream and trial_number is not None:
        print(
            f"  [Trial {trial_number}] starting warmup={n_warmup} eval={n_eval} epochs={epochs} ...",
            file=sys.stderr,
            flush=True,
        )

    try:
        if quiet:
            f = io.StringIO()
            with redirect_stdout(f):
                _, _, metrics = run_walk_forward_with_memory(
                    issues,
                    draws,
                    target_issue,
                    n_eval=n_eval,
                    n_warmup=n_warmup,
                    n_cover=n_cover,
                    epochs=epochs,
                    memory_lr=memory_lr,
                    memory_decay=memory_decay,
                    save_memory="",
                    load_memory="",
                    progress_callback=progress_cb,
                )
        else:
            _, _, metrics = run_walk_forward_with_memory(
                issues,
                draws,
                target_issue,
                n_eval=n_eval,
                n_warmup=n_warmup,
                n_cover=n_cover,
                epochs=epochs,
                memory_lr=memory_lr,
                memory_decay=memory_decay,
                save_memory="",
                load_memory="",
                    progress_callback=progress_cb,
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
) -> float:
    """Optuna objective: maximize cover_best_mean (or chosen metric)."""
    n_warmup = trial.suggest_int("n_warmup", 5, 25)
    n_eval = trial.suggest_int("n_eval", 10, 40)
    n_cover = fixed_n_cover if fixed_n_cover is not None else trial.suggest_int("n_cover", 10, 50)
    epochs = fixed_epochs if fixed_epochs is not None else trial.suggest_int("epochs", 1, 5)
    memory_lr = trial.suggest_float("memory_lr", 0.05, 0.35, log=False)
    memory_decay = trial.suggest_float("memory_decay", 0.85, 0.99, log=False)

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
    )
    if metrics is None:
        return float("-inf")
    return metrics["cover_best_mean"]


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
    parser.add_argument("--verbose", action="store_true", help="Print each trial (no redirect)")
    parser.add_argument(
        "--no-status",
        action="store_true",
        dest="no_status",
        help="Disable live status lines (warmup/eval progress) during trials",
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
    print("Optimizing: cover_best_mean (maximize)\n")

    study.optimize(
        lambda t: objective(
            t, csv_path, args.target, args.n_cover, args.epochs,
            show_status=args.show_status,
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


if __name__ == "__main__":
    main()
