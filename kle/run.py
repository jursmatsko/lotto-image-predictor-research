#!/usr/bin/env python3
"""
One entrypoint for common KLE workflows (run from repo root).

Examples:
  python kle/run.py predict --pick 15 --n-cover 100
  python kle/run.py predict --pick 15 --n-cover 0 --actual "2,3,4,..."
  python kle/run.py eval --target 2026065 --n-cover 50
  python kle/run.py improve --mode estimation --generations 3 --pop-size 10 --backtest-n 80
"""

from __future__ import annotations

import argparse
import os
import sys


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # .../kle
    return os.path.dirname(here)


def _kle_dir() -> str:
    return os.path.join(_repo_root(), "kle")


def _script(path_from_kle: str) -> str:
    return os.path.join(_kle_dir(), path_from_kle)


def main() -> None:
    p = argparse.ArgumentParser(description="KLE workflow runner (predict/eval/improve).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="Fast prediction for next draw (predict-only).")
    sp.add_argument("--pick", type=int, default=15)
    sp.add_argument("--n-cover", type=int, default=20)
    sp.add_argument("--memory", default=os.path.join("kle", "storage", "memory.npz"))
    sp.add_argument("--actual", default="")
    sp.add_argument("--top-k-estimate", type=int, default=20)
    sp.add_argument("--consensus-windows", default="")
    sp.add_argument("--payout", default="none")

    se = sub.add_parser("eval", help="Walk-forward evaluation targeting an issue.")
    se.add_argument("--target", required=True)
    se.add_argument("--pick", type=int, default=15)
    se.add_argument("--n-cover", type=int, default=20)
    se.add_argument("--n-warmup", type=int, default=15)
    se.add_argument("--n-eval", type=int, default=20)
    se.add_argument("--epochs", type=int, default=1)
    se.add_argument("--memory", default=os.path.join("kle", "storage", "memory.npz"))
    se.add_argument("--payout", default="none")

    si = sub.add_parser("improve", help="Self-improve loop (evolution).")
    si.add_argument("--mode", choices=["predictor", "estimation"], default="estimation")
    si.add_argument("--target", default="2026065")
    si.add_argument("--generations", type=int, default=3)
    si.add_argument("--pop-size", type=int, default=10)
    si.add_argument("--elite", type=int, default=3)
    si.add_argument("--backtest-n", type=int, default=80)

    args, extra = p.parse_known_args()

    env = os.environ.copy()
    # Make sure we can import / run from repo root.
    cwd = _repo_root()

    if args.cmd == "predict":
        cmd = [
            sys.executable,
            "-u",
            _script("scripts/generative_memory_predictor.py"),
            "--predict-only",
            "--pick",
            str(args.pick),
            "--n-cover",
            str(args.n_cover),
            "--memory",
            args.memory,
            "--top-k-estimate",
            str(args.top_k_estimate),
        ]
        if args.payout and args.payout != "none":
            cmd += ["--payout", args.payout]
        if args.actual:
            cmd += ["--actual", args.actual]
        if args.consensus_windows:
            cmd += ["--consensus-windows", args.consensus_windows]
        cmd += extra
    elif args.cmd == "eval":
        cmd = [
            sys.executable,
            "-u",
            _script("scripts/generative_memory_predictor.py"),
            "--target",
            str(args.target),
            "--pick",
            str(args.pick),
            "--n-cover",
            str(args.n_cover),
            "--n-warmup",
            str(args.n_warmup),
            "--n-eval",
            str(args.n_eval),
            "--epochs",
            str(args.epochs),
            "--memory",
            args.memory,
        ]
        if args.payout and args.payout != "none":
            cmd += ["--payout", args.payout]
        cmd += extra
    else:
        cmd = [
            sys.executable,
            "-u",
            _script("scripts/self_improve.py"),
            "--mode",
            args.mode,
            "--target",
            args.target,
            "--generations",
            str(args.generations),
            "--pop-size",
            str(args.pop_size),
            "--elite",
            str(args.elite),
            "--backtest-n",
            str(args.backtest_n),
        ]
        cmd += extra

    # Minimal subprocess runner (avoid extra deps).
    import subprocess

    raise SystemExit(subprocess.call(cmd, cwd=cwd, env=env))


if __name__ == "__main__":
    main()

