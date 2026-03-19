"""
kle.predictor CLI — GAN + Transformer prediction pipeline.

Run from the `kle/` directory:
  python -m predictor [OPTIONS]
  # or
  python predictor/cli.py [OPTIONS]

Options
-------
--predict-only           Fast prediction for next draw (no walk-forward).
--full-dataset           Train walk-forward over all draws (oldest → newest).
--latest N               Train on latest N draws only (fast mode).
--target ISSUE           Target issue 期数 for walk-forward (e.g. 2026063).
--n-cover N              Number of generated sets (default: 20).
--n-eval N               Evaluation draws in walk-forward (default: 20).
--n-warmup N             Warmup draws before evaluation (default: 15).
--epochs N               Evaluation passes over same draws (default: 1).
--pick N                 Numbers to predict per set, 1–80 (default: 15).
--model PATH             Single model prefix: load if exists, always save.
--load-model PATH        Load model components from PATH prefix.
--save-model PATH        Save model components to PATH prefix after run.
--memory-lr FLOAT        MemoryBank learning rate (default: 0.15).
--memory-decay FLOAT     MemoryBank attention decay (default: 0.92).
--d-model INT            Transformer embedding dim (default: 64).
--n-heads INT            Transformer attention heads (default: 4).
--noise-dim INT          GAN generator noise dim (default: 32).
--hidden-dim INT         GAN hidden layer width (default: 128).
--fast                   Use fast (7-method) signal subset for --predict-only.
--hit-target INT         Min hits for summary (default: 11 for pick=15).
--gan-perturb INT        ES perturbations per GAN step (default: 20).
--gan-sigma FLOAT        ES perturbation sigma (default: 0.02).

Examples
--------
  # Quick prediction for next draw
  python -m predictor --predict-only

  # Walk-forward evaluation on issue 2026063
  python -m predictor --target 2026063 --n-cover 20

  # Train on entire dataset and save model
  python -m predictor --full-dataset --n-cover 10 --save-model storage/model

  # Walk-forward and save, then load and predict
  python -m predictor --target 2026063 --model storage/model --n-cover 50
  python -m predictor --predict-only --model storage/model --n-cover 50

  # Train on latest 100 draws (fast)
  python -m predictor --latest 100 --n-cover 20 --model storage/model
"""
import argparse
import os
import sys

# Allow running as `python predictor/cli.py` from kle/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor.pipeline import GANTransformerPipeline, _load_data, run_full_dataset

_EXAMPLES = """
examples:
  python -m predictor --predict-only
  python -m predictor --predict-only --n-cover 200 --pick 10
  python -m predictor --target 2026063 --n-cover 20
  python -m predictor --target 2026063 --n-cover 20 --epochs 3 --model storage/model
  python -m predictor --full-dataset --n-cover 10 --save-model storage/model
  python -m predictor --latest 100 --n-cover 20 --model storage/model
  python -m predictor --predict-only --model storage/model --n-cover 50
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GAN + Transformer KL8 predictor (pick N numbers, target 11+ hits).",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predict-only", action="store_true",
                      help="Fast prediction for next draw (no walk-forward)")
    mode.add_argument("--full-dataset", action="store_true",
                      help="Train on entire dataset (walk-forward all draws)")
    mode.add_argument("--latest", type=int, default=None, metavar="N",
                      help="Train on latest N draws only")

    # Walk-forward params
    parser.add_argument("--target", default="2026040",
                        help="Target issue 期数 for walk-forward (default: 2026040)")
    parser.add_argument("--n-cover", type=int, default=20,
                        help="Number of generated sets per run (default: 20)")
    parser.add_argument("--n-eval", type=int, default=20,
                        help="Evaluation draws (default: 20)")
    parser.add_argument("--n-warmup", type=int, default=15,
                        help="Warmup draws before evaluation (default: 15)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Evaluation passes over same draws (default: 1)")
    parser.add_argument("--pick", type=int, default=15,
                        help="Numbers per set (default: 15)")
    parser.add_argument("--hit-target", type=int, default=None,
                        help="Min hits for summary (default auto: 11 for pick=15)")

    # Model persistence
    parser.add_argument("--model", type=str, default="",
                        help="Single model prefix: load if exists, always save")
    parser.add_argument("--load-model", type=str, default="",
                        help="Load model from this prefix")
    parser.add_argument("--save-model", type=str, default="",
                        help="Save model to this prefix after run")

    # Architecture
    parser.add_argument("--d-model", type=int, default=64,
                        help="Transformer embedding dim (default: 64)")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Transformer attention heads (default: 4)")
    parser.add_argument("--noise-dim", type=int, default=32,
                        help="GAN generator noise dim (default: 32)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="GAN hidden layer width (default: 128)")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast signal subset (for --predict-only)")

    # Memory hyperparams
    parser.add_argument("--memory-lr", type=float, default=0.15,
                        help="MemoryBank learning rate (default: 0.15)")
    parser.add_argument("--memory-decay", type=float, default=0.92,
                        help="MemoryBank attention decay (default: 0.92)")

    # GAN training
    parser.add_argument("--gan-perturb", type=int, default=20,
                        help="ES perturbations per GAN training step (default: 20)")
    parser.add_argument("--gan-sigma", type=float, default=0.02,
                        help="ES perturbation sigma (default: 0.02)")
    parser.add_argument("--max-history", type=int, default=100,
                        help="Max history rows fed to signal providers per step (default: 100, lower = faster)")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="Save checkpoint every N steps during --full-dataset (default: 500, 0 = disabled)")

    # Post-generation filter
    parser.add_argument("--filter-top", type=int, default=None, metavar="N",
                        help="Overgenerate then keep top N sets by SetScorer (e.g. --filter-top 20 --overgen-factor 5 → generate 100, keep best 20)")
    parser.add_argument("--overgen-factor", type=int, default=5,
                        help="How many times more candidates to generate before filtering (default: 5)")

    args = parser.parse_args(argv)

    # Resolve --model shorthand
    DEFAULT_MODEL_PATH = "storage/model"
    if args.model:
        if not args.load_model:
            args.load_model = args.model if _model_exists(args.model) else ""
        if not args.save_model:
            args.save_model = args.model
    elif not args.load_model and not args.save_model:
        args.load_model = DEFAULT_MODEL_PATH if _model_exists(DEFAULT_MODEL_PATH) else ""
        args.save_model = DEFAULT_MODEL_PATH

    # Auto hit-target
    if args.hit_target is None:
        args.hit_target = max(1, int(round(args.pick * (11 / 15))))

    csv_path = "data/data.csv"
    issues, draws = _load_data(csv_path)

    # --latest shorthand
    if args.latest is not None:
        args.target = issues[0]
        args.n_eval = args.latest
        args.n_warmup = 0

    # Build pipeline
    pipeline = GANTransformerPipeline(
        d_model=args.d_model,
        n_heads=args.n_heads,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        memory_lr=args.memory_lr,
        memory_decay=args.memory_decay,
        pick=args.pick,
        fast_signals=args.fast or args.predict_only,
    )
    if args.load_model and _model_exists(args.load_model):
        pipeline.load(args.load_model)
        print(f"Model loaded from {args.load_model}_*.npz")
        # Saved GAN metadata can override pick; CLI --pick should win.
        # (Transformer + signals are pick-agnostic; pick only controls selection/generation size.)
        if getattr(pipeline, "_pick", args.pick) != args.pick:
            pipeline._pick = int(args.pick)
            pipeline.generator.pick = int(args.pick)

    if args.predict_only:
        filter_top = args.filter_top or args.n_cover
        print(f"PREDICTION FOR NEXT DRAW (after {issues[0]})")
        if args.filter_top:
            print(f"[filter] generating {args.n_cover * args.overgen_factor} candidates → keeping top {args.n_cover} by SetScorer")
        print("=" * 60)
        top_pick, sets = pipeline.predict(
            draws, n_sets=args.n_cover,
            filter_top=filter_top,
            overgen_factor=args.overgen_factor,
        )
        print(f"\nTop {args.pick} by fused score (target {args.hit_target}+ hits): {top_pick}")
        print(f"\n{args.n_cover} Generated Sets:")
        for i, s in enumerate(sets, 1):
            print(f"  SET_{i:02d}: {' '.join(f'{x:02d}' for x in sorted(s))}")
        if args.save_model:
            pipeline.save(args.save_model)
            print(f"\nModel saved to {args.save_model}_*.npz")

    elif args.full_dataset:
        run_full_dataset(
            csv_path=csv_path,
            n_cover=args.n_cover,
            save_path=args.save_model,
            load_path=args.load_model,
            d_model=args.d_model,
            n_heads=args.n_heads,
            noise_dim=args.noise_dim,
            hidden_dim=args.hidden_dim,
            memory_lr=args.memory_lr,
            memory_decay=args.memory_decay,
            pick=args.pick,
            epochs=args.epochs,
            gan_n_perturb=args.gan_perturb,
            gan_sigma=args.gan_sigma,
            max_history=args.max_history,
            checkpoint_every=args.checkpoint_every,
        )

    else:
        pipeline.walk_forward(
            issues, draws, args.target,
            n_eval=args.n_eval,
            n_warmup=args.n_warmup,
            n_cover=args.n_cover,
            epochs=args.epochs,
            save_path=args.save_model,
            load_path=args.load_model,
            hit_target=args.hit_target,
            gan_n_perturb=args.gan_perturb,
            gan_sigma=args.gan_sigma,
            filter_top=args.filter_top,
            overgen_factor=args.overgen_factor,
        )


def _model_exists(prefix: str) -> bool:
    return os.path.isfile(prefix + "_memory.npz")


if __name__ == "__main__":
    main()
