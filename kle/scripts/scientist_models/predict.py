#!/usr/bin/env python3
"""
CLI entry point for scientist models.

Usage:
  python -m scripts.scientist_models.predict predict --issue 2026062 [--sets 20]
  python -m scripts.scientist_models.predict extreme --issue 2026040 [--sets 20]
  python -m scripts.scientist_models.predict walkforward --issue 2026040 [--eval 20] [--warmup 15]
  python -m scripts.scientist_models.predict compare --issue 2026040
"""
import argparse
import numpy as np
from collections import Counter

from .constants import PICK, TOTAL, DRAW_SIZE
from .signals import ALL_SIGNAL_PROVIDERS
from .memory import MemoryBank
from .generator import GenerativeModel
from .utils import load_data, norm


def _build_memory(draws, idx, method_names, n_warmup=15, n_recent=20):
    """Build memory from warmup + recent evaluation draws."""
    memory = MemoryBank(method_names)
    for step in range(n_warmup):
        t_idx = idx + n_recent + step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        memory.update_after_draw(sc, t_actual)

    for step in range(n_recent):
        t_idx = idx + step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        memory.update_after_draw(sc, t_actual)

    return memory


def cmd_predict(args):
    issues, draws = load_data(args.data)
    method_names = list(ALL_SIGNAL_PROVIDERS.keys())

    if args.issue in issues:
        idx = issues.index(args.issue)
        hist = draws[idx + 1:]
        actual = set(int(x) for x in draws[idx])
        has_actual = True
    else:
        hist = draws
        actual = set()
        has_actual = False
        idx = 0

    memory = _build_memory(draws, idx, method_names)
    gen = GenerativeModel(memory)

    scores = {name: func(hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
    fused = gen.compute_fused_distribution(scores, hist)
    sets = gen.generate_sets(fused, n_sets=args.sets, seed=args.seed)

    print(f'TARGET: {args.issue}')
    print(f'HISTORY: {len(hist)} draws')
    print(f'SETS: {args.sets}')
    print(f'Top {PICK}: {sorted(int(x)+1 for x in np.argsort(fused)[::-1][:PICK])}')
    print('=' * 70)

    for i, s in enumerate(sets, 1):
        if has_actual:
            h = len(set(s) & actual)
            m = sorted(set(s) & actual)
            mark = '★' if h >= 7 else '●' if h >= 5 else ' '
            print(f'{mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')
        else:
            print(f'  SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')

    if has_actual:
        hits = [len(set(s) & actual) for s in sets]
        a = np.array(hits)
        print(f'\nbest={a.max()}, avg={a.mean():.2f}')
        print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')


def cmd_extreme(args):
    issues, draws = load_data(args.data)
    method_names = list(ALL_SIGNAL_PROVIDERS.keys())

    if args.issue not in issues:
        print(f'{args.issue} not in data, using all history for future prediction')
        hist = draws
        actual = set()
        has_actual = False
        idx = 0
    else:
        idx = issues.index(args.issue)
        hist = draws[idx + 1:]
        actual = set(int(x) for x in draws[idx])
        has_actual = True

    print(f'TARGET: {args.issue}')
    if has_actual:
        print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')
    print(f'HISTORY: {len(hist)}')

    print('\nComputing method scores...')
    scores = {name: func(hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}

    print('Building memory...')
    memory = _build_memory(draws, idx, method_names)
    gen = GenerativeModel(memory)

    print('Generating mega pool...')
    pool = gen.generate_mega_pool(scores, hist, pool_size=args.pool_size, seed_base=args.seed)
    print(f'Unique candidates: {len(pool)}')

    if has_actual:
        scored = [(len(set(s) & actual), s) for s in pool]
        scored.sort(key=lambda x: -x[0])

        c = Counter([h for h, _ in scored])
        print('\nHit distribution:')
        for k in sorted(c):
            print(f'  hits={k:>2}: {c[k]:>6} ({c[k]/len(scored)*100:.2f}%)')

        def jaccard(a, b):
            return len(set(a) & set(b)) / len(set(a) | set(b))

        selected = []
        for h, s in scored:
            if len(selected) >= args.sets:
                break
            if not selected or all(jaccard(s, t) < 0.45 for t in selected):
                selected.append(s)

        print(f'\nBEST DIVERSE {args.sets} SETS:')
        final_hits = []
        for i, s in enumerate(selected, 1):
            h = len(set(s) & actual)
            final_hits.append(h)
            m = sorted(set(s) & actual)
            mark = '★' if h >= 7 else '●' if h >= 5 else ' '
            print(f'{mark} SET_{i:02d}|hits={h:>2}/{PICK}|{" ".join(f"{x:02d}" for x in s)}  match={m}')

        fa = np.array(final_hits)
        print(f'\nbest={fa.max()}, avg={fa.mean():.2f}, P>=7={(fa>=7).mean():.2f}, P>=9={(fa>=9).mean():.2f}')
        print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')
    else:
        # No actual: just output top sets by fused score
        fused = gen.compute_fused_distribution(scores, hist)
        sets = gen.generate_sets(fused, n_sets=args.sets, seed=args.seed)
        print(f'\nPREDICTED {args.sets} SETS:')
        for i, s in enumerate(sets, 1):
            print(f'  SET_{i:02d}|{" ".join(f"{x:02d}" for x in s)}')


def cmd_walkforward(args):
    issues, draws = load_data(args.data)
    method_names = list(ALL_SIGNAL_PROVIDERS.keys())
    idx = issues.index(args.issue)

    memory = MemoryBank(method_names)
    gen = GenerativeModel(memory)
    rng = np.random.default_rng(42)

    # Warmup
    print(f'Warmup: {args.warmup} draws')
    for step in range(args.warmup):
        t_idx = idx + args.eval + step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        memory.update_after_draw(sc, t_actual)

    # Evaluate
    results = {'single': [], 'cover': []}
    rand_results = []

    for step in range(args.eval):
        t_idx = idx + args.eval - 1 - step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        t_issue = issues[t_idx]

        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        fused = gen.compute_fused_distribution(sc, t_hist)

        single = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][:PICK])
        sh = len(set(single) & t_actual)

        cover = gen.generate_sets(fused, n_sets=args.sets, seed=42 + t_idx)
        ch = max(len(set(s) & t_actual) for s in cover)

        rh = max(len(set(rng.choice(np.arange(1, 81), PICK, replace=False)) & t_actual) for _ in range(args.sets))

        results['single'].append(sh)
        results['cover'].append(ch)
        rand_results.append(rh)

        memory.update_after_draw(sc, t_actual, predicted_sets=cover)
        print(f'  {t_issue}: single={sh} cover={ch} rand={rh}')

    print(f'\n{"Mode":<20} {"Avg":>6} {"Max":>4} {"P>=5":>6} {"P>=6":>6} {"P>=7":>6}')
    for label, arr in [('Single', results['single']), ('Cover', results['cover']), ('Random', rand_results)]:
        a = np.array(arr)
        print(f'{label:<20} {a.mean():>6.2f} {a.max():>4d} {(a>=5).mean():>6.2f} {(a>=6).mean():>6.2f} {(a>=7).mean():>6.2f}')

    # Save memory
    save_path = args.save_memory
    if save_path:
        memory.save(save_path)
        print(f'\nMemory saved to {save_path}')


def cmd_meta(args):
    from .meta_model import run_meta_pipeline
    issues, draws = load_data(args.data)
    run_meta_pipeline(
        issues, draws, args.issue,
        n_train=args.train, n_sets=args.sets,
        extreme=args.extreme, pool_size=args.pool_size,
    )


def cmd_deepfuse(args):
    from .deep_fusion import run_deep_fusion
    issues, draws = load_data(args.data)
    run_deep_fusion(
        issues, draws, args.issue,
        n_train=args.train, n_sets=args.sets,
        extreme=args.extreme, pool_size=args.pool_size,
        save_model=getattr(args, 'save_model', None),
        load_model=getattr(args, 'load_model', None),
        finetune=getattr(args, 'finetune', 8),
    )


def cmd_compare(args):
    issues, draws = load_data(args.data)
    idx = issues.index(args.issue)
    hist = draws[idx + 1:]
    actual = set(int(x) for x in draws[idx])

    print(f'TARGET: {args.issue}')
    print(f'ACTUAL: {" ".join(f"{x:02d}" for x in sorted(actual))}')
    print('=' * 60)

    for name, func in ALL_SIGNAL_PROVIDERS.items():
        sc = func(hist)
        nums = sorted(int(x) + 1 for x in np.argsort(sc)[::-1][:PICK])
        h = len(set(nums) & actual)
        mark = '★' if h >= 5 else '●' if h >= 4 else ' '
        print(f'{mark} {name:<15} {h:>2}/{PICK}  {" ".join(f"{x:02d}" for x in nums)}')


def cmd_pool_sweep(args):
    """Sweep pool sizes to find minimum needed for 5+ hits (best set in pool)."""
    print('Loading data...', flush=True)
    issues, draws = load_data(args.data)
    method_names = list(ALL_SIGNAL_PROVIDERS.keys())
    idx = issues.index(args.issue)

    pool_sizes = sorted(set(args.pool_sizes))
    n_warmup = args.warmup
    n_eval = args.eval

    # Warmup memory
    memory = MemoryBank(method_names)
    for step in range(n_warmup):
        t_idx = idx + n_eval + step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        memory.update_after_draw(sc, t_actual)

    gen = GenerativeModel(memory)

    # For each pool size: [draw -> best_hit_in_pool]
    results = {ps: [] for ps in pool_sizes}

    print(f'Pool size sweep: target 5+ hits (best set in pool)')
    print(f'Issue anchor: {args.issue}, eval={n_eval} draws, warmup={n_warmup}')
    print('=' * 70)

    for step in range(n_eval):
        t_idx = idx + n_eval - 1 - step
        if t_idx >= len(draws) - 1:
            break
        t_hist = draws[t_idx + 1:]
        t_actual = set(int(x) for x in draws[t_idx])
        t_issue = issues[t_idx]

        sc = {name: func(t_hist) for name, func in ALL_SIGNAL_PROVIDERS.items()}
        fused = gen.compute_fused_distribution(sc, t_hist)

        for pool_size in pool_sizes:
            pool = gen.generate_mega_pool(sc, t_hist, pool_size=pool_size, seed_base=42 + t_idx)
            best_hit = max(len(set(s) & t_actual) for s in pool)
            results[pool_size].append(best_hit)

        memory.update_after_draw(sc, t_actual)

        # Progress
        best_per_ps = [results[ps][-1] for ps in pool_sizes]
        print(f'  {t_issue}: best_hits={best_per_ps}')

    print('=' * 70)
    print(f'{"Pool size":>12} | {"Avg best":>8} | {"Max":>4} | {"P>=5":>6} | {"P>=6":>6} | {"P>=7":>6}')
    print('-' * 55)
    for ps in pool_sizes:
        arr = np.array(results[ps])
        p5 = (arr >= 5).mean()
        p6 = (arr >= 6).mean()
        p7 = (arr >= 7).mean()
        print(f'{ps:>12,} | {arr.mean():>8.2f} | {arr.max():>4d} | {p5:>6.2%} | {p6:>6.2%} | {p7:>6.2%}')

    # Find minimum pool size for P(5+) >= target
    target_p5 = args.target_p5
    for ps in pool_sizes:
        if (np.array(results[ps]) >= 5).mean() >= target_p5:
            print(f'\n→ To achieve P(5+) >= {target_p5:.0%}: pool_size >= {ps:,}')
            break
    else:
        print(f'\n→ No pool size in range achieved P(5+) >= {target_p5:.0%}')


def main():
    parser = argparse.ArgumentParser(description='Scientist Models Predictor')
    parser.add_argument('--data', default='data/data.csv', help='Path to data CSV')
    sub = parser.add_subparsers(dest='command')

    p_pred = sub.add_parser('predict', help='Generate prediction sets')
    p_pred.add_argument('--issue', required=True)
    p_pred.add_argument('--sets', type=int, default=20)
    p_pred.add_argument('--seed', type=int, default=42)

    p_ext = sub.add_parser('extreme', help='Extreme pool search')
    p_ext.add_argument('--issue', required=True)
    p_ext.add_argument('--sets', type=int, default=20)
    p_ext.add_argument('--pool-size', type=int, default=50000)
    p_ext.add_argument('--seed', type=int, default=0)

    p_wf = sub.add_parser('walkforward', help='Walk-forward validation with memory')
    p_wf.add_argument('--issue', required=True)
    p_wf.add_argument('--eval', type=int, default=20)
    p_wf.add_argument('--warmup', type=int, default=15)
    p_wf.add_argument('--sets', type=int, default=20)
    p_wf.add_argument('--save-memory', default=None, help='Path to save memory .npz')

    p_cmp = sub.add_parser('compare', help='Compare all methods single-set')
    p_cmp.add_argument('--issue', required=True)

    p_sweep = sub.add_parser('pool-sweep', help='Sweep pool sizes to find minimum for 5+ hits')
    p_sweep.add_argument('--issue', required=True, help='Anchor issue (eval draws before it)')
    p_sweep.add_argument('--eval', type=int, default=20, help='Number of draws to evaluate')
    p_sweep.add_argument('--warmup', type=int, default=15, help='Warmup draws for memory')
    p_sweep.add_argument('--pool-sizes', type=int, nargs='+',
                        default=[5000, 10000, 20000, 50000, 100000, 200000],
                        help='Pool sizes to test (default: 5k 10k 20k 50k 100k 200k)')
    p_sweep.add_argument('--target-p5', type=float, default=0.5,
                        help='Target P(5+) fraction to achieve (default 0.5)')

    p_meta = sub.add_parser('meta', help='Meta-model: 12 methods as feature layers for learned generator')
    p_meta.add_argument('--issue', required=True)
    p_meta.add_argument('--sets', type=int, default=20)
    p_meta.add_argument('--train', type=int, default=30, help='Training draws')
    p_meta.add_argument('--extreme', action='store_true', help='Enable mega pool search')
    p_meta.add_argument('--pool-size', type=int, default=50000)

    p_deep = sub.add_parser('deepfuse', help='Deep Fusion: stacked cross-signal + agreement-biased generation')
    p_deep.add_argument('--issue', required=True)
    p_deep.add_argument('--sets', type=int, default=30)
    p_deep.add_argument('--train', type=int, default=50, help='Training draws')
    p_deep.add_argument('--extreme', action='store_true', help='Enable multi-strategy extreme generation')
    p_deep.add_argument('--pool-size', type=int, default=80000)
    p_deep.add_argument('--save-model', default=None, metavar='PATH',
                        help='Save trained MLP weights to PATH.npz (e.g. models/2026063)')
    p_deep.add_argument('--load-model', default=None, metavar='PATH',
                        help='Load MLP weights from PATH.npz instead of retraining')
    p_deep.add_argument('--finetune', type=int, default=8, metavar='N',
                        help='Fine-tune on N most recent draws after loading (default 8, 0=off)')

    p_deep10 = sub.add_parser('deepfuse10', help='Deep Fusion Extreme10: huge pool, focus on best-case 9+/10+ hits (no diversity filter)')
    p_deep10.add_argument('--issue', required=True)
    p_deep10.add_argument('--train', type=int, default=80, help='Training draws for fast MLP')
    p_deep10.add_argument('--pool-size', type=int, default=300000, help='Total generated sets (approx)')
    p_deep10.add_argument('--top', type=int, default=40, help='How many top sets to print')

    args = parser.parse_args()
    if args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'extreme':
        cmd_extreme(args)
    elif args.command == 'walkforward':
        cmd_walkforward(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'pool-sweep':
        cmd_pool_sweep(args)
    elif args.command == 'meta':
        cmd_meta(args)
    elif args.command == 'deepfuse':
        cmd_deepfuse(args)
    elif args.command == 'deepfuse10':
        from .deep_fusion import run_extreme10
        issues, draws = load_data(args.data)
        run_extreme10(
            issues, draws, args.issue,
            n_train=args.train, pool_size=args.pool_size, top=args.top,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
