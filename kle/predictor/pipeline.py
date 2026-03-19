"""
GANTransformerPipeline — main prediction pipeline for KL8 (快乐8).

Data flow per draw
------------------
1.  Signal providers (13 methods) → (13, 80) score matrix
2.  SignalTransformerEncoder fuses them with MemoryBank cross-attention → (80,) dist
3.  SetGenerator (GAN) conditions on fused dist + memory state → cover sets
4.  SetDiscriminator (GAN) provides real/fake feedback → trains Generator
5.  MemoryBank updates method weights & attention from actual draw outcome
6.  Transformer encoder checkpoints if this step's hit score improved

Public API
----------
pipeline = GANTransformerPipeline()
pipeline.walk_forward(issues, draws, target_issue, ...)
sets = pipeline.predict(draws, n_sets=20)
pipeline.save(path_prefix)
pipeline.load(path_prefix)
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .gan import SetDiscriminator, SetGenerator, train_gan_step
from .memory import ExperienceReplay, MemoryBank
from .signals import ALL_SIGNAL_PROVIDERS, FAST_SIGNAL_PROVIDERS
from .transformer import SignalTransformerEncoder
from .scorer import SetScorer

TOTAL = 80
DRAW_SIZE = 20
PICK = 15
TARGET_MIN_HITS = 11


def _normalise(scores: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack and normalise signal score dict → (n_signals, TOTAL) array."""
    names = list(scores.keys())
    matrix = np.zeros((len(names), TOTAL))
    for i, name in enumerate(names):
        s = scores[name].copy()
        s -= s.min()
        mx = s.max()
        matrix[i] = s / mx if mx > 0 else np.zeros(TOTAL)
    return matrix


def _load_data(csv_path: str) -> Tuple[List[str], np.ndarray]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    num_cols = [c for c in df.columns if c.startswith("红球")]
    issues = df["期数"].astype(str).tolist()
    draws = df[num_cols].to_numpy(dtype=int)
    return issues, draws


class GANTransformerPipeline:
    """
    End-to-end prediction pipeline combining:
    - 13 statistical signal providers
    - Transformer encoder with cross-attention to MemoryBank
    - GAN (SetGenerator + SetDiscriminator) for cover-set generation
    - Online walk-forward memory updates

    Parameters
    ----------
    d_model       : Transformer embedding dimension
    n_heads       : number of attention heads
    noise_dim     : GAN generator latent noise dimension
    hidden_dim    : GAN hidden layer width
    memory_lr     : MemoryBank learning rate
    memory_decay  : MemoryBank attention decay per draw
    pick          : numbers to predict per set (default 15)
    fast_signals  : use only fast subset of signal providers (for --predict-only)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        noise_dim: int = 32,
        hidden_dim: int = 128,
        memory_lr: float = 0.15,
        memory_decay: float = 0.92,
        pick: int = PICK,
        fast_signals: bool = False,
        seed: int = 42,
    ) -> None:
        global PICK
        PICK = pick

        self.signal_providers = FAST_SIGNAL_PROVIDERS if fast_signals else ALL_SIGNAL_PROVIDERS
        self.method_names = list(self.signal_providers.keys())
        n_signals = len(self.method_names)

        self.memory = MemoryBank(self.method_names, lr=memory_lr, decay=memory_decay)

        # Memory state dim = 2*TOTAL + n_signals
        mem_dim = 2 * TOTAL + n_signals
        self.transformer = SignalTransformerEncoder(
            n_signals=n_signals,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 2,
            seed=seed,
        )

        # GAN conditioning: fused distribution (TOTAL) + memory state (mem_dim)
        cond_dim = TOTAL + mem_dim
        self.generator = SetGenerator(cond_dim=cond_dim, noise_dim=noise_dim, hidden_dim=hidden_dim, pick=pick, seed=seed + 1)
        self.discriminator = SetDiscriminator(cond_dim=cond_dim, hidden_dim=hidden_dim, seed=seed + 2)

        self._rng = np.random.default_rng(seed)
        self._pick = pick

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _compute_signals(self, hist: np.ndarray) -> Dict[str, np.ndarray]:
        scores = {}
        for name, func in self.signal_providers.items():
            scores[name] = func(hist)
        return scores

    def _fuse(self, scores: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (fused_dist, memory_state).
        fused_dist   : (TOTAL,) — transformer output distribution
        memory_state : (mem_dim,) — flat MemoryBank state for GAN conditioning
        """
        signal_matrix = _normalise(scores)
        method_weights = self.memory.get_method_weights()
        memory_state = self.memory.state_vector()

        fused_dist = self.transformer.encode(signal_matrix, method_weights, memory_state)

        # Blend with ExperienceReplay signal
        replay_signal = ExperienceReplay.extract_success_pattern(self.memory, min_hits=4)
        att = self.memory.get_number_attention()
        pair = self.memory.get_pair_boost()

        # Weighted blend (matches original fusion_weights spirit)
        fused_dist = (
            0.45 * fused_dist
            + 0.20 * att
            + 0.15 * pair
            + 0.20 * replay_signal
        )
        fused_dist = np.maximum(fused_dist, 1e-8)
        fused_dist /= fused_dist.sum()

        return fused_dist, memory_state

    def _make_cond(self, fused_dist: np.ndarray, memory_state: np.ndarray) -> np.ndarray:
        return np.concatenate([fused_dist, memory_state])

    def generate_sets(
        self,
        hist: np.ndarray,
        n_sets: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        filter_top: Optional[int] = None,
        overgen_factor: int = 5,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Generate n_sets cover sets, optionally with post-generation filtering.

        Parameters
        ----------
        n_sets         : number of sets to return
        filter_top     : if set, generate (n_sets * overgen_factor) candidates
                         then keep the top n_sets by SetScorer ranking.
                         Increases chance of 5+ hit sets significantly.
        overgen_factor : how many times more candidates to generate before filtering
                         (default 5 → generate 5× then keep best n_sets)

        Returns (sets, fused_dist).
        """
        scores = self._compute_signals(hist)
        fused_dist, memory_state = self._fuse(scores)
        cond = self._make_cond(fused_dist, memory_state)
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        if filter_top is not None:
            # Overgenerate then filter
            n_generate = n_sets * overgen_factor
            candidates = self.generator.generate(cond, n_sets=n_generate, rng=rng, temperature=temperature)
            replay_signal = ExperienceReplay.extract_success_pattern(self.memory, min_hits=4)
            scorer = SetScorer(fused_dist, self.memory, replay_signal=replay_signal)
            sets, set_scores = scorer.filter_top(candidates, top_k=n_sets)
        else:
            sets = self.generator.generate(cond, n_sets=n_sets, rng=rng, temperature=temperature)

        return sets, fused_dist

    # ------------------------------------------------------------------
    # Walk-forward training
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        issues: List[str],
        draws: np.ndarray,
        target_issue: str,
        n_eval: int = 20,
        n_warmup: int = 15,
        n_cover: int = 20,
        epochs: int = 1,
        save_path: str = "",
        load_path: str = "",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        gan_n_perturb: int = 20,
        gan_sigma: float = 0.02,
        hit_target: Optional[int] = None,
        filter_top: Optional[int] = None,
        overgen_factor: int = 5,
    ) -> Dict:
        """
        Walk-forward evaluation & training loop (mirrors original pipeline).

        Returns metrics dict.
        """
        ht = hit_target if hit_target is not None else TARGET_MIN_HITS

        if load_path and os.path.isfile(load_path + "_memory.npz"):
            self.memory = MemoryBank.load(load_path + "_memory.npz", self.method_names)
        if load_path and os.path.isfile(load_path + "_transformer.npz"):
            self.transformer.load(load_path + "_transformer.npz")

        try:
            idx = issues.index(target_issue)
        except ValueError:
            latest = issues[0]
            try:
                if int(target_issue) == int(latest) + 1:
                    return self._predict_next(issues, draws, n_cover, target_issue)
            except (ValueError, TypeError):
                pass
            raise ValueError(
                f"Target issue '{target_issue}' not in data. Latest is '{latest}'."
            )

        total_window = n_warmup + n_eval
        if idx + total_window >= len(draws):
            n_eval = max(1, len(draws) - idx - 30 - n_warmup)

        print(f"TARGET: {target_issue}")
        print(f"WARMUP: {n_warmup}  EVAL: {n_eval}  COVER: {n_cover}  EPOCHS: {epochs}")
        print("=" * 80)

        # Warmup phase
        print("\n--- WARMUP PHASE ---")
        for step in range(n_warmup):
            if progress_callback:
                progress_callback("warmup", step + 1, n_warmup)
            t_idx = idx + n_eval + step
            t_hist = draws[t_idx + 1 :]
            t_actual = set(int(x) for x in draws[t_idx])
            scores = self._compute_signals(t_hist)
            self.memory.update_after_draw(scores, t_actual)
            print(f"  warmup {step + 1:>2}/{n_warmup}: {issues[t_idx]}")

        # Evaluation phase
        eval_results: Dict = {"single": [], "cover_best": [], "cover_all": []}
        signal_cache: Dict[int, Dict[str, np.ndarray]] = {}

        for epoch in range(epochs):
            if progress_callback:
                progress_callback("eval_epoch", epoch + 1, epochs)
            print(f"\n--- EVALUATION EPOCH {epoch + 1}/{epochs} ---")
            eval_results = {"single": [], "cover_best": [], "cover_all": []}
            verbose = epoch == epochs - 1

            for step in range(n_eval):
                if progress_callback:
                    progress_callback("eval_step", step + 1, n_eval)
                t_idx = idx + n_eval - 1 - step
                t_hist = draws[t_idx + 1 :]
                t_actual = set(int(x) for x in draws[t_idx])
                t_issue = issues[t_idx]

                if step not in signal_cache:
                    signal_cache[step] = self._compute_signals(t_hist)
                scores = signal_cache[step]

                fused_dist, memory_state = self._fuse(scores)
                cond = self._make_cond(fused_dist, memory_state)

                # Generate cover sets via GAN (with optional overgenerate+filter)
                if filter_top is not None:
                    n_generate = n_cover * overgen_factor
                    candidates = self.generator.generate(
                        cond,
                        n_sets=n_generate,
                        rng=np.random.default_rng(42 + epoch * n_eval + t_idx),
                        temperature=1.0,
                    )
                    replay_signal = ExperienceReplay.extract_success_pattern(self.memory, min_hits=4)
                    scorer = SetScorer(fused_dist, self.memory, replay_signal=replay_signal)
                    cover, _ = scorer.filter_top(candidates, top_k=n_cover)
                else:
                    cover = self.generator.generate(
                        cond,
                        n_sets=n_cover,
                        rng=np.random.default_rng(42 + epoch * n_eval + t_idx),
                        temperature=1.0,
                    )

                # Single best by fused score
                single = sorted(int(x) + 1 for x in np.argsort(fused_dist)[::-1][: self._pick])
                single_hits = len(set(single) & t_actual)
                cover_hits = [len(set(s) & t_actual) for s in cover]
                best_cover_h = max(cover_hits)

                eval_results["single"].append(single_hits)
                eval_results["cover_best"].append(best_cover_h)
                eval_results["cover_all"].append(cover_hits)

                # GAN training step (use recent draws as real samples)
                recent_actuals = [
                    set(int(x) for x in draws[t_idx + 1 + k])
                    for k in range(min(5, len(draws) - t_idx - 2))
                ]
                train_gan_step(
                    self.generator,
                    self.discriminator,
                    cond,
                    t_actual,
                    recent_actuals,
                    n_fake=min(16, n_cover),
                    sigma=gan_sigma,
                    n_perturb=gan_n_perturb,
                    rng=np.random.default_rng(t_idx),
                )

                # Memory and transformer updates
                self.memory.update_after_draw(scores, t_actual, predicted_sets=cover)
                self.transformer.checkpoint_if_better(best_cover_h)
                self.transformer.ema_toward_best(alpha=0.05)

                if verbose:
                    matched = sorted(set(single) & t_actual)
                    best_set = cover[int(np.argmax(cover_hits))]
                    best_matched = sorted(set(best_set) & t_actual)
                    print(
                        f"  {t_issue}: single={single_hits}/{self._pick} "
                        f"cover_best={best_cover_h}/{self._pick} "
                        f"| single_match={matched}"
                    )
                    if best_cover_h >= ht:
                        print(
                            f"    ★ best_set: {' '.join(f'{x:02d}' for x in sorted(best_set))} "
                            f"match={best_matched}"
                        )

        # Summary
        sa = np.array(eval_results["single"])
        ca = np.array(eval_results["cover_best"])

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Mode':<25} {'Avg':>6} {'Max':>4} {'P>=8':>6} {'P>=' + str(ht):>6}")
        print("-" * 55)
        for label, arr in [("Single (transformer)", sa), (f"Cover best-of-{n_cover} (GAN)", ca)]:
            print(
                f"{label:<25} {arr.mean():>6.2f} {arr.max():>4d} "
                f"{(arr >= 8).mean():>6.2f} {(arr >= ht).mean():>6.2f}"
            )

        # Final prediction for target issue
        print(f"\n{'=' * 80}")
        print(f"FINAL PREDICTION FOR {target_issue}")
        print("=" * 80)
        final_hist = draws[idx + 1 :]
        final_sets, fused_final = self.generate_sets(
            final_hist, n_sets=n_cover, seed=9999,
            filter_top=filter_top, overgen_factor=overgen_factor,
        )
        actual_final = set(int(x) for x in draws[idx])
        top_pick = sorted(int(x) + 1 for x in np.argsort(fused_final)[::-1][: self._pick])
        print(f"\nTop {self._pick} by fused score: {top_pick}")
        print(f"\n{n_cover} GAN-Generated Sets (target {ht}+ hits):")
        final_hits = []
        for i, s in enumerate(final_sets, 1):
            h = len(set(s) & actual_final)
            final_hits.append(h)
            m = sorted(set(s) & actual_final)
            mark = "★" if h >= ht else "●" if h >= 8 else " "
            print(f"{mark} SET_{i:02d}|hits={h:>2}/{self._pick}|{' '.join(f'{x:02d}' for x in sorted(s))}  match={m}")

        fa = np.array(final_hits)
        print(f"\nbest={fa.max()}, avg={fa.mean():.2f}, P>=8={(fa >= 8).mean():.2f}, P>={ht}={(fa >= ht).mean():.2f}")
        print(f"ACTUAL: {' '.join(f'{x:02d}' for x in sorted(actual_final))}")

        # ── Money metrics — FINAL PREDICTION sets only (excludes warmup/eval) ─
        stake = 2.0
        prize_table = {0: stake, 5: 3.0, 6: 5.0, 7: 80.0, 8: 720.0, 9: 8000.0, 10: 5_000_000.0}
        hit_hist = np.zeros(11, dtype=int)
        total_tickets = 0
        for h in final_hits:
            if 0 <= h <= 10:
                hit_hist[h] += 1
                total_tickets += 1

        total_prize = sum(hit_hist[h] * prize_table.get(h, 0.0) for h in range(11))
        investment = total_tickets * stake
        net = total_prize - investment
        ev_per_ticket = net / total_tickets if total_tickets else 0.0
        roi = net / investment if investment else 0.0

        print("\n" + "=" * 80)
        print(f"MONEY METRICS  (FINAL PREDICTION FOR {target_issue} only — {total_tickets} tickets × {stake:.0f} CNY)")
        print(f"快乐8 选十 pay table  |  stake = {stake:.0f} CNY per ticket")
        print("=" * 80)
        print(f"  {'Hits':>5}  {'Count':>7}  {'Prize/ea':>10}  {'Subtotal':>12}")
        print("  " + "-" * 40)
        for h in range(11):
            if hit_hist[h] > 0 or h in prize_table:
                prize_ea = prize_table.get(h, 0.0)
                subtotal = hit_hist[h] * prize_ea
                marker = " ★" if h >= ht else ""
                print(f"  {h:>5}  {hit_hist[h]:>7}  {prize_ea:>10.2f}  {subtotal:>12.2f}{marker}")
        print("  " + "-" * 40)
        print(f"  {'TOTAL':>5}  {total_tickets:>7}  {'':>10}  {total_prize:>12.2f}")
        print()
        print(f"  Investment  : {investment:>10.2f} CNY  ({total_tickets} tickets × {stake:.0f} CNY)")
        print(f"  Total prize : {total_prize:>10.2f} CNY")
        print(f"  Net profit  : {net:>+10.2f} CNY")
        print(f"  ROI         : {roi * 100:>+9.3f}%")
        print(f"  EV/ticket   : {ev_per_ticket:>+10.6f} CNY")
        print("=" * 80)

        if save_path:
            self.save(save_path)
            print(f"\nModel saved to {save_path}_*.npz")

        metrics = {
            "cover_best_mean": float(ca.mean()),
            "cover_best_max": int(ca.max()),
            "single_mean": float(sa.mean()),
            "p_ge_8": float((ca >= 8).mean()),
            "p_ge_hit_target": float((ca >= ht).mean()),
            "investment": float(investment),
            "total_prize": float(total_prize),
            "net_profit": float(net),
            "roi": float(roi),
            "ev_per_ticket": float(ev_per_ticket),
        }
        return metrics

    def _predict_next(
        self,
        issues: List[str],
        draws: np.ndarray,
        n_cover: int,
        target_issue: str,
    ) -> Dict:
        print(f"TARGET {target_issue} is the next draw after latest ({issues[0]}).")
        print("Running prediction-only.")
        print("=" * 80)
        sets, fused = self.generate_sets(draws, n_sets=n_cover, seed=9999)
        top_pick = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][: self._pick])
        print(f"\nTop {self._pick}: {top_pick}")
        print(f"\n{n_cover} Generated Sets:")
        for i, s in enumerate(sets, 1):
            print(f"  SET_{i:02d}: {' '.join(f'{x:02d}' for x in sorted(s))}")
        return {"sets": sets, "top_pick": top_pick}

    def predict(
        self,
        draws: np.ndarray,
        n_sets: int = 20,
        temperature: float = 1.0,
        seed: int = 9999,
        filter_top: Optional[int] = None,
        overgen_factor: int = 5,
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Fast prediction for next draw.

        Returns (top_pick_list, cover_sets).
        """
        sets, fused = self.generate_sets(
            draws, n_sets=n_sets, temperature=temperature, seed=seed,
            filter_top=filter_top, overgen_factor=overgen_factor,
        )
        top_pick = sorted(int(x) + 1 for x in np.argsort(fused)[::-1][: self._pick])
        return top_pick, sets

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path_prefix: str) -> None:
        """Save all components.  Creates {path_prefix}_memory.npz, _transformer.npz, _gan.npz."""
        self.memory.save(path_prefix + "_memory.npz")
        self.transformer.save(path_prefix + "_transformer.npz")
        gen_params = self.generator.net.get_flat_params()
        disc_params = self.discriminator.net.get_flat_params()
        np.savez_compressed(
            path_prefix + "_gan.npz",
            gen_params=gen_params,
            disc_params=disc_params,
            gen_layer_sizes=np.array(self.generator.net.layer_sizes),
            disc_layer_sizes=np.array(self.discriminator.net.layer_sizes),
            pick=np.array([self._pick]),
            noise_dim=np.array([self.generator.noise_dim]),
            n_signals=np.array([len(self.method_names)]),
            fast_signals=np.array([1 if self.signal_providers is FAST_SIGNAL_PROVIDERS else 0]),
        )

    def load(self, path_prefix: str) -> None:
        """Load all components from {path_prefix}_*.npz files."""
        tr_path = path_prefix + "_transformer.npz"
        gan_path = path_prefix + "_gan.npz"

        # ── Step 1: resolve n_signals from saved metadata BEFORE loading anything ──
        # We need to know this first so signal_providers, method_names, and the
        # transformer are all rebuilt with the correct size before weights are loaded.
        n_signals_saved = None
        if os.path.isfile(gan_path):
            _meta = np.load(gan_path)
            if "n_signals" in _meta:
                n_signals_saved = int(_meta["n_signals"][0])
            elif "fast_signals" in _meta:
                n_signals_saved = 7 if int(_meta["fast_signals"][0]) else 13
        if n_signals_saved is None and os.path.isfile(tr_path):
            # Fall back: infer from transformer input_proj rows
            _tr = np.load(tr_path)
            if "input_proj" in _tr:
                n_signals_saved = int(_tr["input_proj"].shape[0])

        if n_signals_saved is not None and n_signals_saved != len(self.method_names):
            if n_signals_saved == len(FAST_SIGNAL_PROVIDERS):
                self.signal_providers = FAST_SIGNAL_PROVIDERS
            else:
                self.signal_providers = ALL_SIGNAL_PROVIDERS
            self.method_names = list(self.signal_providers.keys())
            # Rebuild transformer shell with correct input dimension
            d_model_cur = self.transformer.d_model
            n_heads_cur = self.transformer.attn.n_heads
            if d_model_cur % n_heads_cur != 0:
                n_heads_cur = 1
            self.transformer = SignalTransformerEncoder(
                n_signals=n_signals_saved,
                d_model=d_model_cur,
                n_heads=n_heads_cur,
                d_ff=d_model_cur * 2,
            )

        # ── Step 2: load memory, transformer weights ──
        mem_path = path_prefix + "_memory.npz"
        if os.path.isfile(mem_path):
            self.memory = MemoryBank.load(mem_path, self.method_names)
        if os.path.isfile(tr_path):
            self.transformer.load(tr_path)

        # ── Step 3: load GAN weights ──
        if os.path.isfile(gan_path):
            data = np.load(gan_path)
            gen_layer_sizes = [int(x) for x in data["gen_layer_sizes"]]
            disc_layer_sizes = [int(x) for x in data["disc_layer_sizes"]]
            pick_saved = int(data["pick"][0]) if "pick" in data else self._pick

            n_sig = n_signals_saved if n_signals_saved is not None else len(self.method_names)
            mem_dim_saved = 2 * TOTAL + n_sig
            cond_dim_saved = TOTAL + mem_dim_saved
            noise_dim_saved = gen_layer_sizes[0] - cond_dim_saved
            if noise_dim_saved <= 0:
                noise_dim_saved = int(data["noise_dim"][0]) if "noise_dim" in data else self.generator.noise_dim

            # Rebuild nets from saved architecture before loading params
            from .gan import MLP
            self.generator.net = MLP(gen_layer_sizes, seed=0)
            self.discriminator.net = MLP(disc_layer_sizes, seed=0)
            self.generator.pick = pick_saved
            self.generator.noise_dim = noise_dim_saved
            self._pick = pick_saved
            self.generator.net.set_flat_params(data["gen_params"])
            self.discriminator.net.set_flat_params(data["disc_params"])


# ---------------------------------------------------------------------------
# Convenience: full-dataset walk-forward (train on all draws)
# ---------------------------------------------------------------------------

def run_full_dataset(
    csv_path: str = "data/data.csv",
    n_cover: int = 10,
    save_path: str = "storage/model",
    load_path: str = "",
    max_history: int = 100,
    gan_n_perturb: int = 5,
    gan_sigma: float = 0.02,
    epochs: int = 1,
    checkpoint_every: int = 500,
    **pipeline_kwargs,
) -> GANTransformerPipeline:
    """
    Train on the entire dataset: walk-forward from oldest to newest draw.

    max_history caps the rolling history window fed to signal providers so
    per-step compute stays constant (O(max_history)) regardless of dataset size.
    Saves a checkpoint every `checkpoint_every` steps and on KeyboardInterrupt.
    """
    import time
    import signal as _signal

    issues, draws = _load_data(csv_path)
    n_draws = len(draws)
    min_history = pipeline_kwargs.pop("min_history", 30)

    pipeline = GANTransformerPipeline(**pipeline_kwargs)
    if load_path:
        pipeline.load(load_path)
        print(f"Resumed from {load_path}")

    method_names = pipeline.method_names
    start_i = n_draws - 1 - min_history
    total_steps = start_i + 1

    # Save on Ctrl+C instead of losing progress
    _interrupted = [False]
    def _handle_interrupt(sig, frame):
        _interrupted[0] = True
        print("\n[interrupted] saving checkpoint before exit...")
    _signal.signal(_signal.SIGINT, _handle_interrupt)

    for epoch in range(epochs):
        epoch_label = f" (epoch {epoch + 1}/{epochs})" if epochs > 1 else ""
        print(f"Full-dataset walk-forward{epoch_label}: {total_steps} steps ({issues[start_i]} → {issues[0]})")
        print("=" * 60)
        t0 = time.time()
        report_every = 100

        for step, i in enumerate(range(start_i, -1, -1)):
            if _interrupted[0]:
                break

            # Cap history to max_history rows — keeps signal compute O(1)
            t_hist = draws[i + 1 : i + 1 + max_history]
            t_actual = set(int(x) for x in draws[i])

            scores = {name: pipeline.signal_providers[name](t_hist) for name in method_names}
            fused_dist, memory_state = pipeline._fuse(scores)
            cond = pipeline._make_cond(fused_dist, memory_state)
            cover = pipeline.generator.generate(
                cond, n_sets=n_cover, rng=np.random.default_rng(42 + i + epoch * n_draws)
            )

            # GAN training step
            if gan_n_perturb > 0:
                train_gan_step(
                    pipeline.generator, pipeline.discriminator,
                    cond, t_actual, [],
                    n_perturb=gan_n_perturb, sigma=gan_sigma,
                )

            pipeline.memory.update_after_draw(scores, t_actual, predicted_sets=cover)
            pipeline.transformer.checkpoint_if_better(max(len(set(s) & t_actual) for s in cover))
            pipeline.transformer.ema_toward_best(alpha=0.05)

            if (step + 1) % report_every == 0 or step == total_steps - 1:
                elapsed = time.time() - t0
                sps = (step + 1) / elapsed
                remaining = (total_steps - step - 1) / sps if sps > 0 else 0
                best_h = max(len(set(s) & t_actual) for s in cover)
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                print(
                    f"  {issues[i]} (step {step + 1}/{total_steps})"
                    f"  cover_best={best_h}/{pipeline._pick}"
                    f"  {sps:.1f} steps/s  ETA {eta_min}m{eta_sec:02d}s"
                )

            # Periodic checkpoint
            if save_path and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                pipeline.save(save_path)
                print(f"  [checkpoint] saved at step {step + 1}")

        if save_path:
            pipeline.save(save_path)
            if _interrupted[0]:
                print(f"Model saved to {save_path}_*.npz  (interrupted at step {step + 1}/{total_steps})")
            else:
                print(f"Model saved to {save_path}_*.npz")

        if _interrupted[0]:
            break

    return pipeline

    return pipeline
