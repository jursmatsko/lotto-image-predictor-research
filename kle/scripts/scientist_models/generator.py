"""
GenerativeModel — fused distribution → diverse cover sets
"""
import numpy as np
from typing import List, Dict
from .constants import PICK, TOTAL
from .memory import MemoryBank, ExperienceReplay
from .utils import norm


class GenerativeModel:
    """
    Generates number sets by:
    1. Fusing all method scores with memory-adaptive weights
    2. Adding number attention and pair boost from memory
    3. Adding experience replay signal
    4. Temperature-controlled sampling with diversity enforcement
    """

    def __init__(self, memory: MemoryBank):
        self.memory = memory

    def compute_fused_distribution(
        self,
        method_scores: Dict[str, np.ndarray],
        hist: np.ndarray,
    ) -> np.ndarray:
        weights = self.memory.get_method_weights()
        attention = self.memory.get_number_attention()
        pair_boost = self.memory.get_pair_boost()
        replay_signal = ExperienceReplay.extract_success_pattern(self.memory)

        normed = {}
        for name, sc in method_scores.items():
            normed[name] = norm(sc)

        fused = np.zeros(TOTAL)
        for i, name in enumerate(self.memory.method_names):
            if name in normed:
                fused += weights[i] * normed[name]

        fused = (
            0.45 * fused
            + 0.20 * attention
            + 0.15 * pair_boost
            + 0.20 * replay_signal
        )
        return np.maximum(fused, 1e-8)

    def generate_sets(
        self,
        fused_dist: np.ndarray,
        n_sets: int = 20,
        seed: int = 42,
    ) -> List[List[int]]:
        rng = np.random.default_rng(seed)
        usage = np.zeros(TOTAL)
        sets = []
        temps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

        for i in range(n_sets):
            adj = fused_dist / (1.0 + 0.35 * usage)
            temp = temps[i % len(temps)]
            logits = np.log(np.maximum(adj, 1e-12)) / temp
            p = np.exp(logits - logits.max())
            p /= p.sum()
            picked = rng.choice(TOTAL, size=PICK, replace=False, p=p)
            nums = sorted(int(x) + 1 for x in picked)
            sets.append(nums)
            for num in nums:
                usage[num - 1] += 1
        return sets

    def generate_mega_pool(
        self,
        method_scores: Dict[str, np.ndarray],
        hist: np.ndarray,
        pool_size: int = 50000,
        seed_base: int = 0,
    ) -> List[List[int]]:
        """Generate a massive candidate pool from multiple strategies."""
        fused = self.compute_fused_distribution(method_scores, hist)
        fused_w = norm(fused) + 1e-8

        all_sets = []

        def _pool(weights, n_per_seed=12, seeds=range(500)):
            temps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
            result = []
            for sd in seeds:
                rng = np.random.default_rng(sd + seed_base)
                usage = np.zeros(TOTAL)
                for i in range(n_per_seed):
                    adj = weights / (1.0 + 0.25 * usage)
                    temp = temps[i % len(temps)]
                    logits = np.log(np.maximum(adj, 1e-12)) / temp
                    p = np.exp(logits - logits.max())
                    p /= p.sum()
                    picked = rng.choice(TOTAL, size=PICK, replace=False, p=p)
                    nums = sorted(int(x) + 1 for x in picked)
                    result.append(nums)
                    for num in nums:
                        usage[num - 1] += 1
            return result

        # Strategy 1: Memory-fused
        n_seeds_fused = min(800, pool_size // 12)
        all_sets.extend(_pool(fused_w, 12, range(n_seeds_fused)))

        # Strategy 2: Per-method
        per_method_seeds = min(200, pool_size // (12 * 6))
        for name, sc in method_scores.items():
            w = norm(sc) + 1e-8
            all_sets.extend(_pool(w, 6, range(per_method_seeds)))

        # Strategy 3: Top method blends
        weights = self.memory.get_method_weights()
        top_idx = np.argsort(weights)[::-1][:4]
        top_names = [self.memory.method_names[i] for i in top_idx]
        blend_seeds = min(150, pool_size // (12 * 8))
        for i in range(len(top_names)):
            for j in range(i + 1, len(top_names)):
                for alpha in [0.3, 0.5, 0.7]:
                    if top_names[i] in method_scores and top_names[j] in method_scores:
                        w = norm(
                            alpha * method_scores[top_names[i]]
                            + (1 - alpha) * method_scores[top_names[j]]
                        ) + 1e-8
                        all_sets.extend(_pool(w, 8, range(blend_seeds)))

        # Strategy 4: Attention + pair boost
        att = self.memory.get_number_attention()
        pb = self.memory.get_pair_boost()
        att_seeds = min(300, pool_size // (12 * 10))
        for w_att in [0.4, 0.6, 0.8]:
            w = norm(w_att * att + (1 - w_att) * pb + 0.1 * fused_w) + 1e-8
            all_sets.extend(_pool(w, 10, range(att_seeds)))

        # Deduplicate
        seen = set()
        unique = []
        for s in all_sets:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                unique.append(s)

        return unique
