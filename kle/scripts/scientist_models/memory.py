"""
MemoryBank + ExperienceReplay
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .constants import TOTAL, DRAW_SIZE


class MemoryBank:
    """
    Adaptive memory that learns from sequential draw outcomes.

    Stores:
      - Per-method adaptive weights (updated via softmax after each draw)
      - Per-number attention scores (reinforced by hits, decayed over time)
      - Number pair co-success matrix
      - Experience replay buffer of (predicted_set, actual_set, hits)
    """

    def __init__(self, method_names: List[str], lr: float = 0.15, decay: float = 0.92):
        self.method_names = list(method_names)
        self.n_methods = len(method_names)
        self.lr = lr
        self.decay = decay

        self.method_weights = np.ones(self.n_methods) / self.n_methods
        self.number_attention = np.ones(TOTAL) * 0.5
        self.method_hit_history: Dict[str, List[float]] = {m: [] for m in method_names}
        self.replay_buffer: List[Tuple[List[int], set, int]] = []
        self.max_replay = 200
        self.pair_success = np.zeros((TOTAL, TOTAL), dtype=float)

    def update_after_draw(
        self,
        method_scores: Dict[str, np.ndarray],
        actual: set,
        predicted_sets: Optional[List[List[int]]] = None,
    ):
        actual_list = sorted(actual)

        # 1) Update method weights
        perfs = []
        for i, name in enumerate(self.method_names):
            if name not in method_scores:
                perfs.append(0.0)
                continue
            sc = method_scores[name]
            top20 = set(int(x) + 1 for x in np.argsort(sc)[::-1][:DRAW_SIZE])
            overlap = len(top20 & actual) / DRAW_SIZE
            perfs.append(overlap)
            self.method_hit_history[name].append(overlap)

        perfs = np.array(perfs)
        if perfs.max() > 0:
            exp_p = np.exp(3.0 * (perfs - perfs.mean()))
            target_w = exp_p / exp_p.sum()
            self.method_weights = (1 - self.lr) * self.method_weights + self.lr * target_w

        self.method_weights = np.maximum(self.method_weights, 0.02)
        self.method_weights /= self.method_weights.sum()

        # 2) Update number attention
        self.number_attention *= self.decay
        for num in actual:
            if 1 <= num <= TOTAL:
                self.number_attention[num - 1] += 1.0

        # 3) Update pair success
        for a in actual_list:
            for b in actual_list:
                if a != b and 1 <= a <= TOTAL and 1 <= b <= TOTAL:
                    self.pair_success[a - 1, b - 1] += 0.1
        self.pair_success *= 0.98

        # 4) Experience replay
        if predicted_sets:
            for pset in predicted_sets:
                h = len(set(pset) & actual)
                self.replay_buffer.append((pset, actual, h))
            if len(self.replay_buffer) > self.max_replay:
                self.replay_buffer = self.replay_buffer[-self.max_replay:]

    def get_method_weights(self) -> np.ndarray:
        return self.method_weights.copy()

    def get_number_attention(self) -> np.ndarray:
        att = self.number_attention.copy()
        att -= att.min()
        mx = att.max()
        return att / mx if mx > 0 else att

    def get_pair_boost(self) -> np.ndarray:
        boost = self.pair_success.sum(axis=1)
        mx = boost.max()
        return boost / mx if mx > 0 else boost

    def save(self, path: str):
        np.savez(
            path,
            method_weights=self.method_weights,
            number_attention=self.number_attention,
            pair_success=self.pair_success,
            method_names=np.array(self.method_names),
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.method_weights = data['method_weights']
        self.number_attention = data['number_attention']
        self.pair_success = data['pair_success']


class ExperienceReplay:
    """Extracts patterns from successful predictions in memory."""

    @staticmethod
    def extract_success_pattern(memory: MemoryBank, min_hits: int = 4) -> np.ndarray:
        scores = np.zeros(TOTAL)
        if not memory.replay_buffer:
            return scores

        for pset, actual, hits in memory.replay_buffer:
            if hits >= min_hits:
                weight = (hits - min_hits + 1) ** 2
                matched = set(pset) & actual
                for num in matched:
                    if 1 <= num <= TOTAL:
                        scores[num - 1] += weight * 1.5
                for num in pset:
                    if num in actual and 1 <= num <= TOTAL:
                        scores[num - 1] += weight * 0.5

        mx = scores.max()
        return scores / mx if mx > 0 else scores
