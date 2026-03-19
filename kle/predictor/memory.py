"""
MemoryBank and ExperienceReplay — online adaptive memory for the predictor.

MemoryBank tracks:
  - Per-method adaptive weights (updated after each observed draw)
  - Per-number attention scores (reinforced by actual draws)
  - Pairwise co-occurrence success matrix
  - Experience replay buffer of high-hit prediction patterns
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

TOTAL = 80
DRAW_SIZE = 20
PICK = 15


class MemoryBank:
    """Online adaptive memory updated after every observed draw."""

    def __init__(
        self,
        method_names: List[str],
        lr: float = 0.15,
        decay: float = 0.92,
    ) -> None:
        self.method_names = list(method_names)
        self.n_methods = len(method_names)
        self.lr = lr
        self.decay = decay

        self.method_weights: np.ndarray = np.ones(self.n_methods) / self.n_methods
        self.number_attention: np.ndarray = np.ones(TOTAL) * 0.5
        self.method_hit_history: Dict[str, List[float]] = {m: [] for m in method_names}
        self.replay_buffer: List[Tuple[List[int], set, int]] = []
        self.max_replay = 500
        self.pair_success: np.ndarray = np.zeros((TOTAL, TOTAL), dtype=float)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_after_draw(
        self,
        method_scores: Dict[str, np.ndarray],
        actual: set,
        predicted_sets: Optional[List[List[int]]] = None,
    ) -> None:
        actual_vec = np.zeros(TOTAL)
        for num in actual:
            if 1 <= num <= TOTAL:
                actual_vec[num - 1] = 1

        # Update method weights via rank-correlation performance
        method_perfs = []
        for name in self.method_names:
            if name not in method_scores:
                method_perfs.append(0.0)
                continue
            sc = method_scores[name]
            top20 = set(int(x) + 1 for x in np.argsort(sc)[::-1][:DRAW_SIZE])
            overlap = len(top20 & actual) / DRAW_SIZE
            method_perfs.append(overlap)
            self.method_hit_history[name].append(overlap)

        perfs = np.array(method_perfs)
        if perfs.max() > 0:
            exp_p = np.exp(3.0 * (perfs - perfs.mean()))
            target_w = exp_p / exp_p.sum()
            self.method_weights = (1 - self.lr) * self.method_weights + self.lr * target_w

        self.method_weights = np.maximum(self.method_weights, 0.02)
        self.method_weights /= self.method_weights.sum()

        # Update per-number attention
        self.number_attention *= self.decay
        for num in actual:
            if 1 <= num <= TOTAL:
                self.number_attention[num - 1] += 1.0

        # Update pairwise co-occurrence
        actual_list = sorted(actual)
        for a in actual_list:
            for b in actual_list:
                if a != b and 1 <= a <= TOTAL and 1 <= b <= TOTAL:
                    self.pair_success[a - 1, b - 1] += 0.1
        self.pair_success *= 0.98

        # Experience replay: keep patterns with enough hits
        if predicted_sets:
            min_for_replay = max(5, min(PICK, PICK))
            for pset in predicted_sets:
                h = len(set(pset) & actual)
                if h >= min_for_replay:
                    self.replay_buffer.append((pset, actual, h))
            if len(self.replay_buffer) > self.max_replay:
                self.replay_buffer = self.replay_buffer[-self.max_replay :]

    # ------------------------------------------------------------------
    # Accessors (normalised)
    # ------------------------------------------------------------------

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

    def state_vector(self) -> np.ndarray:
        """Flat conditioning vector for the GAN/Transformer (length 3×TOTAL + n_methods)."""
        att = self.get_number_attention()
        pair = self.get_pair_boost()
        # Normalised method weights broadcast to TOTAL length via repetition
        w_broadcast = np.repeat(self.method_weights.mean() * np.ones(TOTAL), 1)
        return np.concatenate([att, pair, self.method_weights])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        replay_sets = np.zeros((len(self.replay_buffer), PICK), dtype=int)
        replay_actual = np.zeros((len(self.replay_buffer), DRAW_SIZE), dtype=int)
        replay_hits = np.zeros(len(self.replay_buffer), dtype=int)
        for i, (pset, actual, h) in enumerate(self.replay_buffer):
            p = (list(pset)[:PICK] + [0] * PICK)[:PICK]
            replay_sets[i] = p
            replay_actual[i] = sorted(actual)[:DRAW_SIZE]
            replay_hits[i] = h
        np.savez_compressed(
            path,
            method_weights=self.method_weights,
            number_attention=self.number_attention,
            pair_success=self.pair_success,
            method_names=np.array(self.method_names, dtype=object),
            replay_sets=replay_sets,
            replay_actual=replay_actual,
            replay_hits=replay_hits,
            lr=np.array([self.lr]),
            decay=np.array([self.decay]),
        )

    @staticmethod
    def load(path: str, method_names: List[str]) -> "MemoryBank":
        data = np.load(path, allow_pickle=True)
        memory = MemoryBank(method_names, lr=float(data["lr"][0]), decay=float(data["decay"][0]))
        saved_names = list(data["method_names"])
        name_to_idx = {n: i for i, n in enumerate(saved_names)}
        for i, name in enumerate(memory.method_names):
            if name in name_to_idx:
                j = name_to_idx[name]
                memory.method_weights[i] = data["method_weights"][j]
        memory.number_attention = data["number_attention"].copy()
        memory.pair_success = data["pair_success"].copy()
        memory.method_weights /= memory.method_weights.sum()
        n_replay = len(data["replay_hits"])
        for i in range(n_replay):
            pset = [int(x) for x in data["replay_sets"][i] if 1 <= x <= TOTAL]
            actual = set(int(x) for x in data["replay_actual"][i] if 1 <= x <= TOTAL)
            memory.replay_buffer.append((pset, actual, int(data["replay_hits"][i])))
        return memory


class ExperienceReplay:
    """Extracts reinforcement signal from high-hit patterns in the replay buffer."""

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
                        scores[num - 1] += weight
                for num in pset:
                    if num in actual and 1 <= num <= TOTAL:
                        scores[num - 1] += weight * 0.5
        mx = scores.max()
        return scores / mx if mx > 0 else scores
