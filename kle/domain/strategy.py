#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain: Unpopular-number strategy (cold + high-band bias).
"""

import random
from typing import List

import pandas as pd


class UnpopularStrategy:
    """
    Score by cold = recent underperformance vs baseline (baseline from older history only).
    Recent window and baseline are disjoint: cold is not "raw history frequency".
    """

    def __init__(
        self,
        recent_n: int = 100,
        baseline_n: int = 400,
        high_band_min: int = 61,
        high_band_bonus: float = 1.5,
        min_high_band: int = 6,
    ):
        self.recent_n = recent_n
        self.baseline_n = baseline_n
        self.high_band_min = high_band_min
        self.high_band_bonus = high_band_bonus
        self.min_high_band = min_high_band

    def _number_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("红球_") and c[4:].isdigit()]

    def _frequency(self, df: pd.DataFrame) -> dict:
        num_cols = self._number_columns(df)
        if not num_cols:
            return {}
        freq = {n: 0 for n in range(1, 81)}
        for _, row in df.iterrows():
            for c in num_cols:
                v = row[c]
                if isinstance(v, (int, float)) and 1 <= int(v) <= 80:
                    freq[int(v)] = freq.get(int(v), 0) + 1
        return freq

    def score_numbers(self, df: pd.DataFrame) -> pd.Series:
        """
        Cold = recent count below baseline expectation (baseline from older data only).
        Score higher when actual_recent < expected from baseline (not from recent).
        """
        # recent window: head(recent_n)
        recent = df.head(self.recent_n)
        # baseline = older history only (exclude recent): rows [recent_n : recent_n+baseline_n]
        baseline = df.iloc[self.recent_n : self.recent_n + self.baseline_n]
        if len(baseline) == 0:
            baseline = df.iloc[self.recent_n :]
        recent_freq = self._frequency(recent)
        baseline_freq = self._frequency(baseline)
        n_draws_baseline = max(len(baseline), 1)
        n_draws_recent = max(len(recent), 1)
        scores = {}
        for n in range(1, 81):
            actual = recent_freq.get(n, 0)
            base_count = baseline_freq.get(n, 0)
            expected = base_count / n_draws_baseline * n_draws_recent
            under = expected - actual
            score = 1.0 / (1.0 + actual) * (1.0 + max(0, under))
            if n >= self.high_band_min:
                score *= self.high_band_bonus
            scores[n] = score
        return pd.Series(scores)

    def pick_top_20(self, df: pd.DataFrame, shuffle_ties: bool = True) -> List[int]:
        """Return 20 numbers: top by score, at least min_high_band from 61-80.
        When many numbers have the same score (e.g. all 0 frequency), break ties
        randomly so the same 1-12 don't appear every time."""
        scores = self.score_numbers(df)
        low = [n for n in scores.index if n < self.high_band_min]
        high = [n for n in scores.index if n >= self.high_band_min]
        # sort by score desc; within same score, shuffle so not always 1,2,...,12
        def order_low(n):
            return (-scores[n], random.random() if shuffle_ties else 0)
        def order_high(n):
            return (-scores[n], random.random() if shuffle_ties else 0)
        low_sorted = sorted(low, key=order_low)
        high_sorted = sorted(high, key=order_high)
        n_high = min(self.min_high_band, 20, len(high_sorted))
        n_low = 20 - n_high
        chosen = high_sorted[:n_high] + low_sorted[:n_low]
        return sorted(chosen)
