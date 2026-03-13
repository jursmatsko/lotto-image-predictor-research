#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use case: predict unpopular numbers for today (cold + high-band strategy).
"""

from typing import List, Optional

from config.settings import Config
from domain.strategy import UnpopularStrategy
from storage.repository import DataRepository


def run(config: Optional[Config] = None) -> List[int]:
    cfg = config or Config()
    path = cfg.DATA_CONFIG["data_file"]
    repo = DataRepository()
    df = repo.load(path)
    pc = cfg.PREDICT_CONFIG
    strategy = UnpopularStrategy(
        recent_n=pc.get("recent_n", 100),
        baseline_n=pc.get("baseline_n", 400),
        high_band_min=pc.get("high_band_min", 61),
        high_band_bonus=pc.get("high_band_bonus", 1.5),
        min_high_band=pc.get("min_high_band", 8),
    )
    numbers = strategy.pick_top_20(df)
    print("今日冷门预测 (unpopular):", sorted(numbers))
    return numbers
