#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration (single source of truth).
"""

import os
from typing import Dict, Any


class Config:
    """Project configuration."""

    def __init__(self):
        self.ROOT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        self.DATA_CONFIG: Dict[str, Any] = {
            "data_file": os.path.join(self.ROOT_DIR, "data", "data.csv"),
            "issue_col": "期数",
            "date_col": "日期",
            "number_cols": [f"红球_{i}" for i in range(1, 21)],
        }

        self.GAME_RULES = {
            "total_numbers": 80,
            "draw_numbers": 20,
        }

        self.PREDICT_CONFIG = {
            "recent_n": 100,
            "baseline_n": 400,
            "high_band_min": 61,
            "high_band_bonus": 1.5,
            "min_high_band": 8,
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self
        try:
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key, default)
                else:
                    value = getattr(value, key, default)
                if value is default:
                    return default
            return value
        except Exception:
            return default

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        obj = self
        for key in keys[:-1]:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key)
        last_key = keys[-1]
        if isinstance(obj, dict):
            obj[last_key] = value
        else:
            setattr(obj, last_key, value)
