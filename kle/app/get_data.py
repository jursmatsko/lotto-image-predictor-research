#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use case: fetch draw data from remote sources and save to CSV.
"""

from typing import Optional

from config.settings import Config
from storage.fetcher import DataFetcher
from storage.repository import DataRepository


def run(config: Optional[Config] = None) -> None:
    cfg = config or Config()
    path = cfg.DATA_CONFIG["data_file"]
    fetcher = DataFetcher()
    repo = DataRepository()
    df = fetcher.fetch_all()
    repo.save(df, path)
    print(f"已保存 {len(df)} 期 -> {path}")
