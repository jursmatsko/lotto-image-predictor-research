#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLE 快乐8 入口：子命令 get-data / predict。
"""

import os
import sys
import argparse

# 保证从项目根 (kle/) 运行时可导入 config, storage, domain, app
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import Config
from app.get_data import run as run_get_data
from app.predict import run as run_predict


def main() -> None:
    parser = argparse.ArgumentParser(description="KLE 快乐8 数据与预测")
    sub = parser.add_subparsers(dest="command", help="子命令")

    sub.add_parser("get-data", help="拉取开奖数据并保存到 data/data.csv")
    p_pred = sub.add_parser("predict", help="基于冷门策略预测今日号码")
    p_pred.add_argument("--recent", type=int, default=None, help="近期期数 (默认用配置)")

    args = parser.parse_args()
    config = Config()

    if args.command == "get-data":
        run_get_data(config)
    elif args.command == "predict":
        if getattr(args, "recent", None) is not None:
            config.PREDICT_CONFIG["recent_n"] = args.recent
        run_predict(config)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
