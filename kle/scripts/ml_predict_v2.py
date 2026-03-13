#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLE 快乐8 - 统计特征 + 随机森林 综合预测脚本 (v2)

设计目标：
- 使用简单、可解释的特征（近期频率、间隔等）
- 对每个号码 (1–80) 训练独立的概率模型（One-vs-All）
- 用统计 + ML 的加权综合得分进行排序
- 支持「今日预测」和「历史回测」两种模式

用法示例（在 kle 项目根目录执行）：
  # 今日预测（默认 20 注，每注 10 个号）
  python scripts/ml_predict_v2.py today
  python scripts/ml_predict_v2.py today --tickets 20 --numbers-per-ticket 10

  # 简单回测最近 10 期
  python scripts/ml_predict_v2.py backtest --last 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:  # pragma: no cover - 仅在缺库时触发
    RandomForestClassifier = None  # type: ignore
    _SKLEARN_IMPORT_ERROR = e
else:
    _SKLEARN_IMPORT_ERROR = None

# 保证从项目根 (kle/) 运行时可导入 config, storage
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import Config
from storage.repository import DataRepository


TOTAL_NUMBERS = 80
DRAW_NUMBERS = 20


@dataclass
class FeatureConfig:
    """特征配置。"""

    windows: Tuple[int, ...] = (5, 10, 20)
    min_history: int = 20  # 至少多少期历史后才开始作为训练样本


@dataclass
class EnsembleWeights:
    """综合打分权重。"""

    w_ml: float = 0.5       # 机器学习概率权重
    w_freq: float = 0.3     # 近期频率权重
    w_gap: float = 0.2      # 间隔特征权重


def _extract_draw_matrix(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """从 DataFrame 提取开奖矩阵 (n_draws, 20)。"""
    cols = cfg.DATA_CONFIG["number_cols"]
    if not all(c in df.columns for c in cols):
        raise ValueError(f"数据列缺失，期待列: {cols}")
    arr = df[cols].to_numpy(dtype=int)
    return arr


def _compute_single_snapshot_features(
    draws: np.ndarray,
    idx: int,
    feat_cfg: FeatureConfig,
) -> np.ndarray:
    """
    针对某一期（使用其之前的历史 idx 期）计算每个号码的特征。

    draws: shape (n_draws, 20)，倒序/正序均可，这里只看相对位置。
    idx: 使用 [0, idx) 作为历史，idx >= 1。
    返回: X_num, shape (80, feat_dim)
    """
    windows = feat_cfg.windows
    num_win = len(windows)
    feat_dim = num_win * 2  # 每个窗口: 频率 + 间隔
    X = np.zeros((TOTAL_NUMBERS, feat_dim), dtype=float)

    for n in range(1, TOTAL_NUMBERS + 1):
        feats: List[float] = []
        for w in windows:
            start = max(0, idx - w)
            recent = draws[start:idx]  # shape (<=w, 20)
            if recent.size > 0:
                freq = float((recent == n).sum()) / float(recent.size)
            else:
                freq = 0.0

            # 计算在最近 w 期内的“未出现期数”占比作为简化 gap 特征
            gap = 0
            for back in range(1, w + 1):
                j = idx - back
                if j < 0:
                    break
                if n in draws[j]:
                    break
                gap += 1
            gap_norm = float(gap) / float(w)

            feats.append(freq)
            feats.append(gap_norm)

        X[n - 1, :] = np.asarray(feats, dtype=float)

    return X


def build_dataset(
    draws: np.ndarray,
    feat_cfg: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于开奖历史构建训练数据。

    返回:
      X_all: shape (n_samples, 80, feat_dim)
      y_all: shape (n_samples, 80)  # 每个号码是否在该期出现
    """
    n_draws = draws.shape[0]
    start_idx = max(feat_cfg.min_history, 1)
    snapshots: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for idx in range(start_idx, n_draws):
        X_snapshot = _compute_single_snapshot_features(draws, idx, feat_cfg)
        y = np.zeros(TOTAL_NUMBERS, dtype=int)
        for num in draws[idx]:
            if 1 <= num <= TOTAL_NUMBERS:
                y[num - 1] = 1
        snapshots.append(X_snapshot)
        labels.append(y)

    if not snapshots:
        raise ValueError("历史期数不足，无法构建训练集")

    X_all = np.stack(snapshots, axis=0)
    y_all = np.stack(labels, axis=0)
    return X_all, y_all


def train_rf_models(
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> List[RandomForestClassifier | None]:
    """
    训练每个号码的随机森林二分类器。

    返回 models 列表，长度为 80，可能包含 None（表示该号码训练数据退化）。
    """
    if RandomForestClassifier is None:
        raise RuntimeError(
            "scikit-learn 未安装，无法训练随机森林模型。"
        ) from _SKLEARN_IMPORT_ERROR

    n_samples, n_numbers, feat_dim = X_all.shape
    assert n_numbers == TOTAL_NUMBERS

    models: List[RandomForestClassifier | None] = []
    for i in range(TOTAL_NUMBERS):
        X_i = X_all[:, i, :]  # shape (n_samples, feat_dim)
        y_i = y_all[:, i]     # shape (n_samples,)
        positives = int(y_i.sum())
        negatives = int(len(y_i) - positives)

        # 若该号码几乎从未出现或一直出现，则模型没有学习意义，返回 None
        if positives == 0 or negatives == 0:
            models.append(None)
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_i, y_i)
        models.append(clf)

    return models


def predict_next_proba(
    draws: np.ndarray,
    models: Sequence[RandomForestClassifier | None],
    feat_cfg: FeatureConfig,
) -> np.ndarray:
    """
    基于已训练模型预测“下一期”每个号码出现的概率。

    draws: 全部历史，使用最后一作为“当前最新”，预测下一期。
    """
    n_draws = draws.shape[0]
    if n_draws < feat_cfg.min_history:
        raise ValueError("历史期数不足，无法进行预测")

    X_next = _compute_single_snapshot_features(draws, n_draws, feat_cfg)
    prob_ml = np.zeros(TOTAL_NUMBERS, dtype=float)

    for i, model in enumerate(models):
        if model is None:
            prob_ml[i] = 0.0
        else:
            p = model.predict_proba(X_next[i, :].reshape(1, -1))[0, 1]
            prob_ml[i] = float(p)

    return prob_ml


def compute_stats_features(
    draws: np.ndarray,
    windows: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算简单统计特征：
      - freq: 近期出现频率（使用最大窗口）
      - gap: 自最后一次出现以来的“期数”（归一化）
    """
    n_draws = draws.shape[0]
    max_w = max(windows)
    start = max(0, n_draws - max_w)
    recent = draws[start:n_draws]

    freq = np.zeros(TOTAL_NUMBERS, dtype=float)
    if recent.size > 0:
        for n in range(1, TOTAL_NUMBERS + 1):
            freq[n - 1] = float((recent == n).sum()) / float(recent.size)

    gap = np.zeros(TOTAL_NUMBERS, dtype=float)
    for n in range(1, TOTAL_NUMBERS + 1):
        g = 0
        for back in range(1, max_w + 1):
            j = n_draws - back
            if j < 0:
                break
            if n in draws[j]:
                break
            g += 1
        gap[n - 1] = float(g) / float(max_w)

    return freq, gap


def compute_ensemble_score(
    prob_ml: np.ndarray,
    freq: np.ndarray,
    gap: np.ndarray,
    weights: EnsembleWeights,
) -> np.ndarray:
    """计算综合得分并做简单归一化。"""
    # 归一化到 [0,1]
    def _norm(x: np.ndarray) -> np.ndarray:
        xmin = float(x.min())
        xmax = float(x.max())
        if xmax <= xmin:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin)

    p_ml = _norm(prob_ml)
    p_freq = _norm(freq)
    # gap 越大，代表越“冷”，这里给一个适度的正向奖励
    p_gap = _norm(gap)

    score = (
        weights.w_ml * p_ml
        + weights.w_freq * p_freq
        + weights.w_gap * p_gap
    )
    return score


def generate_tickets_from_scores(
    scores: np.ndarray,
    num_tickets: int = 20,
    numbers_per_ticket: int = 10,
    random_seed: int = 42,
) -> List[List[int]]:
    """
    根据综合得分生成若干注号码：
      - 优先覆盖高分号码
      - 同时引入一定随机性增加多样性
    """
    rng = np.random.default_rng(random_seed)
    indices = np.argsort(scores)[::-1]  # 从高到低排序的号码索引 (0-based)

    # 构建候选池：前 30 个为高分池，其余加入少量发散
    top_k = min(30, TOTAL_NUMBERS)
    top_indices = indices[:top_k]
    rest_indices = indices[top_k:]

    tickets: List[List[int]] = []
    for t in range(num_tickets):
        # 每注：从前 15 个里选 6–7 个，从 16–30 里选 2–3 个，剩余从后面随机补足
        rng.shuffle(top_indices)
        high_part = sorted(top_indices[: rng.integers(6, 8)])
        mid_part = sorted(top_indices[rng.integers(10, 15) : rng.integers(15, min(20, top_k))])

        remain = numbers_per_ticket - len(high_part) - len(mid_part)
        if remain < 0:
            remain = 0

        if rest_indices.size > 0 and remain > 0:
            rest_sample = sorted(
                rng.choice(rest_indices, size=min(remain, rest_indices.size), replace=False)
            )
        else:
            rest_sample = []

        nums_idx = sorted(set(high_part + mid_part + rest_sample))
        # 若仍不足 numbers_per_ticket，则从全体中补齐（不重复）
        if len(nums_idx) < numbers_per_ticket:
            all_idx = list(range(TOTAL_NUMBERS))
            rng.shuffle(all_idx)
            for i in all_idx:
                if i not in nums_idx:
                    nums_idx.append(i)
                    if len(nums_idx) >= numbers_per_ticket:
                        break

        nums = sorted([i + 1 for i in nums_idx[:numbers_per_ticket]])
        tickets.append(nums)

    return tickets


def run_today(
    cfg: Config,
    tickets: int,
    numbers_per_ticket: int,
    feat_cfg: FeatureConfig,
    weights: EnsembleWeights,
) -> None:
    """今日预测入口。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)

    print("=" * 80)
    print("KLE 快乐8 - 统计 + 随机森林 综合预测 (v2)")
    print("=" * 80)
    print(f"历史期数: {len(draws)}")

    X_all, y_all = build_dataset(draws, feat_cfg)
    models = train_rf_models(X_all, y_all)
    prob_ml = predict_next_proba(draws, models, feat_cfg)
    freq, gap = compute_stats_features(draws, feat_cfg.windows)
    scores = compute_ensemble_score(prob_ml, freq, gap, weights)

    # 展示前 20 个号码
    top_indices = np.argsort(scores)[::-1][:20]
    print("\n📊 综合得分 Top 20 号码：")
    print("Rank  号码  综合得分")
    for rank, idx in enumerate(top_indices, start=1):
        num = idx + 1
        print(f"{rank:>2d}    {num:02d}   {scores[idx]:.4f}")

    # 生成彩票
    tickets_list = generate_tickets_from_scores(
        scores,
        num_tickets=tickets,
        numbers_per_ticket=numbers_per_ticket,
    )

    print("\n🎫 预测票据：")
    for i, nums in enumerate(tickets_list, start=1):
        s = " ".join(f"{n:02d}" for n in nums)
        print(f"Ticket {i:02d}: {s}")

    print("\n⚠️ 重要免责声明：")
    print("• 彩票开奖为随机事件，本脚本仅基于历史数据做模式分析")
    print("• 任何方法都无法保证提升中奖概率")
    print("• 请务必理性购彩，量力而行，不要超出自己的承受范围")


def run_backtest(
    cfg: Config,
    last_n: int,
    feat_cfg: FeatureConfig,
    weights: EnsembleWeights,
) -> None:
    """简单回测：对最近 last_n 期逐期做“滚动训练 + 预测”。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    n_draws = len(draws)

    if last_n <= 0 or last_n > n_draws - feat_cfg.min_history:
        last_n = min(10, n_draws - feat_cfg.min_history)

    start_test_idx = n_draws - last_n

    print("=" * 80)
    print("KLE 快乐8 - 统计 + 随机森林 回测 (v2)")
    print("=" * 80)
    print(f"总历史期数: {n_draws}, 回测最近: {last_n} 期")

    hit_top20_list: List[int] = []

    for test_idx in range(start_test_idx, n_draws):
        train_draws = draws[:test_idx]
        if len(train_draws) < feat_cfg.min_history:
            continue

        X_all, y_all = build_dataset(train_draws, feat_cfg)
        models = train_rf_models(X_all, y_all)
        prob_ml = predict_next_proba(train_draws, models, feat_cfg)
        freq, gap = compute_stats_features(train_draws, feat_cfg.windows)
        scores = compute_ensemble_score(prob_ml, freq, gap, weights)

        top20_idx = np.argsort(scores)[::-1][:20]
        top20_nums = {i + 1 for i in top20_idx}
        actual_nums = set(int(x) for x in draws[test_idx])

        hits = len(top20_nums & actual_nums)
        hit_top20_list.append(hits)

        issue = df.iloc[test_idx][cfg.DATA_CONFIG["issue_col"]] if cfg.DATA_CONFIG["issue_col"] in df.columns else test_idx
        print(f"期号 {issue}: Top20 命中 {hits}/20")

    if hit_top20_list:
        avg_hits = sum(hit_top20_list) / len(hit_top20_list)
        print("-" * 80)
        print(f"平均 Top20 命中数: {avg_hits:.2f}/20 （随机基线约为 5/20）")
    else:
        print("可用回测样本不足，无法计算统计结果。")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KLE 快乐8 - 统计 + 随机森林 综合预测 (v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_today = sub.add_parser("today", help="基于全部历史数据预测下一期，并生成多注号码")
    p_today.add_argument("--tickets", type=int, default=20, help="生成票数，默认 20")
    p_today.add_argument("--numbers-per-ticket", type=int, default=10, help="每注号码数，默认 10")

    p_bt = sub.add_parser("backtest", help="对最近若干期进行滚动回测")
    p_bt.add_argument("--last", type=int, default=10, help="回测最近 N 期（默认 10）")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config()
    feat_cfg = FeatureConfig()
    weights = EnsembleWeights()

    if args.command == "today":
        run_today(
            cfg=cfg,
            tickets=int(args.tickets),
            numbers_per_ticket=int(args.numbers_per_ticket),
            feat_cfg=feat_cfg,
            weights=weights,
        )
    elif args.command == "backtest":
        run_backtest(
            cfg=cfg,
            last_n=int(args.last),
            feat_cfg=feat_cfg,
            weights=weights,
        )
    else:  # 理论上不会到这里
        raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    main()

