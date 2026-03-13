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


@dataclass
class RealisticConfig:
    """现实折中版配置：频率 + 衰减 + 软约束。"""

    freq_windows: Tuple[int, ...] = (300, 1000)
    freq_window_weights: Tuple[float, ...] = (0.65, 0.35)
    decay_half_life: float = 120.0
    w_freq: float = 0.45
    w_decay: float = 0.55

    draw_size: int = DRAW_NUMBERS
    odd_range: Tuple[int, int] = (9, 11)
    sum_range: Tuple[int, int] = (780, 880)
    max_consecutive_soft: int = 4
    min_decades_covered: int = 7  # 10号一段，共 8 段

    min_constraint_score: float = 0.72
    max_sampling_trials: int = 220


@dataclass
class FastRFConfig:
    """快速 RF 回测配置（单模型、特征缓存）。"""

    n_estimators: int = 120
    max_depth: int = 10
    min_samples_leaf: int = 3
    recency_decay: float = 0.0


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


def build_feature_cache(
    draws: np.ndarray,
    feat_cfg: FeatureConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    预计算所有时点特征与标签，用于加速滚动回测。

    返回:
      X_cache: (n_draws, 80, feat_dim)
      y_matrix: (n_draws, 80)
    """
    n_draws = draws.shape[0]
    feat_dim = len(feat_cfg.windows) * 2
    X_cache = np.zeros((n_draws, TOTAL_NUMBERS, feat_dim), dtype=float)
    y_matrix = np.zeros((n_draws, TOTAL_NUMBERS), dtype=int)

    for idx in range(1, n_draws):
        X_cache[idx] = _compute_single_snapshot_features(draws, idx, feat_cfg)
    for idx in range(n_draws):
        for num in draws[idx]:
            if 1 <= num <= TOTAL_NUMBERS:
                y_matrix[idx, int(num) - 1] = 1
    return X_cache, y_matrix


def train_single_rf_fast(
    X_cache: np.ndarray,
    y_matrix: np.ndarray,
    train_end_idx: int,
    feat_cfg: FeatureConfig,
    rf_cfg: FastRFConfig,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    训练单个 RF（二分类）：
    - 每期 80 个号码视为 80 个样本
    - 标签是该号码在下一期是否出现
    """
    start_idx = max(feat_cfg.min_history, 1)
    if train_end_idx <= start_idx:
        raise ValueError("训练样本不足")

    X_seq = X_cache[start_idx:train_end_idx]   # (T,80,F)
    y_seq = y_matrix[start_idx:train_end_idx]  # (T,80)
    T, _, feat_dim = X_seq.shape

    X_train = X_seq.reshape(T * TOTAL_NUMBERS, feat_dim)
    y_train = y_seq.reshape(T * TOTAL_NUMBERS)

    sample_weight = None
    if rf_cfg.recency_decay > 0:
        # 越近样本权重越高，按“期”赋权后复制给该期 80 个号码
        ages = np.arange(T - 1, -1, -1, dtype=float)
        period_w = np.exp(-rf_cfg.recency_decay * ages)
        sample_weight = np.repeat(period_w, TOTAL_NUMBERS)

    clf = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        max_depth=rf_cfg.max_depth,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf

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


def _normalize(x: np.ndarray) -> np.ndarray:
    """安全归一化到 [0, 1]。"""
    x = np.asarray(x, dtype=float)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def compute_weighted_frequency(
    draws: np.ndarray,
    windows: Sequence[int],
    window_weights: Sequence[float],
) -> np.ndarray:
    """多窗口频率打分。"""
    if len(windows) == 0:
        raise ValueError("freq_windows 不能为空")
    if len(windows) != len(window_weights):
        raise ValueError("freq_windows 与 freq_window_weights 长度必须一致")

    scores = np.zeros(TOTAL_NUMBERS, dtype=float)
    w_arr = np.asarray(window_weights, dtype=float)
    w_sum = float(w_arr.sum())
    if w_sum <= 0:
        w_arr = np.ones_like(w_arr, dtype=float)
        w_sum = float(w_arr.sum())
    w_arr = w_arr / w_sum

    n = draws.shape[0]
    for w, alpha in zip(windows, w_arr):
        start = max(0, n - int(w))
        recent = draws[start:n]
        if recent.size == 0:
            continue
        freq = np.zeros(TOTAL_NUMBERS, dtype=float)
        for num in range(1, TOTAL_NUMBERS + 1):
            freq[num - 1] = float((recent == num).sum()) / float(recent.size)
        scores += float(alpha) * _normalize(freq)
    return _normalize(scores)


def compute_exponential_recency_score(
    draws: np.ndarray,
    half_life: float,
) -> np.ndarray:
    """指数衰减近期打分：越近期开出贡献越大。"""
    if half_life <= 0:
        raise ValueError("decay_half_life 必须大于 0")
    n = draws.shape[0]
    if n <= 0:
        return np.zeros(TOTAL_NUMBERS, dtype=float)

    # age=0 表示最新一期历史
    ages = np.arange(n - 1, -1, -1, dtype=float)
    decay = np.exp(-np.log(2.0) * ages / float(half_life))
    scores = np.zeros(TOTAL_NUMBERS, dtype=float)
    for t in range(n):
        w = decay[t]
        row = draws[t]
        for num in row:
            if 1 <= num <= TOTAL_NUMBERS:
                scores[int(num) - 1] += w
    return _normalize(scores)


def compute_realistic_base_weights(
    draws: np.ndarray,
    rcfg: RealisticConfig,
) -> np.ndarray:
    """现实折中版基础权重：频率 + 衰减。"""
    freq = compute_weighted_frequency(
        draws=draws,
        windows=rcfg.freq_windows,
        window_weights=rcfg.freq_window_weights,
    )
    decay = compute_exponential_recency_score(draws, half_life=rcfg.decay_half_life)
    score = rcfg.w_freq * freq + rcfg.w_decay * decay
    return _normalize(score) + 1e-12


def _max_consecutive_len(nums: Sequence[int]) -> int:
    """计算号码集合中的最长连续长度。"""
    if not nums:
        return 0
    arr = sorted(set(int(x) for x in nums))
    best = 1
    cur = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def evaluate_soft_constraints(
    nums: Sequence[int],
    rcfg: RealisticConfig,
) -> float:
    """
    软约束评分（0~1）：
    - 奇偶平衡
    - 和值范围
    - 连号长度
    - 段位覆盖（1-10 ... 71-80）
    """
    if len(nums) != rcfg.draw_size:
        return 0.0
    arr = sorted(int(x) for x in nums)

    odd_cnt = sum(1 for x in arr if x % 2 == 1)
    odd_low, odd_high = rcfg.odd_range
    if odd_low <= odd_cnt <= odd_high:
        odd_score = 1.0
    else:
        center = (odd_low + odd_high) / 2.0
        odd_score = max(0.0, 1.0 - abs(float(odd_cnt) - center) / 6.0)

    s = int(sum(arr))
    s_low, s_high = rcfg.sum_range
    if s_low <= s <= s_high:
        sum_score = 1.0
    else:
        # 超出区间按比例惩罚
        dist = float(s_low - s) if s < s_low else float(s - s_high)
        sum_score = max(0.0, 1.0 - dist / 220.0)

    max_run = _max_consecutive_len(arr)
    if max_run <= rcfg.max_consecutive_soft:
        run_score = 1.0
    else:
        run_score = max(0.0, 1.0 - float(max_run - rcfg.max_consecutive_soft) / 4.0)

    decades = np.zeros(8, dtype=int)
    for x in arr:
        idx = min(7, max(0, (x - 1) // 10))
        decades[idx] += 1
    covered = int((decades > 0).sum())
    if covered >= rcfg.min_decades_covered:
        cover_score = 1.0
    else:
        cover_score = max(0.0, float(covered) / float(rcfg.min_decades_covered))

    # 均匀性（抑制某一段过度集中）
    max_decade = int(decades.max())
    dense_penalty = max(0.0, float(max_decade - 5) / 6.0)
    dense_score = 1.0 - dense_penalty

    score = (
        0.28 * odd_score
        + 0.28 * sum_score
        + 0.18 * run_score
        + 0.18 * cover_score
        + 0.08 * dense_score
    )
    return float(max(0.0, min(1.0, score)))


def _weighted_sample_without_replacement(
    rng: np.random.Generator,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    """按权重无放回采样 k 个索引。"""
    p = np.asarray(weights, dtype=float).copy()
    p = np.maximum(p, 1e-12)
    p = p / float(p.sum())
    idx = rng.choice(np.arange(len(p)), size=k, replace=False, p=p)
    return np.asarray(idx, dtype=int)


def constrained_sample_from_weights(
    weights: np.ndarray,
    rcfg: RealisticConfig,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> Tuple[List[int], float]:
    """按权重采样并用软约束筛选，返回最佳样本与其约束分。"""
    if temperature <= 0:
        temperature = 1.0
    logits = np.log(np.maximum(weights, 1e-12))
    logits = logits / float(temperature)
    p = np.exp(logits - logits.max())
    p = p / float(p.sum())

    best_nums: List[int] = []
    best_score = -1.0
    for _ in range(rcfg.max_sampling_trials):
        pick_idx = _weighted_sample_without_replacement(rng, p, rcfg.draw_size)
        nums = sorted(int(i) + 1 for i in pick_idx)
        c_score = evaluate_soft_constraints(nums, rcfg)
        if c_score > best_score:
            best_score = c_score
            best_nums = nums
        if c_score >= rcfg.min_constraint_score:
            break
    return best_nums, float(max(0.0, best_score))


def generate_constrained_cover_sets(
    weights: np.ndarray,
    rcfg: RealisticConfig,
    n_sets: int,
    random_seed: int = 42,
) -> List[List[int]]:
    """生成多组覆盖集：在高权重前提下减少组间重复。"""
    rng = np.random.default_rng(random_seed)
    usage = np.zeros(TOTAL_NUMBERS, dtype=float)
    sets: List[List[int]] = []

    for i in range(n_sets):
        # 使用率越高，下一组越降权，增强覆盖
        adjusted = np.asarray(weights, dtype=float) / (1.0 + 0.22 * usage)
        temp = 0.9 + 0.05 * (i % 4)
        nums, _ = constrained_sample_from_weights(
            weights=adjusted,
            rcfg=rcfg,
            rng=rng,
            temperature=temp,
        )
        sets.append(nums)
        for n in nums:
            usage[n - 1] += 1.0
    return sets


def _bootstrap_mean_ci(
    values: Sequence[float],
    n_bootstrap: int = 1200,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """均值 bootstrap 置信区间。"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(sample.mean())
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def run_realistic_today(
    cfg: Config,
    rcfg: RealisticConfig,
    n_sets: int,
    seed: int,
) -> None:
    """现实折中版今日预测：基础权重 + 软约束采样。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    if len(draws) < 30:
        raise ValueError("历史期数太少，至少需要 30 期")

    weights = compute_realistic_base_weights(draws, rcfg)
    top20 = np.argsort(weights)[::-1][:rcfg.draw_size]
    top20_nums = sorted(int(i) + 1 for i in top20)
    top20_score = evaluate_soft_constraints(top20_nums, rcfg)

    sets = generate_constrained_cover_sets(
        weights=weights,
        rcfg=rcfg,
        n_sets=n_sets,
        random_seed=seed,
    )

    print("=" * 80)
    print("KLE 快乐8 - 现实折中版 (频率+衰减+软约束)")
    print("=" * 80)
    print(f"历史期数: {len(draws)}")
    print("\nTop20(按基础权重直接取前 20):")
    print(" ".join(f"{n:02d}" for n in top20_nums))
    print(f"该集合软约束得分: {top20_score:.3f}")

    print("\n覆盖集合（软约束采样）:")
    for i, nums in enumerate(sets, start=1):
        s = " ".join(f"{n:02d}" for n in nums)
        c_score = evaluate_soft_constraints(nums, rcfg)
        print(f"Set {i:02d} [{c_score:.3f}]: {s}")

    print("\n⚠️ 重要免责声明：")
    print("• 公平开奖下不存在稳定可复现的预测优势，本结果仅用于娱乐/实验")
    print("• 输出更偏向“结构化选号与覆盖”，不是提升数学期望的证据")
    print("• 请理性购彩，严格控制预算")


def run_realistic_backtest(
    cfg: Config,
    rcfg: RealisticConfig,
    last_n: int,
    random_trials: int,
    seed: int,
) -> None:
    """现实折中版滚动回测，并给出随机基线对照。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    n_draws = len(draws)

    min_history = max(max(rcfg.freq_windows), 120)
    max_eval = max(0, n_draws - min_history)
    if max_eval <= 0:
        raise ValueError(f"历史期数不足，至少需要 {min_history + 1} 期")
    if last_n <= 0 or last_n > max_eval:
        last_n = min(30, max_eval)
    start_test_idx = n_draws - last_n

    rng = np.random.default_rng(seed)
    model_hits: List[int] = []
    random_mean_hits: List[float] = []
    uplift_hits: List[float] = []
    constraint_scores: List[float] = []

    print("=" * 80)
    print("KLE 快乐8 - 现实折中版滚动回测")
    print("=" * 80)
    print(f"总历史期数: {n_draws}, 回测期数: {last_n}, 随机对照抽样: {random_trials}/期")

    for test_idx in range(start_test_idx, n_draws):
        hist = draws[:test_idx]
        actual = set(int(x) for x in draws[test_idx])
        weights = compute_realistic_base_weights(hist, rcfg)

        pred_nums, c_score = constrained_sample_from_weights(
            weights=weights,
            rcfg=rcfg,
            rng=rng,
            temperature=1.0,
        )
        pred_set = set(pred_nums)
        hit_m = len(pred_set & actual)

        rand_hits = []
        for _ in range(max(1, random_trials)):
            r_idx = rng.choice(np.arange(TOTAL_NUMBERS), size=rcfg.draw_size, replace=False)
            r_set = set(int(i) + 1 for i in r_idx)
            rand_hits.append(len(r_set & actual))
        rand_mean = float(np.mean(rand_hits))

        model_hits.append(hit_m)
        random_mean_hits.append(rand_mean)
        uplift_hits.append(float(hit_m) - rand_mean)
        constraint_scores.append(c_score)

        issue = (
            df.iloc[test_idx][cfg.DATA_CONFIG["issue_col"]]
            if cfg.DATA_CONFIG["issue_col"] in df.columns
            else test_idx
        )
        print(
            f"期号 {issue}: model={hit_m:>2d}/20, "
            f"random≈{rand_mean:.2f}/20, uplift={hit_m - rand_mean:+.2f}, "
            f"constraint={c_score:.3f}"
        )

    m_arr = np.asarray(model_hits, dtype=float)
    r_arr = np.asarray(random_mean_hits, dtype=float)
    u_arr = np.asarray(uplift_hits, dtype=float)
    c_arr = np.asarray(constraint_scores, dtype=float)

    uplift_ci_low, uplift_ci_high = _bootstrap_mean_ci(u_arr.tolist())
    model_ci_low, model_ci_high = _bootstrap_mean_ci(m_arr.tolist())
    rand_ci_low, rand_ci_high = _bootstrap_mean_ci(r_arr.tolist())

    hit_counts = np.bincount(m_arr.astype(int), minlength=rcfg.draw_size + 1)

    print("-" * 80)
    print(f"模型平均命中: {m_arr.mean():.3f}/20  (95% CI [{model_ci_low:.3f}, {model_ci_high:.3f}])")
    print(f"随机平均命中: {r_arr.mean():.3f}/20  (95% CI [{rand_ci_low:.3f}, {rand_ci_high:.3f}])")
    print(f"平均 uplift : {u_arr.mean():+.3f}/20 (95% CI [{uplift_ci_low:+.3f}, {uplift_ci_high:+.3f}])")
    print(f"软约束分数均值: {c_arr.mean():.3f}")
    print("命中数分布(模型):")
    for h in range(len(hit_counts)):
        if hit_counts[h] > 0:
            print(f"  hits={h:2d}: {int(hit_counts[h])} 次")

    print("\n说明：若 uplift CI 覆盖 0，则无法说明相对随机有稳定优势。")


def _parse_int_list(csv: str) -> List[int]:
    vals = [int(x.strip()) for x in csv.split(",") if x.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise ValueError("列表不能为空")
    return sorted(list(set(vals)))


def run_realistic_cover_backtest(
    cfg: Config,
    rcfg: RealisticConfig,
    last_n: int,
    random_trials: int,
    cover_sets_list: Sequence[int],
    seed: int,
    print_picks: bool = False,
    detail_n: int = 0,
    print_hist: bool = True,
) -> None:
    """
    现实折中版覆盖回测：
    - 每期先生成 N 组候选 20 号集合（N 可为 10/20/30...）
    - 以该期“最佳一组命中”评估覆盖策略能力
    """
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    n_draws = len(draws)

    min_history = max(max(rcfg.freq_windows), 120)
    max_eval = max(0, n_draws - min_history)
    if max_eval <= 0:
        raise ValueError(f"历史期数不足，至少需要 {min_history + 1} 期")
    if last_n <= 0 or last_n > max_eval:
        last_n = min(60, max_eval)
    start_test_idx = n_draws - last_n

    cover_sets_list = sorted(list(set(int(x) for x in cover_sets_list if int(x) > 0)))
    if not cover_sets_list:
        raise ValueError("cover_sets_list 不能为空")

    if detail_n <= 0:
        detail_n = int(max(cover_sets_list))
    if detail_n not in cover_sets_list:
        detail_n = int(max(cover_sets_list))

    rng = np.random.default_rng(seed)
    rand_mean_hits: List[float] = []
    strategy_hits = {int(n): [] for n in cover_sets_list}

    print("=" * 80)
    print("KLE 快乐8 - 现实折中版覆盖回测 (best-of-N)")
    print("=" * 80)
    print(
        f"总历史期数: {n_draws}, 回测期数: {last_n}, "
        f"随机对照抽样: {random_trials}/期, N列表: {list(cover_sets_list)}"
    )

    for test_idx in range(start_test_idx, n_draws):
        hist = draws[:test_idx]
        actual = set(int(x) for x in draws[test_idx])
        weights = compute_realistic_base_weights(hist, rcfg)

        # 随机基线（每期均值）
        rand_hits = []
        for _ in range(max(1, random_trials)):
            r_idx = rng.choice(np.arange(TOTAL_NUMBERS), size=rcfg.draw_size, replace=False)
            r_set = set(int(i) + 1 for i in r_idx)
            rand_hits.append(len(r_set & actual))
        rand_mean = float(np.mean(rand_hits))
        rand_mean_hits.append(rand_mean)

        line_parts = []
        detail_best_nums: List[int] = []
        detail_best_hit = -1
        for n_sets in cover_sets_list:
            sets = generate_constrained_cover_sets(
                weights=weights,
                rcfg=rcfg,
                n_sets=int(n_sets),
                random_seed=seed + test_idx + int(n_sets) * 11,
            )
            best_hit = 0
            best_nums_local: List[int] = []
            for nums in sets:
                h = len(set(nums) & actual)
                if h > best_hit:
                    best_hit = h
                    best_nums_local = list(nums)
            strategy_hits[int(n_sets)].append(best_hit)
            mark = "🟢" if best_hit >= 10 else ("🔵" if best_hit >= 8 else "🟡" if best_hit >= 6 else "🔴")
            line_parts.append(f"N={int(n_sets)}->{best_hit:>2d}{mark}")

            if int(n_sets) == int(detail_n):
                detail_best_nums = best_nums_local
                detail_best_hit = best_hit

        issue = (
            df.iloc[test_idx][cfg.DATA_CONFIG["issue_col"]]
            if cfg.DATA_CONFIG["issue_col"] in df.columns
            else test_idx
        )
        print(f"期号 {issue}: random≈{rand_mean:.2f} | " + ", ".join(line_parts))

        if print_picks and detail_best_nums:
            matched = sorted(set(detail_best_nums) & actual)
            pred_s = " ".join(f"{x:02d}" for x in sorted(detail_best_nums))
            act_s = " ".join(f"{x:02d}" for x in sorted(actual))
            m_s = " ".join(f"{x:02d}" for x in matched) if matched else "(none)"
            print(
                f"   detail(N={detail_n}) hits={detail_best_hit:>2d} | "
                f"Pred: {pred_s}"
            )
            print(f"   Actual: {act_s}")
            print(f"   Match : {m_s}")

    print("-" * 80)
    print("覆盖策略汇总（按 uplift 排序）")

    rows = []
    r_arr = np.asarray(rand_mean_hits, dtype=float)
    for n_sets in cover_sets_list:
        arr = np.asarray(strategy_hits[int(n_sets)], dtype=float)
        uplift = arr - r_arr
        low, high = _bootstrap_mean_ci(uplift.tolist(), seed=seed + 100 + int(n_sets))
        rows.append(
            {
                "n_sets": int(n_sets),
                "avg_hits": float(arr.mean()),
                "p_ge_8": float(np.mean(arr >= 8)),
                "p_ge_10": float(np.mean(arr >= 10)),
                "max_hits": int(arr.max()),
                "rand_avg": float(r_arr.mean()),
                "uplift": float(uplift.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )

    rows = sorted(rows, key=lambda x: (x["p_ge_10"], x["p_ge_8"], x["uplift"]), reverse=True)
    for r in rows:
        print(
            f"N={r['n_sets']:>2d}: avg {r['avg_hits']:.3f}, P>=8 {r['p_ge_8']:.3f}, "
            f"P>=10 {r['p_ge_10']:.3f}, max {r['max_hits']}, rand {r['rand_avg']:.3f}, "
            f"uplift {r['uplift']:+.3f}, CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        )

        if print_hist:
            arr = np.asarray(strategy_hits[int(r["n_sets"])], dtype=int)
            cnt = np.bincount(arr, minlength=DRAW_NUMBERS + 1)
            pieces = [f"{h}:{int(cnt[h])}" for h in range(len(cnt)) if cnt[h] > 0]
            print(f"   hist(N={int(r['n_sets'])}) -> " + ", ".join(pieces))

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


def run_rf_backtest_fast(
    cfg: Config,
    last_n: int,
    feat_cfg: FeatureConfig,
    weights: EnsembleWeights,
    rf_cfg: FastRFConfig,
    random_trials: int = 300,
    seed: int = 42,
) -> None:
    """快速 RF 滚动回测：特征缓存 + 单模型训练。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    n_draws = len(draws)

    if last_n <= 0 or last_n > n_draws - feat_cfg.min_history:
        last_n = min(30, n_draws - feat_cfg.min_history)
    start_test_idx = n_draws - last_n

    print("=" * 80)
    print("KLE 快乐8 - Fast RF 滚动回测")
    print("=" * 80)
    print(
        f"总历史期数: {n_draws}, 回测最近: {last_n} 期 | "
        f"trees={rf_cfg.n_estimators}, depth={rf_cfg.max_depth}, leaf={rf_cfg.min_samples_leaf}"
    )

    X_cache, y_matrix = build_feature_cache(draws, feat_cfg)
    rng = np.random.default_rng(seed)
    hit_top20_list: List[int] = []
    random_mean_hits: List[float] = []
    uplift_hits: List[float] = []

    for test_idx in range(start_test_idx, n_draws):
        clf = train_single_rf_fast(
            X_cache=X_cache,
            y_matrix=y_matrix,
            train_end_idx=test_idx,
            feat_cfg=feat_cfg,
            rf_cfg=rf_cfg,
            random_state=seed + test_idx,
        )
        prob_ml = clf.predict_proba(X_cache[test_idx])[:, 1]
        freq, gap = compute_stats_features(draws[:test_idx], feat_cfg.windows)
        scores = compute_ensemble_score(prob_ml, freq, gap, weights)

        top20_idx = np.argsort(scores)[::-1][:20]
        top20_nums = {i + 1 for i in top20_idx}
        actual_nums = set(int(x) for x in draws[test_idx])
        hits = len(top20_nums & actual_nums)
        hit_top20_list.append(hits)

        rand_hits = []
        for _ in range(max(1, random_trials)):
            r_idx = rng.choice(np.arange(TOTAL_NUMBERS), size=DRAW_NUMBERS, replace=False)
            r_set = set(int(i) + 1 for i in r_idx)
            rand_hits.append(len(r_set & actual_nums))
        rand_mean = float(np.mean(rand_hits))
        random_mean_hits.append(rand_mean)
        uplift_hits.append(float(hits) - rand_mean)

        issue = (
            df.iloc[test_idx][cfg.DATA_CONFIG["issue_col"]]
            if cfg.DATA_CONFIG["issue_col"] in df.columns
            else test_idx
        )
        print(
            f"期号 {issue}: model={hits:>2d}/20, "
            f"random≈{rand_mean:.2f}/20, uplift={hits - rand_mean:+.2f}"
        )

    m_arr = np.asarray(hit_top20_list, dtype=float)
    r_arr = np.asarray(random_mean_hits, dtype=float)
    u_arr = np.asarray(uplift_hits, dtype=float)
    u_low, u_high = _bootstrap_mean_ci(u_arr.tolist(), seed=seed)
    print("-" * 80)
    print(f"模型平均命中: {m_arr.mean():.3f}/20")
    print(f"随机平均命中: {r_arr.mean():.3f}/20")
    print(f"平均 uplift : {u_arr.mean():+.3f}/20 (95% CI [{u_low:+.3f}, {u_high:+.3f}])")


def run_strategy_benchmark(
    cfg: Config,
    feat_cfg: FeatureConfig,
    weights: EnsembleWeights,
    rcfg: RealisticConfig,
    rf_cfg: FastRFConfig,
    last_n: int = 60,
    random_trials: int = 300,
    seed: int = 42,
) -> None:
    """同口径策略排行榜（含 Fast RF）。"""
    repo = DataRepository()
    df = repo.load(cfg.DATA_CONFIG["data_file"])
    draws = _extract_draw_matrix(df, cfg)
    n_draws = len(draws)

    min_hist = max(feat_cfg.min_history, max(rcfg.freq_windows))
    if last_n <= 0 or last_n > n_draws - min_hist:
        last_n = min(60, n_draws - min_hist)
    start_test_idx = n_draws - last_n

    X_cache, y_matrix = build_feature_cache(draws, feat_cfg)
    rng = np.random.default_rng(seed)

    hits_map = {
        "Realistic-Mix-Constrained": [],
        "Mix-Top20-Deterministic": [],
        "DecayOnly-Constrained": [],
        "FreqOnly-Constrained": [],
        "FastRF-Mix-Top20": [],
        "Pure-Random": [],
    }
    rand_mean_list: List[float] = []

    for test_idx in range(start_test_idx, n_draws):
        hist = draws[:test_idx]
        actual = set(int(x) for x in draws[test_idx])

        # 统一随机基线
        rand_hits = []
        for _ in range(max(1, random_trials)):
            r_idx = rng.choice(np.arange(TOTAL_NUMBERS), size=DRAW_NUMBERS, replace=False)
            r_set = set(int(i) + 1 for i in r_idx)
            h = len(r_set & actual)
            rand_hits.append(h)
            hits_map["Pure-Random"].append(h)
        rand_mean_list.append(float(np.mean(rand_hits)))

        # Realistic mix constrained
        w_mix = compute_realistic_base_weights(hist, rcfg)
        nums_mix, _ = constrained_sample_from_weights(w_mix, rcfg, rng)
        hits_map["Realistic-Mix-Constrained"].append(len(set(nums_mix) & actual))

        # Deterministic top20 from realistic base
        top20_mix = set(int(i) + 1 for i in np.argsort(w_mix)[::-1][:DRAW_NUMBERS])
        hits_map["Mix-Top20-Deterministic"].append(len(top20_mix & actual))

        # Decay only constrained
        w_decay = compute_exponential_recency_score(hist, rcfg.decay_half_life)
        nums_decay, _ = constrained_sample_from_weights(w_decay + 1e-12, rcfg, rng)
        hits_map["DecayOnly-Constrained"].append(len(set(nums_decay) & actual))

        # Freq only constrained
        w_freq = compute_weighted_frequency(hist, rcfg.freq_windows, rcfg.freq_window_weights)
        nums_freq, _ = constrained_sample_from_weights(w_freq + 1e-12, rcfg, rng)
        hits_map["FreqOnly-Constrained"].append(len(set(nums_freq) & actual))

        # Fast RF + stats
        clf = train_single_rf_fast(
            X_cache=X_cache,
            y_matrix=y_matrix,
            train_end_idx=test_idx,
            feat_cfg=feat_cfg,
            rf_cfg=rf_cfg,
            random_state=seed + test_idx,
        )
        prob_ml = clf.predict_proba(X_cache[test_idx])[:, 1]
        freq, gap = compute_stats_features(hist, feat_cfg.windows)
        score_rf = compute_ensemble_score(prob_ml, freq, gap, weights)
        top20_rf = set(int(i) + 1 for i in np.argsort(score_rf)[::-1][:DRAW_NUMBERS])
        hits_map["FastRF-Mix-Top20"].append(len(top20_rf & actual))

    print("=" * 80)
    print("策略排行榜（同窗口同口径）")
    print("=" * 80)
    print(f"数据期数: {n_draws}, 回测窗口: {last_n}, 随机对照: {random_trials}/期")

    rows = []
    rand_ref = np.asarray(rand_mean_list, dtype=float)
    for name, vals in hits_map.items():
        arr = np.asarray(vals, dtype=float)
        if name == "Pure-Random":
            # Pure-Random 本身就是抽样结果，rand_ref 仅用于展示
            uplift_arr = arr - rand_ref.repeat(max(1, random_trials))
        else:
            uplift_arr = arr - rand_ref
        low, high = _bootstrap_mean_ci(uplift_arr.tolist(), seed=seed + 7)
        rows.append(
            {
                "strategy": name,
                "avg_hits": float(arr.mean()),
                "rand_hits": float(rand_ref.mean()),
                "uplift": float(uplift_arr.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )

    rows = sorted(rows, key=lambda x: x["uplift"], reverse=True)
    for r in rows:
        print(
            f"{r['strategy']}: avg {r['avg_hits']:.3f}, rand {r['rand_hits']:.3f}, "
            f"uplift {r['uplift']:+.3f}, uplift CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KLE 快乐8 - 统计 + 随机森林 综合预测 (v2)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_today = sub.add_parser("today", help="基于全部历史数据预测下一期，并生成多注号码")
    p_today.add_argument("--tickets", type=int, default=20, help="生成票数，默认 20")
    p_today.add_argument("--numbers-per-ticket", type=int, default=10, help="每注号码数，默认 10")

    p_bt = sub.add_parser("backtest", help="对最近若干期进行滚动回测")
    p_bt.add_argument("--last", type=int, default=10, help="回测最近 N 期（默认 10）")

    p_rt = sub.add_parser(
        "realistic-today",
        help="现实折中版：频率+衰减 + 软约束，生成若干 20 号码覆盖集合",
    )
    p_rt.add_argument("--sets", type=int, default=10, help="生成集合数量，默认 10")
    p_rt.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")

    p_rb = sub.add_parser(
        "realistic-backtest",
        help="现实折中版滚动回测，并给出随机基线对照",
    )
    p_rb.add_argument("--last", type=int, default=30, help="回测最近 N 期（默认 30）")
    p_rb.add_argument(
        "--random-trials",
        type=int,
        default=500,
        help="每期随机基线采样次数（默认 500）",
    )
    p_rb.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")

    p_rcb = sub.add_parser(
        "realistic-cover-backtest",
        help="现实折中版覆盖回测（best-of-N）",
    )
    p_rcb.add_argument("--last", type=int, default=60, help="回测最近 N 期（默认 60）")
    p_rcb.add_argument(
        "--random-trials",
        type=int,
        default=300,
        help="每期随机基线采样次数（默认 300）",
    )
    p_rcb.add_argument(
        "--cover-sets",
        type=str,
        default="10,20,30,40",
        help="覆盖集合数量列表，逗号分隔（默认 10,20,30,40）",
    )
    p_rcb.add_argument(
        "--print-picks",
        action="store_true",
        help="逐期打印 detail_n 对应的预测号码、实际号码与命中号码",
    )
    p_rcb.add_argument(
        "--detail-n",
        type=int,
        default=0,
        help="逐期详细打印对应的 N（默认 0=自动取最大 N）",
    )
    p_rcb.add_argument(
        "--no-hist",
        action="store_true",
        help="关闭汇总中的命中分布直方输出",
    )
    p_rcb.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")

    p_rff = sub.add_parser(
        "rf-backtest-fast",
        help="快速 RF 回测（特征缓存 + 单模型训练）",
    )
    p_rff.add_argument("--last", type=int, default=30, help="回测最近 N 期（默认 30）")
    p_rff.add_argument("--trees", type=int, default=120, help="RF 树数（默认 120）")
    p_rff.add_argument("--depth", type=int, default=10, help="RF 最大深度（默认 10）")
    p_rff.add_argument("--leaf", type=int, default=3, help="min_samples_leaf（默认 3）")
    p_rff.add_argument(
        "--recency-decay",
        type=float,
        default=0.0,
        help="训练样本近期加权衰减系数（默认 0）",
    )
    p_rff.add_argument("--random-trials", type=int, default=300, help="随机对照采样次数")
    p_rff.add_argument("--seed", type=int, default=42, help="随机种子")

    p_bench = sub.add_parser(
        "benchmark",
        help="同口径策略排行榜（含 Fast RF）",
    )
    p_bench.add_argument("--last", type=int, default=60, help="回测最近 N 期（默认 60）")
    p_bench.add_argument("--trees", type=int, default=100, help="Fast RF 树数（默认 100）")
    p_bench.add_argument("--depth", type=int, default=9, help="Fast RF 深度（默认 9）")
    p_bench.add_argument("--leaf", type=int, default=3, help="Fast RF min_samples_leaf")
    p_bench.add_argument("--random-trials", type=int, default=300, help="随机对照采样次数")
    p_bench.add_argument("--seed", type=int, default=42, help="随机种子")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = Config()
    feat_cfg = FeatureConfig()
    weights = EnsembleWeights()
    realistic_cfg = RealisticConfig()
    fast_rf_cfg = FastRFConfig()

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
    elif args.command == "realistic-today":
        run_realistic_today(
            cfg=cfg,
            rcfg=realistic_cfg,
            n_sets=int(args.sets),
            seed=int(args.seed),
        )
    elif args.command == "realistic-backtest":
        run_realistic_backtest(
            cfg=cfg,
            rcfg=realistic_cfg,
            last_n=int(args.last),
            random_trials=int(args.random_trials),
            seed=int(args.seed),
        )
    elif args.command == "realistic-cover-backtest":
        run_realistic_cover_backtest(
            cfg=cfg,
            rcfg=realistic_cfg,
            last_n=int(args.last),
            random_trials=int(args.random_trials),
            cover_sets_list=_parse_int_list(str(args.cover_sets)),
            seed=int(args.seed),
            print_picks=bool(args.print_picks),
            detail_n=int(args.detail_n),
            print_hist=not bool(args.no_hist),
        )
    elif args.command == "rf-backtest-fast":
        fast_rf_cfg = FastRFConfig(
            n_estimators=int(args.trees),
            max_depth=int(args.depth),
            min_samples_leaf=int(args.leaf),
            recency_decay=float(args.recency_decay),
        )
        run_rf_backtest_fast(
            cfg=cfg,
            last_n=int(args.last),
            feat_cfg=feat_cfg,
            weights=weights,
            rf_cfg=fast_rf_cfg,
            random_trials=int(args.random_trials),
            seed=int(args.seed),
        )
    elif args.command == "benchmark":
        fast_rf_cfg = FastRFConfig(
            n_estimators=int(args.trees),
            max_depth=int(args.depth),
            min_samples_leaf=int(args.leaf),
            recency_decay=0.0,
        )
        run_strategy_benchmark(
            cfg=cfg,
            feat_cfg=feat_cfg,
            weights=weights,
            rcfg=realistic_cfg,
            rf_cfg=fast_rf_cfg,
            last_n=int(args.last),
            random_trials=int(args.random_trials),
            seed=int(args.seed),
        )
    else:  # 理论上不会到这里
        raise SystemExit(f"未知子命令: {args.command}")


if __name__ == "__main__":
    main()

