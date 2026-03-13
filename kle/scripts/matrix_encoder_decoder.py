#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLE 快乐8 - Matrix Encoder-Decoder 预测框架

核心思想：
1. 将每期开奖号码编码为 8×10 二值矩阵（类似图像）
2. 使用 CNN Autoencoder 学习压缩表示 (Latent Space)
3. 使用 LSTM/GRU 对 Latent 序列建模，预测下一期
4. Decoder 还原为 8×10 概率矩阵，取 Top 20 作为预测

用法：
  python scripts/matrix_encoder_decoder.py train     # 训练模型
  python scripts/matrix_encoder_decoder.py predict   # 预测下一期
  python scripts/matrix_encoder_decoder.py backtest  # 回测验证
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# 项目路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import Config
from storage.repository import DataRepository


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ModelConfig:
    """模型超参数配置"""
    # 矩阵尺寸
    matrix_rows: int = 8
    matrix_cols: int = 10
    total_numbers: int = 80
    draw_numbers: int = 20
    
    # Autoencoder
    latent_dim: int = 32
    encoder_filters: List[int] = field(default_factory=lambda: [16, 32])
    
    # 序列模型
    seq_length: int = 10  # 使用最近多少期作为输入序列
    hidden_dim: int = 64
    
    # 训练
    epochs_ae: int = 100      # Autoencoder 训练轮数
    epochs_seq: int = 100     # 序列模型训练轮数
    learning_rate: float = 0.001
    batch_size: int = 16


# ============================================================================
# 数据编码 / 解码
# ============================================================================

def numbers_to_matrix(numbers: List[int], rows: int = 8, cols: int = 10) -> np.ndarray:
    """
    将开奖号码列表 (20个号) 转换为 rows×cols 的二值矩阵。
    
    号码 1-80 映射到矩阵位置：
      号码 n → row = (n-1) // cols, col = (n-1) % cols
    
    例如 8×10 矩阵：
      号码 1-10 在第 0 行
      号码 11-20 在第 1 行
      ...
      号码 71-80 在第 7 行
    """
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for n in numbers:
        if 1 <= n <= rows * cols:
            r = (n - 1) // cols
            c = (n - 1) % cols
            matrix[r, c] = 1.0
    return matrix


def matrix_to_numbers(matrix: np.ndarray, top_k: int = 20) -> List[int]:
    """
    将概率矩阵转换回号码列表，取概率最高的 top_k 个位置。
    """
    rows, cols = matrix.shape
    flat = matrix.flatten()
    indices = np.argsort(flat)[::-1][:top_k]
    numbers = sorted([int(idx + 1) for idx in indices])
    return numbers


def numbers_to_vector(numbers: List[int], total: int = 80) -> np.ndarray:
    """将号码列表转换为 80 维二值向量。"""
    vec = np.zeros(total, dtype=np.float32)
    for n in numbers:
        if 1 <= n <= total:
            vec[n - 1] = 1.0
    return vec


def vector_to_numbers(vec: np.ndarray, top_k: int = 20) -> List[int]:
    """将概率向量转换回号码列表。"""
    indices = np.argsort(vec)[::-1][:top_k]
    return sorted([int(i + 1) for i in indices])


# ============================================================================
# 纯 NumPy 实现的神经网络组件
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class DenseLayer:
    """全连接层"""
    
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        
        # 缓存用于反向传播
        self.x: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self.z)
        elif self.activation == "sigmoid":
            return sigmoid(self.z)
        else:
            return self.z
    
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        if self.activation == "relu":
            grad = grad * relu_grad(self.z)
        elif self.activation == "sigmoid":
            s = sigmoid(self.z)
            grad = grad * s * (1 - s)
        
        dW = self.x.T @ grad / grad.shape[0]
        db = grad.mean(axis=0)
        dx = grad @ self.W.T
        
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class MLPAutoencoder:
    """
    简单的 MLP Autoencoder：
      Encoder: 80 → 64 → latent_dim
      Decoder: latent_dim → 64 → 80
    """
    
    def __init__(self, input_dim: int = 80, latent_dim: int = 32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = DenseLayer(input_dim, 64, "relu")
        self.enc2 = DenseLayer(64, latent_dim, "relu")
        
        # Decoder
        self.dec1 = DenseLayer(latent_dim, 64, "relu")
        self.dec2 = DenseLayer(64, input_dim, "sigmoid")
        
    def encode(self, x: np.ndarray) -> np.ndarray:
        h = self.enc1.forward(x)
        z = self.enc2.forward(h)
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self.dec1.forward(z)
        out = self.dec2.forward(h)
        return out
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon
    
    def train_step(self, x: np.ndarray, lr: float = 0.001) -> float:
        z, x_recon = self.forward(x)
        loss = binary_cross_entropy(x_recon, x)
        
        # 反向传播
        grad = (x_recon - x) / x.shape[0]
        grad = self.dec2.backward(grad, lr)
        grad = self.dec1.backward(grad, lr)
        grad = self.enc2.backward(grad, lr)
        grad = self.enc1.backward(grad, lr)
        
        return loss


class SequencePredictor:
    """
    简单的序列预测模型 (类似单层 LSTM 的简化版)：
      输入: (seq_len, latent_dim) → 输出: (latent_dim,)
    
    使用简化的 GRU 风格更新，用 MLP 逼近。
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, seq_length: int = 10):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # 将序列展平后用 MLP 处理
        flat_dim = seq_length * latent_dim
        self.fc1 = DenseLayer(flat_dim, hidden_dim, "relu")
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, "relu")
        self.fc3 = DenseLayer(hidden_dim, latent_dim, "linear")
        
    def forward(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: shape (batch, seq_len, latent_dim)
        返回: shape (batch, latent_dim)
        """
        batch = seq.shape[0]
        flat = seq.reshape(batch, -1)
        h = self.fc1.forward(flat)
        h = self.fc2.forward(h)
        out = self.fc3.forward(h)
        return out
    
    def train_step(self, seq: np.ndarray, target: np.ndarray, lr: float = 0.001) -> float:
        pred = self.forward(seq)
        loss = float(np.mean((pred - target) ** 2))
        
        # 反向传播 (MSE loss)
        grad = 2 * (pred - target) / pred.shape[0]
        grad = self.fc3.backward(grad, lr)
        grad = self.fc2.backward(grad, lr)
        grad = self.fc1.backward(grad, lr)
        
        return loss


class OutputDecoder:
    """
    将 latent 向量解码为 80 维概率输出。
    """
    
    def __init__(self, latent_dim: int = 32, output_dim: int = 80):
        self.fc1 = DenseLayer(latent_dim, 64, "relu")
        self.fc2 = DenseLayer(64, output_dim, "sigmoid")
        
    def forward(self, z: np.ndarray) -> np.ndarray:
        h = self.fc1.forward(z)
        out = self.fc2.forward(h)
        return out
    
    def train_step(self, z: np.ndarray, target: np.ndarray, lr: float = 0.001) -> float:
        pred = self.forward(z)
        loss = binary_cross_entropy(pred, target)
        
        grad = (pred - target) / pred.shape[0]
        grad = self.fc2.backward(grad, lr)
        grad = self.fc1.backward(grad, lr)
        
        return loss


# ============================================================================
# 完整的 Encoder-Decoder 预测系统
# ============================================================================

class MatrixEncoderDecoderSystem:
    """
    完整的 Matrix Encoder-Decoder 预测系统。
    
    训练流程：
    1. 将所有历史开奖编码为 80 维向量
    2. 训练 Autoencoder 学习 latent 表示
    3. 构建序列数据 (seq_len 期 latent → 下一期 latent)
    4. 训练序列预测模型
    5. 训练输出解码器 (latent → 80 维概率)
    
    预测流程：
    1. 取最近 seq_len 期数据，编码为 latent 序列
    2. 序列模型预测下一期 latent
    3. 解码为 80 维概率
    4. 取 Top 20 作为预测号码
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # 模型组件
        self.autoencoder = MLPAutoencoder(
            input_dim=config.total_numbers,
            latent_dim=config.latent_dim
        )
        self.seq_predictor = SequencePredictor(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            seq_length=config.seq_length
        )
        self.output_decoder = OutputDecoder(
            latent_dim=config.latent_dim,
            output_dim=config.total_numbers
        )
        
        self.is_trained = False
    
    def _load_draws(self, cfg: Config) -> List[List[int]]:
        """加载开奖数据，返回每期号码列表。"""
        repo = DataRepository()
        df = repo.load(cfg.DATA_CONFIG["data_file"])
        cols = cfg.DATA_CONFIG["number_cols"]
        
        draws = []
        for _, row in df.iterrows():
            nums = [int(row[c]) for c in cols if c in row.index]
            if len(nums) == 20:
                draws.append(nums)
        
        # 按期数升序 (旧的在前)
        return draws[::-1]
    
    def _prepare_data(self, draws: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据。
        
        返回:
          X_ae: (n_draws, 80) 用于 Autoencoder
          X_seq: (n_samples, seq_len, latent_dim) 序列输入 (训练后生成)
          y_seq: (n_samples, latent_dim) 序列目标
        """
        # 编码为向量
        X_ae = np.array([numbers_to_vector(d, self.config.total_numbers) for d in draws])
        return X_ae, None, None  # 序列数据在 AE 训练后生成
    
    def train(self, cfg: Config, verbose: bool = True) -> dict:
        """训练完整系统。"""
        draws = self._load_draws(cfg)
        n_draws = len(draws)
        
        if n_draws < self.config.seq_length + 10:
            raise ValueError(f"历史数据不足，需要至少 {self.config.seq_length + 10} 期")
        
        X_ae, _, _ = self._prepare_data(draws)
        
        print("=" * 70)
        print("KLE 快乐8 - Matrix Encoder-Decoder 训练")
        print("=" * 70)
        print(f"历史期数: {n_draws}")
        print(f"Latent 维度: {self.config.latent_dim}")
        print(f"序列长度: {self.config.seq_length}")
        
        # ========== 阶段 1: 训练 Autoencoder ==========
        print("\n📦 阶段 1: 训练 Autoencoder...")
        ae_losses = []
        for epoch in range(self.config.epochs_ae):
            # Mini-batch 训练
            indices = np.random.permutation(n_draws)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_draws, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch = X_ae[batch_idx]
                loss = self.autoencoder.train_step(batch, self.config.learning_rate)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            ae_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs_ae}: Loss = {avg_loss:.6f}")
        
        # ========== 生成 Latent 表示 ==========
        all_latents = self.autoencoder.encode(X_ae)
        
        # ========== 阶段 2: 准备序列数据 ==========
        seq_len = self.config.seq_length
        X_seq = []
        y_seq = []
        y_out = []
        
        for i in range(seq_len, n_draws):
            seq = all_latents[i - seq_len:i]  # (seq_len, latent_dim)
            target_latent = all_latents[i]    # (latent_dim,)
            target_output = X_ae[i]           # (80,)
            
            X_seq.append(seq)
            y_seq.append(target_latent)
            y_out.append(target_output)
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_out = np.array(y_out)
        
        n_samples = len(X_seq)
        print(f"\n📊 序列样本数: {n_samples}")
        
        # ========== 阶段 3: 训练序列预测器 ==========
        print("\n🔄 阶段 2: 训练序列预测器...")
        seq_losses = []
        for epoch in range(self.config.epochs_seq):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch_x = X_seq[batch_idx]
                batch_y = y_seq[batch_idx]
                loss = self.seq_predictor.train_step(batch_x, batch_y, self.config.learning_rate)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            seq_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs_seq}: Loss = {avg_loss:.6f}")
        
        # ========== 阶段 4: 训练输出解码器 ==========
        print("\n🎯 阶段 3: 训练输出解码器...")
        dec_losses = []
        for epoch in range(self.config.epochs_seq):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                batch_z = y_seq[batch_idx]  # 用真实 latent 训练解码器
                batch_y = y_out[batch_idx]
                loss = self.output_decoder.train_step(batch_z, batch_y, self.config.learning_rate)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            dec_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs_seq}: Loss = {avg_loss:.6f}")
        
        self.is_trained = True
        
        print("\n✅ 训练完成!")
        
        return {
            "ae_losses": ae_losses,
            "seq_losses": seq_losses,
            "dec_losses": dec_losses,
            "n_draws": n_draws,
            "n_samples": n_samples,
        }
    
    def predict(self, cfg: Config, num_tickets: int = 20) -> Tuple[List[int], List[List[int]]]:
        """
        预测下一期号码。
        
        返回:
          top20_numbers: Top 20 预测号码
          tickets: 生成的多注号码
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练，请先调用 train()")
        
        draws = self._load_draws(cfg)
        X_ae, _, _ = self._prepare_data(draws)
        
        # 取最近 seq_len 期
        recent = X_ae[-self.config.seq_length:]
        recent_latents = self.autoencoder.encode(recent)
        
        # 序列预测
        seq_input = recent_latents.reshape(1, self.config.seq_length, self.config.latent_dim)
        pred_latent = self.seq_predictor.forward(seq_input)
        
        # 解码为概率
        probs = self.output_decoder.forward(pred_latent)[0]
        
        # 取 Top 20
        top20 = vector_to_numbers(probs, top_k=20)
        
        # 生成多注号码 (基于概率采样)
        tickets = self._generate_tickets(probs, num_tickets)
        
        return top20, tickets
    
    def _generate_tickets(self, probs: np.ndarray, num_tickets: int) -> List[List[int]]:
        """基于概率生成多注号码。"""
        rng = np.random.default_rng(42)
        tickets = []
        
        # 按概率排序
        sorted_idx = np.argsort(probs)[::-1]
        
        for t in range(num_tickets):
            # 从前 30 个高概率号码中随机选
            pool = list(sorted_idx[:30])
            rng.shuffle(pool)
            
            # 选 7-8 个高概率 + 2-3 个中等概率
            high = sorted(pool[:rng.integers(7, 9)])
            mid = sorted(pool[10:10 + rng.integers(2, 4)])
            
            selected = list(set(high + mid))
            
            # 补足 10 个
            while len(selected) < 10:
                extra = sorted_idx[rng.integers(0, 40)]
                if extra not in selected:
                    selected.append(extra)
            
            nums = sorted([int(i + 1) for i in selected[:10]])
            tickets.append(nums)
        
        return tickets
    
    def backtest(self, cfg: Config, last_n: int = 10, verbose: bool = True) -> dict:
        """
        回测：对最近 last_n 期进行滚动预测验证。
        """
        draws = self._load_draws(cfg)
        n_draws = len(draws)
        
        min_train = self.config.seq_length + 20
        if n_draws < min_train + last_n:
            last_n = max(1, n_draws - min_train)
        
        start_test = n_draws - last_n
        
        print("=" * 70)
        print("KLE 快乐8 - Matrix Encoder-Decoder 回测")
        print("=" * 70)
        print(f"总期数: {n_draws}, 回测: {last_n} 期")
        
        results = []
        
        for test_idx in range(start_test, n_draws):
            # 用 test_idx 之前的数据训练
            train_draws = draws[:test_idx]
            
            # 临时系统
            temp_system = MatrixEncoderDecoderSystem(self.config)
            
            # 简化训练 (减少 epoch)
            temp_config = ModelConfig(
                latent_dim=self.config.latent_dim,
                seq_length=self.config.seq_length,
                epochs_ae=30,
                epochs_seq=30,
            )
            temp_system.config = temp_config
            
            # 准备数据并训练
            X_ae = np.array([numbers_to_vector(d, 80) for d in train_draws])
            
            # 快速训练 AE
            for _ in range(temp_config.epochs_ae):
                idx = np.random.permutation(len(X_ae))[:32]
                temp_system.autoencoder.train_step(X_ae[idx], 0.001)
            
            # 生成 latent
            all_latents = temp_system.autoencoder.encode(X_ae)
            
            # 准备序列数据
            seq_len = temp_config.seq_length
            if len(all_latents) < seq_len + 1:
                continue
                
            X_seq = []
            y_seq = []
            y_out = []
            for i in range(seq_len, len(all_latents)):
                X_seq.append(all_latents[i - seq_len:i])
                y_seq.append(all_latents[i])
                y_out.append(X_ae[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            y_out = np.array(y_out)
            
            # 快速训练序列模型
            for _ in range(temp_config.epochs_seq):
                idx = np.random.permutation(len(X_seq))[:16]
                temp_system.seq_predictor.train_step(X_seq[idx], y_seq[idx], 0.001)
                temp_system.output_decoder.train_step(y_seq[idx], y_out[idx], 0.001)
            
            # 预测
            recent = X_ae[-seq_len:]
            recent_latents = temp_system.autoencoder.encode(recent)
            seq_input = recent_latents.reshape(1, seq_len, self.config.latent_dim)
            pred_latent = temp_system.seq_predictor.forward(seq_input)
            probs = temp_system.output_decoder.forward(pred_latent)[0]
            
            pred_top20 = set(vector_to_numbers(probs, 20))
            actual = set(draws[test_idx])
            
            hits = len(pred_top20 & actual)
            results.append({
                "index": test_idx,
                "hits": hits,
                "predicted": sorted(pred_top20),
                "actual": sorted(actual),
            })
            
            if verbose:
                print(f"期 {test_idx}: Top20 命中 {hits}/20")
        
        if results:
            avg_hits = sum(r["hits"] for r in results) / len(results)
            print("-" * 70)
            print(f"平均 Top20 命中: {avg_hits:.2f}/20 (随机基线: 5/20)")
        
        return {"results": results}


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="KLE Matrix Encoder-Decoder 预测系统")
    sub = parser.add_subparsers(dest="command", required=True)
    
    sub.add_parser("train", help="训练模型")
    
    p_pred = sub.add_parser("predict", help="预测下一期")
    p_pred.add_argument("--tickets", type=int, default=20, help="生成票数")
    
    p_bt = sub.add_parser("backtest", help="回测验证")
    p_bt.add_argument("--last", type=int, default=10, help="回测最近 N 期")
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    model_cfg = ModelConfig()
    
    system = MatrixEncoderDecoderSystem(model_cfg)
    
    if args.command == "train":
        system.train(cfg)
        
    elif args.command == "predict":
        # 先训练
        print("🔧 训练模型中...")
        system.train(cfg, verbose=False)
        
        print("\n" + "=" * 70)
        print("🔮 预测下一期")
        print("=" * 70)
        
        top20, tickets = system.predict(cfg, num_tickets=args.tickets)
        
        print("\n🎯 Top 20 预测号码:")
        print(" ".join(f"{n:02d}" for n in top20))
        
        print(f"\n🎫 生成 {len(tickets)} 注号码:")
        for i, t in enumerate(tickets, 1):
            print(f"Ticket {i:02d}: {' '.join(f'{n:02d}' for n in t)}")
        
        # 可视化为矩阵
        print("\n📊 预测热度矩阵 (8×10):")
        matrix = numbers_to_matrix(top20)
        print("    " + " ".join(f"{i+1:2d}" for i in range(10)))
        for r in range(8):
            row_start = r * 10 + 1
            row_str = f"{row_start:2d}-{row_start+9:2d} "
            for c in range(10):
                if matrix[r, c] > 0:
                    row_str += " ██"
                else:
                    row_str += " ░░"
            print(row_str)
        
        print("\n⚠️ 重要提醒: 彩票开奖为随机事件，此预测仅供娱乐参考！")
        
    elif args.command == "backtest":
        # 直接回测
        system.backtest(cfg, last_n=args.last)


if __name__ == "__main__":
    main()
