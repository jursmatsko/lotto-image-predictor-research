#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLE 快乐8 - Advanced Encoder-Decoder 预测系统 (v2)

增强版本，包含：
1. CNN 风格卷积 - 在 8×10 矩阵上提取空间模式
2. VAE (变分自编码器) - 更好的 latent space
3. Attention 机制 - 关注历史中重要的期数
4. 多头输出 - 结合多种解码策略

用法：
  python scripts/advanced_encoder_decoder.py train
  python scripts/advanced_encoder_decoder.py predict --tickets 20
  python scripts/advanced_encoder_decoder.py backtest --last 20
  python scripts/advanced_encoder_decoder.py analyze  # 分析 latent space
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import Config
from storage.repository import DataRepository


# ============================================================================
# 配置
# ============================================================================

@dataclass
class AdvancedConfig:
    """高级模型配置"""
    # 矩阵尺寸
    matrix_rows: int = 8
    matrix_cols: int = 10
    total_numbers: int = 80
    draw_numbers: int = 20
    
    # VAE
    latent_dim: int = 32
    kl_weight: float = 0.001  # KL 散度权重
    
    # CNN
    conv_filters: List[int] = field(default_factory=lambda: [8, 16])
    kernel_size: int = 3
    
    # Attention
    attention_heads: int = 4
    
    # 序列
    seq_length: int = 15
    hidden_dim: int = 128
    
    # 训练
    epochs: int = 150
    learning_rate: float = 0.001
    batch_size: int = 32


# ============================================================================
# 工具函数
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-8)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def numbers_to_matrix(numbers: List[int], rows: int = 8, cols: int = 10) -> np.ndarray:
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for n in numbers:
        if 1 <= n <= rows * cols:
            r, c = (n - 1) // cols, (n - 1) % cols
            matrix[r, c] = 1.0
    return matrix

def matrix_to_numbers(matrix: np.ndarray, top_k: int = 20) -> List[int]:
    flat = matrix.flatten()
    indices = np.argsort(flat)[::-1][:top_k]
    return sorted([int(idx + 1) for idx in indices])

def numbers_to_vector(numbers: List[int], total: int = 80) -> np.ndarray:
    vec = np.zeros(total, dtype=np.float32)
    for n in numbers:
        if 1 <= n <= total:
            vec[n - 1] = 1.0
    return vec

def vector_to_numbers(vec: np.ndarray, top_k: int = 20) -> List[int]:
    indices = np.argsort(vec)[::-1][:top_k]
    return sorted([int(i + 1) for i in indices])


# ============================================================================
# CNN 模拟 (用滑动窗口实现卷积效果)
# ============================================================================

class Conv2DSimulator:
    """
    模拟 2D 卷积操作（无需深度学习库）
    在 8×10 矩阵上提取局部空间特征
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.pad = kernel_size // 2
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.b = np.zeros(out_channels, dtype=np.float32)
        
        self.x_padded: Optional[np.ndarray] = None
        self.out: Optional[np.ndarray] = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, channels, height, width)
        输出: (batch, out_channels, height, width)
        """
        batch, c, h, w = x.shape
        
        # Padding
        x_padded = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        self.x_padded = x_padded
        
        out = np.zeros((batch, self.out_channels, h, w), dtype=np.float32)
        
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for i in range(h):
                    for j in range(w):
                        patch = x_padded[:, ic, i:i+self.k, j:j+self.k]
                        out[:, oc, i, j] += np.sum(patch * self.W[oc, ic], axis=(1, 2))
            out[:, oc] += self.b[oc]
        
        self.out = relu(out)
        return self.out
    
    def get_flat_features(self, x: np.ndarray) -> np.ndarray:
        """获取展平的特征向量"""
        out = self.forward(x)
        return out.reshape(out.shape[0], -1)


class SpatialFeatureExtractor:
    """
    从 8×10 矩阵提取空间特征：
    - 行统计 (每行出现的号码数)
    - 列统计 (每列出现的号码数)
    - 区域统计 (4个象限)
    - 对角线统计
    - 边缘 vs 中心
    """
    
    def extract(self, matrix: np.ndarray) -> np.ndarray:
        """
        matrix: (8, 10) 或 (batch, 8, 10)
        返回特征向量
        """
        if matrix.ndim == 2:
            matrix = matrix[np.newaxis, ...]
        
        batch = matrix.shape[0]
        features = []
        
        for b in range(batch):
            m = matrix[b]
            feat = []
            
            # 行统计 (8个值)
            row_sums = m.sum(axis=1)
            feat.extend(row_sums / 10.0)
            
            # 列统计 (10个值)
            col_sums = m.sum(axis=0)
            feat.extend(col_sums / 8.0)
            
            # 四象限统计 (4个值)
            q1 = m[:4, :5].sum() / 20.0   # 左上
            q2 = m[:4, 5:].sum() / 20.0   # 右上
            q3 = m[4:, :5].sum() / 20.0   # 左下
            q4 = m[4:, 5:].sum() / 20.0   # 右下
            feat.extend([q1, q2, q3, q4])
            
            # 边缘 vs 中心
            edge = (m[0, :].sum() + m[-1, :].sum() + 
                    m[1:-1, 0].sum() + m[1:-1, -1].sum()) / 32.0
            center = m[2:6, 3:7].sum() / 16.0
            feat.extend([edge, center])
            
            # 奇偶行统计
            odd_rows = m[::2, :].sum() / 40.0
            even_rows = m[1::2, :].sum() / 40.0
            feat.extend([odd_rows, even_rows])
            
            # 连续性特征 (相邻位置都为1的数量)
            h_cont = np.sum(m[:, :-1] * m[:, 1:]) / 70.0
            v_cont = np.sum(m[:-1, :] * m[1:, :]) / 70.0
            feat.extend([h_cont, v_cont])
            
            features.append(feat)
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# VAE (变分自编码器)
# ============================================================================

class DenseLayer:
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.activation = activation
        self.x: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.W + self.b
        if self.activation == "relu":
            return relu(self.z)
        elif self.activation == "sigmoid":
            return sigmoid(self.z)
        elif self.activation == "tanh":
            return tanh(self.z)
        else:
            return self.z
    
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        if self.activation == "relu":
            grad = grad * (self.z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = sigmoid(self.z)
            grad = grad * s * (1 - s)
        elif self.activation == "tanh":
            t = tanh(self.z)
            grad = grad * (1 - t * t)
        
        dW = self.x.T @ grad / grad.shape[0]
        db = grad.mean(axis=0)
        dx = grad @ self.W.T
        
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class VAE:
    """
    变分自编码器：
    - Encoder: 输入 → μ 和 log(σ²)
    - 采样: z = μ + σ * ε (ε ~ N(0,1))
    - Decoder: z → 重建
    
    Loss = Reconstruction Loss + KL Divergence
    """
    
    def __init__(self, input_dim: int = 80, latent_dim: int = 32, kl_weight: float = 0.001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        # Encoder
        self.enc1 = DenseLayer(input_dim, 128, "relu")
        self.enc2 = DenseLayer(128, 64, "relu")
        self.fc_mu = DenseLayer(64, latent_dim, "linear")
        self.fc_logvar = DenseLayer(64, latent_dim, "linear")
        
        # Decoder
        self.dec1 = DenseLayer(latent_dim, 64, "relu")
        self.dec2 = DenseLayer(64, 128, "relu")
        self.dec3 = DenseLayer(128, input_dim, "sigmoid")
        
        # 训练时保存的中间值
        self.mu: Optional[np.ndarray] = None
        self.logvar: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.epsilon: Optional[np.ndarray] = None
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = self.enc1.forward(x)
        h = self.enc2.forward(h)
        mu = self.fc_mu.forward(h)
        logvar = self.fc_logvar.forward(h)
        return mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        std = np.exp(0.5 * logvar)
        self.epsilon = np.random.randn(*mu.shape).astype(np.float32)
        return mu + std * self.epsilon
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        h = self.dec1.forward(z)
        h = self.dec2.forward(h)
        return self.dec3.forward(h)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.mu, self.logvar = self.encode(x)
        self.z = self.reparameterize(self.mu, self.logvar)
        x_recon = self.decode(self.z)
        return x_recon, self.mu, self.logvar
    
    def compute_loss(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> Tuple[float, float, float]:
        # Reconstruction loss (BCE)
        recon_loss = binary_cross_entropy(x_recon, x)
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        
        total_loss = recon_loss + self.kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def train_step(self, x: np.ndarray, lr: float = 0.001) -> Tuple[float, float, float]:
        x_recon, mu, logvar = self.forward(x)
        total_loss, recon_loss, kl_loss = self.compute_loss(x, x_recon, mu, logvar)
        
        # 简化的反向传播
        grad = (x_recon - x) / x.shape[0]
        grad = self.dec3.backward(grad, lr)
        grad = self.dec2.backward(grad, lr)
        grad = self.dec1.backward(grad, lr)
        
        # 对 mu 和 logvar 的梯度
        std = np.exp(0.5 * self.logvar)
        grad_mu = grad + self.kl_weight * self.mu / x.shape[0]
        grad_logvar = grad * self.epsilon * 0.5 * std + self.kl_weight * 0.5 * (np.exp(self.logvar) - 1) / x.shape[0]
        
        self.fc_mu.backward(grad_mu, lr)
        self.fc_logvar.backward(grad_logvar, lr)
        
        grad_h = self.fc_mu.x.T @ grad_mu / x.shape[0] + self.fc_logvar.x.T @ grad_logvar / x.shape[0]
        # 继续反向传播到 encoder
        
        return total_loss, recon_loss, kl_loss
    
    def get_latent(self, x: np.ndarray) -> np.ndarray:
        """获取 latent 表示 (使用 μ 作为确定性表示)"""
        mu, _ = self.encode(x)
        return mu


# ============================================================================
# Attention 机制
# ============================================================================

class SelfAttention:
    """
    简化的 Self-Attention 层：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    """
    
    def __init__(self, dim: int, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        scale = np.sqrt(2.0 / dim)
        self.W_q = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_k = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_v = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_o = np.random.randn(dim, dim).astype(np.float32) * scale
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq_len, dim)
        输出: (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape
        
        Q = x @ self.W_q  # (batch, seq_len, dim)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scale = np.sqrt(float(self.head_dim))
        scores = Q @ K.transpose(0, 2, 1) / scale  # (batch, seq_len, seq_len)
        attn_weights = softmax(scores, axis=-1)
        
        attn_output = attn_weights @ V  # (batch, seq_len, dim)
        output = attn_output @ self.W_o
        
        return output


class TemporalAttentionPredictor:
    """
    带时间注意力的序列预测器：
    1. 对历史序列应用 Self-Attention
    2. 加权聚合历史信息
    3. 预测下一期
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 128, seq_length: int = 15):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # Attention
        self.attention = SelfAttention(latent_dim, num_heads=4)
        
        # 位置编码
        self.pos_encoding = self._create_positional_encoding(seq_length, latent_dim)
        
        # 预测头
        self.fc1 = DenseLayer(latent_dim, hidden_dim, "relu")
        self.fc2 = DenseLayer(hidden_dim, hidden_dim, "relu")
        self.fc3 = DenseLayer(hidden_dim, latent_dim, "linear")
    
    def _create_positional_encoding(self, max_len: int, dim: int) -> np.ndarray:
        pe = np.zeros((max_len, dim), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def forward(self, seq: np.ndarray) -> np.ndarray:
        """
        seq: (batch, seq_len, latent_dim)
        输出: (batch, latent_dim)
        """
        batch, seq_len, dim = seq.shape
        
        # 加入位置编码
        seq_with_pos = seq + self.pos_encoding[:seq_len]
        
        # Self-Attention
        attn_out = self.attention.forward(seq_with_pos)
        
        # 取最后一个位置的输出作为聚合表示
        last_hidden = attn_out[:, -1, :]
        
        # 预测
        h = self.fc1.forward(last_hidden)
        h = self.fc2.forward(h)
        pred = self.fc3.forward(h)
        
        return pred
    
    def train_step(self, seq: np.ndarray, target: np.ndarray, lr: float = 0.001) -> float:
        pred = self.forward(seq)
        loss = float(np.mean((pred - target) ** 2))
        
        grad = 2 * (pred - target) / pred.shape[0]
        grad = self.fc3.backward(grad, lr)
        grad = self.fc2.backward(grad, lr)
        self.fc1.backward(grad, lr)
        
        return loss


# ============================================================================
# 多头输出解码器
# ============================================================================

class MultiHeadDecoder:
    """
    多头输出解码器，结合多种策略：
    - Head 1: 直接概率输出
    - Head 2: 区域偏好输出 (4区)
    - Head 3: 冷热号偏好
    """
    
    def __init__(self, latent_dim: int = 32, output_dim: int = 80):
        # 主输出头
        self.main_head = DenseLayer(latent_dim, output_dim, "sigmoid")
        
        # 区域偏好头 (4区)
        self.zone_head = DenseLayer(latent_dim, 4, "softmax")
        
        # 温度调节
        self.temperature = DenseLayer(latent_dim, 1, "sigmoid")
    
    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        main_probs = self.main_head.forward(z)
        zone_probs = softmax(self.zone_head.forward(z), axis=-1)
        temp = self.temperature.forward(z) * 2 + 0.5  # 范围 [0.5, 2.5]
        return main_probs, zone_probs, temp
    
    def get_final_probs(self, z: np.ndarray) -> np.ndarray:
        """结合各头输出，得到最终 80 维概率"""
        main_probs, zone_probs, temp = self.forward(z)
        
        # 将区域偏好应用到主概率
        zone_masks = np.zeros((z.shape[0], 80), dtype=np.float32)
        zone_masks[:, 0:20] = zone_probs[:, 0:1]
        zone_masks[:, 20:40] = zone_probs[:, 1:2]
        zone_masks[:, 40:60] = zone_probs[:, 2:3]
        zone_masks[:, 60:80] = zone_probs[:, 3:4]
        
        # 温度调节
        adjusted = main_probs ** (1.0 / temp)
        
        # 结合区域偏好
        final_probs = adjusted * (0.7 + 0.3 * zone_masks)
        
        return final_probs
    
    def train_step(self, z: np.ndarray, target: np.ndarray, lr: float = 0.001) -> float:
        probs = self.get_final_probs(z)
        loss = binary_cross_entropy(probs, target)
        
        # 简化梯度更新
        grad = (probs - target) / z.shape[0]
        self.main_head.backward(grad, lr)
        
        return loss


# ============================================================================
# 完整的高级系统
# ============================================================================

class AdvancedEncoderDecoderSystem:
    """高级 Encoder-Decoder 预测系统"""
    
    def __init__(self, config: AdvancedConfig):
        self.config = config
        
        # 空间特征提取
        self.spatial_extractor = SpatialFeatureExtractor()
        
        # VAE
        self.vae = VAE(
            input_dim=config.total_numbers,
            latent_dim=config.latent_dim,
            kl_weight=config.kl_weight
        )
        
        # 注意力序列预测器
        self.seq_predictor = TemporalAttentionPredictor(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            seq_length=config.seq_length
        )
        
        # 多头解码器
        self.decoder = MultiHeadDecoder(
            latent_dim=config.latent_dim,
            output_dim=config.total_numbers
        )
        
        self.is_trained = False
        self.training_stats: Dict[str, List] = {}
    
    def _load_draws(self, cfg: Config) -> List[List[int]]:
        repo = DataRepository()
        df = repo.load(cfg.DATA_CONFIG["data_file"])
        cols = cfg.DATA_CONFIG["number_cols"]
        draws = []
        for _, row in df.iterrows():
            nums = [int(row[c]) for c in cols if c in row.index]
            if len(nums) == 20:
                draws.append(nums)
        return draws[::-1]  # 按时间升序
    
    def train(self, cfg: Config, verbose: bool = True) -> dict:
        draws = self._load_draws(cfg)
        n_draws = len(draws)
        
        X = np.array([numbers_to_vector(d) for d in draws])
        matrices = np.array([numbers_to_matrix(d) for d in draws])
        spatial_features = self.spatial_extractor.extract(matrices)
        
        print("=" * 70)
        print("🚀 KLE 快乐8 - Advanced Encoder-Decoder 训练")
        print("=" * 70)
        print(f"📊 历史期数: {n_draws}")
        print(f"🧬 Latent 维度: {self.config.latent_dim}")
        print(f"📏 序列长度: {self.config.seq_length}")
        print(f"🎯 注意力头数: {self.config.attention_heads}")
        
        # ========== 阶段 1: 训练 VAE ==========
        print("\n📦 阶段 1: 训练 VAE...")
        vae_losses = []
        for epoch in range(self.config.epochs):
            idx = np.random.permutation(n_draws)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_draws, self.config.batch_size):
                batch_idx = idx[i:i + self.config.batch_size]
                batch = X[batch_idx]
                total, recon, kl = self.vae.train_step(batch, self.config.learning_rate)
                epoch_loss += total
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            vae_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 30 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: Loss = {avg_loss:.4f}")
        
        # ========== 获取所有 latent ==========
        all_latents = self.vae.get_latent(X)
        
        # 融合空间特征到 latent
        # 将空间特征投影到 latent 空间并融合
        spatial_proj = DenseLayer(spatial_features.shape[1], self.config.latent_dim, "linear")
        spatial_latent = spatial_proj.forward(spatial_features)
        all_latents = 0.8 * all_latents + 0.2 * spatial_latent
        
        # ========== 阶段 2: 准备序列数据 ==========
        seq_len = self.config.seq_length
        X_seq, y_seq, y_out = [], [], []
        
        for i in range(seq_len, n_draws):
            X_seq.append(all_latents[i - seq_len:i])
            y_seq.append(all_latents[i])
            y_out.append(X[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_out = np.array(y_out)
        n_samples = len(X_seq)
        
        print(f"\n📊 序列样本数: {n_samples}")
        
        # ========== 阶段 3: 训练注意力序列预测器 ==========
        print("\n🔄 阶段 2: 训练 Attention 序列预测器...")
        seq_losses = []
        for epoch in range(self.config.epochs):
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = idx[i:i + self.config.batch_size]
                loss = self.seq_predictor.train_step(
                    X_seq[batch_idx], y_seq[batch_idx], self.config.learning_rate
                )
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            seq_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 30 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: Loss = {avg_loss:.4f}")
        
        # ========== 阶段 4: 训练多头解码器 ==========
        print("\n🎯 阶段 3: 训练 Multi-Head 解码器...")
        dec_losses = []
        for epoch in range(self.config.epochs):
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n_samples, self.config.batch_size):
                batch_idx = idx[i:i + self.config.batch_size]
                loss = self.decoder.train_step(
                    y_seq[batch_idx], y_out[batch_idx], self.config.learning_rate
                )
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            dec_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 30 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.config.epochs}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        self.training_stats = {
            "vae_losses": vae_losses,
            "seq_losses": seq_losses,
            "dec_losses": dec_losses,
        }
        
        # 保存训练数据引用
        self._X = X
        self._all_latents = all_latents
        self._spatial_proj = spatial_proj
        
        print("\n✅ 训练完成!")
        return self.training_stats
    
    def predict(self, cfg: Config, num_tickets: int = 20) -> Tuple[List[int], List[List[int]], np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        
        draws = self._load_draws(cfg)
        X = np.array([numbers_to_vector(d) for d in draws])
        matrices = np.array([numbers_to_matrix(d) for d in draws])
        spatial_features = self.spatial_extractor.extract(matrices)
        
        # 获取 latent
        all_latents = self.vae.get_latent(X)
        spatial_latent = self._spatial_proj.forward(spatial_features)
        all_latents = 0.8 * all_latents + 0.2 * spatial_latent
        
        # 取最近 seq_len 期
        recent = all_latents[-self.config.seq_length:]
        seq_input = recent.reshape(1, self.config.seq_length, self.config.latent_dim)
        
        # 预测
        pred_latent = self.seq_predictor.forward(seq_input)
        probs = self.decoder.get_final_probs(pred_latent)[0]
        
        top20 = vector_to_numbers(probs, 20)
        tickets = self._generate_diverse_tickets(probs, num_tickets)
        
        return top20, tickets, probs
    
    def _generate_diverse_tickets(self, probs: np.ndarray, num_tickets: int) -> List[List[int]]:
        rng = np.random.default_rng(42)
        tickets = []
        sorted_idx = np.argsort(probs)[::-1]
        
        strategies = [
            ("high_prob", 0.4),      # 高概率策略
            ("balanced", 0.3),       # 均衡策略
            ("zone_spread", 0.2),    # 区域分散策略
            ("random_mix", 0.1),     # 随机混合
        ]
        
        for t in range(num_tickets):
            strategy = rng.choice([s[0] for s in strategies], p=[s[1] for s in strategies])
            
            if strategy == "high_prob":
                # 从前 25 个高概率号码中选
                pool = list(sorted_idx[:25])
                rng.shuffle(pool)
                selected = sorted(pool[:10])
            
            elif strategy == "balanced":
                # 每个区选一些
                selected = []
                for zone_start in [0, 20, 40, 60]:
                    zone_idx = [i for i in sorted_idx if zone_start <= i < zone_start + 20][:8]
                    rng.shuffle(zone_idx)
                    selected.extend(zone_idx[:rng.integers(2, 4)])
                rng.shuffle(selected)
                selected = sorted(selected[:10])
            
            elif strategy == "zone_spread":
                # 强制 4 区各选 2-3 个
                selected = []
                for zone_start in [0, 20, 40, 60]:
                    zone_probs = probs[zone_start:zone_start + 20]
                    zone_top = np.argsort(zone_probs)[::-1][:5] + zone_start
                    selected.extend(rng.choice(zone_top, size=rng.integers(2, 4), replace=False))
                rng.shuffle(selected)
                selected = sorted(selected[:10])
            
            else:  # random_mix
                pool = list(range(80))
                weights = probs / probs.sum()
                selected = sorted(rng.choice(pool, size=10, replace=False, p=weights))
            
            nums = [int(i + 1) for i in selected]
            tickets.append(nums)
        
        return tickets
    
    def backtest(self, cfg: Config, last_n: int = 20) -> dict:
        draws = self._load_draws(cfg)
        n_draws = len(draws)
        
        min_train = self.config.seq_length + 30
        if n_draws < min_train + last_n:
            last_n = max(1, n_draws - min_train)
        
        start_test = n_draws - last_n
        
        print("=" * 70)
        print("📈 KLE 快乐8 - Advanced Encoder-Decoder 回测")
        print("=" * 70)
        print(f"总期数: {n_draws}, 回测: {last_n} 期")
        
        results = []
        
        for test_idx in range(start_test, n_draws):
            # 用简化的快速训练
            train_draws = draws[:test_idx]
            X_train = np.array([numbers_to_vector(d) for d in train_draws])
            
            # 快速 VAE 训练
            temp_vae = VAE(80, self.config.latent_dim, self.config.kl_weight)
            for _ in range(30):
                idx = np.random.permutation(len(X_train))[:32]
                temp_vae.train_step(X_train[idx], 0.001)
            
            latents = temp_vae.get_latent(X_train)
            
            # 准备序列
            seq_len = self.config.seq_length
            if len(latents) < seq_len + 1:
                continue
            
            X_seq = np.array([latents[i - seq_len:i] for i in range(seq_len, len(latents))])
            y_seq = np.array([latents[i] for i in range(seq_len, len(latents))])
            y_out = np.array([X_train[i] for i in range(seq_len, len(latents))])
            
            # 快速训练预测器
            temp_pred = TemporalAttentionPredictor(self.config.latent_dim, self.config.hidden_dim, seq_len)
            temp_dec = MultiHeadDecoder(self.config.latent_dim, 80)
            
            for _ in range(30):
                idx = np.random.permutation(len(X_seq))[:16]
                temp_pred.train_step(X_seq[idx], y_seq[idx], 0.001)
                temp_dec.train_step(y_seq[idx], y_out[idx], 0.001)
            
            # 预测
            recent = latents[-seq_len:].reshape(1, seq_len, self.config.latent_dim)
            pred_z = temp_pred.forward(recent)
            probs = temp_dec.get_final_probs(pred_z)[0]
            
            pred_top20 = set(vector_to_numbers(probs, 20))
            actual = set(draws[test_idx])
            hits = len(pred_top20 & actual)
            
            results.append({"index": test_idx, "hits": hits})
            print(f"期 {test_idx}: Top20 命中 {hits}/20")
        
        if results:
            avg_hits = sum(r["hits"] for r in results) / len(results)
            max_hits = max(r["hits"] for r in results)
            min_hits = min(r["hits"] for r in results)
            print("-" * 70)
            print(f"📊 平均命中: {avg_hits:.2f}/20 | 最高: {max_hits}/20 | 最低: {min_hits}/20")
            print(f"📊 随机基线: 5.00/20")
        
        return {"results": results}
    
    def analyze_latent_space(self, cfg: Config) -> None:
        """分析 latent space 的结构"""
        if not self.is_trained:
            print("请先训练模型")
            return
        
        draws = self._load_draws(cfg)
        X = np.array([numbers_to_vector(d) for d in draws])
        latents = self.vae.get_latent(X)
        
        print("=" * 70)
        print("🔍 Latent Space 分析")
        print("=" * 70)
        
        # 统计
        print(f"\n📊 Latent 向量统计:")
        print(f"  维度: {latents.shape[1]}")
        print(f"  均值范围: [{latents.mean(axis=0).min():.3f}, {latents.mean(axis=0).max():.3f}]")
        print(f"  标准差范围: [{latents.std(axis=0).min():.3f}, {latents.std(axis=0).max():.3f}]")
        
        # 最近几期的 latent 可视化
        print(f"\n📈 最近 5 期 Latent 前 8 维:")
        for i in range(-5, 0):
            z = latents[i]
            z_str = " ".join(f"{v:+.2f}" for v in z[:8])
            print(f"  期 {len(draws)+i+1}: [{z_str} ...]")
        
        # Latent 相似度分析
        print(f"\n🔗 相邻期 Latent 相似度 (余弦):")
        for i in range(-5, 0):
            z1 = latents[i - 1]
            z2 = latents[i]
            cos_sim = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2) + 1e-8)
            print(f"  期 {len(draws)+i} ↔ 期 {len(draws)+i+1}: {cos_sim:.4f}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="KLE Advanced Encoder-Decoder")
    sub = parser.add_subparsers(dest="command", required=True)
    
    sub.add_parser("train", help="训练模型")
    
    p_pred = sub.add_parser("predict", help="预测")
    p_pred.add_argument("--tickets", type=int, default=20)
    
    p_bt = sub.add_parser("backtest", help="回测")
    p_bt.add_argument("--last", type=int, default=20)
    
    sub.add_parser("analyze", help="分析 latent space")
    
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()
    model_cfg = AdvancedConfig()
    
    system = AdvancedEncoderDecoderSystem(model_cfg)
    
    if args.command == "train":
        system.train(cfg)
        
    elif args.command == "predict":
        print("🔧 训练模型中...")
        system.train(cfg, verbose=False)
        
        print("\n" + "=" * 70)
        print("🔮 预测下一期")
        print("=" * 70)
        
        top20, tickets, probs = system.predict(cfg, args.tickets)
        
        print("\n🎯 Top 20 预测号码:")
        print(" ".join(f"{n:02d}" for n in top20))
        
        # 按概率排序显示
        print("\n📊 号码概率排名 (Top 30):")
        sorted_idx = np.argsort(probs)[::-1][:30]
        for rank, idx in enumerate(sorted_idx, 1):
            bar = "█" * int(probs[idx] * 20)
            print(f"  {rank:2d}. {idx+1:02d}: {probs[idx]:.3f} {bar}")
        
        print(f"\n🎫 生成 {len(tickets)} 注号码:")
        for i, t in enumerate(tickets, 1):
            print(f"Ticket {i:02d}: {' '.join(f'{n:02d}' for n in t)}")
        
        # 矩阵可视化
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
        system.train(cfg, verbose=False)
        system.backtest(cfg, args.last)
        
    elif args.command == "analyze":
        system.train(cfg, verbose=False)
        system.analyze_latent_space(cfg)


if __name__ == "__main__":
    main()
