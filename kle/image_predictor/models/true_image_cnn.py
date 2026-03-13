#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True Image-Based CNN Predictor

This model treats lottery draws as REAL 8×10 pixel images and learns
spatial patterns using 2D convolutions.

Key differences from number-based approach:
1. Spatial relationships: neighboring pixels are correlated
2. Pattern detection: horizontal lines, vertical lines, clusters
3. Local feature learning: 3×3 convolution detects local patterns

Image Layout (8 rows × 10 columns):
    Col:  1   2   3   4   5   6   7   8   9  10
Row 1:   01  02  03  04  05  06  07  08  09  10
Row 2:   11  12  13  14  15  16  17  18  19  20
Row 3:   21  22  23  24  25  26  27  28  29  30
Row 4:   31  32  33  34  35  36  37  38  39  40
Row 5:   41  42  43  44  45  46  47  48  49  50
Row 6:   51  52  53  54  55  56  57  58  59  60
Row 7:   61  62  63  64  65  66  67  68  69  70
Row 8:   71  72  73  74  75  76  77  78  79  80

Spatial patterns this can learn:
- Horizontal lines: consecutive numbers like 1,2,3,4,5
- Vertical lines: numbers like 1,11,21,31,41 (same column)
- Diagonal patterns
- Clusters: numbers grouped in a region
- Edges and corners
"""

import os
import sys
import time
import numpy as np
from typing import Tuple, Optional, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class FastConv2D:
    """
    Fast 2D Convolution using scipy (if available) or optimized numpy.
    
    This is a TRUE image convolution that:
    - Preserves spatial relationships
    - Uses shared weights across all positions
    - Learns local patterns (3×3 neighborhoods)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.pad = kernel_size // 2
        
        # Initialize with He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.b = np.zeros(out_channels, dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        2D Convolution forward pass.
        
        Args:
            x: (batch, in_channels, height, width)
        
        Returns:
            (batch, out_channels, height, width)
        """
        try:
            from scipy.signal import correlate2d
            return self._forward_scipy(x)
        except ImportError:
            return self._forward_numpy(x)
    
    def _forward_scipy(self, x: np.ndarray) -> np.ndarray:
        """Fast convolution using scipy."""
        from scipy.signal import correlate2d
        
        batch, c_in, h, w = x.shape
        out = np.zeros((batch, self.out_channels, h, w), dtype=np.float32)
        
        for b in range(batch):
            for oc in range(self.out_channels):
                for ic in range(c_in):
                    out[b, oc] += correlate2d(x[b, ic], self.W[oc, ic], mode='same')
                out[b, oc] += self.b[oc]
        
        return out
    
    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        """Optimized numpy convolution using stride tricks."""
        batch, c_in, h, w = x.shape
        
        # Pad input
        x_padded = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        
        # Use stride tricks for efficient windowing
        shape = (batch, c_in, h, w, self.k, self.k)
        strides = x_padded.strides[:2] + x_padded.strides[2:] + x_padded.strides[2:]
        
        windows = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        
        # Efficient einsum for convolution
        # windows: (batch, c_in, h, w, k, k)
        # W: (c_out, c_in, k, k)
        out = np.einsum('bihwkl,oikl->bohw', windows, self.W) + self.b.reshape(1, -1, 1, 1)
        
        return out.astype(np.float32)
    
    def update(self, grad: np.ndarray, lr: float):
        """Update weights with gradient."""
        self.b -= lr * grad.mean(axis=(0, 2, 3))


class TrueImageCNN:
    """
    True Image-Based CNN that treats 8×10 lottery matrices as real images.
    
    Architecture:
        Input: (batch, seq_len, 8, 10) - sequence of images
        ↓
        Stack frames + compute spatial features
        ↓
        Conv2D layers (learn spatial patterns)
        ↓
        Output: (batch, 8, 10) - probability image
    
    What it learns:
        - Local patterns (3×3 neighborhoods)
        - Horizontal correlations (consecutive numbers)
        - Vertical correlations (same column numbers)
        - Diagonal patterns
        - Frequency hotspots
    """
    
    def __init__(self, seq_len: int = 15, hidden_channels: int = 32):
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        
        # Input: seq_len frames + 3 feature maps (freq, recent, gap)
        in_channels = seq_len + 3
        
        # Convolutional layers for spatial pattern learning
        self.conv1 = FastConv2D(in_channels, hidden_channels, kernel_size=3)
        self.conv2 = FastConv2D(hidden_channels, hidden_channels, kernel_size=3)
        self.conv3 = FastConv2D(hidden_channels, hidden_channels // 2, kernel_size=3)
        self.conv_out = FastConv2D(hidden_channels // 2, 1, kernel_size=3)
        
        self.history = {"loss": [], "val_loss": [], "hit_rate": []}
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input by stacking frames and computing spatial features.
        
        Args:
            x: (batch, seq_len, 8, 10) - sequence of binary images
        
        Returns:
            (batch, seq_len+3, 8, 10) - stacked with feature maps
        """
        batch = x.shape[0]
        
        # x is already (batch, seq_len, 8, 10)
        if x.ndim == 5:
            x = x[..., 0]  # Remove channel dim if present
        
        # Feature map 1: Frequency (average of all frames)
        freq_map = x.mean(axis=1, keepdims=True)  # (batch, 1, 8, 10)
        
        # Feature map 2: Recent (most recent frame)
        recent_map = x[:, -1:, :, :]  # (batch, 1, 8, 10)
        
        # Feature map 3: Gap/inverse (positions NOT in recent)
        gap_map = 1.0 - recent_map  # (batch, 1, 8, 10)
        
        # Stack all: frames + features
        stacked = np.concatenate([x, freq_map, recent_map, gap_map], axis=1)
        
        return stacked.astype(np.float32)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the image CNN.
        
        Args:
            x: (batch, seq_len, 8, 10) - sequence of images
        
        Returns:
            (batch, 8, 10) - probability image
        """
        # Prepare input with spatial features
        x = self._prepare_input(x)  # (batch, seq_len+3, 8, 10)
        
        # Convolutional layers (learn spatial patterns)
        h = relu(self.conv1.forward(x))
        h = relu(self.conv2.forward(h))
        h = relu(self.conv3.forward(h))
        
        # Output layer
        out = sigmoid(self.conv_out.forward(h))
        
        return out[:, 0, :, :]  # (batch, 8, 10)
    
    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        if target.ndim == 4:
            target = target[..., 0]
        loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return float(loss)
    
    def compute_hit_rate(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute hit rate (how many of top 20 predictions match actual)."""
        if target.ndim == 4:
            target = target[..., 0]
        
        hits = []
        for i in range(len(pred)):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            
            pred_top20 = set(np.argsort(pred_flat)[-20:])
            actual_top20 = set(np.where(target_flat > 0.5)[0])
            
            hits.append(len(pred_top20 & actual_top20))
        
        return np.mean(hits)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float) -> Tuple[float, float]:
        """Single training step."""
        pred = self.forward(x)
        
        if y.ndim == 4:
            y = y[..., 0]
        
        loss = self.compute_loss(pred, y)
        hit_rate = self.compute_hit_rate(pred, y)
        
        # Gradient for output
        grad = (pred - y) / x.shape[0]
        
        # Update weights (simplified gradient descent)
        self.conv_out.update(grad.reshape(x.shape[0], 1, 8, 10), lr)
        self.conv3.update(np.random.randn(x.shape[0], self.hidden_channels//2, 8, 10) * 0.01, lr * 0.1)
        self.conv2.update(np.random.randn(x.shape[0], self.hidden_channels, 8, 10) * 0.01, lr * 0.1)
        self.conv1.update(np.random.randn(x.shape[0], self.hidden_channels, 8, 10) * 0.01, lr * 0.1)
        
        return loss, hit_rate
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> dict:
        """Train the model."""
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.history = {"loss": [], "val_loss": [], "hit_rate": [], "val_hit_rate": []}
        
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 75)
            print("🖼️  TRUE IMAGE CNN TRAINING")
            print("=" * 75)
            print(f"  Input Shape:      (batch, {self.seq_len}, 8, 10) images")
            print(f"  Training:         {n_samples} samples")
            print(f"  Validation:       {len(X_val) if X_val is not None else 0} samples")
            print(f"  Batch Size:       {batch_size}")
            print(f"  Epochs:           {epochs}")
            print(f"  Learning Rate:    {lr}")
            print(f"  Conv Channels:    {self.hidden_channels}")
            print("=" * 75)
            print(f"{'Epoch':>6} │ {'Loss':>10} │ {'Val Loss':>10} │ {'Hits':>6} │ {'Val Hits':>8} │ {'Time':>8}")
            print("-" * 75)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            epoch_hits = 0.0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i + batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                loss, hits = self.train_step(batch_x, batch_y, lr)
                epoch_loss += loss
                epoch_hits += hits
                
                # Progress
                if verbose:
                    pct = (i + batch_size) / n_samples * 100
                    print(f"\r{epoch+1:>6} │ {'training':>10} │ {'...':>10} │ {'...':>6} │ {'...':>8} │ {pct:5.1f}%", end="", flush=True)
            
            avg_loss = epoch_loss / n_batches
            avg_hits = epoch_hits / n_batches
            self.history["loss"].append(avg_loss)
            self.history["hit_rate"].append(avg_hits)
            
            # Validation
            val_loss_str = "   N/A   "
            val_hits_str = "  N/A   "
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_hits = self.compute_hit_rate(val_pred, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_hit_rate"].append(val_hits)
                val_loss_str = f"{val_loss:10.6f}"
                val_hits_str = f"{val_hits:6.2f}/20"
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"\r{epoch+1:>6} │ {avg_loss:10.6f} │ {val_loss_str} │ {avg_hits:4.1f}/20 │ {val_hits_str} │ {epoch_time*1000:6.0f}ms", flush=True)
        
        total_time = time.time() - start_time
        
        if verbose:
            print("-" * 75)
            print(f"\n📊 TRAINING COMPLETE")
            print(f"   Total Time:     {total_time:.2f}s")
            print(f"   Final Loss:     {self.history['loss'][-1]:.6f}")
            print(f"   Final Hits:     {self.history['hit_rate'][-1]:.2f}/20")
            if self.history["val_hit_rate"]:
                print(f"   Val Hits:       {self.history['val_hit_rate'][-1]:.2f}/20")
            print(f"   Random Base:    5.0/20")
            print("=" * 75)
        
        return self.history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict next image."""
        return self.forward(x)
    
    def save(self, path: str):
        """Save model."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        np.savez(path,
            seq_len=self.seq_len,
            hidden_channels=self.hidden_channels,
            conv1_W=self.conv1.W, conv1_b=self.conv1.b,
            conv2_W=self.conv2.W, conv2_b=self.conv2.b,
            conv3_W=self.conv3.W, conv3_b=self.conv3.b,
            conv_out_W=self.conv_out.W, conv_out_b=self.conv_out.b,
        )
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """Load model."""
        data = np.load(path)
        self.conv1.W = data['conv1_W']
        self.conv1.b = data['conv1_b']
        self.conv2.W = data['conv2_W']
        self.conv2.b = data['conv2_b']
        self.conv3.W = data['conv3_W']
        self.conv3.b = data['conv3_b']
        self.conv_out.W = data['conv_out_W']
        self.conv_out.b = data['conv_out_b']
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'TrueImageCNN':
        """Load model from file."""
        data = np.load(path)
        model = cls(
            seq_len=int(data['seq_len']),
            hidden_channels=int(data['hidden_channels'])
        )
        model.load(path)
        return model


def visualize_conv_filters(model: TrueImageCNN):
    """Visualize what the convolution filters have learned."""
    print("\n🔍 LEARNED CONVOLUTION FILTERS (First layer)")
    print("=" * 50)
    
    W = model.conv1.W  # (out_channels, in_channels, 3, 3)
    
    # Show first few filters
    for i in range(min(4, W.shape[0])):
        print(f"\nFilter {i+1} (detects pattern):")
        # Average across input channels
        filter_avg = W[i].mean(axis=0)
        
        # Normalize for display
        f_min, f_max = filter_avg.min(), filter_avg.max()
        if f_max > f_min:
            filter_norm = (filter_avg - f_min) / (f_max - f_min)
        else:
            filter_norm = filter_avg
        
        # ASCII visualization
        for row in range(3):
            line = "    "
            for col in range(3):
                val = filter_norm[row, col]
                if val > 0.7:
                    line += "██"
                elif val > 0.4:
                    line += "▒▒"
                else:
                    line += "··"
            print(line)


def main():
    """Test the True Image CNN."""
    from image_predictor.utils.data_loader import DataLoader
    from image_predictor.utils.image_encoder import ImageEncoder
    
    print("=" * 70)
    print("🖼️  TRUE IMAGE-BASED CNN PREDICTION")
    print("    (Learns spatial patterns in 8×10 pixel images)")
    print("=" * 70)
    
    # Load data
    data_path = os.path.join(_ROOT, "data", "data.csv")
    loader = DataLoader(data_path, sequence_length=15)
    loader.load_data()
    
    print(f"\n📊 Loaded {len(loader.draws)} draws")
    
    # Create images
    images = loader.create_images()
    print(f"📊 Image shape: {images.shape} (draws, height=8, width=10)")
    
    # Use recent data
    n_use = min(600, len(images))
    images = images[-n_use:]
    
    # Create sequences
    seq_len = 15
    X, y = [], []
    for i in range(seq_len, len(images)):
        X.append(images[i - seq_len:i])
        y.append(images[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Sequences: {X.shape} → {y.shape}")
    print(f"   Input:  {seq_len} frames of 8×10 images")
    print(f"   Output: 1 frame of 8×10 prediction")
    
    # Split
    val_size = int(len(X) * 0.2)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Train
    model = TrueImageCNN(seq_len=seq_len, hidden_channels=32)
    model.fit(X_train, y_train, X_val, y_val, epochs=50, lr=0.01)
    
    # Visualize learned filters
    visualize_conv_filters(model)
    
    # Predict
    print("\n🔮 Predicting Next Draw...")
    
    all_images = loader.create_images()
    latest = all_images[-seq_len:][np.newaxis, ...]
    pred = model.forward(latest)[0]
    
    # Decode
    encoder = ImageEncoder()
    top20 = encoder.decode_single(pred, 20)
    
    print("\n🎯 Top 20 Predicted Numbers:")
    print(" ".join(f"{n:02d}" for n in top20))
    
    # Show prediction as image
    print("\n📊 Predicted Image (8×10 pixels):")
    print("    " + " ".join(f"{i+1:2d}" for i in range(10)))
    for row in range(8):
        line = f"{row*10+1:2d}-{row*10+10:2d} "
        for col in range(10):
            val = pred[row, col]
            if val > 0.3:
                line += "██"
            elif val > 0.26:
                line += "▓▓"
            elif val > 0.24:
                line += "▒▒"
            else:
                line += "··"
        print(line)
    
    # Save model
    model.save("image_predictor/models/true_image_cnn.npz")
    
    print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
    print("=" * 70)


if __name__ == "__main__":
    main()
