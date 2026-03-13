#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN-based Image Predictor for Lottery

Uses convolutional neural networks to predict the next draw image
from a sequence of historical draw images.

Architecture:
  Input: (seq_length, 8, 10, channels)
  ↓
  3D Conv / 2D Conv per frame
  ↓
  Feature aggregation across time
  ↓
  Decoder
  ↓
  Output: (8, 10, 1) probability image
"""

from typing import List, Tuple, Optional
import numpy as np
import time


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class Conv2D:
    """2D Convolution Layer (optimized with im2col)"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: str = 'same'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.padding = padding
        self.pad = kernel_size // 2 if padding == 'same' else 0
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels * kernel_size * kernel_size).astype(np.float32) * scale
        self.b = np.zeros(out_channels, dtype=np.float32)
        
        self.x: Optional[np.ndarray] = None
    
    def _im2col(self, x: np.ndarray, h_out: int, w_out: int) -> np.ndarray:
        """Convert image patches to columns for fast convolution."""
        batch, c, h, w = x.shape
        k = self.k
        
        # Output: (batch * h_out * w_out, c * k * k)
        cols = np.zeros((batch, h_out, w_out, c, k, k), dtype=np.float32)
        
        for i in range(k):
            i_max = i + h_out
            for j in range(k):
                j_max = j + w_out
                cols[:, :, :, :, i, j] = x[:, :, i:i_max, j:j_max].transpose(0, 2, 3, 1)
        
        return cols.reshape(batch * h_out * w_out, -1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, channels, height, width)
        output: (batch, out_channels, height, width)
        """
        self.x = x
        batch, c, h, w = x.shape
        
        # Padding
        if self.pad > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        else:
            x_padded = x
        
        h_out = h if self.padding == 'same' else h - self.k + 1
        w_out = w if self.padding == 'same' else w - self.k + 1
        
        # im2col: (batch * h_out * w_out, c * k * k)
        cols = self._im2col(x_padded, h_out, w_out)
        
        # Matrix multiply: (batch * h_out * w_out, out_channels)
        out = cols @ self.W.T + self.b
        
        # Reshape: (batch, out_channels, h_out, w_out)
        out = out.reshape(batch, h_out, w_out, self.out_channels)
        out = out.transpose(0, 3, 1, 2)
        
        return out.astype(np.float32)
    
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        """Simplified backward pass"""
        db = grad.sum(axis=(0, 2, 3))
        self.b -= lr * db / grad.shape[0]
        return grad


class DenseLayer:
    """Fully connected layer"""
    
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
        else:
            return self.z
    
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        if self.activation == "relu":
            grad = grad * (self.z > 0).astype(float)
        elif self.activation == "sigmoid":
            s = sigmoid(self.z)
            grad = grad * s * (1 - s)
        
        dW = self.x.T @ grad / grad.shape[0]
        db = grad.mean(axis=0)
        dx = grad @ self.W.T
        
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class CNNPredictor:
    """
    CNN-based predictor for lottery images.
    
    Architecture:
    1. Process each frame with shared CNN encoder
    2. Aggregate temporal features
    3. Decode to output image
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (seq_len, height, width, channels)
        conv_filters: List[int] = [32, 64],
        dense_units: List[int] = [256, 128],
    ):
        self.input_shape = input_shape
        self.seq_len, self.height, self.width, self.channels = input_shape
        
        # Calculate flattened feature size
        flat_size = self.height * self.width * conv_filters[-1]
        total_flat = flat_size * self.seq_len
        
        # Feature extraction (shared across frames)
        self.conv1 = Conv2D(self.channels, conv_filters[0], 3)
        self.conv2 = Conv2D(conv_filters[0], conv_filters[1], 3)
        
        # Temporal aggregation
        self.fc1 = DenseLayer(total_flat, dense_units[0], "relu")
        self.fc2 = DenseLayer(dense_units[0], dense_units[1], "relu")
        
        # Output decoder
        self.fc_out = DenseLayer(dense_units[1], self.height * self.width, "sigmoid")
        
        self.conv_filters = conv_filters
        
    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame.
        frame: (batch, height, width, channels)
        """
        # Convert to (batch, channels, height, width)
        x = frame.transpose(0, 3, 1, 2)
        
        # Conv layers
        x = relu(self.conv1.forward(x))
        x = relu(self.conv2.forward(x))
        
        # Flatten
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        
        return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        x: (batch, seq_len, height, width, channels)
        output: (batch, height, width)
        """
        batch = x.shape[0]
        
        # Encode each frame
        frame_features = []
        for t in range(self.seq_len):
            frame = x[:, t]  # (batch, height, width, channels)
            feat = self._encode_frame(frame)
            frame_features.append(feat)
        
        # Concatenate temporal features
        all_features = np.concatenate(frame_features, axis=1)
        
        # Dense layers
        h = self.fc1.forward(all_features)
        h = self.fc2.forward(h)
        
        # Output
        out = self.fc_out.forward(h)
        out = out.reshape(batch, self.height, self.width)
        
        return out
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        if y_true.ndim == 4:
            y_true = y_true.mean(axis=-1)
        return binary_cross_entropy(y_pred, y_true)
    
    def compute_hit_rate(self, y_pred: np.ndarray, y_true: np.ndarray, top_k: int = 20) -> float:
        """
        Compute average hit rate.
        
        Args:
            y_pred: Predicted images (batch, height, width)
            y_true: True images (batch, height, width)
            top_k: Number of top predictions to consider
        
        Returns:
            Average number of correct predictions per sample
        """
        if y_true.ndim == 4:
            y_true = y_true.mean(axis=-1)
        
        hits = []
        for i in range(len(y_pred)):
            pred_flat = y_pred[i].flatten()
            true_flat = y_true[i].flatten()
            
            # Get top K predicted indices
            pred_top = set(np.argsort(pred_flat)[-top_k:])
            # Get indices where true value is 1
            true_pos = set(np.where(true_flat > 0.5)[0])
            
            hits.append(len(pred_top & true_pos))
        
        return np.mean(hits)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single training step."""
        # Forward
        y_pred = self.forward(x)
        
        if y.ndim == 4:
            y = y.mean(axis=-1)
        
        loss = self.compute_loss(y_pred, y)
        
        # Backward (simplified)
        grad = (y_pred - y) / x.shape[0]
        grad_flat = grad.reshape(x.shape[0], -1)
        
        grad_flat = self.fc_out.backward(grad_flat, lr)
        grad_flat = self.fc2.backward(grad_flat, lr)
        self.fc1.backward(grad_flat, lr)
        
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate prediction."""
        return self.forward(x)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True,
        print_every: int = 5,
    ) -> dict:
        """
        Train the model with detailed progress tracking.
        
        Args:
            X_train: Training input
            y_train: Training target
            X_val: Validation input (optional)
            y_val: Validation target (optional)
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Whether to print progress
            print_every: Print every N epochs
        
        Returns:
            Training history dict with losses and metrics
        """
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        history = {
            "loss": [], 
            "val_loss": [],
            "hit_rate": [],
            "val_hit_rate": [],
            "epoch_times": [],
            "best_loss": float('inf'),
            "best_hit_rate": 0.0,
            "best_epoch": 0,
        }
        
        start_time = time.time()
        
        import sys
        
        if verbose:
            print("\n" + "=" * 85, flush=True)
            print("🧠 CNN PREDICTOR TRAINING", flush=True)
            print("=" * 85, flush=True)
            print(f"  Training Samples:   {n_samples}", flush=True)
            print(f"  Validation Samples: {len(X_val) if X_val is not None else 'None'}", flush=True)
            print(f"  Batch Size:         {batch_size}", flush=True)
            print(f"  Batches/Epoch:      {n_batches}", flush=True)
            print(f"  Epochs:             {epochs}", flush=True)
            print(f"  Learning Rate:      {lr}", flush=True)
            print("=" * 85, flush=True)
            print(f"{'Epoch':>6} │ {'Loss':>10} │ {'Val Loss':>10} │ {'Hits':>6} │ {'Time':>8} │ Progress", flush=True)
            print("-" * 85, flush=True)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            batch_count = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i + batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                loss = self.train_step(batch_x, batch_y, lr)
                epoch_loss += loss
                batch_count += 1
                
                # Progress indicator within epoch (show batch progress)
                if verbose:
                    pct = batch_count / n_batches * 100
                    bar_filled = int(15 * batch_count / n_batches)
                    bar = "█" * bar_filled + "░" * (15 - bar_filled)
                    print(f"\r{epoch+1:>6} │ {'training':>10} │ {'...':>10} │ {'...':>6} │ {'...':>8} │ [{bar}] {pct:5.1f}%", end="", flush=True)
                    sys.stdout.flush()
            
            avg_loss = epoch_loss / batch_count
            history["loss"].append(avg_loss)
            
            # Compute training hit rate
            train_pred = self.predict(X_train[:min(100, len(X_train))])
            train_hr = self.compute_hit_rate(train_pred, y_train[:min(100, len(y_train))])
            history["hit_rate"].append(train_hr)
            
            # Validation
            val_loss_str = "   N/A   "
            val_hr = 0.0
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(val_pred, y_val if y_val.ndim == 3 else y_val.mean(axis=-1))
                val_hr = self.compute_hit_rate(val_pred, y_val)
                history["val_loss"].append(val_loss)
                history["val_hit_rate"].append(val_hr)
                val_loss_str = f"{val_loss:10.6f}"
            
            epoch_time = time.time() - epoch_start
            history["epoch_times"].append(epoch_time)
            
            # Track best
            if avg_loss < history["best_loss"]:
                history["best_loss"] = avg_loss
                history["best_epoch"] = epoch + 1
            
            current_hr = val_hr if val_hr > 0 else train_hr
            if current_hr > history["best_hit_rate"]:
                history["best_hit_rate"] = current_hr
            
            # Print progress
            if verbose:
                progress = epoch + 1
                filled = int(15 * progress / epochs)
                bar = "█" * filled + "░" * (15 - filled)
                
                hr_str = f"{current_hr:5.1f}"
                
                # Always print epoch summary (overwrites batch progress)
                print(f"\r{epoch+1:>6} │ {avg_loss:10.6f} │ {val_loss_str} │ {hr_str}/20 │ {epoch_time*1000:6.0f}ms │ [{bar}]", flush=True)
        
        total_time = time.time() - start_time
        history["total_time"] = total_time
        
        if verbose:
            print("-" * 85, flush=True)
            print(f"\n📊 TRAINING COMPLETE", flush=True)
            print(f"   Total Time:       {total_time:.2f}s", flush=True)
            print(f"   Avg Epoch Time:   {np.mean(history['epoch_times'])*1000:.1f}ms", flush=True)
            print(f"   Final Loss:       {history['loss'][-1]:.6f}", flush=True)
            if history["val_loss"]:
                print(f"   Final Val Loss:   {history['val_loss'][-1]:.6f}", flush=True)
            print(f"   Final Hit Rate:   {history['hit_rate'][-1]:.1f}/20", flush=True)
            if history["val_hit_rate"]:
                print(f"   Final Val Hits:   {history['val_hit_rate'][-1]:.1f}/20", flush=True)
            print(f"   Best Loss:        {history['best_loss']:.6f} (Epoch {history['best_epoch']})", flush=True)
            print(f"   Best Hit Rate:    {history['best_hit_rate']:.1f}/20", flush=True)
            print(f"   Random Baseline:  5.0/20", flush=True)
            print("=" * 85 + "\n", flush=True)
        
        return history


    def save(self, path: str):
        """
        Save model weights to file.
        
        Args:
            path: Path to save the model (e.g., 'models/cnn_model.npz')
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Collect all weights
        weights = {
            'input_shape': np.array(self.input_shape),
            'conv1_W': self.conv1.W,
            'conv1_b': self.conv1.b,
            'conv2_W': self.conv2.W,
            'conv2_b': self.conv2.b,
            'fc1_W': self.fc1.W,
            'fc1_b': self.fc1.b,
            'fc2_W': self.fc2.W,
            'fc2_b': self.fc2.b,
            'fc_out_W': self.fc_out.W,
            'fc_out_b': self.fc_out.b,
        }
        
        np.savez(path, **weights)
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """
        Load model weights from file.
        
        Args:
            path: Path to load the model from
        """
        data = np.load(path)
        
        # Load weights
        self.conv1.W = data['conv1_W']
        self.conv1.b = data['conv1_b']
        self.conv2.W = data['conv2_W']
        self.conv2.b = data['conv2_b']
        self.fc1.W = data['fc1_W']
        self.fc1.b = data['fc1_b']
        self.fc2.W = data['fc2_W']
        self.fc2.b = data['fc2_b']
        self.fc_out.W = data['fc_out_W']
        self.fc_out.b = data['fc_out_b']
        
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'CNNPredictor':
        """
        Create a model instance and load weights from file.
        
        Args:
            path: Path to the saved model
        
        Returns:
            Loaded CNNPredictor instance
        """
        data = np.load(path)
        input_shape = tuple(data['input_shape'])
        
        model = cls(input_shape=input_shape)
        model.load(path)
        
        return model


def test_cnn_predictor():
    """Test the CNN predictor"""
    # Create dummy data
    seq_len, h, w, c = 10, 8, 10, 1
    batch_size = 4
    
    X = np.random.rand(batch_size, seq_len, h, w, c).astype(np.float32)
    y = np.random.rand(batch_size, h, w).astype(np.float32)
    
    # Create model
    model = CNNPredictor(input_shape=(seq_len, h, w, c))
    
    # Test forward pass
    pred = model.forward(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {pred.shape}")
    
    # Test training
    loss = model.train_step(X, y)
    print(f"Training loss: {loss:.4f}")


if __name__ == "__main__":
    test_cnn_predictor()
