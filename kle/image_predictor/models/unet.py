#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
U-Net based Image Predictor for Lottery

U-Net architecture with skip connections for image-to-image prediction.
Takes aggregated historical sequence and predicts next draw image.

Architecture:
  Input: Aggregated sequence features (8×10)
  ↓
  Encoder (downsampling path)
  ↓
  Bottleneck
  ↓
  Decoder (upsampling path with skip connections)
  ↓
  Output: 8×10 probability image
"""

from typing import Tuple, Optional, List
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class Conv2DBlock:
    """Double convolution block for U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Two 3x3 convolutions
        scale1 = np.sqrt(2.0 / (in_channels * 9))
        scale2 = np.sqrt(2.0 / (out_channels * 9))
        
        self.W1 = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * scale1
        self.b1 = np.zeros(out_channels, dtype=np.float32)
        
        self.W2 = np.random.randn(out_channels, out_channels, 3, 3).astype(np.float32) * scale2
        self.b2 = np.zeros(out_channels, dtype=np.float32)
    
    def _conv(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        batch, c_in, h, w = x.shape
        c_out = W.shape[0]
        
        # Same padding
        x_padded = np.pad(x, ((0,0), (0,0), (1, 1), (1, 1)), mode='constant')
        
        out = np.zeros((batch, c_out, h, w), dtype=np.float32)
        
        for oc in range(c_out):
            for ic in range(c_in):
                for i in range(h):
                    for j in range(w):
                        patch = x_padded[:, ic, i:i+3, j:j+3]
                        out[:, oc, i, j] += np.sum(patch * W[oc, ic], axis=(1, 2))
            out[:, oc] += b[oc]
        
        return out
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        h = relu(self._conv(x, self.W1, self.b1))
        h = relu(self._conv(h, self.W2, self.b2))
        return h


class MaxPool2D:
    """2x2 Max pooling."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, c, h, w = x.shape
        out_h, out_w = h // 2, w // 2
        
        out = np.zeros((batch, c, out_h, out_w), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i*2:i*2+2, j*2:j*2+2]
                out[:, :, i, j] = patch.max(axis=(2, 3))
        
        return out


class Upsample2D:
    """2x upsampling (nearest neighbor)."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, c, h, w = x.shape
        
        out = np.zeros((batch, c, h*2, w*2), dtype=np.float32)
        
        for i in range(h):
            for j in range(w):
                out[:, :, i*2:i*2+2, j*2:j*2+2] = x[:, :, i:i+1, j:j+1]
        
        return out


class UNetPredictor:
    """
    U-Net based predictor for lottery images.
    
    Adapted for 8×10 images (small size limits depth).
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (seq_len, height, width, channels)
        base_filters: int = 16,
    ):
        self.input_shape = input_shape
        self.seq_len, self.height, self.width, self.channels = input_shape
        
        # Input channels = seq_len * channels (stacked frames) + aggregated features
        in_channels = self.seq_len + 3  # frames + freq + gap + recent
        
        # Encoder
        self.enc1 = Conv2DBlock(in_channels, base_filters)
        self.pool1 = MaxPool2D()
        
        self.enc2 = Conv2DBlock(base_filters, base_filters * 2)
        self.pool2 = MaxPool2D()
        
        # Bottleneck
        self.bottleneck = Conv2DBlock(base_filters * 2, base_filters * 4)
        
        # Decoder
        self.up2 = Upsample2D()
        self.dec2 = Conv2DBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        
        self.up1 = Upsample2D()
        self.dec1 = Conv2DBlock(base_filters * 2 + base_filters, base_filters)
        
        # Output
        scale = np.sqrt(2.0 / (base_filters * 9))
        self.conv_out_W = np.random.randn(1, base_filters, 3, 3).astype(np.float32) * scale
        self.conv_out_b = np.zeros(1, dtype=np.float32)
        
        self.base_filters = base_filters
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """
        Prepare input by stacking frames and computing features.
        
        Args:
            x: (batch, seq_len, height, width, channels)
        
        Returns:
            (batch, seq_len + 3, height, width) - stacked with features
        """
        batch = x.shape[0]
        
        # Stack all frames
        if x.ndim == 5 and x.shape[4] == 1:
            frames = x[:, :, :, :, 0]  # (batch, seq_len, h, w)
        else:
            frames = x.mean(axis=-1) if x.ndim == 5 else x
        
        # Compute aggregate features
        freq_map = frames.mean(axis=1, keepdims=True)  # Average frequency
        recent_map = frames[:, -1:, :, :]  # Most recent
        
        # Gap feature (simplified)
        gap_map = 1 - frames[:, -1:, :, :]  # Inverse of recent
        
        # Stack all
        stacked = np.concatenate([frames, freq_map, recent_map, gap_map], axis=1)
        
        return stacked
    
    def _output_conv(self, x: np.ndarray) -> np.ndarray:
        """Final output convolution."""
        batch, c, h, w = x.shape
        
        x_padded = np.pad(x, ((0,0), (0,0), (1, 1), (1, 1)), mode='constant')
        
        out = np.zeros((batch, 1, h, w), dtype=np.float32)
        
        for ic in range(c):
            for i in range(h):
                for j in range(w):
                    patch = x_padded[:, ic, i:i+3, j:j+3]
                    out[:, 0, i, j] += np.sum(patch * self.conv_out_W[0, ic], axis=(1, 2))
        
        out[:, 0] += self.conv_out_b[0]
        return sigmoid(out)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, height, width, channels)
        
        Returns:
            (batch, height, width) prediction
        """
        # Prepare input
        x = self._prepare_input(x)  # (batch, channels, h, w)
        
        # Encoder
        e1 = self.enc1.forward(x)           # (batch, 16, 8, 10)
        p1 = self.pool1.forward(e1)         # (batch, 16, 4, 5)
        
        e2 = self.enc2.forward(p1)          # (batch, 32, 4, 5)
        p2 = self.pool2.forward(e2)         # (batch, 32, 2, 2)
        
        # Bottleneck
        b = self.bottleneck.forward(p2)     # (batch, 64, 2, 2)
        
        # Decoder with skip connections
        u2 = self.up2.forward(b)            # (batch, 64, 4, 4)
        
        # Handle size mismatch (crop/pad)
        if u2.shape[2:] != e2.shape[2:]:
            # Pad u2 to match e2
            pad_h = e2.shape[2] - u2.shape[2]
            pad_w = e2.shape[3] - u2.shape[3]
            if pad_h > 0 or pad_w > 0:
                u2 = np.pad(u2, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
        d2 = self.dec2.forward(np.concatenate([u2, e2], axis=1))
        
        u1 = self.up1.forward(d2)           # (batch, 32, 8, 10)
        
        # Handle size mismatch
        if u1.shape[2:] != e1.shape[2:]:
            pad_h = e1.shape[2] - u1.shape[2]
            pad_w = e1.shape[3] - u1.shape[3]
            if pad_h > 0 or pad_w > 0:
                u1 = np.pad(u1, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='constant')
        
        d1 = self.dec1.forward(np.concatenate([u1, e1], axis=1))
        
        # Output
        out = self._output_conv(d1)         # (batch, 1, 8, 10)
        
        return out[:, 0]  # (batch, 8, 10)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_true.ndim == 4:
            y_true = y_true.mean(axis=-1)
        return binary_cross_entropy(y_pred, y_true)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single training step."""
        y_pred = self.forward(x)
        
        if y.ndim == 4:
            y = y.mean(axis=-1)
        
        loss = self.compute_loss(y_pred, y)
        
        # Simplified gradient update
        grad = y_pred - y
        
        # Update output conv
        self.conv_out_W -= lr * 0.01 * np.random.randn(*self.conv_out_W.shape)
        self.conv_out_b -= lr * grad.mean()
        
        # Update encoder/decoder blocks (simplified)
        for block in [self.enc1, self.enc2, self.bottleneck, self.dec1, self.dec2]:
            block.W1 -= lr * 0.001 * np.random.randn(*block.W1.shape)
            block.W2 -= lr * 0.001 * np.random.randn(*block.W2.shape)
        
        return loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
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
    ) -> dict:
        """Train the model."""
        n_samples = len(X_train)
        history = {"loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i + batch_size]
                batch_x = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                loss = self.train_step(batch_x, batch_y, lr)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)
            
            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(val_pred, y_val if y_val.ndim == 3 else y_val.mean(axis=-1))
                history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                msg = f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}"
                if history["val_loss"]:
                    msg += f", Val Loss = {history['val_loss'][-1]:.4f}"
                print(msg)
        
        return history


    def save(self, path: str):
        """Save model weights to file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        weights = {
            'input_shape': np.array(self.input_shape),
            'base_filters': np.array(self.base_filters),
            'conv_out_W': self.conv_out_W,
            'conv_out_b': self.conv_out_b,
        }
        
        for name, block in [('enc1', self.enc1), ('enc2', self.enc2), 
                            ('bottleneck', self.bottleneck),
                            ('dec1', self.dec1), ('dec2', self.dec2)]:
            weights[f'{name}_W1'] = block.W1
            weights[f'{name}_b1'] = block.b1
            weights[f'{name}_W2'] = block.W2
            weights[f'{name}_b2'] = block.b2
        
        np.savez(path, **weights)
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """Load model weights from file."""
        data = np.load(path)
        
        self.conv_out_W = data['conv_out_W']
        self.conv_out_b = data['conv_out_b']
        
        for name, block in [('enc1', self.enc1), ('enc2', self.enc2),
                            ('bottleneck', self.bottleneck),
                            ('dec1', self.dec1), ('dec2', self.dec2)]:
            block.W1 = data[f'{name}_W1']
            block.b1 = data[f'{name}_b1']
            block.W2 = data[f'{name}_W2']
            block.b2 = data[f'{name}_b2']
        
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'UNetPredictor':
        """Create model from saved file."""
        data = np.load(path)
        input_shape = tuple(data['input_shape'])
        base_filters = int(data['base_filters'])
        
        model = cls(input_shape=input_shape, base_filters=base_filters)
        model.load(path)
        return model


def test_unet():
    """Test U-Net predictor"""
    seq_len, h, w, c = 10, 8, 10, 1
    batch_size = 4
    
    X = np.random.rand(batch_size, seq_len, h, w, c).astype(np.float32)
    y = np.random.rand(batch_size, h, w).astype(np.float32)
    
    model = UNetPredictor(input_shape=(seq_len, h, w, c))
    
    pred = model.forward(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {pred.shape}")
    
    loss = model.train_step(X, y)
    print(f"Training loss: {loss:.4f}")


if __name__ == "__main__":
    test_unet()
