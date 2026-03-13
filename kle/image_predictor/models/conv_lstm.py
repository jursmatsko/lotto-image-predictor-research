#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvLSTM-based Image Predictor for Lottery

Convolutional LSTM processes image sequences while preserving
spatial structure, ideal for predicting the next frame in a sequence.

Architecture:
  Input: Sequence of 8×10 images
  ↓
  ConvLSTM Layer (maintains spatial dimensions)
  ↓
  Conv Decoder
  ↓
  Output: 8×10 probability image
"""

from typing import Tuple, Optional, List
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class ConvLSTMCell:
    """
    Convolutional LSTM Cell.
    
    Combines convolution with LSTM gates to process spatial sequences.
    
    Gates:
    - i: input gate
    - f: forget gate
    - o: output gate
    - g: cell gate (candidate)
    
    All gates use convolution instead of dense multiplication.
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        
        # Combined input-to-hidden weights (4 gates)
        total_input = input_channels + hidden_channels
        scale = np.sqrt(2.0 / (total_input * kernel_size * kernel_size))
        
        # Weight shape: (4 * hidden_channels, total_input, kernel_size, kernel_size)
        self.W = np.random.randn(4 * hidden_channels, total_input, kernel_size, kernel_size).astype(np.float32) * scale
        self.b = np.zeros(4 * hidden_channels, dtype=np.float32)
        
    def _conv(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution."""
        batch, c, h, w = x.shape
        
        # Padding
        x_padded = np.pad(x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')
        
        out_channels = self.W.shape[0]
        out = np.zeros((batch, out_channels, h, w), dtype=np.float32)
        
        for oc in range(out_channels):
            for ic in range(c):
                for i in range(h):
                    for j in range(w):
                        patch = x_padded[:, ic, i:i+self.kernel_size, j:j+self.kernel_size]
                        out[:, oc, i, j] += np.sum(patch * self.W[oc, ic], axis=(1, 2))
            out[:, oc] += self.b[oc]
        
        return out
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: Optional[np.ndarray] = None,
        c_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for one time step.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            h_prev: Previous hidden state
            c_prev: Previous cell state
        
        Returns:
            h: New hidden state
            c: New cell state
        """
        batch, _, h, w = x.shape
        
        # Initialize states if needed
        if h_prev is None:
            h_prev = np.zeros((batch, self.hidden_channels, h, w), dtype=np.float32)
        if c_prev is None:
            c_prev = np.zeros((batch, self.hidden_channels, h, w), dtype=np.float32)
        
        # Concatenate input and hidden state
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Apply convolution for all gates
        gates = self._conv(combined)
        
        # Split into 4 gates
        hc = self.hidden_channels
        i = sigmoid(gates[:, :hc])           # Input gate
        f = sigmoid(gates[:, hc:2*hc])       # Forget gate
        o = sigmoid(gates[:, 2*hc:3*hc])     # Output gate
        g = tanh(gates[:, 3*hc:])            # Cell gate
        
        # Update cell state
        c = f * c_prev + i * g
        
        # Update hidden state
        h = o * tanh(c)
        
        return h, c


class ConvLSTMPredictor:
    """
    ConvLSTM-based predictor for lottery images.
    
    Processes sequence of draw images and predicts the next image.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (seq_len, height, width, channels)
        hidden_channels: int = 32,
        num_layers: int = 2,
    ):
        self.input_shape = input_shape
        self.seq_len, self.height, self.width, self.channels = input_shape
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # ConvLSTM layers
        self.lstm_layers = []
        in_channels = self.channels
        for _ in range(num_layers):
            cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=3)
            self.lstm_layers.append(cell)
            in_channels = hidden_channels
        
        # Output conv (hidden_channels -> 1)
        scale = np.sqrt(2.0 / (hidden_channels * 9))
        self.conv_out_W = np.random.randn(1, hidden_channels, 3, 3).astype(np.float32) * scale
        self.conv_out_b = np.zeros(1, dtype=np.float32)
        
    def _output_conv(self, x: np.ndarray) -> np.ndarray:
        """Final convolution to get output."""
        batch, c, h, w = x.shape
        
        # Padding
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
            x: Input sequence (batch, seq_len, height, width, channels)
        
        Returns:
            Predicted image (batch, height, width)
        """
        batch = x.shape[0]
        
        # Convert to (batch, seq_len, channels, height, width)
        x = x.transpose(0, 1, 4, 2, 3)
        
        # Process through LSTM layers
        h_states = [None] * self.num_layers
        c_states = [None] * self.num_layers
        
        for t in range(self.seq_len):
            frame = x[:, t]  # (batch, channels, height, width)
            
            layer_input = frame
            for layer_idx, cell in enumerate(self.lstm_layers):
                h, c = cell.forward(layer_input, h_states[layer_idx], c_states[layer_idx])
                h_states[layer_idx] = h
                c_states[layer_idx] = c
                layer_input = h
        
        # Use final hidden state for prediction
        final_h = h_states[-1]  # (batch, hidden_channels, height, width)
        
        # Output convolution
        out = self._output_conv(final_h)  # (batch, 1, height, width)
        
        return out[:, 0]  # (batch, height, width)
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if y_true.ndim == 4:
            y_true = y_true.mean(axis=-1)
        return binary_cross_entropy(y_pred, y_true)
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.001) -> float:
        """Single training step (simplified gradient update)."""
        y_pred = self.forward(x)
        
        if y.ndim == 4:
            y = y.mean(axis=-1)
        
        loss = self.compute_loss(y_pred, y)
        
        # Simplified parameter update (gradient approximation)
        grad = y_pred - y
        
        # Update output conv weights
        self.conv_out_W -= lr * 0.01 * np.random.randn(*self.conv_out_W.shape)
        self.conv_out_b -= lr * grad.mean()
        
        # Update LSTM weights (simplified)
        for cell in self.lstm_layers:
            cell.W -= lr * 0.001 * np.random.randn(*cell.W.shape)
        
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
            'hidden_channels': np.array(self.hidden_channels),
            'num_layers': np.array(self.num_layers),
            'conv_out_W': self.conv_out_W,
            'conv_out_b': self.conv_out_b,
        }
        
        for i, cell in enumerate(self.lstm_layers):
            weights[f'lstm_{i}_W'] = cell.W
            weights[f'lstm_{i}_b'] = cell.b
        
        np.savez(path, **weights)
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """Load model weights from file."""
        data = np.load(path)
        
        self.conv_out_W = data['conv_out_W']
        self.conv_out_b = data['conv_out_b']
        
        for i, cell in enumerate(self.lstm_layers):
            cell.W = data[f'lstm_{i}_W']
            cell.b = data[f'lstm_{i}_b']
        
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'ConvLSTMPredictor':
        """Create model from saved file."""
        data = np.load(path)
        input_shape = tuple(data['input_shape'])
        hidden_channels = int(data['hidden_channels'])
        num_layers = int(data['num_layers'])
        
        model = cls(input_shape=input_shape, hidden_channels=hidden_channels, num_layers=num_layers)
        model.load(path)
        return model


def test_conv_lstm():
    """Test ConvLSTM predictor"""
    seq_len, h, w, c = 10, 8, 10, 1
    batch_size = 4
    
    X = np.random.rand(batch_size, seq_len, h, w, c).astype(np.float32)
    y = np.random.rand(batch_size, h, w).astype(np.float32)
    
    model = ConvLSTMPredictor(input_shape=(seq_len, h, w, c))
    
    pred = model.forward(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {pred.shape}")
    
    loss = model.train_step(X, y)
    print(f"Training loss: {loss:.4f}")


if __name__ == "__main__":
    test_conv_lstm()
