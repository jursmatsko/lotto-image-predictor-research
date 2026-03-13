#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch-based True Image CNN for Lottery Prediction

Fast GPU-accelerated 2D convolutions that learn spatial patterns
from 8×10 pixel lottery images.

Features:
- True 2D convolutions (not flattened vectors)
- GPU acceleration (if available)
- Spatial pattern learning (horizontal, vertical, diagonal)
- Sequence modeling with Conv3D or stacked Conv2D
python3 kle/image_predictor/main.py train --model pytorch_cnn --load kle/image_predictor/models/saved/pytorch_cnn_v2.pt --epochs 40 --lr 0.0006 --save kle/image_predictor/models/saved/pytorch_cnn_v3.pt
"""

import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class ImageSequenceEncoder(nn.Module):
    """
    Encodes a sequence of 8×10 images using 2D convolutions.
    
    Each frame is processed by shared CNN, then temporal info is aggregated.
    """
    
    def __init__(self, seq_len: int = 15, hidden_channels: int = 32):
        super().__init__()
        
        self.seq_len = seq_len
        
        # Spatial feature extractor (shared across frames)
        # Input: (batch, 1, 8, 10) -> learns 2D spatial patterns
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Temporal aggregation
        # Combine seq_len encoded frames
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * seq_len + 3, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 8, 10) - sequence of binary images
        
        Returns:
            (batch, 8, 10) - probability image for next draw
        """
        batch_size = x.shape[0]
        
        # Process each frame through spatial encoder
        frame_features = []
        for t in range(self.seq_len):
            frame = x[:, t:t+1, :, :]  # (batch, 1, 8, 10)
            feat = self.spatial_encoder(frame)  # (batch, hidden, 8, 10)
            frame_features.append(feat)
        
        # Stack all frame features
        stacked = torch.cat(frame_features, dim=1)  # (batch, hidden*seq_len, 8, 10)
        
        # Add aggregate features
        freq_map = x.mean(dim=1, keepdim=True)  # (batch, 1, 8, 10)
        recent_map = x[:, -1:, :, :]  # (batch, 1, 8, 10)
        gap_map = 1 - recent_map  # (batch, 1, 8, 10)
        
        combined = torch.cat([stacked, freq_map, recent_map, gap_map], dim=1)
        
        # Temporal aggregation
        temporal = self.temporal_conv(combined)
        
        # Decode to output
        out = self.decoder(temporal)
        
        return out.squeeze(1)  # (batch, 8, 10)


class LotteryImageCNN(nn.Module):
    """
    Full model for lottery image prediction.
    
    Architecture:
        Input: (batch, seq_len, 8, 10) - sequence of draw images
        ↓
        2D Conv layers (learn spatial patterns)
        ↓
        Temporal aggregation
        ↓
        Output: (batch, 8, 10) - next draw probability
    """
    
    def __init__(self, seq_len: int = 15, hidden_channels: int = 64):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        
        # Input channels: seq_len frames + 3 features (freq, recent, gap)
        in_channels = seq_len + 3
        
        # Convolutional backbone
        self.backbone = nn.Sequential(
            # Block 1: Learn local patterns
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            
            # Block 2: Higher-level patterns
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10),
            
            # Block 3: Even higher-level
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(),
            
            # Output
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 8, 10)
        
        Returns:
            (batch, 8, 10)
        """
        batch_size = x.shape[0]
        
        # Compute features
        freq_map = x.mean(dim=1, keepdim=True)
        recent_map = x[:, -1:, :, :]
        gap_map = 1 - recent_map
        
        # Combine all inputs
        combined = torch.cat([x, freq_map, recent_map, gap_map], dim=1)
        
        # Forward through backbone
        out = self.backbone(combined)
        
        return out.squeeze(1)


class PyTorchImagePredictor:
    """
    High-level interface for PyTorch image-based prediction.
    """
    
    def __init__(self, seq_len: int = 15, hidden_channels: int = 64, device=None):
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        self.device = device or DEVICE
        
        self.model = LotteryImageCNN(seq_len, hidden_channels).to(self.device)
        self.optimizer = None
        self.criterion = None
        self.blend_weight = 1.0  # 1.0 means pure NN, 0.0 means pure baseline
        self.label_smoothing = 0.02
        self.input_noise_std = 0.01
        
        self.history = {"loss": [], "val_loss": [], "hit_rate": [], "val_hit_rate": []}

    def _normalize_input_shape(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize input tensors to the shape expected by this model.
        X: (batch, seq_len, 8, 10), y: (batch, 8, 10)
        """
        if x.ndim in (4, 5) and x.shape[-1] == 1:
            return np.squeeze(x, axis=-1)
        return x

    def _build_loss(self, y_train_t: torch.Tensor):
        """Build a BCE-with-logits loss with dynamic positive weighting."""
        positive_rate = float(y_train_t.mean().item())
        positive_rate = min(max(positive_rate, 1e-4), 1 - 1e-4)
        pos_weight = (1.0 - positive_rate) / positive_rate
        pos_weight_t = torch.tensor([pos_weight], device=self.device, dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    
    def _compute_weighted_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """
        Baseline prediction from time-weighted frequency + recency feature.
        x shape: (batch, seq_len, 8, 10)
        """
        seq_len = x.shape[1]
        weights = torch.linspace(1.0, 2.5, steps=seq_len, device=x.device, dtype=x.dtype)
        weights = weights / weights.sum()
        weighted_freq = (x * weights.view(1, seq_len, 1, 1)).sum(dim=1)
        recent = x[:, -1, :, :]
        momentum = (x[:, -3:, :, :].mean(dim=1) if seq_len >= 3 else recent)
        baseline = 0.65 * weighted_freq + 0.25 * momentum + 0.10 * recent
        return torch.clamp(baseline, 0.0, 1.0)
    
    def _fuse_with_baseline(self, model_prob: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        baseline = self._compute_weighted_baseline(x_input)
        fused = self.blend_weight * model_prob + (1.0 - self.blend_weight) * baseline
        return torch.clamp(fused, 0.0, 1.0)
    
    def _select_best_blend_weight(self, model_prob: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor):
        """
        Tune fusion weight on validation set by maximizing hit rate.
        """
        candidates = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
        baseline = self._compute_weighted_baseline(x_val)
        best_w = self.blend_weight
        best_hits = -1.0
        for w in candidates:
            fused = torch.clamp(w * model_prob + (1.0 - w) * baseline, 0.0, 1.0)
            hits = self.compute_hit_rate(fused, y_val)
            if hits > best_hits:
                best_hits = hits
                best_w = w
        self.blend_weight = best_w
        return best_w, best_hits
    
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = self._normalize_input_shape(x)
            return torch.FloatTensor(x).to(self.device)
        return x.to(self.device)
    
    def compute_hit_rate(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute average hits (top 20 predictions vs actual 20)."""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        hits = []
        for i in range(len(pred_np)):
            pred_flat = pred_np[i].flatten()
            target_flat = target_np[i].flatten()
            
            pred_top20 = set(np.argsort(pred_flat)[-20:])
            actual_top20 = set(np.where(target_flat > 0.5)[0])
            
            hits.append(len(pred_top20 & actual_top20))
        
        return np.mean(hits)
    
    def _compute_hits_per_sample(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """Compute per-sample hit counts (top20 vs actual20)."""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        hits = []
        for i in range(len(pred_np)):
            pred_top20 = set(np.argsort(pred_np[i].flatten())[-20:])
            actual_top20 = set(np.where(target_np[i].flatten() > 0.5)[0])
            hits.append(len(pred_top20 & actual_top20))
        return np.array(hits, dtype=np.float32)
    
    def compute_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute BCE loss on numpy arrays for app-level evaluation."""
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        return float(-np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred)))
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 20,
        grad_clip: float = 1.0,
        use_time_decay: bool = True,
        time_decay: float = 2.0,
        walk_forward_last: int = 0,
        walk_forward_every: int = 1,
        walk_forward_print_samples: int = 5,
        verbose: bool = True,
    ):
        """Train the model."""
        
        # Prepare data
        X_train_t = self._to_tensor(X_train)
        y_train_t = self._to_tensor(y_train)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        sampler = None
        if use_time_decay and len(train_dataset) > 1:
            time_weights = np.exp(np.linspace(0.0, time_decay, len(train_dataset)))
            time_weights = torch.as_tensor(time_weights, dtype=torch.float32)
            sampler = WeightedRandomSampler(
                weights=time_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )
        
        if X_val is not None:
            X_val_t = self._to_tensor(X_val)
            y_val_t = self._to_tensor(y_val)
        
        # Loss + optimizer
        self._build_loss(y_train_t)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1))
        
        # Training
        self.history = {"loss": [], "val_loss": [], "hit_rate": [], "val_hit_rate": [], "wf_hit_rate": []}
        start_time = time.time()
        
        if verbose:
            print("\n" + "=" * 80)
            print("🔥 PYTORCH IMAGE CNN TRAINING")
            print("=" * 80)
            print(f"  Device:          {self.device}")
            print(f"  Input Shape:     (batch, {self.seq_len}, 8, 10) images")
            print(f"  Training:        {len(X_train)} samples")
            print(f"  Validation:      {len(X_val) if X_val is not None else 0} samples")
            print(f"  Batch Size:      {batch_size}")
            print(f"  Epochs:          {epochs}")
            print(f"  Learning Rate:   {lr}")
            print(f"  Weight Decay:    {weight_decay}")
            print(f"  Early Stop Pat.: {early_stopping_patience}")
            print(f"  Grad Clip:       {grad_clip}")
            print(f"  Parameters:      {sum(p.numel() for p in self.model.parameters()):,}")
            print("=" * 80)
            print(f"{'Epoch':>6} │ {'Loss':>10} │ {'Val Loss':>10} │ {'Hits':>6} │ {'Val Hits':>8} │ {'Time':>8}")
            print("-" * 80)
        
        best_val_hit = -1.0
        best_state = None
        no_improve_epochs = 0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            epoch_hits = 0.0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                train_x = batch_x
                if self.input_noise_std > 0:
                    noise = torch.randn_like(train_x) * self.input_noise_std
                    train_x = torch.clamp(train_x + noise, 0.0, 1.0)
                
                logits = self.model(train_x)
                smooth = self.label_smoothing
                if smooth > 0:
                    batch_y_smooth = batch_y * (1.0 - smooth) + 0.5 * smooth
                else:
                    batch_y_smooth = batch_y
                loss = self.criterion(logits, batch_y_smooth)
                
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
                
                pred = torch.sigmoid(logits)
                pred_fused = self._fuse_with_baseline(pred, batch_x)
                epoch_loss += loss.item()
                epoch_hits += self.compute_hit_rate(pred_fused, batch_y)
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_hits = epoch_hits / n_batches
            self.history["loss"].append(avg_loss)
            self.history["hit_rate"].append(avg_hits)
            
            # Validation
            val_loss_str = "   N/A   "
            val_hits_str = "  N/A   "
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = self.criterion(val_logits, y_val_t).item()
                    val_pred = torch.sigmoid(val_logits)
                    self._select_best_blend_weight(val_pred, X_val_t, y_val_t)
                    val_pred_fused = self._fuse_with_baseline(val_pred, X_val_t)
                    val_hits = self.compute_hit_rate(val_pred_fused, y_val_t)
                    
                    self.history["val_loss"].append(val_loss)
                    self.history["val_hit_rate"].append(val_hits)
                    val_loss_str = f"{val_loss:10.6f}"
                    val_hits_str = f"{val_hits:6.2f}/20"
                    
                    # Walk-forward style stats on latest validation window
                    if walk_forward_last > 0 and ((epoch + 1) % max(1, walk_forward_every) == 0):
                        wf_n = min(walk_forward_last, len(y_val_t))
                        wf_pred = val_pred_fused[-wf_n:]
                        wf_true = y_val_t[-wf_n:]
                        wf_hits_arr = self._compute_hits_per_sample(wf_pred, wf_true)
                        wf_mean = float(wf_hits_arr.mean())
                        wf_min = int(wf_hits_arr.min())
                        wf_max = int(wf_hits_arr.max())
                        wf_median = float(np.median(wf_hits_arr))
                        wf_p90 = float(np.percentile(wf_hits_arr, 90))
                        self.history["wf_hit_rate"].append(wf_mean)
                        if verbose:
                            sample_n = min(max(1, walk_forward_print_samples), len(wf_hits_arr))
                            sample_hits = ", ".join(str(int(x)) for x in wf_hits_arr[-sample_n:])
                            print(
                                f"      ↳ WF(last {wf_n}) avg={wf_mean:.2f}/20 "
                                f"min={wf_min} max={wf_max} median={wf_median:.1f} p90={wf_p90:.1f} "
                                f"| recent_hits=[{sample_hits}]",
                                flush=True,
                            )
                    
                    if val_hits > best_val_hit:
                        best_val_hit = val_hits
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
            
            epoch_time = time.time() - epoch_start
            scheduler.step()
            
            if verbose:
                print(f"\r{epoch+1:>6} │ {avg_loss:10.6f} │ {val_loss_str} │ {avg_hits:4.1f}/20 │ {val_hits_str} │ {epoch_time*1000:6.0f}ms", flush=True)
            
            if X_val is not None and early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
                if verbose:
                    print(f"\n⏹️ Early stopping at epoch {epoch + 1} (no val hit improvement for {early_stopping_patience} epochs)")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        total_time = time.time() - start_time
        
        if verbose:
            trained_epochs = max(1, len(self.history["loss"]))
            print("-" * 80)
            print(f"\n📊 TRAINING COMPLETE")
            print(f"   Total Time:     {total_time:.2f}s ({total_time/trained_epochs*1000:.0f}ms/epoch)")
            print(f"   Final Loss:     {self.history['loss'][-1]:.6f}")
            print(f"   Final Hits:     {self.history['hit_rate'][-1]:.2f}/20")
            print(f"   Blend Weight:   {self.blend_weight:.2f} (NN ratio)")
            if self.history["val_hit_rate"]:
                print(f"   Val Hits:       {self.history['val_hit_rate'][-1]:.2f}/20")
                best_val = max(self.history['val_hit_rate'])
                print(f"   Best Val Hits:  {best_val:.2f}/20")
            if self.history["wf_hit_rate"]:
                print(f"   WF Hits:        {self.history['wf_hit_rate'][-1]:.2f}/20")
                print(f"   Best WF Hits:   {max(self.history['wf_hit_rate']):.2f}/20")
            print(f"   Random Base:    5.0/20")
            print("=" * 80)
        
        return self.history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict next draw."""
        self.model.eval()
        with torch.no_grad():
            x_t = self._to_tensor(x)
            logits = self.model(x_t)
            pred = torch.sigmoid(logits)
            pred = self._fuse_with_baseline(pred, x_t)
            return pred.cpu().numpy()
    
    def save(self, path: str, save_optimizer: bool = True):
        """Save model with optimizer state for continue training."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'hidden_channels': self.hidden_channels,
            'history': self.history,
            'epoch': len(self.history.get('loss', [])),
            'blend_weight': float(self.blend_weight),
            'label_smoothing': float(self.label_smoothing),
            'input_noise_std': float(self.input_noise_std),
        }
        if save_optimizer and self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"✅ Model saved to: {path}")
        print(f"   Epochs trained: {checkpoint['epoch']}")
    
    def load(self, path: str, load_optimizer: bool = True):
        """Load model (optionally with optimizer for continue training)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {"loss": [], "val_loss": [], "hit_rate": [], "val_hit_rate": []})
        self.blend_weight = float(checkpoint.get('blend_weight', 1.0))
        self.label_smoothing = float(checkpoint.get('label_smoothing', 0.02))
        self.input_noise_std = float(checkpoint.get('input_noise_std', 0.01))
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', len(self.history.get('loss', [])))
        print(f"✅ Model loaded from: {path}")
        print(f"   Previously trained: {epoch} epochs")
        return epoch
    
    @classmethod
    def from_file(cls, path: str) -> 'PyTorchImagePredictor':
        """Load model from file."""
        checkpoint = torch.load(path, map_location=DEVICE)
        model = cls(
            seq_len=checkpoint['seq_len'],
            hidden_channels=checkpoint['hidden_channels']
        )
        model.load(path)
        return model
    
    def continue_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        additional_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True,
    ):
        """Continue training from saved checkpoint."""
        previous_epochs = len(self.history.get('loss', []))
        
        if verbose:
            print(f"\n🔄 CONTINUE TRAINING")
            print(f"   Previous epochs: {previous_epochs}")
            print(f"   Additional epochs: {additional_epochs}")
            print(f"   Total will be: {previous_epochs + additional_epochs}")
        
        return self.fit(
            X_train, y_train, X_val, y_val,
            epochs=additional_epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose,
        )


def main():
    """Main training script with CLI arguments."""
    import argparse
    from image_predictor.utils.data_loader import DataLoader as LotteryDataLoader
    from image_predictor.utils.image_encoder import ImageEncoder
    
    parser = argparse.ArgumentParser(description="PyTorch Image CNN for Lottery Prediction")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden channels")
    parser.add_argument("--load", type=str, default=None, help="Load model from path")
    parser.add_argument("--save", type=str, default="image_predictor/models/pytorch_cnn.pt", help="Save model to path")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from loaded model")
    parser.add_argument("--predict-only", action="store_true", help="Only predict, no training")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔥 PYTORCH TRUE IMAGE CNN PREDICTION")
    print("   (Fast GPU-accelerated 2D convolutions)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Load data
    data_path = os.path.join(_ROOT, "data", "data.csv")
    loader = LotteryDataLoader(data_path, sequence_length=15)
    loader.load_data()
    
    print(f"\n📊 Loaded {len(loader.draws)} draws")
    
    # Create images
    images = loader.create_images()
    print(f"📊 Image shape: {images.shape}")
    
    # Use recent data
    n_use = min(800, len(images))
    images = images[-n_use:]
    
    # Create sequences
    seq_len = 15
    X, y = [], []
    for i in range(seq_len, len(images)):
        X.append(images[i - seq_len:i])
        y.append(images[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"📊 Sequences: X={X.shape}, y={y.shape}")
    
    # Split
    val_size = int(len(X) * 0.2)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    encoder = ImageEncoder()
    all_images = loader.create_images()
    latest = all_images[-seq_len:][np.newaxis, ...].astype(np.float32)
    
    # Load or create model
    if args.load and os.path.exists(args.load):
        print(f"\n📂 Loading model from: {args.load}")
        model = PyTorchImagePredictor.from_file(args.load)
        
        if args.continue_training and not args.predict_only:
            print(f"\n🔄 CONTINUE TRAINING for {args.epochs} more epochs...")
            model.continue_training(
                X_train, y_train, X_val, y_val,
                additional_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            model.save(args.save)
    else:
        if args.predict_only:
            print("❌ No model to load for prediction!")
            return
        
        # Train new model
        model = PyTorchImagePredictor(seq_len=seq_len, hidden_channels=args.hidden)
        model.fit(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        model.save(args.save)
    
    # Predict
    print("\n🔮 Predicting Next Draw...")
    pred = model.predict(latest)[0]
    
    top20 = encoder.decode_single(pred, 20)
    
    print("\n🎯 Top 20 Predicted Numbers:")
    print(" ".join(f"{n:02d}" for n in top20))
    
    # Show prediction as image
    print("\n📊 Predicted Image (8×10 pixels):")
    print("     " + " ".join(f"{i+1:2d}" for i in range(10)))
    for row in range(8):
        line = f"{row*10+1:2d}-{row*10+10:2d} "
        for col in range(10):
            val = pred[row, col]
            if val > 0.35:
                line += "██"
            elif val > 0.28:
                line += "▓▓"
            elif val > 0.24:
                line += "▒▒"
            else:
                line += "··"
        print(line)
    
    # Quick backtest
    print("\n📈 Quick Backtest (Last 10 draws):")
    hits_list = []
    for test_i in range(-10, 0):
        test_idx = len(all_images) + test_i
        seq = all_images[test_idx - seq_len:test_idx][np.newaxis, ...].astype(np.float32)
        p = model.predict(seq)[0]
        
        pred_nums = set(encoder.decode_single(p, 20))
        actual_nums = set(encoder.decode_single(all_images[test_idx], 20))
        
        hits = len(pred_nums & actual_nums)
        hits_list.append(hits)
        
        issue = loader.issues[test_idx]
        print(f"  Draw {issue}: {hits}/20 hits")
    
    print(f"\n📊 Average: {np.mean(hits_list):.2f}/20 (Random: 5/20)")
    
    print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
    print("=" * 70)


if __name__ == "__main__":
    main()
