#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional Diffusion Model for Lottery Image Prediction

Diffusion models learn to generate images by:
1. Forward process: gradually add noise to images
2. Reverse process: learn to denoise step by step

For lottery prediction:
- Condition on sequence of previous draws
- Generate the "next" draw image from noise

Based on DDPM (Denoising Diffusion Probabilistic Models)
"""

import os
import sys
import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embeddings (from Transformer positional encoding).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalUNet(nn.Module):
    """
    U-Net for denoising, conditioned on:
    - Timestep t (how much noise)
    - Context (sequence of previous draws)
    
    Architecture:
        Encoder → Bottleneck → Decoder with skip connections
    """
    
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=64, context_dim=64, base_channels=64):
        super().__init__()
        
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Context encoder (process sequence of previous draws)
        self.context_encoder = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, padding=1),  # 15 previous frames
            nn.SiLU(),
            nn.Conv2d(32, context_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Encoder
        self.enc1 = self._conv_block(in_channels + context_dim, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        
        # Bottleneck with time conditioning
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )
        self.time_proj = nn.Linear(time_emb_dim, base_channels * 4)
        
        # Decoder
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )
    
    def forward(self, x, t, context):
        """
        Args:
            x: (batch, 1, 8, 10) - noisy image
            t: (batch,) - timestep
            context: (batch, 15, 8, 10) - previous draws
        
        Returns:
            (batch, 1, 8, 10) - predicted noise
        """
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        
        # Context encoding
        ctx = self.context_encoder(context)  # (batch, context_dim, 8, 10)
        
        # Concatenate input with context
        x_ctx = torch.cat([x, ctx], dim=1)  # (batch, 1+context_dim, 8, 10)
        
        # Encoder
        e1 = self.enc1(x_ctx)  # (batch, base, 8, 10)
        e2 = self.enc2(e1)     # (batch, base*2, 8, 10)
        
        # Bottleneck with time conditioning
        b = self.bottleneck(e2)  # (batch, base*4, 8, 10)
        t_proj = self.time_proj(t_emb)[:, :, None, None]
        b = b + t_proj  # Add time information
        
        # Decoder with skip connections
        d2 = self.dec2(torch.cat([b, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return self.out(d1)


class DiffusionScheduler:
    """
    Noise schedule for diffusion process.
    
    Forward: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """
    
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device=DEVICE):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
        # For reverse process
        self.posterior_variance = self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
    
    def add_noise(self, x_0, t, noise=None):
        """Add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def sample_timesteps(self, batch_size):
        """Sample random timesteps."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)


class LotteryDiffusionModel:
    """
    Complete Diffusion Model for lottery prediction.
    
    Workflow:
    1. Train: Learn to denoise lottery images conditioned on history
    2. Predict: Start from noise, iteratively denoise to generate prediction
    """
    
    def __init__(self, seq_len=15, num_timesteps=500, device=None):
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps
        self.device = device or DEVICE
        
        self.model = ConditionalUNet(
            in_channels=1,
            out_channels=1,
            time_emb_dim=64,
            context_dim=64,
            base_channels=64,
        ).to(self.device)
        
        self.scheduler = DiffusionScheduler(num_timesteps, device=self.device)
        self.optimizer = None
        self.history = {"loss": [], "val_loss": []}
    
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.FloatTensor(x).to(self.device)
        return x.to(self.device)
    
    def fit(
        self,
        X_train: np.ndarray,  # (N, seq_len, 8, 10)
        y_train: np.ndarray,  # (N, 8, 10)
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ):
        """Train the diffusion model."""
        
        # Prepare data
        X_train_t = self._to_tensor(X_train)
        y_train_t = self._to_tensor(y_train).unsqueeze(1)  # Add channel dim
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_t = self._to_tensor(X_val)
            y_val_t = self._to_tensor(y_val).unsqueeze(1)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        if verbose:
            print("\n" + "=" * 80)
            print("🌀 DIFFUSION MODEL TRAINING")
            print("=" * 80)
            print(f"  Device:          {self.device}")
            print(f"  Timesteps:       {self.num_timesteps}")
            print(f"  Training:        {len(X_train)} samples")
            print(f"  Validation:      {len(X_val) if X_val is not None else 0} samples")
            print(f"  Epochs:          {epochs}")
            print(f"  Parameters:      {sum(p.numel() for p in self.model.parameters()):,}")
            print("=" * 80)
            print(f"{'Epoch':>6} │ {'Loss':>12} │ {'Val Loss':>12} │ {'Time':>10}")
            print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Sample timesteps
                t = self.scheduler.sample_timesteps(batch_y.shape[0])
                
                # Add noise
                x_t, noise = self.scheduler.add_noise(batch_y, t)
                
                # Predict noise
                pred_noise = self.model(x_t, t, batch_x)
                
                # Loss
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.history["loss"].append(avg_loss)
            
            # Validation
            val_loss_str = "    N/A     "
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    t = self.scheduler.sample_timesteps(y_val_t.shape[0])
                    x_t, noise = self.scheduler.add_noise(y_val_t, t)
                    pred_noise = self.model(x_t, t, X_val_t)
                    val_loss = F.mse_loss(pred_noise, noise).item()
                    self.history["val_loss"].append(val_loss)
                    val_loss_str = f"{val_loss:12.6f}"
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"\r{epoch+1:>6} │ {avg_loss:12.6f} │ {val_loss_str} │ {epoch_time*1000:8.0f}ms", flush=True)
        
        total_time = time.time() - start_time
        
        if verbose:
            print("-" * 80)
            print(f"\n📊 TRAINING COMPLETE")
            print(f"   Total Time:     {total_time:.2f}s")
            print(f"   Final Loss:     {self.history['loss'][-1]:.6f}")
            print("=" * 80)
        
        return self.history
    
    @torch.no_grad()
    def sample(self, context: np.ndarray, num_inference_steps: int = 50) -> np.ndarray:
        """
        Generate prediction using reverse diffusion process.
        
        Args:
            context: (batch, seq_len, 8, 10) - previous draws
            num_inference_steps: number of denoising steps (fewer = faster)
        
        Returns:
            (batch, 8, 10) - generated prediction
        """
        self.model.eval()
        
        context_t = self._to_tensor(context)
        batch_size = context_t.shape[0]
        
        # Start from pure noise
        x = torch.randn(batch_size, 1, 8, 10, device=self.device)
        
        # Subsample timesteps for faster inference
        step_size = self.num_timesteps // num_inference_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]
        
        # Reverse diffusion
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            pred_noise = self.model(x, t_tensor, context_t)
            
            # Get alpha values
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            alpha_bar_prev = self.scheduler.alpha_bars_prev[t] if t > 0 else torch.tensor(1.0)
            
            # Compute x_{t-1}
            beta = self.scheduler.betas[t]
            
            # Mean
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1, 1)  # Clamp for stability
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            
            # x_{t-1}
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(x) * 0.1  # Small noise
                x = x + noise
        
        # Convert to probability (sigmoid)
        x = torch.sigmoid(x * 5)  # Scale for sharper output
        
        return x.squeeze(1).cpu().numpy()
    
    def predict(self, context: np.ndarray, num_inference_steps: int = 50) -> np.ndarray:
        """Predict next draw."""
        return self.sample(context, num_inference_steps)
    
    def compute_hit_rate(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute average hits."""
        hits = []
        for i in range(len(pred)):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            
            pred_top20 = set(np.argsort(pred_flat)[-20:])
            actual_top20 = set(np.where(target_flat > 0.5)[0])
            
            hits.append(len(pred_top20 & actual_top20))
        return np.mean(hits)
    
    def save(self, path: str, save_optimizer: bool = True):
        """Save model with optimizer state for continue training."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.seq_len,
            'num_timesteps': self.num_timesteps,
            'history': self.history,
            'epoch': len(self.history.get('loss', [])),
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
        self.history = checkpoint.get('history', {"loss": [], "val_loss": []})
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', len(self.history.get('loss', [])))
        print(f"✅ Model loaded from: {path}")
        print(f"   Previously trained: {epoch} epochs")
        return epoch
    
    @classmethod
    def from_file(cls, path: str) -> 'LotteryDiffusionModel':
        """Load from file."""
        checkpoint = torch.load(path, map_location=DEVICE)
        model = cls(
            seq_len=checkpoint['seq_len'],
            num_timesteps=checkpoint['num_timesteps'],
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
        lr: float = 1e-3,
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


def visualize_diffusion_process(model, context, steps=[0, 10, 25, 50]):
    """Visualize the denoising process."""
    print("\n🌀 Diffusion Process Visualization:")
    print("   (From noise → prediction)")
    
    model.model.eval()
    context_t = model._to_tensor(context)
    
    # Start from noise
    x = torch.randn(1, 1, 8, 10, device=model.device)
    
    step_size = model.num_timesteps // 50
    timesteps = list(range(0, model.num_timesteps, step_size))[::-1]
    
    images = {}
    
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i in steps:
                img = torch.sigmoid(x * 5).squeeze().cpu().numpy()
                images[i] = img
            
            t_tensor = torch.full((1,), t, device=model.device, dtype=torch.long)
            pred_noise = model.model(x, t_tensor, context_t)
            
            alpha_bar = model.scheduler.alpha_bars[t]
            alpha_bar_prev = model.scheduler.alpha_bars_prev[t] if t > 0 else torch.tensor(1.0)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_noise
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
            
            if t > 0:
                x = x + torch.randn_like(x) * 0.1
    
    # Final image
    images[50] = torch.sigmoid(x * 5).squeeze().cpu().numpy()
    
    # Print
    for step in steps + [50]:
        if step in images:
            print(f"\n   Step {step}:")
            img = images[step]
            for row in range(8):
                line = "   "
                for col in range(10):
                    val = img[row, col]
                    if val > 0.4:
                        line += "██"
                    elif val > 0.3:
                        line += "▓▓"
                    elif val > 0.2:
                        line += "▒▒"
                    else:
                        line += "··"
                print(line)


def main():
    """Main training script with CLI arguments."""
    import argparse
    from image_predictor.utils.data_loader import DataLoader as LotteryDataLoader
    from image_predictor.utils.image_encoder import ImageEncoder
    
    parser = argparse.ArgumentParser(description="Diffusion Model for Lottery Prediction")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--timesteps", type=int, default=500, help="Diffusion timesteps")
    parser.add_argument("--load", type=str, default=None, help="Load model from path")
    parser.add_argument("--save", type=str, default="image_predictor/models/diffusion.pt", help="Save model to path")
    parser.add_argument("--continue-training", action="store_true", help="Continue training from loaded model")
    parser.add_argument("--predict-only", action="store_true", help="Only predict, no training")
    parser.add_argument("--num-tickets", type=int, default=5, help="Number of diverse predictions")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌀 DIFFUSION MODEL FOR LOTTERY PREDICTION")
    print("   (Denoising Diffusion Probabilistic Model)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Load data
    data_path = os.path.join(_ROOT, "data", "data.csv")
    loader = LotteryDataLoader(data_path, sequence_length=15)
    loader.load_data()
    
    print(f"\n📊 Loaded {len(loader.draws)} draws")
    
    # Create images (on-the-fly from CSV!)
    print("\n🖼️ Converting draws to 8×10 images (on-the-fly, no pre-generation needed):")
    print("   CSV row → 20 numbers → 80-bit binary → 8×10 matrix")
    
    images = loader.create_images()
    print(f"   Result: {images.shape} images in memory")
    
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
    
    # Split
    val_size = int(len(X) * 0.2)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    print(f"\n📊 Dataset:")
    print(f"   Training:   {len(X_train)} sequences")
    print(f"   Validation: {len(X_val)} sequences")
    
    encoder = ImageEncoder()
    all_images = loader.create_images()
    latest = all_images[-seq_len:][np.newaxis, ...].astype(np.float32)
    
    # Load or create model
    if args.load and os.path.exists(args.load):
        print(f"\n📂 Loading model from: {args.load}")
        model = LotteryDiffusionModel.from_file(args.load)
        
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
        model = LotteryDiffusionModel(seq_len=seq_len, num_timesteps=args.timesteps)
        model.fit(
            X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        model.save(args.save)
    
    # Visualize diffusion process
    if not args.predict_only:
        visualize_diffusion_process(model, latest)
    
    # Predict
    print("\n🔮 Final Prediction:")
    pred = model.predict(latest, num_inference_steps=50)[0]
    
    top20 = encoder.decode_single(pred, 20)
    
    print("\n🎯 Top 20 Predicted Numbers:")
    print(" ".join(f"{n:02d}" for n in top20))
    
    # Evaluate
    print("\n📈 Backtest (Last 10 draws):")
    hits_list = []
    for test_i in range(-10, 0):
        test_idx = len(all_images) + test_i
        seq = all_images[test_idx - seq_len:test_idx][np.newaxis, ...].astype(np.float32)
        p = model.predict(seq, num_inference_steps=50)[0]
        
        pred_nums = set(encoder.decode_single(p, 20))
        actual_nums = set(encoder.decode_single(all_images[test_idx], 20))
        
        hits = len(pred_nums & actual_nums)
        hits_list.append(hits)
        
        issue = loader.issues[test_idx]
        print(f"  Draw {issue}: {hits}/20 hits")
    
    print(f"\n📊 Average: {np.mean(hits_list):.2f}/20 (Random: 5/20)")
    
    # Generate multiple predictions (diffusion can generate diverse samples!)
    print(f"\n🎲 Diffusion Advantage: Generate {args.num_tickets} Diverse Predictions:")
    for i in range(args.num_tickets):
        p = model.predict(latest, num_inference_steps=50)[0]
        nums = encoder.decode_single(p, 20)
        print(f"  Ticket {i+1}: {' '.join(f'{n:02d}' for n in nums)}")
    
    print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
    print("=" * 70)


if __name__ == "__main__":
    main()
