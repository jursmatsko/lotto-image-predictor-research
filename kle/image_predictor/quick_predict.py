#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Image-Based Prediction (Optimized for Speed)

Fast version using simplified model for quick testing.
With detailed training metrics and progress display.
"""

import os
import sys
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from image_predictor.utils.image_encoder import ImageEncoder
from image_predictor.utils.data_loader import DataLoader


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def progress_bar(current, total, width=40, prefix="", suffix=""):
    """Generate a text progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}|{bar}| {current}/{total} ({percent*100:.1f}%) {suffix}"


class TrainingMetrics:
    """Track and display training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.hit_rates = []
        self.epoch_times = []
        self.start_time = None
        self.best_loss = float('inf')
        self.best_hit_rate = 0.0
        self.best_epoch = 0
    
    def start_training(self):
        self.start_time = time.time()
    
    def record_epoch(self, epoch, train_loss, val_loss=None, hit_rate=None, epoch_time=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if hit_rate is not None:
            self.hit_rates.append(hit_rate)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        
        # Track best
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self.best_epoch = epoch
        if hit_rate is not None and hit_rate > self.best_hit_rate:
            self.best_hit_rate = hit_rate
    
    def get_summary(self):
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            "total_epochs": len(self.train_losses),
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "final_hit_rate": self.hit_rates[-1] if self.hit_rates else None,
            "best_loss": self.best_loss,
            "best_hit_rate": self.best_hit_rate,
            "best_epoch": self.best_epoch,
            "total_time": total_time,
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
        }
    
    def print_summary(self):
        s = self.get_summary()
        print("\n" + "=" * 60)
        print("📊 TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Total Epochs:      {s['total_epochs']}")
        print(f"  Total Time:        {s['total_time']:.2f}s")
        print(f"  Avg Epoch Time:    {s['avg_epoch_time']*1000:.1f}ms")
        print("-" * 60)
        print(f"  Final Train Loss:  {s['final_train_loss']:.6f}")
        if s['final_val_loss']:
            print(f"  Final Val Loss:    {s['final_val_loss']:.6f}")
        if s['final_hit_rate']:
            print(f"  Final Hit Rate:    {s['final_hit_rate']:.2f}/20")
        print("-" * 60)
        print(f"  Best Loss:         {s['best_loss']:.6f} (Epoch {s['best_epoch']+1})")
        if s['best_hit_rate'] > 0:
            print(f"  Best Hit Rate:     {s['best_hit_rate']:.2f}/20")
        print("=" * 60)
    
    def plot_curves(self, save_path: str = None):
        """Plot training curves to ASCII or file."""
        if not self.train_losses:
            print("No training data to plot.")
            return
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # ASCII plot
        print("\n📈 TRAINING CURVES (ASCII)")
        print("-" * 60)
        
        # Loss curve
        print("Loss:")
        max_loss = max(self.train_losses)
        min_loss = min(self.train_losses)
        height = 10
        width = min(50, len(self.train_losses))
        
        for row in range(height):
            threshold = max_loss - (max_loss - min_loss) * row / (height - 1)
            line = ""
            step = max(1, len(self.train_losses) // width)
            for i in range(0, len(self.train_losses), step):
                if self.train_losses[i] >= threshold:
                    line += "█"
                elif self.val_losses and i < len(self.val_losses) and self.val_losses[i] >= threshold:
                    line += "▒"
                else:
                    line += " "
            if row == 0:
                print(f"  {max_loss:.4f} |{line}")
            elif row == height - 1:
                print(f"  {min_loss:.4f} |{line}")
            else:
                print(f"         |{line}")
        
        print("          " + "-" * width)
        print(f"          Epoch 1 -> {len(self.train_losses)}")
        print("          █ Train Loss  ▒ Val Loss")
        
        # Hit rate curve
        if self.hit_rates:
            print("\nHit Rate:")
            max_hr = max(self.hit_rates)
            min_hr = min(self.hit_rates)
            if max_hr > min_hr:
                for row in range(height):
                    threshold = max_hr - (max_hr - min_hr) * row / (height - 1)
                    line = ""
                    step = max(1, len(self.hit_rates) // width)
                    for i in range(0, len(self.hit_rates), step):
                        if self.hit_rates[i] >= threshold:
                            line += "●"
                        else:
                            line += " "
                    if row == 0:
                        print(f"  {max_hr:.1f}/20 |{line}")
                    elif row == height - 1:
                        print(f"  {min_hr:.1f}/20 |{line}")
                    else:
                        print(f"         |{line}")
                print("          " + "-" * width)
                print(f"          Epoch 1 -> {len(self.hit_rates)}")
        
        # Save to PNG if matplotlib available
        if save_path:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Loss plot
                axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
                if self.val_losses:
                    axes[0].plot(epochs[:len(self.val_losses)], self.val_losses, 'r--', label='Val Loss', linewidth=2)
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Hit rate plot
                if self.hit_rates:
                    axes[1].plot(epochs[:len(self.hit_rates)], self.hit_rates, 'g-', label='Hit Rate', linewidth=2)
                    axes[1].axhline(y=5.0, color='gray', linestyle='--', label='Random (5/20)')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Hits/20')
                    axes[1].set_title('Hit Rate')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\n📊 Training curves saved to: {save_path}")
            except ImportError:
                print("\n⚠️ matplotlib not available for PNG export")


class QuickImagePredictor:
    """
    Fast image-based predictor using simple MLP.
    """
    
    def __init__(self, seq_len: int = 15, hidden_dim: int = 128):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.encoder = ImageEncoder()
        
        # Simple MLP weights
        input_dim = seq_len * 80
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, 80).astype(np.float32) * 0.01
        self.b3 = np.zeros(80, dtype=np.float32)
        
        self.metrics = TrainingMetrics()
    
    def forward(self, X):
        """X: (batch, seq_len, 8, 10)"""
        batch = X.shape[0]
        X_flat = X.reshape(batch, -1)
        
        h1 = np.maximum(0, X_flat @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        out = sigmoid(h2 @ self.W3 + self.b3)
        
        return out.reshape(batch, 8, 10)
    
    def compute_loss(self, pred, y):
        """Compute binary cross-entropy loss."""
        pred_flat = pred.reshape(pred.shape[0], -1)
        y_flat = y.reshape(y.shape[0], -1)
        eps = 1e-7
        loss = -np.mean(y_flat * np.log(pred_flat + eps) + (1 - y_flat) * np.log(1 - pred_flat + eps))
        return loss
    
    def compute_hit_rate(self, pred, y):
        """Compute average hit rate (predicted top 20 vs actual top 20)."""
        hits = []
        for i in range(len(pred)):
            pred_nums = set(self.encoder.decode_single(pred[i], 20))
            actual_nums = set(self.encoder.decode_single(y[i], 20))
            hits.append(len(pred_nums & actual_nums))
        return np.mean(hits)
    
    def train_step(self, X, y, lr=0.001):
        pred = self.forward(X)
        y_flat = y.reshape(y.shape[0], -1)
        pred_flat = pred.reshape(pred.shape[0], -1)
        
        loss = self.compute_loss(pred, y)
        
        # Simplified gradient update
        grad = (pred_flat - y_flat) / X.shape[0]
        self.W3 -= lr * np.outer(np.ones(self.hidden_dim), grad.mean(axis=0))
        
        return loss, pred
    
    def save(self, path: str):
        """Save model weights to file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        np.savez(path,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
        )
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """Load model weights from file."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'QuickImagePredictor':
        """Create model from saved file."""
        data = np.load(path)
        model = cls(
            seq_len=int(data['seq_len']),
            hidden_dim=int(data['hidden_dim'])
        )
        model.load(path)
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=50, lr=0.001, verbose=True):
        """
        Train the model with detailed progress display.
        
        Args:
            X: Training input (batch, seq_len, 8, 10)
            y: Training target (batch, 8, 10)
            X_val: Validation input (optional)
            y_val: Validation target (optional)
            epochs: Number of training epochs
            lr: Learning rate
            verbose: Whether to print progress
        """
        self.metrics = TrainingMetrics()
        self.metrics.start_training()
        
        n_samples = len(X)
        batch_size = min(32, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        if verbose:
            print("\n" + "=" * 60)
            print("🧠 TRAINING IMAGE PREDICTOR")
            print("=" * 60)
            print(f"  Samples:     {n_samples}")
            print(f"  Batch Size:  {batch_size}")
            print(f"  Epochs:      {epochs}")
            print(f"  LR:          {lr}")
            print("=" * 60 + "\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Shuffle
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0
            epoch_preds = []
            epoch_targets = []
            
            for i in range(0, n_samples, batch_size):
                batch_idx = idx[i:i + batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]
                
                loss, pred = self.train_step(batch_X, batch_y, lr)
                epoch_loss += loss
                epoch_preds.append(pred)
                epoch_targets.append(batch_y)
            
            # Compute metrics
            avg_loss = epoch_loss / n_batches
            all_preds = np.concatenate(epoch_preds, axis=0)
            all_targets = np.concatenate(epoch_targets, axis=0)
            train_hit_rate = self.compute_hit_rate(all_preds[:len(y)], y)
            
            # Validation
            val_loss = None
            val_hit_rate = None
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = self.compute_loss(val_pred, y_val)
                val_hit_rate = self.compute_hit_rate(val_pred, y_val)
            
            epoch_time = time.time() - epoch_start
            
            # Record metrics
            self.metrics.record_epoch(
                epoch, avg_loss, val_loss, 
                val_hit_rate if val_hit_rate else train_hit_rate,
                epoch_time
            )
            
            # Print progress
            if verbose:
                bar = progress_bar(epoch + 1, epochs, width=30, prefix=f"Epoch {epoch+1:3d}")
                
                status = f"Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    status += f" | Val: {val_loss:.4f}"
                status += f" | Hits: {train_hit_rate:.1f}/20"
                if val_hit_rate is not None:
                    status += f" (Val: {val_hit_rate:.1f})"
                status += f" | {epoch_time*1000:.0f}ms"
                
                # Show every epoch or every 5 for long training
                if epochs <= 20 or (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"\r{bar} {status}")
                else:
                    print(f"\r{bar}", end="", flush=True)
        
        if verbose:
            print()  # New line after progress
            self.metrics.print_summary()
            self.metrics.plot_curves(save_path=os.path.join(_ROOT, "image_predictor", "output", "training_curves.png"))
        
        return self.metrics.get_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Image-Based Lottery Prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--seq-len", type=int, default=15, help="Sequence length")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest")
    parser.add_argument("--save", type=str, default=None, help="Path to save model (e.g., models/quick.npz)")
    parser.add_argument("--load", type=str, default=None, help="Path to load model from (continue training)")
    parser.add_argument("--predict-only", action="store_true", help="Only predict, skip training")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🖼️  Quick Image-Based Lottery Prediction")
    print("=" * 70)
    
    # Load data
    data_path = os.path.join(_ROOT, "data", "data.csv")
    loader = DataLoader(data_path, sequence_length=args.seq_len)
    loader.load_data()
    
    print(f"\n📊 Loaded {len(loader.draws)} draws")
    
    # Create images
    images = loader.create_images()
    print(f"📊 Image shape: {images.shape}")
    
    # Create sequences (use only recent 500 for speed)
    n_use = min(500, len(images))
    images = images[-n_use:]
    
    seq_len = args.seq_len
    X, y = [], []
    for i in range(seq_len, len(images)):
        X.append(images[i - seq_len:i])
        y.append(images[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/validation (80/20)
    val_size = int(len(X) * 0.2)
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Create or load model
    if args.load:
        print(f"\n📂 Loading model from: {args.load}")
        model = QuickImagePredictor.from_file(args.load)
    else:
        model = QuickImagePredictor(seq_len=seq_len, hidden_dim=args.hidden)
    
    # Train (unless predict-only mode)
    if not args.predict_only:
        if args.load:
            print(f"📈 Continuing training for {args.epochs} more epochs...")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, epochs=args.epochs, lr=args.lr)
        
        # Save model if specified
        if args.save:
            model.save(args.save)
    else:
        if not args.load:
            print("⚠️ Warning: --predict-only requires --load to specify a model file")
            return
    
    # Predict next
    print("\n🔮 Predicting Next Draw...")
    
    # Use full data for prediction
    all_images = loader.create_images()
    latest = all_images[-seq_len:][np.newaxis, ...]
    pred = model.forward(latest)[0]
    
    # Decode
    top20 = model.encoder.decode_single(pred, 20)
    
    print("\n🎯 Top 20 Predicted Numbers:")
    print(" ".join(f"{n:02d}" for n in top20))
    
    # Show heatmap
    print("\n📊 Prediction Heatmap:")
    print(model.encoder.image_to_heatmap_ascii(pred))
    
    # Generate tickets
    print("\n🎫 Generated 20 Tickets:")
    rng = np.random.default_rng(42)
    flat = pred.flatten()
    sorted_idx = np.argsort(flat)[::-1]
    
    for t in range(20):
        if t % 3 == 0:
            pool = list(sorted_idx[:25])
            rng.shuffle(pool)
            selected = sorted(pool[:10])
        elif t % 3 == 1:
            selected = []
            for z in [0, 20, 40, 60]:
                zone = [i for i in sorted_idx if z <= i < z + 20][:6]
                rng.shuffle(zone)
                selected.extend(zone[:rng.integers(2, 4)])
            selected = sorted(selected[:10])
        else:
            probs = flat / flat.sum()
            selected = sorted(rng.choice(80, 10, replace=False, p=probs))
        
        nums = [i + 1 for i in selected]
        print(f"Ticket {t+1:02d}: {' '.join(f'{n:02d}' for n in nums)}")
    
    # Quick backtest
    if not args.no_backtest:
        print("\n📈 Quick Backtest (Last 10 Draws):")
        
        all_images = loader.create_images()
        hits_list = []
        
        for test_i in range(-10, 0):
            test_idx = len(all_images) + test_i
            seq = all_images[test_idx - seq_len:test_idx][np.newaxis, ...]
            p = model.forward(seq)[0]
            
            pred_nums = set(model.encoder.decode_single(p, 20))
            actual_nums = set(model.encoder.decode_single(all_images[test_idx], 20))
            
            hits = len(pred_nums & actual_nums)
            hits_list.append(hits)
            
            issue = loader.issues[test_idx]
            print(f"  Draw {issue}: {hits}/20 hits")
        
        print(f"\n📊 Average: {np.mean(hits_list):.2f}/20 (Random: 5/20)")
    
    print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
    print("=" * 70)


if __name__ == "__main__":
    main()
