#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KLE 快乐8 - Image-Based Lottery Prediction System

Scientific Research Application

Main entry point for training, prediction, and analysis.

Usage:
  python image_predictor/main.py generate-images   # Generate all draw images
  python image_predictor/main.py train --model cnn # Train model
  python image_predictor/main.py predict           # Predict next draw
  python image_predictor/main.py backtest          # Run backtest
  python image_predictor/main.py visualize         # Generate visualizations
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add paths
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

from image_predictor.config import AppConfig, ImageConfig, ModelConfig, TrainingConfig, PathConfig
from image_predictor.utils.image_encoder import ImageEncoder
from image_predictor.utils.data_loader import DataLoader
from image_predictor.utils.visualization import Visualizer
from image_predictor.models.cnn_predictor import CNNPredictor
from image_predictor.models.conv_lstm import ConvLSTMPredictor
from image_predictor.models.unet import UNetPredictor


class ImagePredictionApp:
    """
    Main application class for image-based lottery prediction.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.config.paths = PathConfig(_APP_ROOT)
        
        self.encoder = ImageEncoder(
            height=self.config.image.height,
            width=self.config.image.width,
        )
        
        # Data source
        data_path = os.path.join(_ROOT, "data", "data.csv")
        self.loader = DataLoader(
            data_path=data_path,
            encoder=self.encoder,
            sequence_length=self.config.model.sequence_length,
        )
        
        self.visualizer = Visualizer(self.config.paths.output_dir)
        
        self.model = None
        self.model_type = None
    
    @staticmethod
    def _parse_seq_lens(seq_lens: str) -> List[int]:
        """Parse comma-separated sequence lengths."""
        vals = [int(x.strip()) for x in seq_lens.split(",") if x.strip()]
        vals = [v for v in vals if v >= 5]
        if not vals:
            raise ValueError("seq_lens must contain at least one integer >= 5")
        return sorted(list(set(vals)))
    
    @staticmethod
    def _build_sequences_from_images(images: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build (X, y) samples from image timeline."""
        if len(images) <= seq_len:
            raise ValueError(f"Not enough images ({len(images)}) for seq_len={seq_len}")
        X = []
        y = []
        for i in range(seq_len, len(images)):
            X.append(images[i - seq_len:i])
            y.append(images[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def _train_single_pytorch_model(
        self,
        images: np.ndarray,
        seq_len: int,
        epochs: int,
        lr: float,
        seed: int,
        verbose: bool = False,
    ):
        """Train one PyTorch model and return predictor."""
        from image_predictor.models.pytorch_image_cnn import PyTorchImagePredictor
        import torch
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        X, y = self._build_sequences_from_images(images, seq_len)
        val_size = max(1, int(len(X) * 0.2))
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]
        
        model = PyTorchImagePredictor(seq_len=seq_len, hidden_channels=64)
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=self.config.training.batch_size,
            lr=lr,
            verbose=verbose,
        )
        return model
    
    def _create_model(self, model_type: str):
        """Create model based on type."""
        input_shape = (
            self.config.model.sequence_length,
            self.config.image.height,
            self.config.image.width,
            self.config.image.channels,
        )
        
        if model_type == "cnn":
            self.model = CNNPredictor(input_shape=input_shape)
        elif model_type == "conv_lstm":
            self.model = ConvLSTMPredictor(input_shape=input_shape)
        elif model_type == "unet":
            self.model = UNetPredictor(input_shape=input_shape)
        elif model_type == "pytorch_cnn":
            from image_predictor.models.pytorch_image_cnn import PyTorchImagePredictor
            self.model = PyTorchImagePredictor(
                seq_len=self.config.model.sequence_length,
                hidden_channels=64,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def generate_images(self, save: bool = True) -> np.ndarray:
        """
        Generate heatmap images for all draws.
        
        Args:
            save: Whether to save images to disk
        
        Returns:
            Array of images
        """
        print("=" * 70)
        print("🖼️  Generating Heatmap Images for All Draws")
        print("=" * 70)
        
        self.loader.load_data()
        images = self.loader.create_images()
        
        print(f"Generated {len(images)} images of shape {images.shape[1:]}")
        
        if save:
            # Save some sample images
            for i in [-1, -2, -3, -4, -5]:  # Last 5 draws
                idx = len(images) + i
                img = images[idx]
                draw, issue, date = self.loader.get_draw_by_index(idx)
                
                print(f"\n📊 Draw {issue} ({date}):")
                print(f"Numbers: {sorted(draw)}")
                print(self.encoder.image_to_ascii(img))
                
                # Save as image file if matplotlib available
                filepath = self.visualizer.save_heatmap_image(
                    img,
                    f"draw_{issue}.png",
                    title=f"Draw {issue} - {date}"
                )
                if filepath:
                    print(f"Saved to: {filepath}")
        
        return images
    
    def save_model(self, path: str):
        """
        Save trained model to file.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            print("⚠️ No model to save. Train a model first.")
            return
        
        # Ensure directory exists
        import os
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.model.save(path)
    
    def load_model(self, path: str, model_type: str = "cnn"):
        """
        Load model from file.
        
        Args:
            path: Path to load the model from
            model_type: Type of model to create
        """
        import os
        if not os.path.exists(path):
            print(f"⚠️ Model file not found: {path}")
            return False
        
        # Create model structure first
        self._create_model(model_type)
        
        # Load weights
        self.model.load(path)
        return True
    
    def train(
        self,
        model_type: str = "cnn",
        epochs: Optional[int] = None,
        verbose: bool = True,
        lr: Optional[float] = None,
        wf_last: int = 0,
        wf_every: int = 1,
        wf_samples: int = 5,
    ) -> dict:
        """
        Train the prediction model.
        
        Args:
            model_type: "cnn", "conv_lstm", "unet", or "pytorch_cnn"
            epochs: Number of training epochs
            verbose: Whether to print progress
            lr: Learning rate (optional, uses config default if not specified)
            wf_last: Walk-forward window size during training (PyTorch model only)
            wf_every: Print walk-forward stats every N epochs
            wf_samples: How many recent per-draw hits to print
        
        Returns:
            Training history
        """
        print("=" * 70)
        print(f"🧠 Training {model_type.upper()} Model")
        print("=" * 70)
        
        # Load data
        self.loader.load_data()
        X, y = self.loader.create_sequences()
        
        # Add channel dimension if needed
        if X.ndim == 4:
            X = X[..., np.newaxis]
        if y.ndim == 2:
            y = y[..., np.newaxis] if self.config.image.channels > 1 else y
        
        # Split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.loader.split_data(X, y)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        # Create model if not already loaded
        if self.model is None or self.model_type != model_type:
            self._create_model(model_type)
        
        # Train
        n_epochs = epochs or self.config.training.epochs
        learning_rate = lr or self.config.training.learning_rate
        
        if model_type == "pytorch_cnn":
            history = self.model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=n_epochs,
                batch_size=self.config.training.batch_size,
                lr=learning_rate,
                walk_forward_last=max(0, int(wf_last)),
                walk_forward_every=max(1, int(wf_every)),
                walk_forward_print_samples=max(1, int(wf_samples)),
                verbose=verbose,
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=n_epochs,
                batch_size=self.config.training.batch_size,
                lr=learning_rate,
                verbose=verbose,
            )
        
        # Evaluate on test set
        print("\n" + "=" * 70)
        print("📊 TEST SET EVALUATION")
        print("=" * 70)
        
        test_pred = self.model.predict(X_test)
        test_loss = self.model.compute_loss(test_pred, y_test if y_test.ndim == 3 else y_test.mean(axis=-1))
        
        # Calculate hit rates
        hit_rates = []
        for i in range(len(X_test)):
            pred_nums = self.encoder.decode_single(test_pred[i], top_k=20)
            y_img = y_test[i] if y_test.ndim == 3 else y_test[i, ..., 0]
            actual_nums = self.encoder.decode_single(y_img, top_k=20)
            hits = len(set(pred_nums) & set(actual_nums))
            hit_rates.append(hits)
        
        avg_hits = np.mean(hit_rates)
        min_hits = np.min(hit_rates)
        max_hits = np.max(hit_rates)
        
        print(f"  Test Samples:      {len(X_test)}")
        print(f"  Test Loss:         {test_loss:.6f}")
        print(f"  Average Hits:      {avg_hits:.2f}/20")
        print(f"  Min Hits:          {min_hits}/20")
        print(f"  Max Hits:          {max_hits}/20")
        print(f"  Random Baseline:   5.0/20")
        print(f"  vs Random:         {'+' if avg_hits > 5 else ''}{(avg_hits - 5):.2f}")
        print("=" * 70)
        
        # Save training curve
        self.visualizer.plot_training_history(history)
        
        print("\n✅ Training complete!")
        
        return history
    
    def predict(self, num_tickets: int = 20) -> tuple:
        """
        Predict the next draw.
        
        Args:
            num_tickets: Number of tickets to generate
        
        Returns:
            (top20_numbers, tickets, probability_matrix)
        """
        print("=" * 70)
        print("🔮 Predicting Next Draw")
        print("=" * 70)
        
        if self.model is None:
            print("No trained model found. Training CNN model first...")
            self.train(model_type="cnn", epochs=100, verbose=False)
        
        # Get latest sequence
        self.loader.load_data()
        latest_seq = self.loader.get_latest_sequence()
        
        if latest_seq.ndim == 4:
            latest_seq = latest_seq[..., np.newaxis]
        
        # Predict
        pred_image = self.model.predict(latest_seq)[0]
        
        # Decode to numbers
        top20 = self.encoder.decode_single(pred_image, top_k=20)
        
        # Generate diverse tickets
        tickets = self._generate_tickets(pred_image, num_tickets)
        
        # Print results
        print("\n🎯 Top 20 Predicted Numbers:")
        print(" ".join(f"{n:02d}" for n in top20))
        
        print("\n📊 Probability Heatmap:")
        print(self.encoder.image_to_heatmap_ascii(pred_image))
        
        print(f"\n🎫 Generated {num_tickets} Tickets:")
        for i, ticket in enumerate(tickets, 1):
            print(f"Ticket {i:02d}: {' '.join(f'{n:02d}' for n in ticket)}")
        
        # Save prediction image
        self.visualizer.save_heatmap_image(
            pred_image,
            "prediction_next.png",
            title="Predicted Next Draw"
        )
        
        print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
        
        return top20, tickets, pred_image
    
    def predict_ensemble(
        self,
        seq_lens: List[int],
        models_per_seq: int = 2,
        epochs: int = 25,
        lr: float = 0.0008,
        num_tickets: int = 20,
    ) -> tuple:
        """
        Ensemble prediction using multiple seq_len and random seeds.
        """
        print("=" * 70)
        print("🧪 Ensemble Predicting Next Draw")
        print("=" * 70)
        
        self.loader.load_data()
        images = self.loader.create_images().astype(np.float32)
        
        preds = []
        model_count = 0
        base_seed = 20260313
        
        for seq_len in seq_lens:
            if len(images) <= seq_len + 30:
                print(f"Skipping seq_len={seq_len}: not enough samples")
                continue
            
            print(f"\n🔧 Training ensemble group seq_len={seq_len} ...")
            for m in range(models_per_seq):
                seed = base_seed + seq_len * 100 + m
                model = self._train_single_pytorch_model(
                    images=images,
                    seq_len=seq_len,
                    epochs=epochs,
                    lr=lr,
                    seed=seed,
                    verbose=False,
                )
                latest = images[-seq_len:][np.newaxis, ...]
                pred = model.predict(latest)[0]
                preds.append(pred)
                model_count += 1
                print(f"  Trained model {m + 1}/{models_per_seq} for seq_len={seq_len}")
        
        if not preds:
            raise RuntimeError("No ensemble models were trained.")
        
        ensemble_pred = np.mean(np.stack(preds, axis=0), axis=0)
        top20 = self.encoder.decode_single(ensemble_pred, top_k=20)
        tickets = self._generate_tickets(ensemble_pred, num_tickets)
        
        print(f"\n✅ Ensemble size: {model_count}")
        print("🎯 Top 20 Predicted Numbers:")
        print(" ".join(f"{n:02d}" for n in top20))
        
        print("\n📊 Ensemble Probability Heatmap:")
        print(self.encoder.image_to_heatmap_ascii(ensemble_pred))
        
        print(f"\n🎫 Generated {num_tickets} Tickets:")
        for i, ticket in enumerate(tickets, 1):
            print(f"Ticket {i:02d}: {' '.join(f'{n:02d}' for n in ticket)}")
        
        self.visualizer.save_heatmap_image(
            ensemble_pred,
            "prediction_ensemble_next.png",
            title=f"Ensemble Predicted Next Draw ({model_count} models)",
        )
        
        return top20, tickets, ensemble_pred
    
    def walk_forward_ensemble(
        self,
        last_n: int = 20,
        seq_lens: Optional[List[int]] = None,
        models_per_seq: int = 1,
        epochs: int = 8,
        lr: float = 0.0008,
        min_train_samples: int = 320,
    ) -> Dict:
        """
        Walk-forward rolling evaluation with ensemble models.
        """
        seq_lens = seq_lens or [12, 16, 20]
        print("=" * 70)
        print(f"📈 Walk-Forward Ensemble Backtest (last_n={last_n})")
        print("=" * 70)
        
        self.loader.load_data()
        images = self.loader.create_images().astype(np.float32)
        n_total = len(images)
        
        start_idx = max(max(seq_lens) + min_train_samples, n_total - last_n)
        results = []
        base_seed = 20260313
        
        for test_idx in range(start_idx, n_total):
            train_images = images[:test_idx]
            pred_list = []
            
            for seq_len in seq_lens:
                if len(train_images) <= seq_len + 20:
                    continue
                for m in range(models_per_seq):
                    seed = base_seed + test_idx * 10 + seq_len * 100 + m
                    model = self._train_single_pytorch_model(
                        images=train_images,
                        seq_len=seq_len,
                        epochs=epochs,
                        lr=lr,
                        seed=seed,
                        verbose=False,
                    )
                    test_seq = images[test_idx - seq_len:test_idx][np.newaxis, ...]
                    pred = model.predict(test_seq)[0]
                    pred_list.append(pred)
            
            if not pred_list:
                continue
            
            ensemble_pred = np.mean(np.stack(pred_list, axis=0), axis=0)
            pred_nums = set(self.encoder.decode_single(ensemble_pred, 20))
            actual_nums = set(self.encoder.decode_single(images[test_idx], 20))
            hits = len(pred_nums & actual_nums)
            
            issue = self.loader.issues[test_idx] if test_idx < len(self.loader.issues) else str(test_idx)
            results.append({"index": test_idx, "issue": issue, "hits": hits})
            print(f"Draw {issue}: Hits={hits}/20 ({len(pred_list)} models)")
        
        if not results:
            raise RuntimeError("No walk-forward results generated. Try reducing seq_len or min_train_samples.")
        
        hit_arr = np.array([r["hits"] for r in results], dtype=np.float32)
        summary = {
            "n_draws": int(len(results)),
            "avg_hits": float(hit_arr.mean()),
            "min_hits": int(hit_arr.min()),
            "max_hits": int(hit_arr.max()),
            "vs_random": float(hit_arr.mean() - 5.0),
            "seq_lens": seq_lens,
            "models_per_seq": models_per_seq,
            "epochs": epochs,
        }
        
        print("-" * 70)
        print(f"📊 Average: {summary['avg_hits']:.2f}/20")
        print(f"📊 Min/Max: {summary['min_hits']}/{summary['max_hits']}")
        print(f"📊 Random baseline: 5.00/20")
        print(f"📊 vs Random: {'+' if summary['vs_random'] > 0 else ''}{summary['vs_random']:.2f}")
        
        return {"summary": summary, "results": results}
    
    def _generate_tickets(self, prob_image: np.ndarray, num_tickets: int) -> List[List[int]]:
        """Generate diverse tickets from probability image."""
        rng = np.random.default_rng(42)
        flat_probs = prob_image.flatten()
        sorted_idx = np.argsort(flat_probs)[::-1]
        
        tickets = []
        
        for t in range(num_tickets):
            # Mix strategies
            if t % 3 == 0:
                # High probability
                pool = list(sorted_idx[:25])
                rng.shuffle(pool)
                selected = sorted(pool[:10])
            elif t % 3 == 1:
                # Zone balanced
                selected = []
                for zone_start in [0, 20, 40, 60]:
                    zone_idx = [i for i in sorted_idx if zone_start <= i < zone_start + 20][:6]
                    rng.shuffle(zone_idx)
                    selected.extend(zone_idx[:rng.integers(2, 4)])
                selected = sorted(selected[:10])
            else:
                # Probabilistic sampling
                probs_norm = flat_probs / flat_probs.sum()
                selected = sorted(rng.choice(80, size=10, replace=False, p=probs_norm))
            
            nums = [int(i + 1) for i in selected]
            tickets.append(nums)
        
        return tickets
    
    def backtest(self, last_n: int = 20) -> dict:
        """
        Run backtest on historical data.
        
        Args:
            last_n: Number of recent draws to test
        
        Returns:
            Backtest results
        """
        print("=" * 70)
        print(f"📈 Backtesting on Last {last_n} Draws")
        print("=" * 70)
        
        self.loader.load_data()
        
        # Need to train on earlier data and test on recent
        images = self.loader.create_images()
        if images.ndim == 3:
            images = images[..., np.newaxis]
        
        seq_len = self.config.model.sequence_length
        n_total = len(images)
        
        if n_total < seq_len + last_n + 50:
            last_n = min(last_n, n_total - seq_len - 50)
        
        test_start = n_total - last_n
        
        results = []
        
        for test_idx in range(test_start, n_total):
            # Use data before test_idx for training
            train_end = test_idx
            
            if train_end < seq_len + 20:
                continue
            
            # Create training sequences
            X_train = []
            y_train = []
            for i in range(seq_len, train_end):
                X_train.append(images[i - seq_len:i])
                y_train.append(images[i])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Quick train
            self._create_model("cnn")
            self.model.fit(X_train, y_train, epochs=30, verbose=False)
            
            # Predict
            test_seq = images[test_idx - seq_len:test_idx][np.newaxis, ...]
            pred = self.model.predict(test_seq)[0]
            
            pred_nums = set(self.encoder.decode_single(pred, 20))
            actual_nums = set(self.encoder.decode_single(images[test_idx, ..., 0], 20))
            
            hits = len(pred_nums & actual_nums)
            results.append({
                "index": test_idx,
                "hits": hits,
                "predicted": sorted(pred_nums),
                "actual": sorted(actual_nums),
            })
            
            issue = self.loader.issues[test_idx] if test_idx < len(self.loader.issues) else test_idx
            print(f"Draw {issue}: Hits = {hits}/20")
        
        # Summary
        if results:
            avg_hits = np.mean([r["hits"] for r in results])
            max_hits = max(r["hits"] for r in results)
            min_hits = min(r["hits"] for r in results)
            
            print("-" * 70)
            print(f"📊 Average: {avg_hits:.2f}/20 | Max: {max_hits}/20 | Min: {min_hits}/20")
            print(f"📊 Random baseline: 5.00/20")
        
        return {"results": results}
    
    def visualize_sequence(self, n: int = 10):
        """Visualize recent draw sequence."""
        print("=" * 70)
        print(f"🎨 Visualizing Last {n} Draws")
        print("=" * 70)
        
        self.loader.load_data()
        images = self.loader.create_images()
        
        recent = images[-n:]
        
        for i, img in enumerate(recent):
            idx = len(images) - n + i
            draw, issue, date = self.loader.get_draw_by_index(idx)
            
            print(f"\n📊 Draw {issue} ({date}):")
            print(f"Numbers: {sorted(draw)}")
            print(self.encoder.image_to_ascii(img))
        
        # Save strip image
        self.visualizer.save_sequence_strip(recent, "recent_sequence.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="KLE Image-Based Lottery Prediction System"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Generate images
    sub.add_parser("generate-images", help="Generate heatmap images for all draws")
    
    # Train
    p_train = sub.add_parser("train", help="Train prediction model")
    p_train.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "conv_lstm", "unet", "pytorch_cnn"],
                        help="Model type")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--save", type=str, default=None,
                        help="Path to save trained model (e.g., models/cnn.npz)")
    p_train.add_argument("--load", type=str, default=None,
                        help="Path to load model from (continue training)")
    p_train.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    p_train.add_argument("--wf-last", type=int, default=0,
                        help="Walk-forward window during training (PyTorch model)")
    p_train.add_argument("--wf-every", type=int, default=1,
                        help="Print walk-forward stats every N epochs")
    p_train.add_argument("--wf-samples", type=int, default=5,
                        help="How many recent walk-forward hits to print")
    
    # Predict
    p_pred = sub.add_parser("predict", help="Predict next draw")
    p_pred.add_argument("--tickets", type=int, default=20)
    p_pred.add_argument("--model", type=str, default="cnn",
                       choices=["cnn", "conv_lstm", "unet", "pytorch_cnn"])
    p_pred.add_argument("--load", type=str, default=None,
                       help="Path to load model from")
    
    # Backtest
    p_bt = sub.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--last", type=int, default=20)
    
    # Visualize
    p_vis = sub.add_parser("visualize", help="Visualize recent draws")
    p_vis.add_argument("-n", type=int, default=10)
    
    # Ensemble predict
    p_ens = sub.add_parser("ensemble-predict", help="Ensemble prediction with multi seq_len models")
    p_ens.add_argument("--seq-lens", type=str, default="12,16,20",
                       help="Comma-separated sequence lengths, e.g. 12,16,20")
    p_ens.add_argument("--models-per-seq", type=int, default=2)
    p_ens.add_argument("--epochs", type=int, default=25)
    p_ens.add_argument("--lr", type=float, default=0.0008)
    p_ens.add_argument("--tickets", type=int, default=20)
    
    # Walk-forward backtest
    p_wf = sub.add_parser("walk-forward", help="Walk-forward ensemble backtest")
    p_wf.add_argument("--last", type=int, default=20)
    p_wf.add_argument("--seq-lens", type=str, default="12,16,20",
                      help="Comma-separated sequence lengths, e.g. 12,16,20")
    p_wf.add_argument("--models-per-seq", type=int, default=1)
    p_wf.add_argument("--epochs", type=int, default=8)
    p_wf.add_argument("--lr", type=float, default=0.0008)
    
    return parser.parse_args()


def main():
    args = parse_args()
    app = ImagePredictionApp()
    
    if args.command == "generate-images":
        app.generate_images()
        
    elif args.command == "train":
        # Load existing model if specified
        if args.load:
            app.load_model(args.load, args.model)
            print(f"📂 Continuing training from: {args.load}")
        
        # Train
        app.train(
            model_type=args.model,
            epochs=args.epochs,
            lr=args.lr,
            wf_last=args.wf_last,
            wf_every=args.wf_every,
            wf_samples=args.wf_samples,
        )
        
        # Save model if specified
        if args.save:
            app.save_model(args.save)
        
    elif args.command == "predict":
        if args.load:
            app.load_model(args.load, args.model)
        else:
            # Train first if no model loaded
            app.train(model_type=args.model, epochs=50, verbose=False)
        app.predict(num_tickets=args.tickets)
        
    elif args.command == "backtest":
        app.backtest(last_n=args.last)
        
    elif args.command == "visualize":
        app.visualize_sequence(n=args.n)
    
    elif args.command == "ensemble-predict":
        seq_lens = app._parse_seq_lens(args.seq_lens)
        app.predict_ensemble(
            seq_lens=seq_lens,
            models_per_seq=args.models_per_seq,
            epochs=args.epochs,
            lr=args.lr,
            num_tickets=args.tickets,
        )
    
    elif args.command == "walk-forward":
        seq_lens = app._parse_seq_lens(args.seq_lens)
        app.walk_forward_ensemble(
            last_n=args.last,
            seq_lens=seq_lens,
            models_per_seq=args.models_per_seq,
            epochs=args.epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
