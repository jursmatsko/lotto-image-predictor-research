#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Random Forest for Lottery Prediction

Combines:
1. Image-based features (spatial patterns from 8×10 images)
2. Classical statistical features (frequency, gap)
3. Random Forest classifier (interpretable, fast)

Key insight: Use convolution-like feature extraction manually,
then feed to Random Forest for interpretable predictions.
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    raise RuntimeError("请安装 scikit-learn: pip install scikit-learn") from e

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


TOTAL_NUMBERS = 80
DRAW_NUMBERS = 20


@dataclass
class VisionFeatureConfig:
    """Vision feature extraction configuration."""
    sequence_length: int = 15       # Number of historical draws to use
    use_spatial_features: bool = True
    use_frequency_features: bool = True
    use_pattern_features: bool = True
    use_neighbor_features: bool = True


def numbers_to_image(numbers: np.ndarray) -> np.ndarray:
    """
    Convert lottery numbers to 8×10 binary image.
    
    Args:
        numbers: Array of winning numbers (1-80)
    
    Returns:
        8×10 binary matrix (1=selected, 0=not selected)
    """
    image = np.zeros((8, 10), dtype=np.float32)
    for num in numbers:
        if 1 <= num <= 80:
            row = (num - 1) // 10
            col = (num - 1) % 10
            image[row, col] = 1.0
    return image


def extract_spatial_features(image: np.ndarray) -> np.ndarray:
    """
    Extract spatial features from 8×10 image.
    
    Features:
    - Row sums (8 features): how many numbers in each row
    - Column sums (10 features): how many numbers in each column
    - Diagonal sums (various diagonals)
    - Quadrant distributions
    """
    features = []
    
    # Row sums (8 features)
    row_sums = image.sum(axis=1)
    features.extend(row_sums)
    
    # Column sums (10 features)
    col_sums = image.sum(axis=0)
    features.extend(col_sums)
    
    # Quadrant sums (4 features)
    # Top-left, Top-right, Bottom-left, Bottom-right
    features.append(image[:4, :5].sum())
    features.append(image[:4, 5:].sum())
    features.append(image[4:, :5].sum())
    features.append(image[4:, 5:].sum())
    
    # Diagonal patterns
    # Main diagonal (approximate for 8x10)
    main_diag = sum(image[i, min(i, 9)] for i in range(8))
    features.append(main_diag)
    
    # Anti-diagonal
    anti_diag = sum(image[i, min(9-i, 9)] for i in range(8))
    features.append(anti_diag)
    
    # Center density (middle 4x6 region)
    center = image[2:6, 2:8].sum()
    features.append(center)
    
    # Edge density
    edge = image[0, :].sum() + image[-1, :].sum() + image[:, 0].sum() + image[:, -1].sum()
    features.append(edge)
    
    return np.array(features, dtype=np.float32)


def extract_pattern_features(image: np.ndarray) -> np.ndarray:
    """
    Extract pattern features using simple convolution-like operations.
    
    Features:
    - Horizontal patterns (consecutive numbers)
    - Vertical patterns (numbers in same column across rows)
    - 2x2 block patterns
    """
    features = []
    
    # Horizontal consecutive pairs
    h_pairs = 0
    for row in range(8):
        for col in range(9):
            if image[row, col] == 1 and image[row, col+1] == 1:
                h_pairs += 1
    features.append(h_pairs)
    
    # Horizontal consecutive triples
    h_triples = 0
    for row in range(8):
        for col in range(8):
            if image[row, col] == 1 and image[row, col+1] == 1 and image[row, col+2] == 1:
                h_triples += 1
    features.append(h_triples)
    
    # Vertical consecutive pairs
    v_pairs = 0
    for row in range(7):
        for col in range(10):
            if image[row, col] == 1 and image[row+1, col] == 1:
                v_pairs += 1
    features.append(v_pairs)
    
    # 2x2 blocks with 2+ numbers
    blocks_2plus = 0
    blocks_3plus = 0
    for row in range(7):
        for col in range(9):
            block_sum = image[row:row+2, col:col+2].sum()
            if block_sum >= 2:
                blocks_2plus += 1
            if block_sum >= 3:
                blocks_3plus += 1
    features.append(blocks_2plus)
    features.append(blocks_3plus)
    
    # L-shaped patterns
    l_patterns = 0
    for row in range(7):
        for col in range(9):
            # Check various L shapes
            if image[row, col] == 1 and image[row+1, col] == 1 and image[row+1, col+1] == 1:
                l_patterns += 1
            if image[row, col] == 1 and image[row, col+1] == 1 and image[row+1, col] == 1:
                l_patterns += 1
    features.append(l_patterns)
    
    return np.array(features, dtype=np.float32)


def extract_neighbor_features(image: np.ndarray, position: int) -> np.ndarray:
    """
    Extract features about a specific position's neighborhood.
    
    Args:
        image: 8×10 image
        position: number position (0-79)
    
    Returns:
        Neighbor features for this position
    """
    row = position // 10
    col = position % 10
    features = []
    
    # 8 immediate neighbors
    neighbor_sum = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 10:
                neighbor_sum += image[nr, nc]
    features.append(neighbor_sum)
    
    # Row neighbors (same row)
    row_sum = image[row, :].sum()
    features.append(row_sum)
    
    # Column neighbors (same column)
    col_sum = image[:, col].sum()
    features.append(col_sum)
    
    # Diagonal neighbors
    diag_sum = 0
    for d in [-2, -1, 1, 2]:
        nr1, nc1 = row + d, col + d
        nr2, nc2 = row + d, col - d
        if 0 <= nr1 < 8 and 0 <= nc1 < 10:
            diag_sum += image[nr1, nc1]
        if 0 <= nr2 < 8 and 0 <= nc2 < 10:
            diag_sum += image[nr2, nc2]
    features.append(diag_sum)
    
    return np.array(features, dtype=np.float32)


def extract_frequency_features(
    images: np.ndarray,
    windows: List[int] = [5, 10, 20]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract frequency and gap features from image sequence.
    
    Args:
        images: Sequence of images (seq_len, 8, 10)
        windows: Window sizes for frequency calculation
    
    Returns:
        freq_features: (80,) frequency for each position
        gap_features: (80,) gap for each position
    """
    seq_len = len(images)
    freq_features = np.zeros(80, dtype=np.float32)
    gap_features = np.zeros(80, dtype=np.float32)
    
    # Flatten images for easier indexing
    flat_images = images.reshape(seq_len, 80)
    
    # Frequency in last window
    max_w = min(max(windows), seq_len)
    recent = flat_images[-max_w:]
    freq_features = recent.mean(axis=0)
    
    # Gap: how many draws since last appearance
    for pos in range(80):
        gap = 0
        for t in range(seq_len - 1, -1, -1):
            if flat_images[t, pos] == 1:
                break
            gap += 1
        gap_features[pos] = gap / max(seq_len, 1)
    
    return freq_features, gap_features


def build_vision_features(
    images: np.ndarray,
    target_position: int,
    cfg: VisionFeatureConfig,
) -> np.ndarray:
    """
    Build complete feature vector for a single position.
    
    Args:
        images: Sequence of images (seq_len, 8, 10)
        target_position: Position (0-79) to predict
        cfg: Feature configuration
    
    Returns:
        Feature vector for this position
    """
    features = []
    
    # Aggregate image (frequency map)
    freq_map = images.mean(axis=0)
    
    # Recent image
    recent_image = images[-1] if len(images) > 0 else np.zeros((8, 10))
    
    if cfg.use_spatial_features:
        # Spatial features from frequency map
        spatial = extract_spatial_features(freq_map)
        features.extend(spatial)
        
        # Spatial features from recent
        spatial_recent = extract_spatial_features(recent_image)
        features.extend(spatial_recent)
    
    if cfg.use_pattern_features:
        # Pattern features from frequency map (binarized)
        pattern = extract_pattern_features((freq_map > 0.25).astype(float))
        features.extend(pattern)
    
    if cfg.use_neighbor_features:
        # Neighbor features for target position
        neighbor = extract_neighbor_features(freq_map, target_position)
        features.extend(neighbor)
        
        neighbor_recent = extract_neighbor_features(recent_image, target_position)
        features.extend(neighbor_recent)
    
    if cfg.use_frequency_features:
        # Frequency and gap for target position
        freq, gap = extract_frequency_features(images)
        features.append(freq[target_position])
        features.append(gap[target_position])
        
        # Also add global stats
        features.append(freq.mean())
        features.append(freq.std())
    
    return np.array(features, dtype=np.float32)


class VisionRandomForest:
    """
    Random Forest with Vision Features for lottery prediction.
    """
    
    def __init__(
        self,
        seq_len: int = 15,
        n_estimators: int = 100,
        max_depth: int = 10,
        use_gradient_boosting: bool = False,
    ):
        self.seq_len = seq_len
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.use_gradient_boosting = use_gradient_boosting
        
        self.cfg = VisionFeatureConfig(sequence_length=seq_len)
        self.models: List[Optional[RandomForestClassifier]] = [None] * 80
        self.scalers: List[Optional[StandardScaler]] = [None] * 80
        self.history = {"train_time": 0, "feature_dim": 0}
    
    def _create_model(self):
        """Create a new classifier."""
        if self.use_gradient_boosting:
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
    
    def fit(self, images: np.ndarray, verbose: bool = True):
        """
        Train models for all 80 positions.
        
        Args:
            images: All historical images (n_draws, 8, 10)
        """
        n_draws = len(images)
        start_idx = self.seq_len
        
        if verbose:
            print("\n" + "=" * 70)
            print("🌲 VISION RANDOM FOREST TRAINING")
            print("=" * 70)
            print(f"   Total draws:      {n_draws}")
            print(f"   Sequence length:  {self.seq_len}")
            print(f"   Training samples: {n_draws - start_idx}")
            print(f"   Model type:       {'GradientBoosting' if self.use_gradient_boosting else 'RandomForest'}")
            print(f"   Estimators:       {self.n_estimators}")
            print("=" * 70)
        
        start_time = time.time()
        
        # Build dataset
        X_all = {pos: [] for pos in range(80)}
        y_all = {pos: [] for pos in range(80)}
        
        if verbose:
            print("\n📊 Extracting vision features...")
        
        for idx in range(start_idx, n_draws):
            seq = images[idx - self.seq_len:idx]
            target = images[idx].flatten()
            
            for pos in range(80):
                features = build_vision_features(seq, pos, self.cfg)
                X_all[pos].append(features)
                y_all[pos].append(int(target[pos]))
            
            if verbose and (idx - start_idx + 1) % 100 == 0:
                print(f"   Processed {idx - start_idx + 1}/{n_draws - start_idx} samples", end="\r")
        
        if verbose:
            print(f"\n   Feature dimension: {len(X_all[0][0])}")
            self.history["feature_dim"] = len(X_all[0][0])
        
        # Train models
        if verbose:
            print("\n🔧 Training 80 position models...")
        
        for pos in range(80):
            X = np.array(X_all[pos])
            y = np.array(y_all[pos])
            
            # Check if we have both classes
            if y.sum() == 0 or y.sum() == len(y):
                self.models[pos] = None
                self.scalers[pos] = None
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = self._create_model()
            model.fit(X_scaled, y)
            
            self.models[pos] = model
            self.scalers[pos] = scaler
            
            if verbose and (pos + 1) % 20 == 0:
                print(f"   Trained {pos + 1}/80 models", end="\r")
        
        train_time = time.time() - start_time
        self.history["train_time"] = train_time
        
        if verbose:
            print(f"\n\n✅ Training complete in {train_time:.2f}s")
            print("=" * 70)
    
    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """
        Predict probability for each position.
        
        Args:
            images: Recent images (seq_len, 8, 10)
        
        Returns:
            Probabilities for each position (80,)
        """
        probs = np.zeros(80, dtype=np.float32)
        
        for pos in range(80):
            if self.models[pos] is None:
                probs[pos] = 0.25  # Default
                continue
            
            features = build_vision_features(images, pos, self.cfg)
            features_scaled = self.scalers[pos].transform(features.reshape(1, -1))
            
            prob = self.models[pos].predict_proba(features_scaled)[0]
            probs[pos] = prob[1] if len(prob) > 1 else prob[0]
        
        return probs
    
    def predict_top_k(self, images: np.ndarray, k: int = 20) -> List[int]:
        """Predict top k numbers."""
        probs = self.predict_proba(images)
        top_indices = np.argsort(probs)[::-1][:k]
        return sorted([idx + 1 for idx in top_indices])
    
    def get_feature_importance(self, position: int) -> Optional[np.ndarray]:
        """Get feature importance for a position."""
        if self.models[position] is None:
            return None
        return self.models[position].feature_importances_
    
    def save(self, path: str):
        """Save models to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'models': self.models,
            'scalers': self.scalers,
            'seq_len': self.seq_len,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'use_gradient_boosting': self.use_gradient_boosting,
            'history': self.history,
        }
        joblib.dump(data, path)
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str):
        """Load models from file."""
        data = joblib.load(path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.seq_len = data['seq_len']
        self.n_estimators = data['n_estimators']
        self.max_depth = data['max_depth']
        self.use_gradient_boosting = data.get('use_gradient_boosting', False)
        self.history = data.get('history', {})
        print(f"✅ Model loaded from: {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'VisionRandomForest':
        """Load from file."""
        data = joblib.load(path)
        model = cls(
            seq_len=data['seq_len'],
            n_estimators=data['n_estimators'],
            max_depth=data['max_depth'],
            use_gradient_boosting=data.get('use_gradient_boosting', False),
        )
        model.load(path)
        return model


def main():
    """Main training and prediction script."""
    parser = argparse.ArgumentParser(description="Vision Random Forest for Lottery")
    parser.add_argument("--epochs", type=int, default=1, help="Not used (RF trains once)")
    parser.add_argument("--estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=10, help="Max tree depth")
    parser.add_argument("--seq-len", type=int, default=15, help="Sequence length")
    parser.add_argument("--load", type=str, default=None, help="Load model from path")
    parser.add_argument("--save", type=str, default="image_predictor/models/vision_rf.joblib", help="Save model to path")
    parser.add_argument("--predict-only", action="store_true", help="Only predict")
    parser.add_argument("--gradient-boosting", action="store_true", help="Use GradientBoosting instead of RF")
    parser.add_argument("--quick", action="store_true", help="Quick mode: use recent data only")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip rolling backtest")
    args = parser.parse_args()
    
    from image_predictor.utils.data_loader import DataLoader as LotteryDataLoader
    from image_predictor.utils.image_encoder import ImageEncoder
    
    print("=" * 70)
    print("🌲 VISION RANDOM FOREST FOR LOTTERY PREDICTION")
    print("   (Classical ML + Image Features)")
    print("=" * 70)
    
    # Load data
    data_path = os.path.join(_ROOT, "data", "data.csv")
    loader = LotteryDataLoader(data_path, sequence_length=args.seq_len)
    loader.load_data()
    
    print(f"\n📊 Loaded {len(loader.draws)} draws")
    
    # Create images
    images = loader.create_images()
    print(f"📊 Image shape: {images.shape}")
    
    # Quick mode: use only recent data
    if args.quick:
        n_use = min(500, len(images))
        images = images[-n_use:]
        print(f"📊 Quick mode: using last {n_use} draws")
    
    encoder = ImageEncoder()
    
    # Load or train model
    if args.load and os.path.exists(args.load):
        print(f"\n📂 Loading model from: {args.load}")
        model = VisionRandomForest.from_file(args.load)
    else:
        if args.predict_only:
            print("❌ No model to load!")
            return
        
        model = VisionRandomForest(
            seq_len=args.seq_len,
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            use_gradient_boosting=args.gradient_boosting,
        )
        model.fit(images)
        model.save(args.save)
    
    # Predict
    print("\n🔮 Predicting Next Draw...")
    latest = images[-args.seq_len:]
    
    probs = model.predict_proba(latest)
    top20 = model.predict_top_k(latest, 20)
    
    print("\n🎯 Top 20 Predicted Numbers:")
    print(" ".join(f"{n:02d}" for n in top20))
    
    # Show as image
    print("\n📊 Probability Heatmap (8×10):")
    print("     " + " ".join(f"{i+1:2d}" for i in range(10)))
    for row in range(8):
        line = f"{row*10+1:2d}-{row*10+10:2d} "
        for col in range(10):
            pos = row * 10 + col
            val = probs[pos]
            if val > 0.35:
                line += "██"
            elif val > 0.28:
                line += "▓▓"
            elif val > 0.24:
                line += "▒▒"
            else:
                line += "··"
        print(line)
    
    # Proper Rolling Backtest (train on past, test on future)
    if not args.skip_backtest:
        print("\n📈 Rolling Backtest (Last 10 draws):")
        print("   (Each test uses model trained ONLY on prior data)")
        hits_list = []
        
        all_images = loader.create_images()  # Use all for backtest
        
        for test_i in range(-10, 0):
            test_idx = len(all_images) + test_i
            
            # Train model on data BEFORE test_idx only
            train_images = all_images[:test_idx]
            if len(train_images) < args.seq_len + 50:
                continue
            
            # Use only recent data for training
            train_images = train_images[-500:]
            
            # Create a fresh model for this test
            test_model = VisionRandomForest(
                seq_len=args.seq_len,
                n_estimators=50,  # Fewer trees for speed
                max_depth=8,
            )
            test_model.fit(train_images, verbose=False)
            
            # Predict using only prior data
            seq = train_images[-args.seq_len:]
            pred_nums = set(test_model.predict_top_k(seq, 20))
            actual_nums = set(encoder.decode_single(all_images[test_idx], 20))
            
            hits = len(pred_nums & actual_nums)
            hits_list.append(hits)
            
            issue = loader.issues[test_idx]
            print(f"  Draw {issue}: {hits}/20 hits")
        
        print(f"\n📊 Average: {np.mean(hits_list):.2f}/20 (Random: 5/20)")
    else:
        print("\n⏭️ Skipping backtest (use --skip-backtest=false to enable)")
    
    # Feature importance analysis
    print("\n📊 Top Feature Importances (Position 1):")
    importance = model.get_feature_importance(0)
    if importance is not None:
        top_feat_idx = np.argsort(importance)[::-1][:10]
        for i, idx in enumerate(top_feat_idx):
            print(f"   Feature {idx}: {importance[idx]:.4f}")
    
    print("\n⚠️ DISCLAIMER: Lottery is random. This is for scientific research only!")
    print("=" * 70)


if __name__ == "__main__":
    main()
