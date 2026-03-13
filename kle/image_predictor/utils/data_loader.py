#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loader for Image-Based Lottery Prediction

Loads lottery draw data and converts to image sequences for training.
"""

import os
import sys
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# Add parent paths for imports
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from image_predictor.utils.image_encoder import ImageEncoder


class DataLoader:
    """
    Loads lottery data and prepares image sequences for training.
    
    Creates (X, y) pairs where:
    - X: Sequence of N historical draw images
    - y: The next draw image (target)
    """
    
    def __init__(
        self,
        data_path: str,
        encoder: Optional[ImageEncoder] = None,
        sequence_length: int = 20,
        encoding_mode: str = "binary",
        multi_channel: bool = False,
    ):
        """
        Args:
            data_path: Path to CSV data file
            encoder: ImageEncoder instance
            sequence_length: Number of past draws to use as input
            encoding_mode: "binary", "frequency", or "heatmap"
            multi_channel: Whether to use multi-channel images
        """
        self.data_path = data_path
        self.encoder = encoder or ImageEncoder()
        self.sequence_length = sequence_length
        self.encoding_mode = encoding_mode
        self.multi_channel = multi_channel
        
        self.draws: List[List[int]] = []
        self.issues: List[str] = []
        self.dates: List[str] = []
        
    def load_data(self) -> None:
        """Load lottery draw data from CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        df = pd.read_csv(self.data_path, encoding="utf-8")
        
        # Find columns
        issue_col = "期数" if "期数" in df.columns else "期号"
        date_col = "日期" if "日期" in df.columns else None
        number_cols = [f"红球_{i}" for i in range(1, 21)]
        
        # Sort by issue (ascending - oldest first)
        df = df.sort_values(issue_col, ascending=True).reset_index(drop=True)
        
        self.draws = []
        self.issues = []
        self.dates = []
        
        for _, row in df.iterrows():
            numbers = []
            for col in number_cols:
                if col in row.index:
                    try:
                        num = int(row[col])
                        if 1 <= num <= 80:
                            numbers.append(num)
                    except (ValueError, TypeError):
                        pass
            
            if len(numbers) == 20:
                self.draws.append(numbers)
                self.issues.append(str(row[issue_col]))
                if date_col and date_col in row.index:
                    self.dates.append(str(row[date_col]))
                else:
                    self.dates.append("")
        
        print(f"Loaded {len(self.draws)} draws from {self.data_path}")
    
    def create_images(
        self,
        frequency_window: int = 50,
    ) -> np.ndarray:
        """
        Create image representations for all draws.
        
        Args:
            frequency_window: Window for frequency calculation
        
        Returns:
            Array of shape (n_draws, height, width) or (n_draws, height, width, channels)
        """
        if not self.draws:
            self.load_data()
        
        if self.multi_channel:
            # Create multi-channel images
            n_draws = len(self.draws)
            images = np.zeros((n_draws, 8, 10, 3), dtype=np.float32)
            
            for i, draw in enumerate(self.draws):
                history = self.draws[:i] if i > 0 else []
                images[i] = self.encoder.create_multi_channel_image(
                    draw, history, frequency_window
                )
        else:
            # Single channel images
            images = self.encoder.encode_batch(
                self.draws,
                mode=self.encoding_mode,
                compute_running_frequency=(self.encoding_mode != "binary"),
                frequency_window=frequency_window,
            )
        
        return images
    
    def create_sequences(
        self,
        frequency_window: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence data for training.
        
        Returns:
            X: Input sequences of shape (n_samples, seq_length, height, width, channels)
            y: Target images of shape (n_samples, height, width, channels)
        """
        images = self.create_images(frequency_window)
        n_draws = len(images)
        seq_len = self.sequence_length
        
        if n_draws <= seq_len:
            raise ValueError(f"Not enough draws ({n_draws}) for sequence length {seq_len}")
        
        n_samples = n_draws - seq_len
        
        # Determine image shape
        if images.ndim == 3:
            h, w = images.shape[1], images.shape[2]
            X = np.zeros((n_samples, seq_len, h, w), dtype=np.float32)
            y = np.zeros((n_samples, h, w), dtype=np.float32)
        else:
            h, w, c = images.shape[1], images.shape[2], images.shape[3]
            X = np.zeros((n_samples, seq_len, h, w, c), dtype=np.float32)
            y = np.zeros((n_samples, h, w, c), dtype=np.float32)
        
        for i in range(n_samples):
            X[i] = images[i:i + seq_len]
            y[i] = images[i + seq_len]
        
        return X, y
    
    def get_latest_sequence(
        self,
        frequency_window: int = 50,
    ) -> np.ndarray:
        """
        Get the most recent sequence for prediction.
        
        Returns:
            Array of shape (1, seq_length, height, width) or (1, seq_length, height, width, channels)
        """
        images = self.create_images(frequency_window)
        seq_len = self.sequence_length
        
        if len(images) < seq_len:
            raise ValueError(f"Not enough draws for sequence length {seq_len}")
        
        latest = images[-seq_len:]
        return latest[np.newaxis, ...]  # Add batch dimension
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        test_split: float = 0.1,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets.
        
        Note: Uses temporal split (not random) to respect time series nature.
        """
        n = len(X)
        test_size = int(n * test_split)
        val_size = int(n * validation_split)
        train_size = n - val_size - test_size
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_draw_by_index(self, index: int) -> Tuple[List[int], str, str]:
        """Get draw numbers, issue, and date by index."""
        if not self.draws:
            self.load_data()
        return self.draws[index], self.issues[index], self.dates[index]
    
    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        if not self.draws:
            self.load_data()
        
        all_numbers = [num for draw in self.draws for num in draw]
        
        return {
            "total_draws": len(self.draws),
            "first_issue": self.issues[0] if self.issues else None,
            "last_issue": self.issues[-1] if self.issues else None,
            "first_date": self.dates[0] if self.dates else None,
            "last_date": self.dates[-1] if self.dates else None,
            "total_numbers": len(all_numbers),
            "unique_numbers": len(set(all_numbers)),
        }


def test_data_loader():
    """Test the data loader"""
    # Find data file
    data_path = os.path.join(_ROOT, "data", "data.csv")
    
    loader = DataLoader(data_path, sequence_length=10)
    loader.load_data()
    
    stats = loader.get_statistics()
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Create sequences
    X, y = loader.create_sequences()
    print(f"\nSequence Data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    # Get latest for prediction
    latest = loader.get_latest_sequence()
    print(f"  Latest sequence shape: {latest.shape}")


if __name__ == "__main__":
    test_data_loader()
