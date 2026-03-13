#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Encoder/Decoder for Lottery Draws

Converts lottery draws to/from 8×10 pixel heatmap images.

Number Layout in 8×10 Matrix:
┌────────────────────────────────────────────┐
│  01  02  03  04  05  06  07  08  09  10    │  Row 0
│  11  12  13  14  15  16  17  18  19  20    │  Row 1
│  21  22  23  24  25  26  27  28  29  30    │  Row 2
│  31  32  33  34  35  36  37  38  39  40    │  Row 3
│  41  42  43  44  45  46  47  48  49  50    │  Row 4
│  51  52  53  54  55  56  57  58  59  60    │  Row 5
│  61  62  63  64  65  66  67  68  69  70    │  Row 6
│  71  72  73  74  75  76  77  78  79  80    │  Row 7
└────────────────────────────────────────────┘
"""

from typing import List, Tuple, Optional
import numpy as np


class ImageEncoder:
    """
    Encodes lottery draws as 8×10 pixel images and decodes back.
    
    Encoding Modes:
    1. Binary: 1 if number present, 0 otherwise
    2. Frequency: Value based on historical frequency
    3. Heatmap: Gradient based on recency and frequency
    """
    
    def __init__(
        self,
        height: int = 8,
        width: int = 10,
        channels: int = 1,
        present_value: float = 1.0,
        absent_value: float = 0.0,
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.total_numbers = height * width
        self.present_value = present_value
        self.absent_value = absent_value
    
    def number_to_position(self, number: int) -> Tuple[int, int]:
        """Convert number (1-80) to (row, col) position."""
        if not 1 <= number <= self.total_numbers:
            raise ValueError(f"Number must be between 1 and {self.total_numbers}")
        idx = number - 1
        row = idx // self.width
        col = idx % self.width
        return row, col
    
    def position_to_number(self, row: int, col: int) -> int:
        """Convert (row, col) position to number (1-80)."""
        return row * self.width + col + 1
    
    def encode_single(
        self,
        numbers: List[int],
        mode: str = "binary",
        frequency_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Encode a single draw to an image.
        
        Args:
            numbers: List of drawn numbers (typically 20)
            mode: "binary", "frequency", or "heatmap"
            frequency_map: Optional (8, 10) array of historical frequencies
        
        Returns:
            Image array of shape (height, width) or (height, width, channels)
        """
        image = np.full((self.height, self.width), self.absent_value, dtype=np.float32)
        
        for num in numbers:
            if 1 <= num <= self.total_numbers:
                row, col = self.number_to_position(num)
                if mode == "binary":
                    image[row, col] = self.present_value
                elif mode == "frequency" and frequency_map is not None:
                    image[row, col] = frequency_map[row, col]
                elif mode == "heatmap" and frequency_map is not None:
                    # Combine presence with frequency
                    image[row, col] = 0.7 * self.present_value + 0.3 * frequency_map[row, col]
                else:
                    image[row, col] = self.present_value
        
        if self.channels > 1:
            # Expand to multi-channel
            image = np.stack([image] * self.channels, axis=-1)
        
        return image
    
    def encode_batch(
        self,
        draws: List[List[int]],
        mode: str = "binary",
        compute_running_frequency: bool = True,
        frequency_window: int = 50,
    ) -> np.ndarray:
        """
        Encode multiple draws to a batch of images.
        
        Args:
            draws: List of draws, each draw is a list of numbers
            mode: Encoding mode
            compute_running_frequency: Whether to compute running frequency
            frequency_window: Window size for frequency calculation
        
        Returns:
            Array of shape (n_draws, height, width) or (n_draws, height, width, channels)
        """
        n_draws = len(draws)
        
        if self.channels == 1:
            images = np.zeros((n_draws, self.height, self.width), dtype=np.float32)
        else:
            images = np.zeros((n_draws, self.height, self.width, self.channels), dtype=np.float32)
        
        for i, draw in enumerate(draws):
            freq_map = None
            if compute_running_frequency and i > 0:
                start_idx = max(0, i - frequency_window)
                freq_map = self.compute_frequency_map(draws[start_idx:i])
            
            images[i] = self.encode_single(draw, mode=mode, frequency_map=freq_map)
        
        return images
    
    def decode_single(
        self,
        image: np.ndarray,
        top_k: int = 20,
        threshold: Optional[float] = None,
    ) -> List[int]:
        """
        Decode an image back to lottery numbers.
        
        Args:
            image: Image array
            top_k: Number of positions to select (default 20)
            threshold: Optional threshold for selection
        
        Returns:
            List of predicted numbers
        """
        if image.ndim == 3:
            # Take mean across channels
            image = image.mean(axis=-1)
        
        flat = image.flatten()
        
        if threshold is not None:
            # Select all positions above threshold
            indices = np.where(flat >= threshold)[0]
            if len(indices) > top_k:
                # Sort by value and take top_k
                sorted_indices = indices[np.argsort(flat[indices])[::-1]]
                indices = sorted_indices[:top_k]
        else:
            # Select top_k positions by value
            indices = np.argsort(flat)[::-1][:top_k]
        
        numbers = sorted([int(idx + 1) for idx in indices])
        return numbers
    
    def compute_frequency_map(
        self,
        draws: List[List[int]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute frequency map from historical draws.
        
        Args:
            draws: List of historical draws
            normalize: Whether to normalize to [0, 1]
        
        Returns:
            Frequency map of shape (height, width)
        """
        freq_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        for draw in draws:
            for num in draw:
                if 1 <= num <= self.total_numbers:
                    row, col = self.number_to_position(num)
                    freq_map[row, col] += 1
        
        if normalize and len(draws) > 0:
            freq_map = freq_map / len(draws)
        
        return freq_map
    
    def compute_gap_map(
        self,
        draws: List[List[int]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute gap map (periods since last appearance) for each number.
        
        Args:
            draws: List of historical draws (oldest first)
            normalize: Whether to normalize
        
        Returns:
            Gap map of shape (height, width)
        """
        gap_map = np.zeros((self.height, self.width), dtype=np.float32)
        last_seen = {}
        
        for i, draw in enumerate(draws):
            for num in draw:
                last_seen[num] = i
        
        total = len(draws)
        for num in range(1, self.total_numbers + 1):
            row, col = self.number_to_position(num)
            if num in last_seen:
                gap = total - last_seen[num] - 1
            else:
                gap = total
            gap_map[row, col] = gap
        
        if normalize and total > 0:
            gap_map = gap_map / total
        
        return gap_map
    
    def create_multi_channel_image(
        self,
        numbers: List[int],
        draws_history: List[List[int]],
        frequency_window: int = 50,
    ) -> np.ndarray:
        """
        Create a multi-channel image with different feature maps.
        
        Channel 0: Binary presence (current draw)
        Channel 1: Historical frequency
        Channel 2: Gap (recency)
        
        Args:
            numbers: Current draw numbers
            draws_history: Historical draws for computing frequency/gap
            frequency_window: Window for frequency calculation
        
        Returns:
            Image of shape (height, width, 3)
        """
        # Channel 0: Binary
        binary = self.encode_single(numbers, mode="binary")
        
        # Channel 1: Frequency
        recent_draws = draws_history[-frequency_window:] if len(draws_history) > 0 else []
        freq_map = self.compute_frequency_map(recent_draws, normalize=True)
        
        # Channel 2: Gap
        gap_map = self.compute_gap_map(draws_history, normalize=True)
        
        # Stack channels
        image = np.stack([binary, freq_map, gap_map], axis=-1)
        
        return image
    
    def image_to_ascii(self, image: np.ndarray, threshold: float = 0.5) -> str:
        """
        Convert image to ASCII art for terminal display.
        
        Convention:
        - ██ (dark) = 1 (number present/high probability)
        - ·· (light) = 0 (number absent/low probability)
        
        Args:
            image: Image array
            threshold: Threshold for "on" pixels
        
        Returns:
            ASCII string representation
        """
        if image.ndim == 3:
            image = image.mean(axis=-1)
        
        lines = []
        lines.append("    " + " ".join(f"{i+1:2d}" for i in range(self.width)))
        
        for r in range(self.height):
            row_start = r * self.width + 1
            row_end = row_start + self.width - 1
            line = f"{row_start:2d}-{row_end:2d} "
            for c in range(self.width):
                if image[r, c] >= threshold:
                    line += " ██"  # Black = 1 (present)
                else:
                    line += " ··"  # White = 0 (absent)
            lines.append(line)
        
        return "\n".join(lines)
    
    def image_to_heatmap_ascii(self, image: np.ndarray) -> str:
        """
        Convert image to heatmap ASCII art with intensity levels.
        
        Args:
            image: Image array with continuous values
        
        Returns:
            ASCII string with intensity levels
        """
        if image.ndim == 3:
            image = image.mean(axis=-1)
        
        # Intensity levels
        levels = " ░▒▓█"
        
        lines = []
        lines.append("    " + " ".join(f"{i+1:2d}" for i in range(self.width)))
        
        for r in range(self.height):
            row_start = r * self.width + 1
            row_end = row_start + self.width - 1
            line = f"{row_start:2d}-{row_end:2d} "
            for c in range(self.width):
                val = image[r, c]
                level_idx = min(int(val * len(levels)), len(levels) - 1)
                char = levels[level_idx]
                line += f" {char}{char}"
            lines.append(line)
        
        return "\n".join(lines)


def test_encoder():
    """Test the image encoder"""
    encoder = ImageEncoder()
    
    # Test single draw
    draw = [1, 5, 12, 23, 34, 45, 56, 67, 78, 80, 
            2, 15, 28, 39, 41, 52, 63, 74, 77, 79]
    
    image = encoder.encode_single(draw)
    print("Binary Image (8×10):")
    print(encoder.image_to_ascii(image))
    
    # Decode back
    decoded = encoder.decode_single(image, top_k=20)
    print(f"\nOriginal: {sorted(draw)}")
    print(f"Decoded:  {decoded}")
    print(f"Match: {set(draw) == set(decoded)}")


if __name__ == "__main__":
    test_encoder()
