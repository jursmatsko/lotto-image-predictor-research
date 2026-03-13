#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for Image-Based Lottery Prediction

Creates heatmap visualizations, training curves, and prediction displays.
"""

import os
from typing import List, Optional, Tuple
import numpy as np


class Visualizer:
    """
    Visualization tools for lottery image prediction.
    
    Can work without matplotlib by providing ASCII alternatives.
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if matplotlib is available
        self.has_matplotlib = False
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            self.plt = plt
            self.has_matplotlib = True
        except ImportError:
            self.plt = None
    
    def draw_to_ascii(
        self,
        numbers: List[int],
        width: int = 10,
        height: int = 8,
    ) -> str:
        """
        Convert draw numbers to ASCII grid.
        
        Convention:
        - ██ (dark/filled) = Number is present (1)
        - ·· (empty/light) = Number is absent (0)
        
        Args:
            numbers: List of drawn numbers
            width: Grid width (default 10)
            height: Grid height (default 8)
        
        Returns:
            ASCII string representation
        """
        number_set = set(numbers)
        lines = []
        
        # Header
        header = "    " + " ".join(f"{i+1:2d}" for i in range(width))
        lines.append(header)
        lines.append("    " + "-" * (width * 3 - 1))
        
        for r in range(height):
            row_start = r * width + 1
            row_end = row_start + width - 1
            line = f"{row_start:2d}-{row_end:2d}|"
            
            for c in range(width):
                num = r * width + c + 1
                if num in number_set:
                    line += " ██"  # Black = 1 (present)
                else:
                    line += " ··"  # White/empty = 0 (absent)
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def heatmap_to_ascii(
        self,
        matrix: np.ndarray,
        show_values: bool = False,
    ) -> str:
        """
        Convert probability matrix to ASCII heatmap.
        
        Args:
            matrix: 2D array of probabilities
            show_values: Whether to show numeric values
        
        Returns:
            ASCII heatmap string
        """
        if matrix.ndim > 2:
            matrix = matrix.mean(axis=-1)
        
        height, width = matrix.shape
        
        # Intensity characters
        chars = " ░▒▓█"
        
        lines = []
        
        # Header
        header = "    " + " ".join(f"{i+1:2d}" for i in range(width))
        lines.append(header)
        lines.append("    " + "-" * (width * 3 - 1))
        
        for r in range(height):
            row_start = r * width + 1
            row_end = row_start + width - 1
            line = f"{row_start:2d}-{row_end:2d}|"
            
            for c in range(width):
                val = matrix[r, c]
                char_idx = min(int(val * len(chars)), len(chars) - 1)
                
                if show_values:
                    line += f"{val:.1f}"
                else:
                    line += f" {chars[char_idx]}{chars[char_idx]}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def compare_prediction_ascii(
        self,
        predicted: List[int],
        actual: List[int],
        width: int = 10,
        height: int = 8,
    ) -> str:
        """
        Compare predicted vs actual numbers in ASCII.
        
        Legend:
        - ██ : Hit (predicted and actual)
        - ▓▓ : Miss (predicted but not actual)
        - ░░ : Not predicted, not actual
        - ▒▒ : Not predicted, but actual (missed)
        """
        pred_set = set(predicted)
        actual_set = set(actual)
        
        lines = []
        lines.append("Legend: ██=Hit  ▓▓=FalsePos  ▒▒=FalseNeg  ░░=TrueNeg")
        lines.append("")
        
        header = "    " + " ".join(f"{i+1:2d}" for i in range(width))
        lines.append(header)
        lines.append("    " + "-" * (width * 3 - 1))
        
        hits = 0
        false_pos = 0
        false_neg = 0
        
        for r in range(height):
            row_start = r * width + 1
            row_end = row_start + width - 1
            line = f"{row_start:2d}-{row_end:2d}|"
            
            for c in range(width):
                num = r * width + c + 1
                in_pred = num in pred_set
                in_actual = num in actual_set
                
                if in_pred and in_actual:
                    line += " ██"
                    hits += 1
                elif in_pred and not in_actual:
                    line += " ▓▓"
                    false_pos += 1
                elif not in_pred and in_actual:
                    line += " ▒▒"
                    false_neg += 1
                else:
                    line += " ░░"
            
            lines.append(line)
        
        lines.append("")
        lines.append(f"Statistics: Hits={hits}/20  FalsePos={false_pos}  FalseNeg={false_neg}")
        
        return "\n".join(lines)
    
    def save_heatmap_image(
        self,
        matrix: np.ndarray,
        filename: str,
        title: str = "Lottery Heatmap",
        cmap: str = "gray_r",
        invert: bool = True,
    ) -> Optional[str]:
        """
        Save matrix as heatmap image (requires matplotlib).
        
        Convention:
        - Black (1) = Number is present/selected
        - White (0) = Number is absent/not selected
        
        Args:
            matrix: 2D probability matrix
            filename: Output filename
            title: Plot title
            cmap: Colormap name (gray_r = black=high, white=low)
            invert: If True, use black=1, white=0 convention
        
        Returns:
            Path to saved file or None if matplotlib unavailable
        """
        if not self.has_matplotlib:
            print("Warning: matplotlib not available, cannot save image")
            return None
        
        if matrix.ndim > 2:
            matrix = matrix.mean(axis=-1)
        
        fig, ax = self.plt.subplots(figsize=(12, 8))
        
        # Use gray_r colormap: high values = black, low values = white
        # This matches convention: 1 (present) = black, 0 (absent) = white
        display_matrix = matrix if invert else (1 - matrix)
        im = ax.imshow(display_matrix, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        
        # Add colorbar with correct labels
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value (Black=1, White=0)", rotation=-90, va="bottom")
        
        # Set labels
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(8))
        ax.set_xticklabels([str(i+1) for i in range(10)])
        ax.set_yticklabels([f"{i*10+1}-{i*10+10}" for i in range(8)])
        
        # Add grid
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
        
        # Add number annotations
        for i in range(8):
            for j in range(10):
                num = i * 10 + j + 1
                val = matrix[i, j]
                # White text on black cells, black text on white cells
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, str(num), ha="center", va="center", 
                       color=color, fontsize=8, fontweight='bold')
        
        ax.set_title(title)
        
        # Save
        filepath = os.path.join(self.output_dir, filename)
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        return filepath
    
    def save_sequence_strip(
        self,
        images: np.ndarray,
        filename: str,
        title: str = "Draw Sequence",
    ) -> Optional[str]:
        """
        Save a sequence of draw images as a strip (requires matplotlib).
        
        Args:
            images: Array of shape (n_images, height, width)
            filename: Output filename
            title: Plot title
        
        Returns:
            Path to saved file or None
        """
        if not self.has_matplotlib:
            return None
        
        if images.ndim > 3:
            images = images.mean(axis=-1)
        
        n_images = len(images)
        fig, axes = self.plt.subplots(1, n_images, figsize=(n_images * 2, 3))
        
        if n_images == 1:
            axes = [axes]
        
        for i, (ax, img) in enumerate(zip(axes, images)):
            ax.imshow(img, cmap='hot', aspect='equal', vmin=0, vmax=1)
            ax.set_title(f"t-{n_images-i-1}" if i < n_images - 1 else "Current")
            ax.axis('off')
        
        fig.suptitle(title)
        
        filepath = os.path.join(self.output_dir, filename)
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        return filepath
    
    def plot_training_history(
        self,
        history: dict,
        filename: str = "training_history.png",
    ) -> Optional[str]:
        """
        Plot training loss/accuracy curves (requires matplotlib).
        
        Args:
            history: Dict with 'loss', 'val_loss', etc.
            filename: Output filename
        
        Returns:
            Path to saved file or None
        """
        if not self.has_matplotlib:
            return None
        
        fig, axes = self.plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy or other metrics
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Acc')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Val Acc')
        if 'hit_rate' in history:
            axes[1].plot(history['hit_rate'], label='Hit Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training Metrics')
        axes[1].legend()
        axes[1].grid(True)
        
        filepath = os.path.join(self.output_dir, filename)
        self.plt.savefig(filepath, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        return filepath
    
    def print_prediction_report(
        self,
        predicted: List[int],
        probabilities: np.ndarray,
        actual: Optional[List[int]] = None,
    ) -> str:
        """
        Generate a formatted prediction report.
        
        Args:
            predicted: Predicted numbers
            probabilities: Probability matrix
            actual: Optional actual numbers for comparison
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PREDICTION REPORT")
        lines.append("=" * 70)
        
        # Top predictions
        lines.append("\n🎯 Predicted Numbers (Top 20):")
        lines.append(" ".join(f"{n:02d}" for n in predicted))
        
        # Probability ranking
        if probabilities.ndim > 2:
            probs = probabilities.mean(axis=-1)
        else:
            probs = probabilities
        
        flat_probs = probs.flatten()
        sorted_idx = np.argsort(flat_probs)[::-1]
        
        lines.append("\n📊 Probability Ranking (Top 30):")
        for rank, idx in enumerate(sorted_idx[:30], 1):
            num = idx + 1
            prob = flat_probs[idx]
            bar = "█" * int(prob * 20)
            mark = "✓" if num in predicted else " "
            lines.append(f"  {rank:2d}. [{mark}] {num:02d}: {prob:.3f} {bar}")
        
        # Heatmap
        lines.append("\n📈 Probability Heatmap:")
        lines.append(self.heatmap_to_ascii(probs))
        
        # Comparison with actual (if provided)
        if actual is not None:
            lines.append("\n🎲 Comparison with Actual:")
            lines.append(self.compare_prediction_ascii(predicted, actual))
            
            hits = len(set(predicted) & set(actual))
            lines.append(f"\n🎯 Hit Rate: {hits}/20 ({hits/20*100:.1f}%)")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


def test_visualizer():
    """Test the visualizer"""
    viz = Visualizer()
    
    # Test ASCII drawing
    draw = [1, 5, 12, 23, 34, 45, 56, 67, 78, 80,
            2, 15, 28, 39, 41, 52, 63, 74, 77, 79]
    
    print("Draw as ASCII:")
    print(viz.draw_to_ascii(draw))
    
    # Test heatmap
    probs = np.random.rand(8, 10)
    print("\nRandom Heatmap:")
    print(viz.heatmap_to_ascii(probs))
    
    # Test comparison
    predicted = [1, 5, 12, 23, 34, 45, 56, 67, 78, 80,
                 3, 16, 29, 40, 42, 53, 64, 75, 76, 77]
    actual = [1, 5, 12, 23, 34, 45, 56, 67, 78, 80,
              2, 15, 28, 39, 41, 52, 63, 74, 77, 79]
    
    print("\nPrediction vs Actual:")
    print(viz.compare_prediction_ascii(predicted, actual))


if __name__ == "__main__":
    test_visualizer()
