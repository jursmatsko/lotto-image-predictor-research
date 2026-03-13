#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for Image-Based Lottery Prediction System
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ImageConfig:
    """Image encoding configuration"""
    height: int = 8
    width: int = 10
    channels: int = 1  # 1 for grayscale, 3 for RGB
    total_numbers: int = 80
    draw_numbers: int = 20
    
    # Heatmap settings
    use_frequency_overlay: bool = True  # Overlay historical frequency
    frequency_window: int = 50  # Window for frequency calculation
    
    # Color mapping
    present_value: float = 1.0   # Value when number is present
    absent_value: float = 0.0    # Value when number is absent
    

@dataclass
class ModelConfig:
    """Neural network model configuration"""
    # Sequence settings
    sequence_length: int = 20  # Number of past draws to use
    
    # CNN Architecture
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: Tuple[int, int] = (3, 3)
    
    # ConvLSTM settings
    lstm_filters: int = 64
    lstm_kernel: Tuple[int, int] = (3, 3)
    
    # Dense layers
    dense_units: List[int] = field(default_factory=lambda: [256, 128])
    
    # Output
    output_activation: str = "sigmoid"
    

@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Loss weights
    bce_weight: float = 1.0
    dice_weight: float = 0.5
    
    # Regularization
    dropout_rate: float = 0.3
    l2_reg: float = 0.001
    
    # Early stopping
    patience: int = 20
    min_delta: float = 0.001
    
    # Validation
    validation_split: float = 0.2
    

@dataclass
class PathConfig:
    """Path configuration"""
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            root_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.image_dir = os.path.join(root_dir, "data", "images")
        self.model_dir = os.path.join(root_dir, "models", "saved")
        self.output_dir = os.path.join(root_dir, "output")
        
        # Create directories
        for d in [self.data_dir, self.image_dir, self.model_dir, self.output_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass 
class AppConfig:
    """Main application configuration"""
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Data source
    source_data_file: str = "../data/data.csv"
    
    def __post_init__(self):
        if isinstance(self.paths, type):
            self.paths = PathConfig()
