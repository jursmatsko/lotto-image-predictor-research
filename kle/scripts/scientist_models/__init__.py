"""
Scientist Models — Generative Memory-Augmented Lottery Prediction
================================================================
Modules:
  signals    : 12 independent signal providers (scoring methods)
  memory     : MemoryBank + ExperienceReplay
  generator  : GenerativeModel (fused distribution → cover sets)
  predict    : CLI entry point (walk-forward, predict, extreme search)
"""
from .signals import ALL_SIGNAL_PROVIDERS
from .memory import MemoryBank, ExperienceReplay
from .generator import GenerativeModel
from .constants import PICK, TOTAL, DRAW_SIZE
