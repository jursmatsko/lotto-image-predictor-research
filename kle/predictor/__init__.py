"""
kle.predictor — GAN + Transformer prediction pipeline for KL8 lottery.

Main entry points
-----------------
from kle.predictor.pipeline import GANTransformerPipeline
pipeline = GANTransformerPipeline()
pipeline.walk_forward(issues, draws, target_issue)
sets = pipeline.predict(draws, n_sets=20)
"""
from .pipeline import GANTransformerPipeline
from .memory import MemoryBank, ExperienceReplay
from .transformer import SignalTransformerEncoder
from .gan import SetGenerator, SetDiscriminator
from .scorer import SetScorer

__all__ = [
    "GANTransformerPipeline",
    "MemoryBank",
    "ExperienceReplay",
    "SignalTransformerEncoder",
    "SetGenerator",
    "SetDiscriminator",
    "SetScorer",
]
