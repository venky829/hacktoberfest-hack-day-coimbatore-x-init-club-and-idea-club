"""
NutriLens Model Harness Package
Provides hybrid orchestration between local edge models (RTX 3050) and cloud Gemini API.
"""

from .engine import HybridInferenceHarness, ExtractedNutritionData
from .gemini_client import GeminiGemmaClient

__all__ = [
    "HybridInferenceHarness",
    "ExtractedNutritionData",
    "GeminiGemmaClient",
]