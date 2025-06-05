"""
Moneypatch Attention - TPU-optimized attention mechanisms

This package provides drop-in replacements for standard attention mechanisms
with TPU-optimized implementations including Splash Attention and XLA Flash Attention.

Available wrappers:
- SplashAttentionWrapper: Standard Splash Attention
- SplashAttentionQKNormWrapper: Splash Attention with QK normalization
- XLAFlashAttentionWrapper: Standard XLA Flash Attention  
- XLAFlashAttentionQKNormWrapper: XLA Flash Attention with QK normalization
"""

from .splash_wrappers import (
    SplashAttentionWrapper,
    SplashAttentionQKNormWrapper,
    SPLASH_ATTENTION_AVAILABLE,
)

from .flash_wrappers import (
    XLAFlashAttentionWrapper, 
    XLAFlashAttentionQKNormWrapper,
    FLASH_ATTENTION_AVAILABLE,
    repeat_kv,
)

__all__ = [
    "SplashAttentionWrapper",
    "SplashAttentionQKNormWrapper", 
    "XLAFlashAttentionWrapper",
    "XLAFlashAttentionQKNormWrapper",
    "SPLASH_ATTENTION_AVAILABLE",
    "FLASH_ATTENTION_AVAILABLE",
    "repeat_kv",
]

__version__ = "0.1.0"