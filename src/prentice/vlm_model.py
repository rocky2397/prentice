"""Shared mlx-vlm model loading, cached — reused by Stage 3 (Interpret, with
keyframe images) and Stage 4 (Refine, text-only, reasoning over Stage 3's
structured output instead of pixels) so the same checkpoint loads once per
process regardless of which stage calls it, and neither stage re-implements
the loading/caching logic.
"""

from __future__ import annotations

from functools import lru_cache

from mlx_vlm import load
from mlx_vlm.utils import load_config


@lru_cache(maxsize=2)
def load_vlm_model(model_name: str):
    """Cached so a batch run over many sessions loads the checkpoint once."""
    model, processor = load(model_name)
    config = load_config(model_name)
    return model, processor, config
