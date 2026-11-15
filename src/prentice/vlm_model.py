"""Shared mlx-vlm model loading, cached — reused by Stage 3 (Interpret, with
keyframe images) and Stage 4 (Refine, text-only, reasoning over Stage 3's
structured output instead of pixels) so the same checkpoint loads once per
process regardless of which stage calls it, and neither stage re-implements
the loading/caching logic.
"""

from __future__ import annotations

import os
from functools import lru_cache

from mlx_vlm import load
from mlx_vlm.utils import load_config


def resolve_model(model_name: str) -> str:
    """The local copy of a model when one is cached, else its name (which downloads it).

    Loading by name checks the Hub for a newer revision every time; when one
    appeared, a run silently re-downloaded the whole 13 GB model first. Set
    PRENTICE_UPDATE_MODEL=1 to check for (and fetch) a newer revision.
    """
    if os.environ.get("PRENTICE_UPDATE_MODEL") == "1":
        return model_name
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:  # not cached yet: the first run downloads it
        return model_name


@lru_cache(maxsize=2)
def load_vlm_model(model_name: str):
    """Cached so a batch run over many sessions loads the checkpoint once."""
    path = resolve_model(model_name)
    model, processor = load(path)
    config = load_config(path)
    return model, processor, config
