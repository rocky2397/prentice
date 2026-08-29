"""Prentice: records a desktop workflow and converts it into a Claude Code
SKILL.md. See ARCHITECTURE.md for the full pipeline design.

Model checkpoints (CLIP in Stage 2, the VLM in Stage 3) are large, and left
to Hugging Face's own default they land silently in the home-directory
cache — easy to forget about and not necessarily the disk you meant. Default
HF_HOME to a directory colocated with the repo instead, so the checkpoints
live on the same disk as the project and the location is never a surprise.
Only a default — an HF_HOME already set in the environment is left alone.
This has to run before anything in the project imports huggingface_hub
(transitively, via open_clip / mlx_vlm), since it reads HF_HOME once at
import time — which is why it's here, at the top of the package's own
__init__, guaranteed to run before any submodule import.
"""

import os
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", str(Path(__file__).resolve().parent.parent.parent / ".model_cache" / "huggingface")
)
