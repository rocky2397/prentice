"""Small shared helper for parsing LLM/VLM text responses that are expected
to be JSON, possibly wrapped in a markdown code fence (some models wrap
structured output in ```json ... ``` even when explicitly told not to).
"""

from __future__ import annotations


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text
