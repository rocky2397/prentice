"""Shared JSON/JSONL loading helpers for pydantic-validated per-stage output
files (``events.jsonl``, ``session.json``, ``segments.jsonl``,
``segment_meta.json``) — used by both capture/schema.py and segment/schema.py
so the parse-one-object-per-line and parse-whole-file patterns exist once.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import TypeAdapter

T = TypeVar("T")


def load_json(path: str, adapter: TypeAdapter[T]) -> T:
    with open(path, encoding="utf-8") as f:
        return adapter.validate_json(f.read())


def load_jsonl(path: str, adapter: TypeAdapter[T]) -> list[T]:
    items: list[T] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(adapter.validate_json(line))
    return items
