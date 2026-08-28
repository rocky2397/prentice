"""Accessibility-tree lookups via macOS's AX API (through PyObjC).

This is a supplementary signal per ARCHITECTURE.md ("where available: the
accessibility-tree element under the cursor") — never required for capture
to function. Every lookup here is best-effort: it returns ``None`` rather
than raising if the AX API is unavailable, Accessibility permission hasn't
been granted, or no element resolves at the given point.
"""

from __future__ import annotations

from typing import Any

try:
    from ApplicationServices import (
        AXUIElementCopyAttributeValue,
        AXUIElementCopyElementAtPosition,
        AXUIElementCreateSystemWide,
        kAXDescriptionAttribute,
        kAXRoleAttribute,
        kAXTitleAttribute,
        kAXValueAttribute,
    )

    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


def describe_element_at(x: float, y: float) -> dict[str, Any] | None:
    """Best-effort description of the accessibility element under (x, y) in screen coords."""
    if not _AX_AVAILABLE:
        return None
    try:
        system_wide = AXUIElementCreateSystemWide()
        err, element = AXUIElementCopyElementAtPosition(system_wide, x, y, None)
        if err != 0 or element is None:
            return None
        return {
            "role": _get_attr(element, kAXRoleAttribute),
            "title": _get_attr(element, kAXTitleAttribute),
            "description": _get_attr(element, kAXDescriptionAttribute),
            "value": _get_attr(element, kAXValueAttribute),
        }
    except Exception:
        return None


def _get_attr(element: Any, attribute: str) -> str | None:
    try:
        err, value = AXUIElementCopyAttributeValue(element, attribute, None)
        if err != 0 or value is None:
            return None
        return str(value)
    except Exception:
        return None
