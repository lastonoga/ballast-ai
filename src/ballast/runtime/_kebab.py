"""Shared kebab-case helper."""
from __future__ import annotations

import re


def kebab_case(name: str) -> str:
    """``BrainstormFlow`` → ``brainstorm-flow``; ``MyXMLFlow`` → ``my-xml-flow``."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1)
    return s2.lower()


__all__ = ["kebab_case"]
