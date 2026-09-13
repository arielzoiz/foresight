"""Adapter registry.

Config names an adapter by string; an unknown name must fail at startup rather
than at the first request. Milestones 3-4 each add one import and one entry.
"""

from __future__ import annotations

from .base import AdapterOptions, AuxResult, CallerAdapter
from .generic import GenericAdapter
from .local import LocalAdapter

ADAPTERS: dict[str, type[CallerAdapter]] = {
    GenericAdapter.name: GenericAdapter,
    LocalAdapter.name: LocalAdapter,
}

__all__ = [
    "ADAPTERS",
    "AdapterOptions",
    "AuxResult",
    "CallerAdapter",
    "GenericAdapter",
    "LocalAdapter",
]
