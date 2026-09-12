"""Adapter registry.

Config names an adapter by string; an unknown name must fail at startup rather
than at the first request. Milestones 2-4 each add one import and one entry.
"""

from __future__ import annotations

from .base import AuxResult, CallerAdapter
from .generic import GenericAdapter

ADAPTERS: dict[str, type[CallerAdapter]] = {
    GenericAdapter.name: GenericAdapter,
}

__all__ = ["ADAPTERS", "AuxResult", "CallerAdapter", "GenericAdapter"]
