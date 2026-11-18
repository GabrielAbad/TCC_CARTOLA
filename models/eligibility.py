"""Shared athlete eligibility rules for prediction evaluation and lineup selection."""

from __future__ import annotations

PROBABLE_STATUS_ID = 7
STATUS_CANDIDATOS: frozenset[int] = frozenset({PROBABLE_STATUS_ID})
