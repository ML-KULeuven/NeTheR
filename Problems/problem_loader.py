"""Helpers for loading the saved benchmark problem definitions."""

from __future__ import annotations

import json
from pathlib import Path


PROBLEMS_DIR = Path(__file__).resolve().parent


def load_problem(dataset: str, filename: str) -> dict:
    """Load one saved expert-theory problem from the repository.

    Parameters
    ----------
    dataset:
        Dataset directory below ``Problems`` (for example ``"animals"``).
    filename:
        JSON filename below that dataset's ``expert_theories`` directory.
    """
    path = PROBLEMS_DIR / dataset / "expert_theories" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Problem definition not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)
