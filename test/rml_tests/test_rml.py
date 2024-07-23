"""
Tests for RML code parsing
"""
import os
from pathlib import Path

import pytest

from src.rosemary_ai.rosemary import _build, Rosemary


def _path(path: str) -> str:
    return str(Path(os.path.abspath(__file__)).parent / path)


def test_not_found():
    with pytest.raises(FileNotFoundError):
        rosemary: Rosemary = _build(_path('not_exist.rml'))


def test_empty():
    rosemary: Rosemary = _build(_path('empty.rml'))
    assert rosemary is not None
    assert len(rosemary.namespace) == 0
