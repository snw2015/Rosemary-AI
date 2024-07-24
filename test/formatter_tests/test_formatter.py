"""
Tests for formatters of petals
"""
import os
from pathlib import Path

import pytest

from src.rosemary_ai.rosemary import _build, Rosemary


def _path(path: str) -> str:
    return str(Path(os.path.abspath(__file__)).parent / path)


@pytest.fixture(scope='module')
def simple_rml() -> Rosemary:
    return _build(_path('simple.rml'))


def test_empty_formatter(simple_rml):
    format = simple_rml.get_formatter('empty_formatter')

    assert format() is None


def test_fixed_str_formatter(simple_rml):
    format = simple_rml.get_formatter('fixed_str')

    assert format() == 'fixed'
