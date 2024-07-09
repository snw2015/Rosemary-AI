import os
from pathlib import Path

from ..root import ROOT_PATH


def _get_proj_root() -> Path:
    return ROOT_PATH


def read_and_close_file_to_root(path_to_root_str: str) -> str:
    return read_and_close_file(_get_proj_root() / path_to_root_str)


def read_and_close_file(path: str | Path) -> str:
    if isinstance(path, str):
        path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise FileNotFoundError(f'File {path} not found')

    return text
