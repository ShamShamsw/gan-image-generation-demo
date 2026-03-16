"""storage.py – persistence helpers for the GAN demo."""

import json
import os


def ensure_directory(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, filepath: str) -> None:
    """Serialise *data* to *filepath* as pretty-printed JSON."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        ensure_directory(dirpath)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_json(filepath: str) -> dict:
    """Load and return the JSON object stored at *filepath*."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)
