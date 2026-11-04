"""Discovery and loading of frame files.

The whole pipeline is built around one idea: frames live in a single flat
directory and are picked up by a glob, so dropping more frames into it (and
re-running the CLI) extends the sequence without touching any code.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

_NUMBER_RE = re.compile(r"(\d+)")


def natural_key(path: Path) -> tuple:
    """Sort key that orders ``frame_2`` before ``frame_10``."""
    parts: list[tuple[int, object]] = []
    for chunk in _NUMBER_RE.split(path.stem):
        parts.append((0, int(chunk)) if chunk.isdigit() else (1, chunk.lower()))
    return tuple(parts)


def discover_frames(directory: str | Path) -> list[Path]:
    """Return frame files in *directory*, sorted in natural (human) order."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"no such directory: {directory}")
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=natural_key)


def load_gray(path: str | Path, scale: float = 1.0) -> np.ndarray:
    """Load a frame as a float32 grayscale image, optionally downscaled.

    ``scale`` is the resize factor (0 < scale <= 1). Flow vectors are always
    reported in *original* pixel units, so downscaling does not change the
    physical meaning of the result.
    """
    if not 0 < scale <= 1:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return image.astype(np.float32) / 255.0


def load_frames(
    directory: str | Path, scale: float = 1.0
) -> tuple[list[Path], list[np.ndarray]]:
    """Load every frame of a directory as grayscale."""
    paths = discover_frames(directory)
    return paths, [load_gray(p, scale) for p in paths]


def frame_size(directory: str | Path) -> tuple[int, int] | None:
    """(width, height) of the first frame, or None if the directory is empty."""
    for path in discover_frames(directory):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            height, width = image.shape[:2]
            return width, height
    return None
