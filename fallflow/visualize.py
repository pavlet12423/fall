"""Colour-coding and overlays so a flow field can actually be looked at."""

from __future__ import annotations

import cv2
import numpy as np

from .flow import polar


def flow_to_bgr(
    flow: np.ndarray, max_magnitude: float | None = None, brightness: int = 255
) -> np.ndarray:
    """Standard HSV colour wheel: hue = direction, value = magnitude.

    If ``max_magnitude`` is None the field is normalised to its own maximum,
    which makes every image colourful but hides absolute speed differences.
    Pass a fixed value to compare frames against each other.
    """
    magnitude, angle = polar(flow)
    value = magnitude / max_magnitude if max_magnitude else cv2.normalize(
        magnitude, None, 0, 1, cv2.NORM_MINMAX
    )
    value = np.clip(value, 0, 1) * (brightness / 255.0)

    hsv = np.zeros((*magnitude.shape, 3), np.uint8)
    hsv[..., 0] = np.rint(np.degrees(angle) / 2.0).astype(np.uint8)  # hue 0..179
    hsv[..., 1] = 255
    hsv[..., 2] = np.rint(value * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(
    image: np.ndarray,
    flow: np.ndarray,
    step: int = 24,
    threshold: float = 0.5,
    scale: float = 1.0,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Sparse arrow grid over a frame; arrows shorter than *threshold* are dropped."""
    if image.ndim == 2:
        image = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2BGR)
    else:
        image = _to_uint8(image).copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if np.hypot(dx, dy) < threshold:
                continue
            end = (int(round(x + dx * scale)), int(round(y + dy * scale)))
            cv2.arrowedLine(image, (x, y), end, color, 1, cv2.LINE_AA, tipLength=0.3)
    return image


def side_by_side(images: list[np.ndarray], pad: int = 8) -> np.ndarray:
    """Horizontal stack of images, padded to a common height."""
    images = [_to_uint8(im) for im in images]
    height = max(im.shape[0] for im in images)
    resized = []
    for im in images:
        if im.shape[0] != height:
            k = height / im.shape[0]
            im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        resized.append(im)
    canvas = np.full(
        (height, sum(im.shape[1] for im in resized) + pad * (len(resized) - 1), 3),
        255,
        np.uint8,
    )
    x = 0
    for im in resized:
        canvas[:, x : x + im.shape[1]] = im
        x += im.shape[1] + pad
    return canvas


def montage(
    images: list[np.ndarray],
    cols: int = 5,
    cell: tuple[int, int] = (320, 180),
    labels: list[str] | None = None,
    pad: int = 4,
    label_height: int = 18,
) -> np.ndarray:
    """Contact sheet: a labelled grid of images, handy for a whole sequence."""
    if not images:
        raise ValueError("no images to montage")
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    cw, ch = cell
    tile_h = ch + (label_height if labels else 0)
    canvas = np.full(
        (rows * (tile_h + pad) + pad, cols * (cw + pad) + pad, 3), 40, np.uint8
    )
    for idx, image in enumerate(images):
        im = _to_uint8(image)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (cw, ch), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        y = pad + r * (tile_h + pad)
        x = pad + c * (cw + pad)
        canvas[y : y + ch, x : x + cw] = im
        if labels:
            cv2.putText(
                canvas,
                str(labels[idx]),
                (x + 2, y + ch + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image, copy=False)
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)
