"""Optical flow computation, serialisation and motion statistics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

METHODS = ("farneback", "dis")


@dataclass(frozen=True)
class FlowConfig:
    """Parameters of the dense optical flow estimator.

    ``scale`` downsamples frames before matching (speed vs. detail). Output
    vectors are rescaled back, so flow is always in original-pixel units.
    """

    method: str = "farneback"
    scale: float = 1.0
    blur: int = 0  # gaussian kernel size before matching (0 = off, use odd numbers)

    # Farneback
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

    # DIS
    preset: str = "medium"  # ultrafast | fast | medium

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}, choose from {METHODS}")
        if not 0 < self.scale <= 1:
            raise ValueError(f"scale must be in (0, 1], got {self.scale}")
        if self.preset not in ("ultrafast", "fast", "medium"):
            raise ValueError(f"unknown DIS preset {self.preset!r}")

    def with_scale(self, scale: float) -> "FlowConfig":
        return replace(self, scale=scale)


def _dis_preset_code(preset: str) -> int:
    return {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }[preset]


def _create_dis(cfg: FlowConfig):
    dis = cv2.DISOpticalFlow_create(_dis_preset_code(cfg.preset))
    dis.setUseSpatialPropagation(True)
    return dis


def compute_flow(
    prev: np.ndarray, next_frame: np.ndarray, cfg: FlowConfig | None = None
) -> np.ndarray:
    """Dense flow from *prev* to *next_frame* as ``float32[H, W, 2]``.

    Channel 0 is horizontal displacement (x), channel 1 vertical (y), in
    pixels of the original frames.
    """
    cfg = cfg or FlowConfig()
    # both estimators need 8-bit single-channel input
    prev_u8 = _to_uint8(prev)
    next_u8 = _to_uint8(next_frame)
    if cfg.blur:
        k = cfg.blur | 1  # force odd
        prev_u8 = cv2.GaussianBlur(prev_u8, (k, k), 0)
        next_u8 = cv2.GaussianBlur(next_u8, (k, k), 0)
    if prev_u8.shape != next_u8.shape:
        raise ValueError(f"frame size mismatch: {prev_u8.shape} vs {next_u8.shape}")

    if cfg.method == "farneback":
        flow = cv2.calcOpticalFlowFarneback(
            prev_u8,
            next_u8,
            None,
            cfg.pyr_scale,
            cfg.levels,
            cfg.winsize,
            cfg.iterations,
            cfg.poly_n,
            cfg.poly_sigma,
            0,
        )
    else:  # dis
        flow = _create_dis(cfg).calc(prev_u8, next_u8, None)

    flow = flow.astype(np.float32)
    if cfg.scale < 1.0:
        flow /= cfg.scale
    return flow


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """DIS expects uint8; accept float images in [0, 1] or [0, 255]."""
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image, copy=False)
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def polar(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, angle in radians) of a flow field."""
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx.astype(np.float32), fy.astype(np.float32))
    return magnitude, angle


def flow_stats(flow: np.ndarray, motion_threshold: float = 1.0) -> dict:
    """Scalar summary of a flow field — handy for spotting the interesting pairs."""
    magnitude, angle = polar(flow)
    flat = magnitude.reshape(-1)
    # circular mean of direction: average of unit vectors, weighted by magnitude
    weight = flat
    total = float(weight.sum())
    if total > 0:
        ux = float(np.sum(np.cos(angle.reshape(-1)) * weight) / total)
        uy = float(np.sum(np.sin(angle.reshape(-1)) * weight) / total)
        mean_direction = float(np.degrees(np.arctan2(uy, ux)))
        coherence = float(np.hypot(ux, uy))  # 1 = all motion in one direction
    else:
        mean_direction, coherence = 0.0, 0.0
    return {
        "mean_magnitude": float(flat.mean()),
        "median_magnitude": float(np.median(flat)),
        "p95_magnitude": float(np.percentile(flat, 95)),
        "max_magnitude": float(flat.max()),
        "moving_pixels_pct": float(100.0 * np.mean(flat > motion_threshold)),
        "mean_direction_deg": mean_direction,
        "direction_coherence": coherence,
        "mean_dx": float(flow[..., 0].mean()),
        "mean_dy": float(flow[..., 1].mean()),
    }


def global_shift(
    prev: np.ndarray, next_frame: np.ndarray, max_side: int = 640
) -> dict:
    """Dominant whole-frame translation via phase correlation.

    Independent of the dense estimators: it is one number for the whole frame,
    it cannot be fooled by texture noise the way a per-pixel field can, and it
    does not saturate at large displacements the way Farneback does. Cheap
    (a couple of ms), so it is worth keeping next to the dense flow.
    """
    a, b = _to_uint8(prev), _to_uint8(next_frame)
    h, w = a.shape[:2]
    k = min(1.0, max_side / float(max(h, w)))
    if k < 1.0:
        a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    dx, dy = dx / k, dy / k  # back to original pixels
    return {
        "shift_x": float(dx),
        "shift_y": float(dy),
        "shift_magnitude": float(np.hypot(dx, dy)),
        "shift_response": float(response),
    }


def save_flow(path: str | Path, flow: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, flow.astype(np.float32))
    return path


def load_flow(path: str | Path) -> np.ndarray:
    return np.load(Path(path)).astype(np.float32)
