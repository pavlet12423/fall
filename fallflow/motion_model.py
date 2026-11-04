"""Global camera motion model for a rocking platform (boat, handheld camera).

On a boat the camera does not translate: it yaws, rolls and vibrates. That is
why a pure-translation test finds almost no agreeing tracks (16%) while the
frame is obviously moving — every point moves differently under a rotation.
Fitting a similarity transform (rotation + scale + translation) with RANSAC
separates the platform motion from whatever moves on its own.

    python -m fallflow.motion_model --frames data/frames --out outputs --fps 25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .frames import discover_frames, load_gray
from .tracks import track_corners

MODELS = ("similarity", "affine", "homography")


def fit(
    src: np.ndarray, dst: np.ndarray, model: str = "similarity", threshold: float = 3.0
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """RANSAC fit of the dominant motion from *src* (N,2) to *dst* (N,2)."""
    kwargs = dict(
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold,
        maxIters=5000,
        confidence=0.999,
    )
    if model == "homography":
        h, mask = cv2.findHomography(src, dst, **kwargs)
        return h, mask
    if model == "affine":
        return cv2.estimateAffine2D(src, dst, **kwargs)
    return cv2.estimateAffinePartial2D(src, dst, **kwargs)


def decompose(m: np.ndarray, model: str) -> dict:
    """Rotation (deg), scale and translation of the fitted transform."""
    if m is None:
        return {"rotation_deg": 0.0, "scale": 1.0, "tx": 0.0, "ty": 0.0}
    if model == "homography":
        # rotation/scale are only meaningful near the image centre for a homography
        a, b = float(m[0, 0]), float(m[0, 1])
        tx, ty = float(m[0, 2]), float(m[1, 2])
    else:
        a, b = float(m[0, 0]), float(m[1, 0])
        tx, ty = float(m[0, 2]), float(m[1, 2])
    return {
        "rotation_deg": float(np.degrees(np.arctan2(b, a))),
        "scale": float(np.hypot(a, b)),
        "tx": tx,
        "ty": ty,
    }


def apply_model(m: np.ndarray, points: np.ndarray, model: str) -> np.ndarray:
    src = points.reshape(-1, 1, 2).astype(np.float64)
    if model == "homography":
        warped = cv2.perspectiveTransform(src, m)
    else:
        warped = cv2.transform(src, m)
    return warped.reshape(-1, 2)


def analyse_pair(
    prev: np.ndarray,
    curr: np.ndarray,
    model: str = "similarity",
    threshold: float = 3.0,
    cols: int = 8,
    rows: int = 5,
    max_corners: int = 400,
) -> dict:
    """Fit the platform motion, then describe whatever does not fit it."""
    src, dst = track_corners(prev, curr, max_corners)
    base = {
        "corners": 0,
        "model_inliers": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "tx": 0.0,
        "ty": 0.0,
        "residual_median": 0.0,
        "outlier_share": 0.0,
        "outlier_peak_cell": -1,
        "outlier_concentration": 0.0,
        "outlier_cells": [],
    }
    if len(src) < 8:
        return base

    m, mask = fit(src, dst, model, threshold)
    if m is None or mask is None:
        return base
    inliers = mask.reshape(-1).astype(bool)

    predicted = apply_model(m, src, model)
    residual = np.linalg.norm(predicted - dst, axis=1)
    outliers = ~inliers
    h, w = prev.shape[:2]

    grid = np.zeros((rows, cols), np.int32)
    if outliers.any():
        pts = dst[outliers]
        cy = np.clip((pts[:, 1] * rows / h).astype(int), 0, rows - 1)
        cx = np.clip((pts[:, 0] * cols / w).astype(int), 0, cols - 1)
        np.add.at(grid, (cy, cx), 1)
    peak = int(grid.max()) if grid.any() else 0

    base.update(
        {
            "corners": int(len(src)),
            "model_inliers": float(inliers.mean()),
            **decompose(m, model),
            "residual_median": float(np.median(residual)),
            "outlier_share": float(outliers.mean()),
            "outlier_peak_cell": int(np.argmax(grid)) if peak else -1,
            "outlier_concentration": float(peak / (grid.mean() + 1e-6)) if peak else 0.0,
            "outlier_cells": grid.tolist(),
        }
    )
    return base


def run(
    frames_dir: str | Path,
    model: str = "similarity",
    threshold: float = 3.0,
    fps: float = 0.0,
    stride: int = 1,
    limit: int = 0,
    cols: int = 8,
    rows: int = 5,
    progress=None,
) -> list[dict]:
    paths = discover_frames(frames_dir)
    pairs = [(i, i + stride) for i in range(1, len(paths) - stride + 1)]
    if limit:
        pairs = pairs[:limit]

    cache: dict[int, np.ndarray] = {}
    out = []
    for n, (i, j) in enumerate(pairs, start=1):
        for idx in (i, j):
            if idx not in cache:
                cache[idx] = load_gray(paths[idx - 1])
        record = {
            "pair": f"flow_{i:04d}_{j:04d}",
            "from": paths[i - 1].name,
            "to": paths[j - 1].name,
            **analyse_pair(cache[i], cache[j], model, threshold, cols, rows),
        }
        if fps:
            record["t_from"] = (i - 1) / fps
            record["rotation_deg_s"] = (
                record["rotation_deg"] * fps / (j - i)
            )
        out.append(record)
        if progress:
            progress(n, len(pairs), record)
        cache.pop(min(cache), None)
    return out


FIELDS = [
    "pair", "from", "to", "corners", "model_inliers", "rotation_deg", "scale",
    "tx", "ty", "residual_median", "outlier_share", "outlier_peak_cell",
    "outlier_concentration", "t_from", "rotation_deg_s",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.motion_model",
        description="RANSAC platform-motion fit and independent-motion outliers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default="data/frames", help="directory with frames")
    p.add_argument("--out", default="outputs", help="where to write the results")
    p.add_argument("--model", choices=list(MODELS), default="similarity")
    p.add_argument("--threshold", type=float, default=3.0, help="RANSAC inlier radius, px")
    p.add_argument("--cols", type=int, default=8, help="outlier grid columns")
    p.add_argument("--rows", type=int, default=5, help="outlier grid rows")
    p.add_argument("--fps", type=float, default=0.0, help="source frame rate")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(n: int, total: int, record: dict) -> None:
        if not args.quiet:
            print(
                f"  [{n}/{total}] {record['pair'][5:]}: inliers {record['model_inliers'] * 100:5.1f}%, "
                f"rot {record['rotation_deg']:+6.2f}deg, outliers {record['outlier_share'] * 100:5.1f}%",
                flush=True,
            )

    rows = run(
        args.frames, args.model, args.threshold, args.fps, args.stride,
        args.limit, args.cols, args.rows, progress,
    )
    with (out_dir / "motion_model.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "motion_model.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "outlier_cells"} for r in rows], indent=2)
    )

    inl = np.array([r["model_inliers"] for r in rows])
    rot = np.array([r["rotation_deg"] for r in rows])
    out_share = np.array([r["outlier_share"] for r in rows])
    print(f"model     : {args.model} (RANSAC, {args.threshold} px)")
    print(f"inliers   : median {np.median(inl) * 100:.1f}%  (vs 16.5% for pure translation)")
    print(f"rotation  : median {np.median(np.abs(rot)):.2f} deg/frame, max {np.abs(rot).max():.2f} deg")
    print(f"outliers  : median {np.median(out_share) * 100:.1f}% of tracks move on their own")
    print(
        "note      : per-frame outlier clusters do not separate a rigid object\n"
        "            from water (the platform residual is locally smooth); use\n"
        "            fallflow.persist for the temporal test"
    )
    print(f"written   : {out_dir / 'motion_model.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
