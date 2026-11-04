"""Temporal persistence test: object motion vs. water motion.

Per-frame outlier clusters cannot tell a rigid object from a wavy surface:
neighbouring points share the same error of the platform model, so every
cluster looks "coherent". What actually separates them is time — a wave
decorrelates within a frame or two, an object keeps moving the same way.

For every window of K frames this module tracks corners, fits the platform
motion (homography) at each step, keeps only the points that disagree with
the model at *every* step, and clusters those survivors in space. A compact
cluster of persistent outliers is what a solid object looks like.

    python -m fallflow.persist --frames data/frames --out outputs --fps 25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .frames import discover_frames, load_gray
from .motion_model import fit
from .tracks import detect


def track_window(
    images: list[np.ndarray], max_corners: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Track corners through a window of frames.

    Returns ``(points, alive)`` with points of shape (steps, N, 2) — the
    position of each tracked corner in every frame of the window — and a
    boolean mask of the points that survived to the end.
    """
    corners = detect(images[0], max_corners)
    if corners is None or len(corners) < 12:
        return np.zeros((len(images), 0, 2)), np.zeros((0,), bool)

    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    n = len(corners)
    track = np.zeros((len(images), n, 2), np.float32)
    track[0] = corners.reshape(-1, 2)
    alive = np.ones(n, bool)
    # indices of the corners still being tracked; they shrink as points die
    current_idx = np.arange(n)
    current_pts = corners
    for step in range(1, len(images)):
        if len(current_idx) < 12:
            break
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            images[step - 1],
            images[step],
            current_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        status = status.reshape(-1).astype(bool)
        alive[current_idx[~status]] = False
        current_idx = current_idx[status]
        if len(current_idx) == 0:
            break
        track[step, current_idx] = nxt.reshape(-1, 2)[status]
        current_pts = track[step : step + 1, current_idx].reshape(-1, 1, 2)
    alive[:] = False
    alive[current_idx] = True  # only points tracked to the last frame count
    return track, alive


def persistent_outliers(
    track: np.ndarray,
    alive: np.ndarray,
    threshold: float = 3.0,
    model: str = "homography",
) -> tuple[np.ndarray, np.ndarray]:
    """Points that disagree with the platform model at every step.

    Returns the mask over the surviving points and their mean residual
    direction (unit vector); waves point every which way, a solid object
    keeps one.
    """
    steps = len(track)
    n = track.shape[1]
    always = alive.copy()
    residuals = np.zeros((n, 2), np.float64)
    for step in range(steps - 1):
        src = track[step][alive].astype(np.float64)
        dst = track[step + 1][alive].astype(np.float64)
        if len(src) < 12:
            return np.zeros(n, bool), np.zeros(2)
        m, mask = fit(src, dst, model, threshold)
        if m is None:
            return np.zeros(n, bool), np.zeros(2)
        inliers = mask.reshape(-1).astype(bool)
        pred = cv2.perspectiveTransform(
            src.reshape(-1, 1, 2), m
        ).reshape(-1, 2) if model == "homography" else cv2.transform(
            src.reshape(-1, 1, 2), m
        ).reshape(-1, 2)
        step_residual = dst - pred
        step_outlier = ~inliers
        idx = np.where(alive)[0]
        always[idx[~step_outlier]] = False
        residuals[idx] += step_residual
    persistent = always & alive
    direction = (
        residuals[persistent].mean(axis=0) if persistent.any() else np.zeros(2)
    )
    norm = np.linalg.norm(direction)
    return persistent, (direction / norm if norm > 1e-9 else direction)


def cluster(
    points: np.ndarray,
    residuals: np.ndarray,
    shape: tuple[int, int],
    cols: int = 8,
    rows: int = 5,
    min_points: int = 6,
) -> list[dict]:
    """Group persistent outliers on a grid, merging 4-connected cells."""
    if len(points) < min_points:
        return []
    h, w = shape
    cy = np.clip((points[:, 1] * rows / h).astype(int), 0, rows - 1)
    cx = np.clip((points[:, 0] * cols / w).astype(int), 0, cols - 1)
    grid = np.zeros((rows, cols), np.int32)
    for r, c in zip(cy, cx):
        grid[r, c] += 1

    clusters = []
    seen = np.zeros_like(grid, bool)
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] == 0:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            member = np.zeros(len(points), bool)
            for cr, cc in cells:
                member |= (cy == cr) & (cx == cc)
            count = int(member.sum())
            if count < min_points:
                continue
            block = residuals[member]
            lengths = np.linalg.norm(block, axis=1)
            coherence = float(
                np.linalg.norm(block.mean(axis=0)) / (lengths.mean() + 1e-9)
            )
            clusters.append(
                {
                    "points": count,
                    "cells": [[cr, cc] for cr, cc in cells],
                    "centroid": [
                        float(points[member][:, 0].mean()),
                        float(points[member][:, 1].mean()),
                    ],
                    "coherence": coherence,
                    "mean_residual": float(lengths.mean()),
                    "score": count * coherence,
                }
            )
    return sorted(clusters, key=lambda c: -c["score"])


def analyse_window(
    images: list[np.ndarray],
    threshold: float = 3.0,
    model: str = "homography",
    cols: int = 8,
    rows: int = 5,
    min_points: int = 6,
    max_corners: int = 400,
) -> dict:
    track, alive = track_window(images, max_corners)
    result = {
        "tracked": int(alive.sum()),
        "persistent": 0,
        "persistent_share": 0.0,
        "best_score": 0.0,
        "best_points": 0,
        "best_coherence": 0.0,
        "best_centroid": None,
        "best_cells": [],
        "best_cells_count": 0,
        "mean_residual": 0.0,
    }
    if alive.sum() < 12:
        return result
    mask, _direction = persistent_outliers(track, alive, threshold, model)
    points = track[-1][mask]
    residuals = (track[-1] - track[0])[mask]
    result["persistent"] = int(mask.sum())
    result["persistent_share"] = float(mask.sum() / max(1, alive.sum()))
    if len(points) < min_points:
        return result
    result["mean_residual"] = float(np.linalg.norm(residuals, axis=1).mean())
    groups = cluster(points, residuals, images[0].shape, cols, rows, min_points)
    if groups:
        best = groups[0]
        result.update(
            {
                "best_score": float(best["score"]),
                "best_points": int(best["points"]),
                "best_coherence": float(best["coherence"]),
                "best_centroid": best["centroid"],
                "best_cells": best["cells"],
                "best_cells_count": len(best["cells"]),
            }
        )
    return result


def run(
    frames_dir: str | Path,
    window: int = 5,
    stride: int = 1,
    fps: float = 0.0,
    threshold: float = 3.0,
    model: str = "homography",
    cols: int = 8,
    rows: int = 5,
    min_points: int = 6,
    limit: int = 0,
    progress=None,
) -> list[dict]:
    paths = discover_frames(frames_dir)
    starts = list(range(1, len(paths) - window + 2, stride))
    if limit:
        starts = starts[:limit]
    cache: dict[int, np.ndarray] = {}
    out = []
    for n, start in enumerate(starts, start=1):
        indices = list(range(start, start + window))
        for idx in indices:
            if idx not in cache:
                cache[idx] = load_gray(paths[idx - 1])
        images = [_to_uint8(cache[i]) for i in indices]
        record = {
            "window": f"{start:04d}_{start + window - 1:04d}",
            "start": start,
            "end": start + window - 1,
            **analyse_window(
                images, threshold, model, cols, rows, min_points
            ),
        }
        if fps:
            record["t_from"] = (start - 1) / fps
            record["t_to"] = (start + window - 2) / fps
        out.append(record)
        if progress:
            progress(n, len(starts), record)
        for idx in list(cache):
            if idx < start:
                cache.pop(idx)
    return out


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image, copy=False)
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


FIELDS = [
    "window", "start", "end", "t_from", "t_to", "tracked", "persistent",
    "persistent_share", "mean_residual", "best_score", "best_points",
    "best_coherence", "best_cells_count", "best_centroid", "best_cells",
]


def plot(rows: list[dict], path: str | Path, fps: float = 0.0) -> Path:
    """Persistent-outlier share and the size of the strongest cluster."""
    path = Path(path)
    width, height, left, top, gap = 1400, 560, 70, 46, 40
    panel_h = (height - top - 60 - gap) // 2
    plot_w = width - left - 30
    canvas = np.full((height, width, 3), 255, np.uint8)
    n = len(rows)
    xs = (left + np.arange(n) * (plot_w / max(1, n - 1))).astype(int)

    def panel(y0, values, color, label, hi=None):
        lo, high = 0.0, float(hi if hi else values.max())
        span = (high - lo) or 1.0
        cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + panel_h), (246, 246, 246), -1)
        cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + panel_h), (200, 200, 200), 1)
        ys = (y0 + panel_h - (values - lo) / span * (panel_h - 12) - 6).astype(int)
        cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{label}  [{lo:.1f} .. {high:.1f}]", (left + 8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        return ys

    share = np.array([r["persistent_share"] for r in rows]) * 100.0
    cells = np.array([r["best_cells_count"] for r in rows], np.float64)
    panel(top, share, (0, 90, 220), "tracks disagreeing with the platform model, %", 100.0)
    ys = panel(top + panel_h + gap, cells, (200, 60, 60),
               "cells covered by the strongest cluster (of 40)", 40.0)
    for idx, r in enumerate(rows):
        if 0 < r["best_cells_count"] <= 3 and r["best_points"] >= 6:
            cv2.circle(canvas, (int(xs[idx]), int(ys[idx])), 5, (0, 200, 0), -1, cv2.LINE_AA)
    for k in range(0, n, max(1, n // 24)):
        label = f"{rows[k]['t_from']:.1f}s" if fps and "t_from" in rows[k] else rows[k]["window"][:4]
        cv2.putText(canvas, label, (int(xs[k]) - 14, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"persistence test over {n} windows (green = compact cluster)",
                (left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.persist",
        description="Find independently moving objects that survive over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", default="outputs")
    p.add_argument("--window", type=int, default=5, help="frames per window")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--fps", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=3.0, help="RANSAC inlier radius")
    p.add_argument("--model", choices=["homography", "affine", "similarity"], default="homography")
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--min-points", type=int, default=6, help="min points per cluster")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-plot", action="store_true", help="skip the chart")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(n: int, total: int, record: dict) -> None:
        if not args.quiet:
            print(
                f"  [{n}/{total}] {record['window']}: tracked {record['tracked']:3d}, "
                f"persistent {record['persistent']:3d} ({record['persistent_share'] * 100:4.1f}%), "
                f"best cluster {record['best_points']:2d} pts coh {record['best_coherence']:.2f}",
                flush=True,
            )

    rows = run(
        args.frames, args.window, args.stride, args.fps, args.threshold,
        args.model, args.cols, args.rows, args.min_points, args.limit, progress,
    )
    with (out_dir / "persist.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "persist.json").write_text(json.dumps(rows, indent=2))

    share = np.array([r["persistent_share"] for r in rows])
    score = np.array([r["best_score"] for r in rows])
    print(f"windows   : {len(rows)} of {args.window} frames ({args.window / max(args.fps, 1):.2f}s each)")
    print(f"persistent: median {np.median(share) * 100:.1f}% of tracked points survive the whole window")
    print(f"clusters  : best score median {np.median(score):.1f}, max {score.max():.1f}")
    print()
    print("strongest independently moving clusters:")
    for r in sorted(rows, key=lambda x: -x["best_score"])[:8]:
        print(
            f"  {r['window']} t={r.get('t_from', 0):.2f}-{r.get('t_to', 0):.2f}s "
            f"score {r['best_score']:5.1f}  {r['best_points']:2d} pts  "
            f"coh {r['best_coherence']:.2f}  residual {r['mean_residual']:5.1f} px"
        )
    print(f"\nwritten   : {out_dir / 'persist.csv'}")
    if not args.no_plot:
        chart = plot(rows, out_dir / "persist_chart.png", args.fps)
        print(f"chart     : {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
