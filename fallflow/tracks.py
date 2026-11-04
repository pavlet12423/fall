"""Sparse feature tracking — the tie-breaker between the dense estimators.

Farneback, DIS and phase correlation disagree on this footage, so this module
tracks a few hundred corner features with Lucas-Kanade and reports the
*median* displacement and how consistent the tracks are. Sparse tracking on
corners is far less sensitive to texture noise than a dense field, and the
inlier share tells us directly whether the frame moves as a whole (camera)
or whether the tracks disagree (noise / independent local motion).

    python -m fallflow.tracks --frames data/frames --out outputs --fps 25
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .flow import _to_uint8
from .frames import discover_frames, load_gray


def detect(gray: np.ndarray, max_corners: int = 300) -> np.ndarray | None:
    """Corners to track. `gray` may be uint8 or float in [0, 1]."""
    return cv2.goodFeaturesToTrack(
        _to_uint8(gray),
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )


def track_pair(
    prev: np.ndarray,
    curr: np.ndarray,
    inlier_px: float = 2.0,
    max_corners: int = 300,
) -> dict:
    """Track corners from *prev* to *curr*; summarise the displacement."""
    corners = detect(prev, max_corners)
    if corners is None or len(corners) < 8:
        return {
            "tracked": 0,
            "track_rate": 0.0,
            "median_dx": 0.0,
            "median_dy": 0.0,
            "median_magnitude": 0.0,
            "inlier_share": 0.0,
            "spread": 0.0,
        }
    points, status, _ = cv2.calcOpticalFlowPyrLK(
        _to_uint8(prev),
        _to_uint8(curr),
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    status = status.reshape(-1).astype(bool)
    if status.sum() < 8:
        return {
            "tracked": int(status.sum()),
            "track_rate": float(status.mean()),
            "median_dx": 0.0,
            "median_dy": 0.0,
            "median_magnitude": 0.0,
            "inlier_share": 0.0,
            "spread": 0.0,
        }
    delta = (points[status] - corners[status]).reshape(-1, 2).astype(np.float64)
    median = np.median(delta, axis=0)
    distance = np.linalg.norm(delta - median, axis=1)
    return {
        "tracked": int(status.sum()),
        "track_rate": float(status.mean()),
        "median_dx": float(median[0]),
        "median_dy": float(median[1]),
        "median_magnitude": float(np.linalg.norm(median)),
        "inlier_share": float(np.mean(distance <= inlier_px)),
        "spread": float(np.median(distance)),
    }


def run(
    frames_dir: str | Path,
    stride: int = 1,
    fps: float = 0.0,
    limit: int = 0,
    inlier_px: float = 2.0,
    progress=None,
) -> list[dict]:
    paths = discover_frames(frames_dir)
    pairs = [(i, i + stride) for i in range(1, len(paths) - stride + 1)]
    if limit:
        pairs = pairs[:limit]

    cache: dict[int, np.ndarray] = {}
    rows = []
    for n, (i, j) in enumerate(pairs, start=1):
        for idx in (i, j):
            if idx not in cache:
                cache[idx] = load_gray(paths[idx - 1])
        result = track_pair(cache[i], cache[j], inlier_px)
        record = {
            "pair": f"flow_{i:04d}_{j:04d}",
            "from": paths[i - 1].name,
            "to": paths[j - 1].name,
            **result,
        }
        if fps:
            dt = (j - i) / fps
            record["t_from"] = (i - 1) / fps
            record["speed_px_s"] = result["median_magnitude"] / dt
        rows.append(record)
        if progress:
            progress(n, len(pairs), record)
        cache.pop(min(cache), None)  # keep memory flat
    return rows


FIELDS = [
    "pair", "from", "to", "tracked", "track_rate", "median_dx", "median_dy",
    "median_magnitude", "inlier_share", "spread", "t_from", "speed_px_s",
]


def plot(rows: list[dict], path: str | Path, fps: float = 0.0) -> Path:
    """Two panels: median displacement and the share of agreeing tracks."""
    path = Path(path)
    width, height, left, top, gap = 1400, 560, 70, 46, 40
    panel_h = (height - top - 60 - gap) // 2
    plot_w = width - left - 30
    canvas = np.full((height, width, 3), 255, np.uint8)
    n = len(rows)
    xs = (left + np.arange(n) * (plot_w / max(1, n - 1))).astype(int)

    def panel(y0, values, color, label, pct=False):
        lo, hi = float(values.min()), float(values.max())
        if pct:
            lo, hi = 0.0, max(1.0, hi)
        span = (hi - lo) or 1.0
        cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + panel_h), (246, 246, 246), -1)
        cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + panel_h), (200, 200, 200), 1)
        ys = (y0 + panel_h - (values - lo) / span * (panel_h - 12) - 6).astype(int)
        cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label}  [{lo:.1f} .. {hi:.1f}]",
            (left + 8, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    panel(top, np.array([r["median_magnitude"] for r in rows]), (0, 90, 220),
          "median tracked displacement, px")
    panel(top + panel_h + gap, np.array([r["inlier_share"] for r in rows]) * 100.0,
          (0, 150, 60), "tracks agreeing with the median, %", pct=True)

    for k in range(0, n, max(1, n // 24)):
        label = f"{rows[k]['t_from']:.1f}s" if fps and "t_from" in rows[k] else rows[k]["pair"][-4:]
        cv2.putText(canvas, label, (int(xs[k]) - 14, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"sparse feature tracking over {n} pairs", (left, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.tracks",
        description="Sparse corner tracking — an independent motion estimate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default="data/frames", help="directory with frames")
    p.add_argument("--out", default="outputs", help="where to write the results")
    p.add_argument("--fps", type=float, default=0.0, help="source frame rate")
    p.add_argument("--stride", type=int, default=1, help="compare frame i with i+stride")
    p.add_argument("--limit", type=int, default=0, help="max pairs (0 = all)")
    p.add_argument("--inlier-px", type=float, default=2.0, help="inlier radius")
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
                f"  [{n}/{total}] {record['pair'][5:]}: median {record['median_magnitude']:6.2f} px, "
                f"inliers {record['inlier_share'] * 100:5.1f}%, tracked {record['track_rate'] * 100:5.1f}%",
                flush=True,
            )

    rows = run(args.frames, args.stride, args.fps, args.limit, args.inlier_px, progress)
    csv_path = out_dir / "tracks.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "tracks.json").write_text(json.dumps(rows, indent=2))

    mag = np.array([r["median_magnitude"] for r in rows])
    inl = np.array([r["inlier_share"] for r in rows])
    tr = np.array([r["track_rate"] for r in rows])
    print(f"pairs      : {len(rows)}")
    print(
        f"median |d| : median {np.median(mag):.2f} px, p90 {np.percentile(mag, 90):.2f} px, "
        f"max {mag.max():.2f} px"
    )
    print(
        f"inliers    : median {np.median(inl) * 100:.1f}% "
        f"(high = the frame moves as a whole, low = tracks disagree)"
    )
    print(f"tracked    : median {np.median(tr) * 100:.1f}% of corners")
    print(f"tracks.csv : {csv_path}")
    if not args.no_plot:
        chart = plot(rows, out_dir / "tracks_chart.png", args.fps)
        print(f"chart      : {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
