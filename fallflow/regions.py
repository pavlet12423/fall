"""Regional (per-cell) flow features.

Whole-frame averages cannot tell *where* the motion happened. This module
splits every flow field into a grid and tracks each cell over time, so a
localised event (a person falling in one part of the frame) stands out from
the background motion of the whole scene.

    python -m fallflow.regions --out outputs --cols 8 --rows 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .analyze import robust_z
from .flow import load_flow, polar
from .frames import discover_frames


def cell_features(flow: np.ndarray, cols: int = 8, rows: int = 5) -> dict:
    """Per-cell mean magnitude, mean dx/dy and coherence of one flow field."""
    magnitude, angle = polar(flow)
    h, w = magnitude.shape
    cell_h, cell_w = h // rows, w // cols
    mag = np.zeros((rows, cols), np.float64)
    dx = np.zeros((rows, cols), np.float64)
    dy = np.zeros((rows, cols), np.float64)
    coh = np.zeros((rows, cols), np.float64)
    for r in range(rows):
        for c in range(cols):
            ys, ye = r * cell_h, (r + 1) * cell_h
            xs, xe = c * cell_w, (c + 1) * cell_w
            block = magnitude[ys:ye, xs:xe]
            mag[r, c] = block.mean()
            dx[r, c] = flow[ys:ye, xs:xe, 0].mean()
            dy[r, c] = flow[ys:ye, xs:xe, 1].mean()
            # coherence of the block: length of the mean vector / mean length
            ux = np.cos(angle[ys:ye, xs:xe]) * block
            uy = np.sin(angle[ys:ye, xs:xe]) * block
            total = block.sum()
            coh[r, c] = (
                float(np.hypot(ux.sum(), uy.sum()) / total) if total > 0 else 0.0
            )
    return {"magnitude": mag, "dx": dx, "dy": dy, "coherence": coh}


def camera_vector(
    flow: np.ndarray, mode: str = "median"
) -> np.ndarray:
    """Estimate the whole-frame (camera) motion to subtract.

    ``median`` takes the per-axis median of the flow field: a moving object
    usually covers well under half the frame, so the median tracks the camera
    and ignores the object. Unlike phase correlation it never fails to return
    a number, which matters on blurry frames.
    """
    flat = flow.reshape(-1, 2)
    if mode == "median":
        return np.median(flat, axis=0).astype(np.float32)
    if mode == "mean":
        return flat.mean(axis=0).astype(np.float32)
    return np.zeros(2, np.float32)


def load_shifts(out_dir: str | Path) -> dict[str, tuple[float, float, float]]:
    """pair -> (shift_x, shift_y, response) from report.json, if present."""
    report = Path(out_dir) / "report.json"
    if not report.exists():
        return {}
    return {
        r["pair"]: (
            float(r.get("shift_x", 0.0)),
            float(r.get("shift_y", 0.0)),
            float(r.get("shift_response", 0.0)),
        )
        for r in json.loads(report.read_text())
    }


def collect(
    flow_dir: str | Path,
    cols: int = 8,
    rows: int = 5,
    compensate: str = "none",
    shifts: dict[str, tuple[float, float, float]] | None = None,
    min_response: float = 0.2,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-cell magnitudes/coherences, optionally with camera motion removed.

    ``compensate`` is one of none | median | mean | shift. With ``shift`` the
    phase-correlation estimate is used where it is trustworthy
    (response >= min_response) and the median of the flow elsewhere.
    """
    files = sorted(Path(flow_dir).glob("flow_*.npy"))
    if not files:
        raise FileNotFoundError(f"no flow_*.npy in {flow_dir}")
    mags, cohs, names, cameras = [], [], [], []
    for path in files:
        flow = load_flow(path)
        name = path.stem
        if compensate == "shift":
            sx, sy, response = (shifts or {}).get(name, (0.0, 0.0, 0.0))
            camera = (
                np.array([sx, sy], np.float32)
                if response >= min_response
                else camera_vector(flow, "median")
            )
        elif compensate in ("median", "mean"):
            camera = camera_vector(flow, compensate)
        else:
            camera = np.zeros(2, np.float32)
        cameras.append(camera)
        flow = flow - camera
        cf = cell_features(flow, cols, rows)
        mags.append(cf["magnitude"])
        cohs.append(cf["coherence"])
        names.append(name)
    return names, np.stack(mags), np.stack(cohs), np.stack(cameras)


def analyse(
    names: list[str],
    mags: np.ndarray,
    cohs: np.ndarray,
    window: int = 15,
    cameras: np.ndarray | None = None,
) -> list[dict]:
    """Per-cell z-scores over time -> local spike, its position, concentration."""
    n, rows, cols = mags.shape
    z = np.zeros_like(mags)
    for r in range(rows):
        for c in range(cols):
            z[:, r, c] = robust_z(mags[:, r, c], window)

    records = []
    for i, name in enumerate(names):
        grid = mags[i]
        frame_mean = float(grid.mean())
        peak = float(grid.max())
        # peak-to-mean: how much the hottest cell stands out. Bounded by the
        # number of cells and well behaved when most cells are near zero
        # (a peak/median ratio explodes there).
        concentration = float(peak / (frame_mean + 1e-6))
        r, c = np.unravel_index(np.argmax(z[i]), z[i].shape)
        camera = cameras[i] if cameras is not None else np.zeros(2)
        records.append(
            {
                "pair": name,
                "camera_dx": float(camera[0]),
                "camera_dy": float(camera[1]),
                "camera_speed": float(np.hypot(camera[0], camera[1])),
                "frame_mean_magnitude": frame_mean,
                "peak_cell_magnitude": peak,
                "concentration": float(concentration),
                "max_cell_z": float(z[i].max()),
                "peak_row": int(r),
                "peak_col": int(c),
                "active_cells": int(np.sum(z[i] > 3.0)),
                "grid_mean": grid.round(3).tolist(),
                "grid_z": z[i].round(2).tolist(),
                "coherence_peak": float(cohs[i].max()),
            }
        )

    # a localised event = a cell spikes while the rest of the frame does not
    local = np.array([r["max_cell_z"] for r in records])
    frame_z = robust_z(np.array([r["frame_mean_magnitude"] for r in records]), window)
    conc_z = robust_z(np.array([r["concentration"] for r in records]), window)
    for i, rec in enumerate(records):
        rec.update(
            {
                "z_frame": float(frame_z[i]),
                "z_concentration": float(conc_z[i]),
                "local_score": float(
                    max(0.0, local[i]) + max(0.0, conc_z[i]) + 0.5 * max(0.0, frame_z[i])
                ),
            }
        )
    return records


def heatmap(
    grid: np.ndarray,
    cell_px: int = 64,
    vmax: float | None = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Grid of numbers -> colour image, for a quick look at where motion is."""
    rows, cols = grid.shape
    norm = grid / (vmax or grid.max() or 1.0)
    small = np.clip(norm * 255, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(small, colormap)
    image = cv2.resize(image, (cols * cell_px, rows * cell_px), interpolation=cv2.INTER_NEAREST)
    for r in range(1, rows):
        cv2.line(image, (0, r * cell_px), (image.shape[1], r * cell_px), (60, 60, 60), 1)
    for c in range(1, cols):
        cv2.line(image, (c * cell_px, 0), (c * cell_px, image.shape[0]), (60, 60, 60), 1)
    return image


def _chart(records: list[dict], path: Path, threshold: float) -> Path:
    width, height = 1400, 420
    left, top, bottom = 70, 40, 40
    canvas = np.full((height, width, 3), 255, np.uint8)

    def series(values: np.ndarray, color: tuple[int, int, int], label: str) -> None:
        lo, hi = float(values.min()), float(values.max())
        span = (hi - lo) or 1.0
        n = len(values)
        xs = (left + np.arange(n) * ((width - left - 30) / max(1, n - 1))).astype(int)
        ys = (height - bottom - (values - lo) / span * (height - top - bottom)).astype(int)
        cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{label} [{lo:.2f}..{hi:.2f}]",
            (left + 8, top + 18 + 20 * SERIES_ORDER.index(label)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    SERIES_ORDER = ["max cell z", "concentration", "frame mean |v|"]
    series(np.array([r["max_cell_z"] for r in records]), (0, 0, 220), "max cell z")
    series(np.array([r["concentration"] for r in records]), (0, 150, 0), "concentration")
    series(np.array([r["frame_mean_magnitude"] for r in records]), (200, 120, 0), "frame mean |v|")

    for i, r in enumerate(records):
        if r["local_score"] >= threshold:
            x = int(left + i * ((width - left - 30) / max(1, len(records) - 1)))
            cv2.line(canvas, (x, top), (x, height - bottom), (255, 200, 200), 1)
            cv2.circle(canvas, (x, top + 6), 4, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"regional features over {len(records)} pairs (red = local_score >= {threshold})",
        (left, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.regions",
        description="Per-cell flow features and heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="outputs", help="directory with flow/*.npy")
    p.add_argument("--cols", type=int, default=8, help="grid columns")
    p.add_argument("--rows", type=int, default=5, help="grid rows")
    p.add_argument("--window", type=int, default=15, help="rolling baseline window")
    p.add_argument("--threshold", type=float, default=8.0, help="local_score cutoff")
    p.add_argument("--top", type=int, default=8, help="how many pairs to print")
    p.add_argument(
        "--compensate",
        choices=["none", "median", "mean", "shift"],
        default="none",
        help="subtract the whole-frame (camera) motion before the cell stats",
    )
    p.add_argument("--heatmaps", type=int, default=6, help="heatmaps to render (0 = none)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    analysis = out_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    names, mags, cohs, cameras = collect(
        out_dir / "flow",
        args.cols,
        args.rows,
        args.compensate,
        load_shifts(out_dir),
    )
    records = analyse(names, mags, cohs, args.window, cameras)
    suffix = f"_{args.compensate}" if args.compensate != "none" else ""
    for name in ("regions.csv", "regions.json", "regions_chart.png"):
        (analysis / name).unlink(missing_ok=True)
    for r in records:
        r["is_local_event"] = bool(r["local_score"] >= args.threshold)

    fields = [
        "pair", "frame_mean_magnitude", "peak_cell_magnitude", "concentration",
        "max_cell_z", "peak_row", "peak_col", "active_cells", "coherence_peak",
        "z_frame", "z_concentration", "local_score", "is_local_event",
        "camera_dx", "camera_dy", "camera_speed",
    ]
    csv_path = analysis / f"regions{suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (analysis / f"regions{suffix}.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k not in ("grid_mean", "grid_z")}
                for r in records
            ],
            indent=2,
        )
    )

    print(f"grid      : {args.rows}x{args.cols} ({args.rows * args.cols} cells per pair)")
    if args.compensate != "none":
        cam = np.linalg.norm(cameras, axis=1)
        print(f"camera    : compensated with '{args.compensate}', "
              f"median |camera| {np.median(cam):.2f} px (max {cam.max():.2f})")
    print(f"pairs     : {len(records)}")
    print(f"events    : {sum(r['is_local_event'] for r in records)} (local_score >= {args.threshold})")
    print()
    print(f"  {'pair':>9s} {'score':>6s} {'maxz':>6s} {'conc':>5s} {'peak':>5s} {'cell':>7s}")
    for r in sorted(records, key=lambda x: -x["local_score"])[: args.top]:
        print(
            f"  {r['pair'][5:]:>9s} {r['local_score']:6.2f} {r['max_cell_z']:6.1f} "
            f"{r['concentration']:5.2f} {r['peak_cell_magnitude']:5.2f} "
            f"({r['peak_row']},{r['peak_col']})"
            f"{'  <== LOCAL EVENT' if r['is_local_event'] else ''}"
        )

    if args.heatmaps:
        hm_dir = analysis / "heatmaps"
        hm_dir.mkdir(exist_ok=True)
        vmax = float(mags.max())
        for r in sorted(records, key=lambda x: -x["local_score"])[: args.heatmaps]:
            grid = np.array(r["grid_mean"])
            cv2.imwrite(
                str(hm_dir / f"{r['pair']}{suffix}.png"), heatmap(grid, vmax=vmax)
            )
        print(f"\nheatmaps  : {hm_dir} (top {args.heatmaps}, shared colour scale)")

    chart = _chart(records, analysis / f"regions_chart{suffix}.png", args.threshold)
    print(f"chart     : {chart}")
    print(f"regions   : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
