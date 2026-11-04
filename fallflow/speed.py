"""Boat speed from image motion, calibrated by the distance to the shore.

Physics used (pinhole camera, height h above a flat water surface, camera
looking along the direction of travel):

    a point on the water at depth Z appears at normalised row y = h / Z
    forward motion V  ->  vertical image velocity  dy/dt = (V/h) * y^2
    lateral motion U  ->  horizontal image velocity dx/dt = (U/h) * y

in pixels, with y0 the horizon row and f the focal length in pixels:

    dy_px/frame = K * (y_px - y0)^2 ,   K = V / (h * f * fps)
    dx_px/frame = L * (y_px - y0)   ,   L = U / (h * f * fps)

So fitting K and the horizon row y0 to the measured per-row flow gives V once
h and f are known. The shore removes the need for both: a shoreline is at
water level and at a known distance D, so

    f = D * (y_shore - y0) / h      and therefore
    V = D * K * (y_shore - y0) * fps          <- h and f cancel

which is why the 30 m to the shore is the useful calibration here.

    python -m fallflow.speed --frames data/frames --out outputs --fps 25 --end 70
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

FIELDS = [
    "bin", "row_center", "count", "median_dx", "median_dy", "median_magnitude",
]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = np.nan_to_num(image, copy=False)
    if image.max(initial=0.0) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def row_profile(
    frames_dir: str | Path,
    start: int = 1,
    end: int = 0,
    stride: int = 1,
    bins: int = 18,
    far_fraction: float = 0.2,
    max_corners: int = 500,
    progress=None,
) -> tuple[list[dict], int]:
    """Per-row median displacement, with the far-field drift removed.

    The far-field (top) rows are almost unaffected by translation — their
    motion is the camera rotation — so their median is subtracted as the
    rotation reference. What is left grows downwards and is the boat's
    translation over the water.
    """
    paths = discover_frames(frames_dir)
    end = end or len(paths)
    pairs = [(i, i + stride) for i in range(start, min(end, len(paths)) - stride + 1)]

    height = load_gray(paths[0]).shape[0]
    edges = np.linspace(0, height, bins + 1)
    acc_dx = [[] for _ in range(bins)]
    acc_dy = [[] for _ in range(bins)]

    cache: dict[int, np.ndarray] = {}
    for n, (i, j) in enumerate(pairs, start=1):
        for idx in (i, j):
            if idx not in cache:
                cache[idx] = _to_uint8(load_gray(paths[idx - 1]))
        src, dst = track_corners(cache[i], cache[j], max_corners)
        if len(src) < 20:
            if progress:
                progress(n, len(pairs), None)
            cache.pop(min(cache), None)
            continue
        delta = dst - src
        # rotation reference = median displacement of the topmost points
        far = src[:, 1] < height * far_fraction
        ref = np.median(delta[far], axis=0) if far.sum() >= 8 else np.median(delta, axis=0)
        delta = delta - ref
        which = np.clip(np.digitize(src[:, 1], edges) - 1, 0, bins - 1)
        for b in range(bins):
            sel = which == b
            if sel.sum() >= 4:
                acc_dx[b].append(np.median(delta[sel][:, 0]))
                acc_dy[b].append(np.median(delta[sel][:, 1]))
        if progress:
            progress(n, len(pairs), len(src))
        cache.pop(min(cache), None)

    rows = []
    for b in range(bins):
        dx = np.array(acc_dx[b])
        dy = np.array(acc_dy[b])
        rows.append(
            {
                "bin": b,
                "row_center": float((edges[b] + edges[b + 1]) / 2),
                "count": int(len(dx)),
                "median_dx": float(np.median(dx)) if len(dx) else 0.0,
                "median_dy": float(np.median(dy)) if len(dy) else 0.0,
                "median_magnitude": float(np.hypot(np.median(dx), np.median(dy)))
                if len(dx)
                else 0.0,
            }
        )
    return rows, height


def fit_horizon(profile: list[dict], height: int) -> dict:
    """Fit dy = K*(y - y0)^2 and dx = L*(y - y0) by searching over y0."""
    y = np.array([r["row_center"] for r in profile])
    dy = np.array([r["median_dy"] for r in profile])
    dx = np.array([r["median_dx"] for r in profile])
    weight = np.array([min(r["count"], 30) for r in profile], np.float64)

    best = {"y0": float(height) * 0.5, "K": 0.0, "L": 0.0, "error": np.inf}
    for y0 in np.arange(-height * 0.5, height * 0.9, 2.0):
        below = (y > y0 + 5) & (weight > 0)
        if below.sum() < 5:
            continue
        u = (y[below] - y0) ** 2
        w = weight[below]
        # weighted least squares through the origin
        k = float(np.sum(w * u * dy[below]) / np.sum(w * u * u))
        err = float(np.sum(w * (dy[below] - k * u) ** 2) / np.sum(w))
        if err < best["error"]:
            v = y[below] - y0
            l = float(np.sum(w * v * dx[below]) / np.sum(w * v * v))
            best = {"y0": float(y0), "K": k, "L": l, "error": err}
    return best


def speeds(
    K: float, y0: float, fps: float, distance: float, shore_row: float,
    heights=(1.0, 1.3, 1.5, 1.7, 2.0), fovs=(60.0, 70.0, 80.0, 90.0), width=1280
) -> list[dict]:
    """Speed implied by the fit, either from the shore (h cancels) or from h and FOV."""
    out = []
    for h in heights:
        for fov in fovs:
            f = (width / 2.0) / np.tan(np.radians(fov) / 2.0)
            v_both = K * h * f * fps
            row = {"height_m": h, "fov_deg": fov, "f_px": float(f), "v_from_h_fov": float(v_both)}
            if shore_row and shore_row > y0:
                f_shore = distance * (shore_row - y0) / h
                row["f_from_shore_px"] = float(f_shore)
                row["fov_from_shore_deg"] = float(
                    np.degrees(2 * np.arctan((width / 2.0) / f_shore))
                )
                row["v_from_shore"] = float(distance * K * (shore_row - y0) * fps)
            out.append(row)
    return out


def long_baseline_map(
    frames_dir: str | Path,
    start: int,
    end: int,
    path: str | Path,
    cols: int = 12,
    rows: int = 8,
    min_points: int = 8,
    max_corners: int = 1500,
) -> tuple[Path, list[dict]]:
    """Per-cell displacement between two distant frames, drawn on the frame.

    Over a long baseline water decorrelates and static objects do not, so the
    cells that still move as one are the ones worth identifying: they are the
    shore, a buoy, another boat — anything that gives a scale reference.
    """
    paths = discover_frames(frames_dir)
    prev = _to_uint8(load_gray(paths[start - 1]))
    curr = _to_uint8(load_gray(paths[end - 1]))
    src, dst = track_corners(prev, curr, max_corners)
    h, w = prev.shape
    base = cv2.cvtColor(prev, cv2.COLOR_GRAY2BGR)

    cells = []
    if len(src) >= min_points:
        delta = dst - src
        for r in range(rows):
            for c in range(cols):
                sel = (
                    (src[:, 1] >= r * h / rows) & (src[:, 1] < (r + 1) * h / rows)
                    & (src[:, 0] >= c * w / cols) & (src[:, 0] < (c + 1) * w / cols)
                )
                n = int(sel.sum())
                if n < min_points:
                    continue
                block = delta[sel]
                lengths = np.linalg.norm(block, axis=1)
                coh = float(np.linalg.norm(block.mean(axis=0)) / (lengths.mean() + 1e-9))
                median = np.median(block, axis=0)
                cells.append(
                    {
                        "row": r,
                        "col": c,
                        "points": n,
                        "coherence": coh,
                        "dx": float(median[0]),
                        "dy": float(median[1]),
                        "magnitude": float(np.linalg.norm(median)),
                    }
                )
                x0, y0 = int(c * w / cols), int(r * h / rows)
                x1, y1 = int((c + 1) * w / cols), int((r + 1) * h / rows)
                intensity = int(min(255, 60 + 195 * coh))
                color = (60, 60, intensity) if coh < 0.9 else (60, 220, 60)
                cv2.rectangle(base, (x0, y0), (x1, y1), color, 2)
                cv2.putText(
                    base, f"{np.linalg.norm(median):.0f}", (x0 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
                centre = (int((x0 + x1) / 2), int((y0 + y1) / 2))
                tip = (
                    int(centre[0] + median[0] * 0.25),
                    int(centre[1] + median[1] * 0.25),
                )
                cv2.arrowedLine(base, centre, tip, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        base,
        f"cells moving as one between frames {start} and {end} "
        f"(green = coherence >= 0.9, arrows = median shift x0.25)",
        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        base,
        f"cells moving as one between frames {start} and {end} "
        f"(green = coherence >= 0.9, arrows = median shift x0.25)",
        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
    )
    path = Path(path)
    cv2.imwrite(str(path), base)
    return path, sorted(cells, key=lambda c: -c["coherence"])


def plot(profile: list[dict], fit: dict, path: str | Path, height: int) -> Path:
    width, left, top = 1200, 80, 50
    panel_h = 300
    canvas = np.full((top + panel_h * 2 + 80, width, 3), 255, np.uint8)
    y = np.array([r["row_center"] for r in profile])

    def panel(y0, values, color, label, model=None):
        lo, hi = float(values.min()), float(values.max())
        if model is not None:
            lo, hi = min(lo, float(model.min())), max(hi, float(model.max()))
        span = (hi - lo) or 1.0
        cv2.rectangle(canvas, (left, y0), (width - 30, y0 + panel_h), (247, 247, 247), -1)
        cv2.rectangle(canvas, (left, y0), (width - 30, y0 + panel_h), (200, 200, 200), 1)
        xs = (left + (y / height) * (width - 30 - left)).astype(int)
        ys = (y0 + panel_h - (values - lo) / span * (panel_h - 16) - 8).astype(int)
        cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
        for k in range(len(profile)):
            cv2.circle(canvas, (int(xs[k]), int(ys[k])), 3, color, -1, cv2.LINE_AA)
        if model is not None:
            ys = (y0 + panel_h - (model - lo) / span * (panel_h - 16) - 8).astype(int)
            cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, (220, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (left + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0", (left - 22, y0 + panel_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{height}", (left - 34, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (90, 90, 90), 1, cv2.LINE_AA)

    dy = np.array([r["median_dy"] for r in profile])
    dx = np.array([r["median_dx"] for r in profile])
    u = (y - fit["y0"]) ** 2
    panel(top, dy, (0, 90, 220), "vertical flow dy, px/frame (forward motion)", fit["K"] * u)
    panel(top + panel_h + 30, dx, (0, 150, 60),
          "horizontal flow dx, px/frame (lateral motion)", fit["L"] * (y - fit["y0"]))
    horizon_x = int(left + (fit["y0"] / height) * (width - 30 - left))
    for y0p in (top, top + panel_h + 30):
        if left <= horizon_x <= width - 30:
            cv2.line(canvas, (horizon_x, y0p), (horizon_x, y0p + panel_h), (90, 90, 90), 1)
    cv2.putText(canvas, f"fitted horizon row {fit['y0']:.0f}", (left, top + panel_h * 2 + 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.speed",
        description="Boat speed from per-row image motion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", default="outputs")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=0, help="last frame to use (0 = all)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bins", type=int, default=18, help="row bins")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--distance", type=float, default=30.0, help="distance to the shore, m")
    p.add_argument("--shore-row", type=float, default=0.0,
                   help="image row of the shoreline (0 = unknown)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--map-from", type=int, default=0, help="draw the long-baseline map from this frame")
    p.add_argument("--map-to", type=int, default=0, help="...to this frame")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def progress(n: int, total: int, tracked) -> None:
        if not args.quiet and n % 10 == 0:
            print(f"  [{n}/{total}] tracked {tracked}", flush=True)

    profile, height = row_profile(
        args.frames, args.start, args.end, args.stride, args.bins, progress=progress
    )
    fit = fit_horizon(profile, height)

    with (out_dir / "speed_profile.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(profile)
    (out_dir / "speed_fit.json").write_text(json.dumps({"fit": fit, "profile": profile}, indent=2))

    print(f"frames    : {args.start}..{args.end or 'end'} at {args.fps:g} fps")
    print(f"horizon   : row {fit['y0']:.0f} of {height} (fitted)")
    print(f"forward   : K = {fit['K']:.3e} px/frame per (row - horizon)^2")
    print(f"lateral   : L = {fit['L']:.3e} px/frame per (row - horizon)")
    print(f"fit error : {fit['error']:.3f} px^2 (weighted)")

    table = speeds(fit["K"], fit["y0"], args.fps, args.distance, args.shore_row)
    print()
    print("speed implied by the fit (needs camera height and field of view):")
    print(f"  {'h, m':>5s} {'FOV':>5s} {'f, px':>7s} {'V, m/s':>8s} {'V, km/h':>8s}"
          + (f" {'f(shore)':>9s} {'FOV(shore)':>11s} {'V(shore)':>9s}" if args.shore_row else ""))
    for row in table:
        line = (
            f"  {row['height_m']:5.1f} {row['fov_deg']:5.0f} {row['f_px']:7.0f} "
            f"{row['v_from_h_fov']:8.2f} {row['v_from_h_fov'] * 3.6:8.1f}"
        )
        if "v_from_shore" in row:
            line += (
                f" {row['f_from_shore_px']:9.0f} {row['fov_from_shore_deg']:11.1f} "
                f"{row['v_from_shore']:9.2f}"
            )
        print(line)
    if not args.shore_row:
        print()
        print("add --shore-row R (image row of the waterline on the shore, 30 m away)")
        print("and the estimate stops depending on both h and the FOV")
    chart = plot(profile, fit, out_dir / "speed_profile.png", height)
    print(f"\nchart     : {chart}")

    if args.map_from and args.map_to:
        map_path, cells = long_baseline_map(
            args.frames, args.map_from, args.map_to,
            out_dir / f"long_baseline_{args.map_from:04d}_{args.map_to:04d}.png",
        )
        dt = (args.map_to - args.map_from) / args.fps
        print(f"map       : {map_path}")
        print(f"\ncells moving as one over {dt:.2f}s (frames {args.map_from}->{args.map_to}):")
        for c in cells[:6]:
            v = None
            if args.distance and c["coherence"] >= 0.9:
                # lateral shift of a static object at distance D: s = D * dx / f
                v = [
                    args.distance * c["magnitude"] / ((args.width / 2) / np.tan(np.radians(fov) / 2)) / dt
                    for fov in (60, 70, 80, 90)
                ]
            extra = (
                "  -> "
                + ", ".join(f"{x:.1f} m/s @ {fov}deg" for x, fov in zip(v, (60, 70, 80, 90)))
                if v
                else ""
            )
            print(
                f"  cell (row {c['row']}, col {c['col']}): {c['points']:3d} pts, "
                f"coherence {c['coherence']:.2f}, shift {c['magnitude']:6.1f} px "
                f"(dx {c['dx']:+6.1f}, dy {c['dy']:+6.1f}){extra}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
