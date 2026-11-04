"""CLI: compute dense optical flow for a directory of frames.

    python -m fallflow --frames data/frames --out outputs

Run it again after dropping more frames into the directory: pairs that are
already computed are skipped, only the new ones are processed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .flow import FlowConfig, compute_flow, flow_stats, polar, save_flow
from .frames import discover_frames, load_gray
from .visualize import draw_arrows, flow_to_bgr, montage

DEFAULT_FRAMES = "data/frames"
DEFAULT_OUT = "outputs"

STAT_FIELDS = (
    "mean_magnitude",
    "median_magnitude",
    "p95_magnitude",
    "max_magnitude",
    "moving_pixels_pct",
    "mean_direction_deg",
    "direction_coherence",
    "mean_dx",
    "mean_dy",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow",
        description="Dense optical flow over a sequence of frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default=DEFAULT_FRAMES, help="directory with frames")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument(
        "--method",
        default="farneback",
        choices=["farneback", "dis"],
        help="flow estimator",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="downscale factor before matching (flow is rescaled back)",
    )
    p.add_argument(
        "--stride", type=int, default=1, help="compare frame i with i+stride"
    )
    p.add_argument("--limit", type=int, default=0, help="max number of pairs (0 = all)")
    p.add_argument(
        "--arrow-step", type=int, default=0, help="also save arrow overlay (grid step)"
    )
    p.add_argument(
        "--max-magnitude",
        type=float,
        default=0.0,
        help="fixed colour scale for HSV images (0 = auto per pair)",
    )
    p.add_argument(
        "--viz-normalize",
        choices=["max", "p99", "p95"],
        default="p99",
        help="auto colour scale: by max magnitude or a percentile (brighter)",
    )
    p.add_argument(
        "--montage",
        action="store_true",
        help="also build a contact sheet of all visualisations",
    )
    p.add_argument("--montage-cols", type=int, default=5, help="columns in the montage")
    p.add_argument("--no-flow", action="store_true", help="do not save .npy fields")
    p.add_argument("--no-viz", action="store_true", help="do not save visualisations")
    p.add_argument(
        "--overwrite", action="store_true", help="recompute already computed pairs"
    )
    p.add_argument("--dry-run", action="store_true", help="only list what would run")
    p.add_argument("--quiet", action="store_true", help="less console output")
    p.add_argument("--version", action="version", version=f"fallflow {__version__}")
    return p


PERCENTILES = {"p99": 99, "p95": 95}


def _colour_scale(flow: np.ndarray, args) -> float | None:
    """Value mapped to full brightness in the HSV image (None = OpenCV auto/max)."""
    if args.max_magnitude:
        return args.max_magnitude
    if args.viz_normalize == "max":
        return None
    magnitude = polar(flow)[0]
    value = float(np.percentile(magnitude, PERCENTILES[args.viz_normalize]))
    return value or None


def pair_name(i: int, j: int) -> str:
    return f"flow_{i:04d}_{j:04d}"


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def warn_duplicates(paths: list[Path], log) -> None:
    seen: dict[str, list[int]] = {}
    for idx, path in enumerate(paths, start=1):
        seen.setdefault(file_md5(path), []).append(idx)
    dupes = {h: idxs for h, idxs in seen.items() if len(idxs) > 1}
    if not dupes:
        return
    log("! identical frames detected (zero flow between them):")
    for idxs in dupes.values():
        log(f"    frames {idxs}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def log(*msg: object) -> None:
        if not args.quiet:
            print(*msg, flush=True)

    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    flow_dir = out_dir / "flow"
    viz_dir = out_dir / "viz"

    paths = discover_frames(frames_dir)
    if len(paths) < 2:
        print(f"need at least 2 frames in {frames_dir}, found {len(paths)}", file=sys.stderr)
        return 1

    stride = max(1, args.stride)
    pairs = [(i, i + stride) for i in range(1, len(paths) - stride + 1)]
    if args.limit:
        pairs = pairs[: args.limit]

    log(f"frames      : {len(paths)} in {frames_dir}")
    log(f"pairs       : {len(pairs)} (stride {stride})")
    log(f"method      : {args.method} @ scale {args.scale}")

    if args.dry_run:
        for i, j in pairs:
            log(f"  {pair_name(i, j)}: {paths[i - 1].name} -> {paths[j - 1].name}")
        return 0

    warn_duplicates(paths, log)

    cfg = FlowConfig(method=args.method, scale=args.scale)
    rows: list[dict] = []
    started = time.perf_counter()
    skipped = 0

    for n, (i, j) in enumerate(pairs, start=1):
        name = pair_name(i, j)
        flow_path = flow_dir / f"{name}.npy"
        viz_path = viz_dir / f"{name}.png"
        done = (args.no_flow or flow_path.exists()) and (args.no_viz or viz_path.exists())
        if done and not args.overwrite:
            skipped += 1
            continue

        prev = load_gray(paths[i - 1], cfg.scale)
        curr = load_gray(paths[j - 1], cfg.scale)
        flow = compute_flow(prev, curr, cfg)
        stats = flow_stats(flow)

        if not args.no_flow:
            save_flow(flow_path, flow)
        if not args.no_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(viz_path), flow_to_bgr(flow, _colour_scale(flow, args)))
            if args.arrow_step:
                overlay = draw_arrows(
                    cv2.imread(str(paths[j - 1])), flow, step=args.arrow_step
                )
                cv2.imwrite(str(viz_dir / f"{name}_arrows.png"), overlay)

        rows.append({"pair": name, "from": paths[i - 1].name, "to": paths[j - 1].name, **stats})
        log(
            f"  [{n}/{len(pairs)}] {name}: mean |v| = {stats['mean_magnitude']:6.3f} px, "
            f"moving = {stats['moving_pixels_pct']:5.1f}%"
        )

    elapsed = time.perf_counter() - started
    log(f"done in {elapsed:.1f}s ({len(rows)} computed, {skipped} already on disk)")

    if rows:
        report_csv = out_dir / "report.csv"
        report_json = out_dir / "report.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        fields = ["pair", "from", "to", *STAT_FIELDS]
        with report_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report_json.write_text(json.dumps(rows, indent=2))
        log(f"report      : {report_csv}")

        top = sorted(rows, key=lambda r: -r["mean_magnitude"])[:5]
        log("busiest pairs (mean |v|, px):")
        for r in top:
            log(f"    {r['pair']}: {r['mean_magnitude']:.3f} px, dir {r['mean_direction_deg']:.0f}°")
    elif not args.quiet:
        log("nothing new to compute")

    if args.montage and not args.no_viz:
        tiles = sorted(
            p for p in viz_dir.glob("flow_*.png") if not p.name.endswith("_arrows.png")
        )
        if tiles:
            sheet = montage(
                [cv2.imread(str(p)) for p in tiles],
                cols=args.montage_cols,
                labels=[p.stem for p in tiles],
            )
            sheet_path = out_dir / "montage.png"
            cv2.imwrite(str(sheet_path), sheet)
            log(f"montage     : {sheet_path} ({len(tiles)} tiles)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
