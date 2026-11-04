"""Exploratory analysis of the flow statistics: find anomalous pairs.

No labels are needed — the baseline is a robust rolling statistic (median +
MAD), so a "fall-like" event shows up as a spike against the local background.

    python -m fallflow.analyze --out outputs --top 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

SERIES = (
    ("mean_magnitude", "mean |v|, px", (0, 200, 0)),
    ("moving_pixels_pct", "moving pixels, %", (200, 120, 0)),
    ("direction_coherence", "direction coherence", (0, 100, 220)),
    ("shift_magnitude", "global shift, px", (180, 0, 0)),
    ("mean_dy", "mean dy, px (down +)", (200, 0, 160)),
    ("speed_px_s", "flow speed, px/s", (0, 150, 150)),
    ("shift_speed_px_s", "global shift speed, px/s", (150, 0, 110)),
    ("sharpness", "sharpness (Laplacian var)", (90, 90, 90)),
    ("brightness", "brightness", (120, 90, 40)),
)


def load_frame_quality(out_dir: str | Path) -> dict[str, dict]:
    """frame filename -> {brightness, contrast, sharpness}."""
    path = Path(out_dir) / "frame_quality.csv"
    if not path.exists():
        return {}
    with path.open() as fh:
        return {
            row["frame"]: {
                "brightness": float(row["brightness"]),
                "contrast": float(row["contrast"]),
                "sharpness": float(row["sharpness"]),
            }
            for row in csv.DictReader(fh)
        }


def _shift_key(records: list[dict]) -> str:
    return "shift_speed_px_s" if "shift_speed_px_s" in records[0] else "shift_magnitude"


# when the frame rate is known the px/s series replace their px counterparts
REDUNDANT_WITH_TIME = {
    "mean_magnitude": "speed_px_s",
    "shift_magnitude": "shift_speed_px_s",
}


def series_for(records: list[dict]) -> tuple:
    """SERIES entries available in these records, without px/px-s duplicates."""
    available = {name for name, _, _ in SERIES if name in records[0]}
    out = []
    for name, label, color in SERIES:
        if name not in available:
            continue
        if REDUNDANT_WITH_TIME.get(name) in available:
            continue
        out.append((name, label, color))
    return tuple(out)


# heuristics tuned for "a person falls": motion gets faster, more coherent
# and more vertical than the surrounding background motion
def features_for(records: list[dict]) -> tuple:
    """Feature set for the event score, adapted to the available columns."""
    available = {name for name, _, _ in SERIES if name in records[0]}
    speed = "speed_px_s" if "speed_px_s" in available else "mean_magnitude"
    shift = _shift_key(records)
    return (
        (speed, 1.0),  # faster than usual
        ("direction_coherence", 1.0),  # motion lines up in one direction
        ("verticality", 1.0),  # vertical component dominates
        ("moving_pixels_pct", 0.5),  # more of the frame is moving
        (shift, 1.5),  # whole-frame displacement jumps
    )


def load_records(out_dir: str | Path) -> list[dict]:
    """Read report.json (falling back to per-pair stats sidecars)."""
    out_dir = Path(out_dir)
    report = out_dir / "report.json"
    if report.exists():
        records = json.loads(report.read_text())
    else:
        records = [
            json.loads(p.read_text())
            for p in sorted((out_dir / "stats").glob("flow_*.json"))
        ]
    return sorted(records, key=lambda r: r["pair"])


def robust_z(values: np.ndarray, window: int = 15, min_scale_frac: float = 0.5) -> np.ndarray:
    """Local z-score against a rolling median/MAD baseline (outlier-proof).

    The local scale is floored at a fraction of the *global* scale: inside a
    very quiet stretch the local MAD can collapse to almost zero and turn an
    insignificant wiggle into a z of 10.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 3:
        return np.zeros(n)
    window = max(5, min(window | 1, n if n % 2 else n - 1))
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    baseline = np.array(
        [np.median(padded[i : i + window]) for i in range(n)], dtype=np.float64
    )
    mad = np.array(
        [np.median(np.abs(padded[i : i + window] - baseline[i])) for i in range(n)]
    )
    local = 1.4826 * mad
    global_scale = 1.4826 * float(np.median(np.abs(values - np.median(values))))
    # absolute floor: a series that is essentially flat (or essentially zero)
    # must not turn a rounding error into a z of hundreds
    level = float(np.median(np.abs(values))) or 1.0
    floor = max(min_scale_frac * global_scale, 0.02 * level, 1e-6)
    scale = np.maximum(local, floor)
    return (values - baseline) / scale


def verticality(record: dict) -> float:
    """How vertical the dominant motion is: 0 = horizontal, 1 = straight up/down.

    Uses the magnitude-weighted mean direction, which is far more stable than
    the plain mean of dy (that one is dominated by pixels that do not move).
    """
    if "mean_direction_deg" in record:
        return float(abs(np.sin(np.radians(record["mean_direction_deg"]))))
    dx, dy = abs(record.get("mean_dx", 0.0)), abs(record.get("mean_dy", 0.0))
    return dy / (dx + dy) if (dx + dy) > 1e-9 else 0.0


def analyse(
    records: list[dict],
    window: int = 15,
    min_magnitude: float = 0.1,
    min_response: float = 0.2,
    min_sharpness: float = 0.0,
    quality: dict[str, dict] | None = None,
) -> list[dict]:
    """`min_magnitude` filters out (near-)static pairs: with no motion the
    direction and coherence are pure noise and must not seed the baseline.
    `min_response` / `min_sharpness` mark measurements the estimators could
    not make reliably; those are excluded from the baseline and the score."""
    """Enrich records with z-scores and a combined event score."""
    if not records:
        return []
    if quality:
        for record in records:
            q = quality.get(record.get("to", ""), {})
            record.update(q)

    values = {
        name: np.array([r.get(name, 0.0) for r in records]) for name, _, _ in SERIES
    }
    vert = np.array([verticality(r) for r in records])
    shift_key = _shift_key(records)
    trustworthy = np.array(
        [
            r.get("shift_response", 1.0) >= min_response
            and r.get("sharpness", 1e9) >= min_sharpness
            for r in records
        ]
    )
    zs = {}
    for name, v in values.items():
        if name in ("shift_magnitude", "shift_speed_px_s"):
            # baseline from the trustworthy samples only, so a failed estimate
            # cannot inflate it and cannot score as an event either
            z = np.zeros(len(v))
            if trustworthy.sum() >= 3:
                z[trustworthy] = robust_z(v[trustworthy], window)
            else:
                z = robust_z(v, window)
            zs[name] = z
        else:
            zs[name] = robust_z(v, window)
    zs["verticality"] = robust_z(vert, window)
    features = features_for(records)

    out = []
    for i, record in enumerate(records):
        static = bool(values["mean_magnitude"][i] < min_magnitude)
        score = 0.0
        if not static:
            for name, weight in features:
                score += weight * max(0.0, float(zs[name][i]))  # only positive spikes
        enriched = dict(record)
        if static:  # keep the numbers, but they carry no signal
            for name in ("mean_magnitude", "moving_pixels_pct", "direction_coherence", "verticality"):
                zs[name][i] = 0.0
        enriched.update(
            {
                "verticality": float(vert[i]),
                "z_shift_magnitude": float(zs["shift_magnitude"][i]),
                "z_mean_magnitude": float(zs["mean_magnitude"][i]),
                "z_moving_pixels_pct": float(zs["moving_pixels_pct"][i]),
                "z_direction_coherence": float(zs["direction_coherence"][i]),
                "z_verticality": float(zs["verticality"][i]),
                "event_score": float(score),
                "is_event": bool(not static and score >= 6.0),
                "static": static,
                "low_quality": bool(not trustworthy[i]),
            }
        )
        out.append(enriched)
    return out


def top_events(records: list[dict], count: int = 10) -> list[dict]:
    return sorted(records, key=lambda r: -r["event_score"])[:count]


def write_csv(path: str | Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair", "from", "to", "mean_magnitude", "moving_pixels_pct",
        "direction_coherence", "mean_dx", "mean_dy", "verticality",
        "shift_magnitude", "shift_x", "shift_y", "shift_response",
        "z_mean_magnitude", "z_moving_pixels_pct", "z_direction_coherence",
        "z_verticality", "z_shift_magnitude", "event_score", "is_event", "static",
        "sharpness", "brightness", "contrast", "low_quality",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def _xlabel(record: dict) -> str:
    return f"{record['t_from']:.1f}s" if "t_from" in record else record["pair"][-4:]


def _panel(
    canvas: np.ndarray,
    y0: int,
    height: int,
    values: np.ndarray,
    z: np.ndarray,
    label: str,
    color: tuple[int, int, int],
    events: list[int],
    left: int,
    plot_w: int,
) -> None:
    lo, hi = float(values.min()), float(values.max())
    span = (hi - lo) or 1.0
    cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + height), (245, 245, 245), -1)
    cv2.rectangle(canvas, (left, y0), (left + plot_w, y0 + height), (200, 200, 200), 1)

    n = len(values)
    xs = (left + np.arange(n) * (plot_w / max(1, n - 1))).astype(int)
    ys = (y0 + height - (values - lo) / span * (height - 10) - 5).astype(int)
    cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)

    for idx in events:  # red vertical band + dot on anomalies
        x = int(xs[idx])
        cv2.line(canvas, (x, y0), (x, y0 + height), (255, 210, 210), 1)
        cv2.circle(canvas, (x, int(ys[idx])), 4, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{label}   [{lo:.2f} .. {hi:.2f}]   max z {z.max():.1f}",
        (left + 6, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )


def plot(records: list[dict], path: str | Path, title: str = "") -> Path:  # noqa: C901
    """Stacked chart of every series with anomalies marked in red."""
    if not records:
        raise ValueError("nothing to plot")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    left, right, top, bottom = 70, 30, 46, 34
    panel_h, gap = 150, 26
    panels = len(series_for(records))
    width = 1400
    plot_w = width - left - right
    height = top + panels * (panel_h + gap) + bottom
    canvas = np.full((height, width, 3), 255, np.uint8)

    events = [i for i, r in enumerate(records) if r.get("is_event")]
    cv2.putText(
        canvas,
        title or f"optical flow over {len(records)} pairs",
        (left, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    for panel, (name, label, color) in enumerate(series_for(records)):
        values = np.array([r[name] for r in records], dtype=np.float64)
        z = np.array([r.get(f"z_{name}", 0.0) for r in records])
        _panel(
            canvas,
            top + panel * (panel_h + gap),
            panel_h,
            values,
            z,
            label,
            color,
            events,
            left,
            plot_w,
        )

    for k in range(0, len(records), max(1, len(records) // 20)):
        x = int(left + k * (plot_w / max(1, len(records) - 1)))
        cv2.putText(
            canvas,
            _xlabel(records[k]),
            (x - 14, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fallflow.analyze",
        description="Flag anomalous pairs in the flow statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="outputs", help="directory with report.json / stats")
    p.add_argument("--window", type=int, default=15, help="rolling baseline window")
    p.add_argument("--threshold", type=float, default=6.0, help="event score cutoff")
    p.add_argument("--top", type=int, default=10, help="how many events to print")
    p.add_argument(
        "--min-response",
        type=float,
        default=0.2,
        help="phase-correlation response below this = unreliable shift",
    )
    p.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="frames with Laplacian variance below this = too blurry to trust",
    )
    p.add_argument(
        "--min-magnitude",
        type=float,
        default=0.1,
        help="pairs slower than this (px) are treated as static",
    )
    p.add_argument("--no-plot", action="store_true", help="skip the chart")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    records = analyse(
        load_records(out_dir),
        args.window,
        args.min_magnitude,
        args.min_response,
        args.min_sharpness,
        load_frame_quality(out_dir),
    )
    if not records:
        print(f"no stats found in {out_dir} — run `python -m fallflow` first", )
        return 1
    for r in records:
        r["is_event"] = bool(not r.get("static") and r["event_score"] >= args.threshold)

    csv_path = write_csv(out_dir / "analysis" / "anomalies.csv", records)
    (out_dir / "analysis" / "anomalies.json").write_text(json.dumps(records, indent=2))
    print(f"pairs    : {len(records)}")
    print(f"events   : {sum(r['is_event'] for r in records)} (score >= {args.threshold})")
    print(f"static   : {sum(r.get('static', False) for r in records)} pair(s) below "
          f"{args.min_magnitude} px (excluded)")
    print(f"untrusted: {sum(r.get('low_quality', False) for r in records)} pair(s) with "
          f"weak correlation or blur (excluded from the score)")

    print("\ntop anomalies by event score:")
    print(f"  {'pair':>9s} {'score':>6s} {'|v|':>6s} {'mov%':>5s} {'coh':>5s} {'vert':>5s} {'z|v|':>5s} {'zcoh':>5s}")
    for r in top_events(records, args.top):
        print(
            f"  {r['pair'][5:]:>9s} {r['event_score']:6.2f} {r['mean_magnitude']:6.2f} "
            f"{r['moving_pixels_pct']:5.1f} {r['direction_coherence']:5.2f} "
            f"{r['verticality']:5.2f} {r['z_mean_magnitude']:5.1f} "
            f"{r['z_direction_coherence']:5.1f}{'  <== EVENT' if r['is_event'] else ''}"
        )
    print(f"\nanomalies : {csv_path}")

    if not args.no_plot:
        chart = plot(records, out_dir / "analysis" / "chart.png")
        print(f"chart     : {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
