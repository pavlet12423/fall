#!/usr/bin/env python3
"""Скорость судна относительно берега по всему ролику.

    pip install numpy opencv-python-headless
    python shore_speed.py --frames data/frames --fps 25 --distance 30 --fov 70

Идея. Берег — статичный объект на известном расстоянии D. Его смещение в кадре
даёт УГЛОВУЮ скорость взаимного движения «камера <-> берег»:

        w_app = (dx_px / f) * fps      [рад/с]

Если камера не поворачивается, то поперечная скорость равна V = w_app * D.
Если камера ещё и рыскает, то w_app = V/D + w_рыск, то есть V = D*(w_app - w_рыск).
Поэтому результат по берегу — ВЕРХНЯЯ ОЦЕНКА поперечной скорости.

От рысканья можно избавиться другим путём — по воде. Для точки воды на строке y
    Z(y) = h*f/(y - y_гор),   px/с(y) = f*V/Z = V*(y - y_гор)/h
То есть скорость потока воды ЛИНЕЙНО растёт со строкой, и наклон равен V/h.
Поворот камеры сдвигает весь график целиком (одинаково для всех строк), но
наклон не меняет. Значит наклон даёт V, не требующий знания фокуса:
    V = slope * h * fps        (slope в px/кадр на строку)

Скрипт считает обе оценки и печатает, насколько они согласуются.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


# ----------------------------------------------------------------- загрузка
def frame_paths(frames_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for e in exts:
        paths.extend(sorted(frames_dir.glob(e)))
    if not paths:
        raise SystemExit(f"нет кадров в {frames_dir}")
    return paths


def read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"не читается {p}")
    return img


# ------------------------------------------------------------------ берег
def detect_shoreline(g: np.ndarray) -> int:
    """Строка уреза воды: самый резкий рост яркости сверху вниз."""
    prof = g.mean(axis=1).astype(np.float64)
    prof = np.convolve(prof, np.ones(9) / 9.0, mode="same")
    lo, hi = int(g.shape[0] * 0.10), int(g.shape[0] * 0.90)
    d = np.diff(prof)
    return int(lo + int(np.argmax(d[lo:hi])))


# --------------------------------------------------------------- трассировка
def track_band(prev: np.ndarray, curr: np.ndarray, band, corners: int):
    """Трассирует точки внутри полосы band=(y0,y1). Возвращает (y, dx, dy)."""
    y0, y1 = band
    if y1 - y0 < 10:
        return np.zeros((0, 3))
    mask = np.zeros(prev.shape, np.uint8)
    mask[y0:y1, :] = 255
    p0 = cv2.goodFeaturesToTrack(prev, maxCorners=corners, qualityLevel=0.008,
                                 minDistance=8, mask=mask, blockSize=9)
    if p0 is None or len(p0) < 8:
        return np.zeros((0, 3))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, winSize=(25, 25),
                                         maxLevel=5,
                                         criteria=(cv2.TERM_CRITERIA_EPS |
                                                   cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if p1 is None:
        return np.zeros((0, 3))
    back, stb, _ = cv2.calcOpticalFlowPyrLK(curr, prev, p1, None, winSize=(25, 25),
                                            maxLevel=5,
                                            criteria=(cv2.TERM_CRITERIA_EPS |
                                                      cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if back is None:
        return np.zeros((0, 3))
    a = p0.reshape(-1, 2)
    b = p1.reshape(-1, 2)
    c = back.reshape(-1, 2)
    err = np.hypot(c[:, 0] - a[:, 0], c[:, 1] - a[:, 1])
    ok = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & (err < 1.0)
    if ok.sum() < 8:
        return np.zeros((0, 3))
    return np.column_stack([a[ok, 1], b[ok, 0] - a[ok, 0], b[ok, 1] - a[ok, 1]])


def robust_slope(y: np.ndarray, dx: np.ndarray, bin_h: int = 20):
    """Наклон dx(px/кадр) от строки y по медианам в полосах. Возвращает
    (slope, intercept, r2, n_bins)."""
    if len(y) < 50:
        return None
    h = int(y.max()) + 1
    edges = np.arange(0, h + bin_h, bin_h)
    ys, xs = [], []
    for i in range(len(edges) - 1):
        m = (y >= edges[i]) & (y < edges[i + 1])
        if m.sum() < 15:
            continue
        ys.append(float(np.median(y[m])))
        xs.append(float(np.median(dx[m])))
    if len(ys) < 4:
        return None
    ys = np.array(ys)
    xs = np.array(xs)
    A = np.vstack([ys, np.ones_like(ys)]).T
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    pred = A @ coef
    ss_res = float(((xs - pred) ** 2).sum())
    ss_tot = float(((xs - xs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    return float(coef[0]), float(coef[1]), float(r2), len(ys)


def mad(a: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


# ------------------------------------------------------------------- график
def plot(out_png: Path, times, v_shore, v_water, px_land, slope_r2, fps: int,
         min_r2: float = 0.7) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax[0].plot(times, px_land, lw=1.2, color="#4da3ff")
    ax[0].axhline(0, color="#555", lw=0.8)
    ax[0].set_ylabel("берег: сдвиг,\nпикселей за кадр")
    ax[0].set_title("Движение берега относительно камеры (отрицательное = судно идёт вправо)")
    ax[0].grid(alpha=0.25)

    ax[1].plot(times, np.abs(v_shore), lw=1.4, color="#4ad07a",
               label="берег (верхняя оценка, включает рысканье)")
    acc = slope_r2 >= min_r2
    ax[1].plot(times[acc], np.abs(v_water[acc]), "o", ms=4, color="#ffb454",
               label=f"вода, R² ≥ {min_r2:g} (не зависит от рысканья)")
    ax[1].plot(times[~acc], np.abs(v_water[~acc]), "x", ms=4, color="#666",
               label="вода, отброшено")
    if acc.sum():
        med = float(np.median(np.abs(v_water[acc])))
        ax[1].axhline(med, color="#ffb454", ls="--", lw=1.2)
        ax[1].text(times[-1], med, f"  медиана {med:.2f} м/с ({med*3.6:.1f} км/ч)",
                   va="center", ha="right", color="#ffb454", fontsize=9)
    ax[1].set_ylabel("м/с")
    ax[1].set_title("Скорость судна")
    ax[1].legend(loc="upper left", fontsize=9)
    ax[1].grid(alpha=0.25)

    ax[2].plot(times, slope_r2, lw=1.0, color="#ffb454")
    ax[2].set_ylabel("R²")
    ax[2].set_xlabel("время, с")
    ax[2].set_title("Достоверность оценки по воде (наклон потока по строкам)")
    ax[2].set_ylim(0, 1)
    ax[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser(description="Скорость судна относительно берега.")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", default="shore_speed_out")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--distance", type=float, default=30.0, help="расстояние до берега, м")
    p.add_argument("--height", type=float, default=1.5, help="высота камеры над водой, м")
    p.add_argument("--focal", type=float, default=0.0, help="фокус, пиксели")
    p.add_argument("--fov", type=float, default=0.0, help="угол обзора по ширине, град")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--step", type=int, default=1, help="база трассировки, кадров")
    p.add_argument("--corners", type=int, default=800)
    p.add_argument("--shore-row", type=float, default=0.0)
    p.add_argument("--min-r2", type=float, default=0.7,
                   help="порог достоверности подгонки по воде")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    paths = frame_paths(Path(args.frames))
    i0 = max(0, args.start - 1)
    i1 = len(paths) if args.end <= 0 else min(len(paths), args.end)
    paths = paths[i0:i1]
    if len(paths) < args.step + 2:
        raise SystemExit("слишком мало кадров")

    first = read_gray(paths[0])
    H, W = first.shape
    shore = int(args.shore_row) if args.shore_row > 0 else detect_shoreline(first)

    f = args.focal
    if f <= 0:
        if args.fov <= 0:
            raise SystemExit("нужен --focal или --fov")
        f = (W / 2.0) / np.tan(np.radians(args.fov) / 2.0)

    land_band = (max(0, shore - 150), max(10, shore - 12))
    water_band = (min(H - 5, shore + 30), H - 5)

    print(f"кадры      : {args.start}..{i1} ({len(paths)} шт), {args.fps:g} fps, "
          f"{(len(paths)-1)/args.fps:.2f} с")
    print(f"размер     : {W}x{H}")
    print(f"урез воды  : строка {shore}")
    print(f"берег (полоса трассировки): строки {land_band[0]}..{land_band[1]}")
    print(f"вода  (полоса трассировки): строки {water_band[0]}..{water_band[1]}")
    print(f"камера     : f = {f:.0f} px" + (f" (FOV {args.fov:g}°)" if args.fov else ""))
    print(f"расстояние : {args.distance:g} м, высота {args.height:g} м")
    print()

    prev = first
    rows = []
    water_all_y = []
    water_all_dx = []
    for i in range(1, len(paths), args.step):
        j = min(i + args.step, len(paths) - 1)
        if j == i:
            break
        curr = read_gray(paths[j])
        base = j - (i - 1)

        land = track_band(prev, curr, land_band, args.corners)
        water = track_band(prev, curr, water_band, args.corners)

        wsp = robust_slope(water[:, 0], water[:, 1] / base) if len(water) else None
        v_water = float(wsp[0] * args.height * args.fps) if wsp else float("nan")
        r2 = wsp[2] if wsp else float("nan")
        horizon = float(-wsp[1] / wsp[0]) if wsp and abs(wsp[0]) > 1e-9 else float("nan")

        if len(water) > 100:
            water_all_y.append(water[:, 0])
            water_all_dx.append(water[:, 1] / base)

        dx_land = float(np.median(land[:, 1])) / base if len(land) else float("nan")
        w_app = dx_land * args.fps / f
        rows.append({
            "frame": j + args.start - 1 + 1 - 1,
            "t": (j) / args.fps,
            "n_land": int(len(land)),
            "n_water": int(len(water)),
            "dx_land": dx_land,
            "dy_land": float(np.median(land[:, 2])) / base if len(land) else float("nan"),
            "w_app_deg_s": float(np.degrees(w_app)),
            "v_shore": float(w_app * args.distance),
            "v_water": v_water,
            "r2_water": r2,
            "horizon_row": horizon,
        })
        prev = curr

    t = np.array([r["t"] for r in rows])
    dx = np.array([r["dx_land"] for r in rows])
    vs = np.array([r["v_shore"] for r in rows])
    vw = np.array([r["v_water"] for r in rows])
    r2s = np.array([r["r2_water"] for r in rows])

    good = ~np.isnan(vs)
    print("=" * 72)
    print("РЕЗУЛЬТАТ")
    print("=" * 72)
    if good.sum() == 0:
        print("берег не трассируется — оценка невозможна")
        return 1

    v_med = float(np.median(vs[good]))
    v_mad = mad(vs[good])
    print(f"  оценка по берегу (верхняя, если рысканья нет):")
    print(f"     медиана V = {abs(v_med):.2f} м/с = {abs(v_med)*3.6:.1f} км/ч")
    print(f"     разброс MAD = ±{v_mad:.2f} м/с, диапазон "
          f"{np.nanmin(np.abs(vs[good])):.2f}..{np.nanmax(np.abs(vs[good])):.2f} м/с")
    print(f"     угловая скорость берега: {np.median(np.degrees(vs[good]/args.distance)):.1f} °/с")
    print(f"     кадров с трассировкой берега: {good.sum()} из {len(rows)}")
    print()

    wacc = (~np.isnan(vw)) & (~np.isnan(r2s)) & (r2s >= args.min_r2)
    wgood = ~np.isnan(vw)
    if wgood.sum() > 5:
        print(f"  оценка по воде (наклон потока; не зависит ни от фокуса, ни от рысканья):")
        print(f"     по всем кадрам : медиана R² = {np.nanmedian(r2s[wgood]):.2f}, "
              f"V = {abs(np.nanmedian(vw[wgood])):.2f} м/с")
        if wacc.sum() >= 5:
            wm = float(np.median(vw[wacc]))
            print(f"     по надёжным    : {wacc.sum()} кадров из {len(rows)} "
                  f"(R² ≥ {args.min_r2:g})")
            print(f"     МЕДИАНА V = {abs(wm):.2f} м/с = {abs(wm)*3.6:.1f} км/ч")
            print(f"     разброс  MAD = ±{mad(vw[wacc]):.2f} м/с")
            print(f"     горизонт из подгонки: {np.nanmedian(np.array([r['horizon_row'] for r in rows])[wacc]):.0f} строка "
                  f"(урез воды на {shore})")
            print("     -> ЭТУ оценку и следует брать за скорость судна")
        else:
            print(f"     надёжных кадров мало ({wacc.sum()}) — оценка по воде недостоверна")
        print()

    # чувствительность к углу обзора
    print("  чувствительность оценки по берегу к углу обзора:")
    print("     FOV     f,px    V, м/с   V, км/ч")
    for fov in (50, 60, 70, 80, 90):
        ff = (W / 2.0) / np.tan(np.radians(fov) / 2.0)
        vv = float(np.median(dx[good])) * args.fps / ff * args.distance
        print(f"     {fov:3d}°   {ff:6.0f}   {abs(vv):6.2f}   {abs(vv)*3.6:6.1f}")
    print()

    # по отрезкам
    print("  по отрезкам ролика (оценка по берегу):")
    n = max(1, len(rows) // 6)
    for k in range(0, len(rows), n):
        sl = slice(k, min(k + n, len(rows)))
        seg = vs[sl]
        seg = seg[~np.isnan(seg)]
        m = wacc[sl]
        wseg = vw[sl][m]
        wtxt = f", вода {abs(np.median(wseg)):5.2f} м/с ({m.sum():3d} кадров)" if m.sum() >= 3 else ""
        if len(seg) == 0:
            continue
        print(f"     t = {t[sl][0]:5.2f}..{t[sl][-1]:5.2f} с   "
              f"берег {abs(np.median(seg)):5.2f} м/с ({len(seg):3d} кадров){wtxt}")
    print()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "series.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "report.json").write_text(json.dumps({
        "frames": f"{args.start}..{i1}",
        "fps": args.fps,
        "shore_row": shore,
        "land_band": list(land_band),
        "water_band": list(water_band),
        "focal_px": float(f),
        "fov_deg": args.fov or None,
        "distance_m": args.distance,
        "height_m": args.height,
        "v_shore_median_ms": v_med,
        "v_shore_mad_ms": v_mad,
        "v_shore_min_ms": float(np.nanmin(vs[good])),
        "v_shore_max_ms": float(np.nanmax(vs[good])),
        "yaw_deg_s": float(np.median(np.degrees(vs[good] / args.distance))),
        "v_water_median_ms": float(np.median(vw[wacc])) if wacc.sum() >= 5 else None,
        "v_water_mad_ms": float(mad(vw[wacc])) if wacc.sum() >= 5 else None,
        "v_water_accepted_frames": int(wacc.sum()),
        "r2_water_median": float(np.nanmedian(r2s[wgood])) if wgood.sum() else None,
        "height_sensitivity_ms": {f"{hh:g}": abs(float(np.median(vw[wacc])) / args.height * hh)
                                  for hh in (1.0, 1.2, 1.5, 1.8, 2.0, 2.5)} if wacc.sum() >= 5 else None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_viz:
        png = out / "shore_speed.png"
        if plot(png, t, vs, vw, dx, np.nan_to_num(r2s), args.fps, args.min_r2):
            print(f"график     : {png}")
        else:
            print("график     : matplotlib не установлен, пропущен")
    print(f"таблица    : {out/'series.csv'}")
    print(f"отчёт      : {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
