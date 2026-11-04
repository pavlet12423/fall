#!/usr/bin/env python3
"""Скорость катера по видео: отдельный скрипт без зависимостей от репозитория.

Нужно только:
    pip install numpy opencv-python-headless

Запуск (минимальный):
    python boat_speed.py --frames data/frames --start 1 --end 70 --fps 25 --distance 30

Что можно указать из известного (чем больше, тем точнее):
    --height 1.5     высота камеры над водой, м (пассажир с телефоном: 1.2-1.8)
    --focal 914      фокусное расстояние в пикселях
    --fov 70         угол обзора по горизонтали, градусов (если фокус неизвестен)
    --shore-row 326  строка, где вода встречается с берегом (иначе найдётся сама)

Скрипт считает три оценки скорости независимо и печатает для каждой показатель
достоверности. Если ни фокус, ни высота не заданы — печатается таблица чувствительности,
и это честнее, чем выдавать одно число.

Что внутри:
  1. строка уреза воды находится по перепаду яркости;
  2. поток воды по строкам кадра, наклон которого даёт скорость/высоту и не портится
     поворотом катера (поворот сдвигает кадр целиком, наклон остаётся);
  3. длинная база: трассировка точек с первого по последний кадр, поиск статичных
     областей (берег) по когерентности;
  4. существенная матрица + восстановение движения: поворот и перенос разделяются,
     берег в 30 м даёт масштаб. Работает, если известно фокусное расстояние.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- IO


def imread_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"не читается: {path}")
    return image


def frame_paths(frames_dir: Path) -> list[Path]:
    suffixes = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [p for p in frames_dir.iterdir() if p.suffix.lower() in suffixes]
    return sorted(files, key=lambda p: p.name)


# ------------------------------------------------------------------ геометрия


def detect_shoreline(images: list[np.ndarray], smooth: int = 31) -> tuple[int, float]:
    """Строка самого сильного перепада яркости — линия воды на берегу."""
    profile = np.mean([im.mean(axis=1) for im in images], axis=0)
    smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, smooth), 0).ravel()
    gradient = np.gradient(smoothed)
    row = int(np.argmax(np.abs(gradient)))
    return row, float(gradient[row])


def track_chain(
    images: list[np.ndarray], start: int, end: int, max_corners: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Трассировка точек от кадра start к кадру end через все промежуточные.

    Прямой Лукас-Канаде между далёкими кадрами не берёт большие смещения,
    поэтому идём по цепочке кадр-в-кадр и сохраняем только выжившие точки.
    """
    corners = cv2.goodFeaturesToTrack(
        images[start], maxCorners=max_corners, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    if corners is None or len(corners) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    src = corners.reshape(-1, 2).astype(np.float64)
    pts = corners.astype(np.float32)
    alive = np.arange(len(src))
    for k in range(start, end):
        if len(alive) < 20:
            break
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            images[k],
            images[k + 1],
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        status = status.reshape(-1).astype(bool)
        alive = alive[status]
        pts = nxt[status]
        if len(alive) == 0:
            break
    if len(alive) < 20:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return src[alive], pts.reshape(-1, 2).astype(np.float64)


def coherent_cells(
    src: np.ndarray,
    delta: np.ndarray,
    shape: tuple[int, int],
    cols: int = 12,
    rows: int = 8,
    min_points: int = 8,
) -> tuple[list[dict], np.ndarray]:
    """Ячейки, точки которых сместились согласованно — это статичные объекты."""
    h, w = shape
    cells = []
    mask = np.zeros(len(src), bool)
    for r in range(rows):
        for c in range(cols):
            sel = (
                (src[:, 1] >= r * h / rows)
                & (src[:, 1] < (r + 1) * h / rows)
                & (src[:, 0] >= c * w / cols)
                & (src[:, 0] < (c + 1) * w / cols)
            )
            n = int(sel.sum())
            if n < min_points:
                continue
            block = delta[sel]
            lengths = np.linalg.norm(block, axis=1)
            coherence = float(np.linalg.norm(block.mean(axis=0)) / (lengths.mean() + 1e-9))
            median = np.median(block, axis=0)
            cells.append(
                {
                    "row": r,
                    "col": c,
                    "points": n,
                    "coherence": coherence,
                    "dx": float(median[0]),
                    "dy": float(median[1]),
                    "magnitude": float(np.linalg.norm(median)),
                }
            )
            if coherence >= 0.9:
                mask |= sel
    return sorted(cells, key=lambda c: -c["coherence"]), mask


# -------------------------------------------------------------- оценки скорости


def water_plane_profile(
    frames_dir: Path,
    paths: list[Path],
    start: int,
    end: int,
    water_rows: tuple[float, float],
    bins: int = 36,
    max_corners: int = 800,
) -> list[dict]:
    """Поток воды по строкам, усреднённый по всем парам кадра.

    Возвращает медианные смещения по полосам строк. Никакой опорной точки не
    вычитается: поворот катера сдвигает весь кадр целиком и уйдёт в свободный
    член линейной аппроксимации, а не в наклон.
    """
    images: dict[int, np.ndarray] = {}
    acc_dx: list[list[float]] = [[] for _ in range(bins)]
    acc_dy: list[list[float]] = [[] for _ in range(bins)]
    low, high = water_rows
    edges = np.linspace(low, high, bins + 1)

    for i in range(start, end):
        for idx in (i, i + 1):
            if idx not in images:
                images[idx] = imread_gray(paths[idx - 1])
        corners = cv2.goodFeaturesToTrack(
            images[i], maxCorners=max_corners, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        if corners is None or len(corners) < 20:
            images.pop(min(images), None)
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            images[i],
            images[i + 1],
            corners,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        status = status.reshape(-1).astype(bool)
        a = corners.reshape(-1, 2)[status]
        b = nxt.reshape(-1, 2)[status]
        delta = b - a
        which = np.clip(np.digitize(a[:, 1], edges) - 1, 0, bins - 1)
        for k in range(bins):
            sel = which == k
            if sel.sum() >= 5:
                acc_dx[k].append(float(np.median(delta[sel][:, 0])))
                acc_dy[k].append(float(np.median(delta[sel][:, 1])))
        images.pop(min(images), None)

    return [
        {
            "row": float((edges[k] + edges[k + 1]) / 2),
            "samples": int(len(acc_dx[k])),
            "dx": float(np.median(acc_dx[k])) if acc_dx[k] else 0.0,
            "dy": float(np.median(acc_dy[k])) if acc_dy[k] else 0.0,
        }
        for k in range(bins)
    ]


def fit_line(profile: list[dict]) -> dict:
    """dx = slope * row + c.  slope = V * dt / h,  c вбирает поворот камеры."""
    rows = np.array([r["row"] for r in profile if r["samples"] > 0])
    dx = np.array([r["dx"] for r in profile if r["samples"] > 0])
    w = np.array([min(r["samples"], 40) for r in profile if r["samples"] > 0], float)
    if len(rows) < 4:
        return {"slope": 0.0, "c": 0.0, "r2": 0.0, "horizon": 0.0, "points": len(rows)}
    x = np.vstack([rows, np.ones_like(rows)]).T
    coef = np.linalg.lstsq(x * w[:, None], dx * w, rcond=None)[0]
    resid = dx - x @ coef
    mean = np.average(dx, weights=w)
    r2 = 1.0 - float(np.sum(w * resid**2) / np.sum(w * (dx - mean) ** 2))
    slope, c = float(coef[0]), float(coef[1])
    return {
        "slope": slope,
        "c": c,
        "r2": r2,
        "horizon": float(-c / slope) if abs(slope) > 1e-12 else float("nan"),
        "points": int(len(rows)),
    }


def estimate_from_water(fit: dict, height: float, fps: float) -> dict:
    """V = slope * h * fps. Фокусное расстояние не нужно — его заменяет высота."""
    if not height or not fit["slope"]:
        return {"v": None}
    v = fit["slope"] * height * fps
    return {"v": float(v), "kmh": float(v * 3.6)}


def estimate_from_shore(
    cells: list[dict], focal: float, distance: float, dt: float
) -> dict:
    """V = D * (сдвиг берега / f) / dt — для бокового перемещения мимо берега.

    Поворот катера эту оценку портит точно так же, как перенос, поэтому она
    годится только вместе с оценкой поворота (см. estimate_from_essential).
    """
    if not focal:
        return {"v": None}
    strong = [c for c in cells if c["coherence"] >= 0.9 and c["points"] >= 8]
    if not strong:
        return {"v": None}
    shift = float(np.median([c["magnitude"] for c in strong]))
    v = distance * shift / focal / dt
    return {"v": float(v), "kmh": float(v * 3.6), "shift_px": shift, "cells": len(strong)}


def estimate_from_essential(
    src: np.ndarray,
    dst: np.ndarray,
    focal: float,
    center: tuple[float, float],
    distance: float,
    dt: float,
    static_mask: np.ndarray | None = None,
) -> dict:
    """Разделение поворота и переноса через существенную матрицу.

    Требует фокусное расстояние. Масштаб берётся из известного расстояния до
    берега: восстановленные глубины точек берега приводятся к 30 м.
    """
    if not focal or len(src) < 30:
        return {"v": None}
    p1 = src if static_mask is None else src[static_mask]
    p2 = dst if static_mask is None else dst[static_mask]
    if len(p1) < 30:
        p1, p2 = src, dst

    e, emask = cv2.findEssentialMat(
        p1, p2, focal=focal, pp=center, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if e is None:
        return {"v": None}
    _, r, t, pmask = cv2.recoverPose(e, p1, p2, focal=focal, pp=center)
    inliers = pmask.reshape(-1).astype(bool) if pmask is not None else np.ones(len(p1), bool)
    if inliers.sum() < 20:
        return {"v": None, "inliers": int(inliers.sum())}

    k = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], float)
    p1m = np.hstack([np.eye(3), np.zeros((3, 1))])
    p2m = np.hstack([r, t.reshape(3, 1)])
    proj1 = k @ p1m
    proj2 = k @ p2m
    pts1 = p1[inliers].astype(np.float64).T
    pts2 = p2[inliers].astype(np.float64).T
    cloud = cv2.triangulatePoints(proj1, proj2, pts1, pts2)
    cloud = cloud[:3] / cloud[3]
    depth = cloud[2]
    positive = depth > 0
    if positive.sum() < 20:
        return {"v": None, "inliers": int(inliers.sum())}

    # масштаб: медианная глубина статичных точек берега = distance метров
    scale = distance / float(np.median(depth[positive]))
    displacement = float(np.linalg.norm(t)) * scale  # |t| == 1 после recoverPose
    v = displacement / dt

    rvec = cv2.Rodrigues(r)[0].ravel()
    return {
        "v": float(v),
        "kmh": float(v * 3.6),
        "displacement_m": displacement,
        "yaw_deg": float(np.degrees(rvec[1])),
        "pitch_deg": float(np.degrees(rvec[0])),
        "roll_deg": float(np.degrees(rvec[2])),
        "inliers": int(inliers.sum()),
        "inlier_share": float(inliers.mean()),
        "translation_dir": (t.reshape(-1) / np.linalg.norm(t)).round(3).tolist(),
    }


# -------------------------------------------------------------------- отрисовка


def draw_profile(profile: list[dict], fit: dict, path: Path, water_rows: tuple[float, float]) -> Path:
    width, height = 1100, 520
    canvas = np.full((height, width, 3), 255, np.uint8)
    rows = np.array([r["row"] for r in profile])
    dx = np.array([r["dx"] for r in profile])
    lo, hi = float(dx.min()), float(dx.max())
    span = (hi - lo) or 1.0
    left, top = 70, 60
    plot_w, plot_h = width - left - 30, height - top - 70
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (247, 247, 247), -1)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (200, 200, 200), 1)
    xs = (left + (rows - water_rows[0]) / max(1e-6, water_rows[1] - water_rows[0]) * plot_w).astype(int)
    ys = (top + plot_h - (dx - lo) / span * (plot_h - 16) - 8).astype(int)
    cv2.polylines(canvas, [np.stack([xs, ys], 1)], False, (0, 90, 220), 2, cv2.LINE_AA)
    for x, y in zip(xs, ys):
        cv2.circle(canvas, (int(x), int(y)), 3, (0, 90, 220), -1, cv2.LINE_AA)
    if fit["slope"]:
        model = fit["slope"] * rows + fit["c"]
        my = (top + plot_h - (model - lo) / span * (plot_h - 16) - 8).astype(int)
        cv2.polylines(canvas, [np.stack([xs, my], 1)], False, (60, 60, 220), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"water flow vs image row: slope {fit['slope']:.5f} px/frame per px, R^2 {fit['r2']:.2f}",
        (left, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"rows {water_rows[0]:.0f} .. {water_rows[1]:.0f}",
        (left, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)
    return path


def draw_map(
    image: np.ndarray, cells: list[dict], path: Path, cols: int = 12, rows: int = 8
) -> Path:
    h, w = image.shape
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for c in cells:
        x0, y0 = int(c["col"] * w / cols), int(c["row"] * h / rows)
        x1, y1 = int((c["col"] + 1) * w / cols), int((c["row"] + 1) * h / rows)
        good = c["coherence"] >= 0.9
        color = (60, 220, 60) if good else (60, 60, 200)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            canvas,
            f"{c['magnitude']:.0f}",
            (x0 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        centre = ((x0 + x1) // 2, (y0 + y1) // 2)
        tip = (int(centre[0] + c["dx"] * 0.25), int(centre[1] + c["dy"] * 0.25))
        cv2.arrowedLine(canvas, centre, tip, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(
        canvas,
        "green = coherence >= 0.9 (static: shore), arrows = shift x0.25",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green = coherence >= 0.9 (static: shore), arrows = shift x0.25",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)
    return path


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Скорость катера по видео (нужны numpy и opencv-python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--frames", default="data/frames", help="папка с кадрами")
    p.add_argument("--out", default="boat_speed_out", help="куда писать отчёт и графики")
    p.add_argument("--start", type=int, default=1, help="первый кадр")
    p.add_argument("--end", type=int, default=0, help="последний кадр (0 = все)")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--distance", type=float, default=30.0, help="расстояние до берега, м")
    p.add_argument("--height", type=float, default=0.0, help="высота камеры над водой, м")
    p.add_argument("--focal", type=float, default=0.0, help="фокусное расстояние, пиксели")
    p.add_argument("--fov", type=float, default=0.0, help="угол обзора по ширине, град (если нет фокуса)")
    p.add_argument("--shore-row", type=float, default=0.0, help="строка уреза воды (0 = найти)")
    p.add_argument("--bins", type=int, default=36, help="полос строк для профиля")
    p.add_argument("--corners", type=int, default=1500, help="сколько точек трассировать")
    p.add_argument("--window", type=int, default=0,
                   help="считать по скользящим окнам такой длины (0 = одна база)")
    p.add_argument("--step", type=int, default=5, help="шаг окна в кадрах")
    p.add_argument("--max-yaw", type=float, default=25.0,
                   help="окна с рысканьем больше этого градуса отбрасываются")
    p.add_argument("--no-viz", action="store_true", help="не писать графики")
    return p


def main() -> int:
    args = build_parser().parse_args()
    frames_dir = Path(args.frames)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = frame_paths(frames_dir)
    if len(paths) < 2:
        print(f"в {frames_dir} меньше двух кадров")
        return 1
    end = args.end or len(paths)
    if end > len(paths):
        print(f"кадров всего {len(paths)}, --end уменьшен до {len(paths)}")
        end = len(paths)

    first = imread_gray(paths[args.start - 1])
    h, w = first.shape
    center = (w / 2.0, h / 2.0)
    duration = (end - args.start) / args.fps

    focal = args.focal
    if not focal and args.fov:
        focal = (w / 2.0) / math.tan(math.radians(args.fov) / 2.0)

    print(f"кадры      : {args.start}..{end} ({len(paths)} в папке), {args.fps:g} fps, {duration:.2f} с")
    print(f"размер     : {w}x{h}")
    print(f"берег      : {args.distance:g} м")
    print(f"камера     : f = {focal:.0f} px" if focal else "камера     : фокус НЕ задан")
    print(f"высота     : {args.height:g} м" if args.height else "высота     : НЕ задана")

    # 1. где вода встречается с берегом
    sample = [imread_gray(paths[i - 1]) for i in range(args.start, end + 1, max(1, (end - args.start) // 10 or 1))]
    shore_row, gradient = (
        (int(args.shore_row), 0.0) if args.shore_row else detect_shoreline(sample)
    )
    print(f"урез воды  : строка {shore_row} (перепад яркости {gradient:+.2f})")
    water_rows = (shore_row + 15.0, float(h))

    # 2. профиль потока воды по строкам
    profile = water_plane_profile(frames_dir, paths, args.start, end, water_rows, args.bins, args.corners)
    fit = fit_line(profile)
    print()
    print(f"профиль    : {fit['points']} полос, наклон {fit['slope']:.5f} px/кадр на px, R^2 = {fit['r2']:.2f}")
    if fit["slope"]:
        print(f"             горизонт по аппроксимации: строка {fit['horizon']:.0f}")

    # 3. длинная база: статичные области (берег)
    images = [imread_gray(paths[i - 1]) for i in range(args.start, end + 1)]
    src, dst = track_chain(images, 0, len(images) - 1, args.corners)
    print(f"трассировка: {len(src)} точек дожило от кадра {args.start} до {end}")
    delta = dst - src if len(src) else np.zeros((0, 2))
    cells, static_mask = (
        coherent_cells(src, delta, (h, w)) if len(src) else ([], np.zeros(0, bool))
    )
    strong = [c for c in cells if c["coherence"] >= 0.9 and c["points"] >= 8]
    print(f"берег      : {len(strong)} согласованных ячеек, сдвиг "
          f"{np.median([c['magnitude'] for c in strong]) if strong else 0:.0f} px за {duration:.2f} с")

    # 4. три оценки
    water = estimate_from_water(fit, args.height, args.fps)
    shore = estimate_from_shore(cells, focal, args.distance, duration)
    essential = estimate_from_essential(
        src, dst, focal, center, args.distance, duration, static_mask if static_mask.any() else None
    )

    print()
    print("=" * 72)
    print("ОЦЕНКИ СКОРОСТИ")
    print("=" * 72)
    if water.get("v"):
        print(f"  1. по наклону потока воды (нужна высота, фокус не нужен)")
        print(f"     V = {water['v']:.2f} м/с = {water['kmh']:.1f} км/ч   при h = {args.height:g} м")
        print(f"     достоверность: R^2 = {fit['r2']:.2f} "
              f"({'надёжно' if fit['r2'] > 0.6 else 'СЛАБО, не доверять'})")
    else:
        print("  1. по наклону потока воды: нужно --height (высота камеры над водой)")

    if shore.get("v"):
        print(f"  2. по угловой скорости берега (нужен фокус)")
        print(f"     V = {shore['v']:.2f} м/с = {shore['kmh']:.1f} км/ч")
        print(f"     но поворот катера входит в ту же величину — это ВЕРХНЯЯ граница")
    else:
        print("  2. по угловой скорости берега: нужен --focal или --fov")

    if essential.get("v"):
        print(f"  3. разделение поворота и переноса (нужен фокус) — основная оценка")
        print(f"     V = {essential['v']:.2f} м/с = {essential['kmh']:.1f} км/ч")
        print(f"     перемещение {essential['displacement_m']:.1f} м за {duration:.2f} с")
        print(f"     поворот: рысканье {essential['yaw_deg']:+.1f}°, "
              f"тангаж {essential['pitch_deg']:+.1f}°, крен {essential['roll_deg']:+.1f}°")
        print(f"     согласных точек {essential['inliers']} ({essential['inlier_share'] * 100:.0f}%)")
    else:
        print("  3. разделение поворота и переноса: нужен --focal или --fov "
              f"(сейчас: {essential.get('inliers', 0)} согласных точек)")

    if focal:
        print()
        print("  Чувствительность основной оценки к углу обзора (он неизвестен — это и есть погрешность):")
        base_fov = args.fov or math.degrees(2 * math.atan((w / 2.0) / focal))
        print(f"  {'FOV':>5s} {'f,px':>7s} {'V, м/с':>9s} {'V, км/ч':>9s} {'рысканье':>10s}")
        for fov in (base_fov - 15, base_fov - 10, base_fov - 5, base_fov,
                    base_fov + 5, base_fov + 10, base_fov + 15):
            if fov < 20 or fov > 120:
                continue
            f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
            est = estimate_from_essential(
                src, dst, f, center, args.distance, duration,
                static_mask if static_mask.any() else None,
            )
            if est.get("v") is None:
                continue
            mark = " <-- задано" if abs(fov - base_fov) < 0.01 else ""
            print(f"  {fov:5.0f} {f:7.0f} {est['v']:9.2f} {est['kmh']:9.1f} {est['yaw_deg']:+9.1f}°{mark}")

    if not focal and not args.height:
        print()
        print("  Ни фокус, ни высота не заданы — таблица чувствительности (это не измерение):")
        print(f"  {'FOV':>5s} {'f,px':>7s} | {'V (наклон, h=1.5м)':>20s} | {'V (берег)':>10s}")
        for fov in (50, 60, 70, 80, 90):
            f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
            v_water = fit["slope"] * 1.5 * args.fps if fit["slope"] else float("nan")
            v_shore = args.distance * (np.median([c["magnitude"] for c in strong]) if strong else 0) / f / duration
            print(f"  {fov:5.0f} {f:7.0f} | {v_water:17.2f} м/с | {v_shore:8.2f} м/с")

    # 4b. надёжный режим: много коротких окон и медиана по ним
    windows_report: dict = {}
    if args.window and focal and len(images) > args.window:
        print()
        print("=" * 72)
        print(f"ОКНА: длина {args.window} кадров, шаг {args.step}")
        print("=" * 72)
        speeds, yaws, shares = [], [], []
        for begin in range(0, len(images) - args.window, max(1, args.step)):
            wsrc, wdst = track_chain(images, begin, begin + args.window, args.corners)
            if len(wsrc) < 40:
                continue
            wdelta = wdst - wsrc
            wcells, wmask = coherent_cells(wsrc, wdelta, (h, w))
            wdt = args.window / args.fps
            est = estimate_from_essential(
                wsrc, wdst, focal, center, args.distance, wdt,
                wmask if wmask.any() else None,
            )
            if est.get("v") is None:
                continue
            ok = (
                est.get("inlier_share", 0) >= 0.5
                and abs(est.get("yaw_deg", 999)) <= args.max_yaw
            )
            speeds.append(est["v"])
            yaws.append(est["yaw_deg"])
            shares.append(est["inlier_share"])
            print(
                f"  кадры {args.start + begin:3d}..{args.start + begin + args.window:3d}: "
                f"V = {est['v']:5.2f} м/с ({est['kmh']:5.1f} км/ч), "
                f"рысканье {est['yaw_deg']:+6.1f}°, согласных {est['inlier_share'] * 100:3.0f}%"
                f"{'' if ok else '   <- отброшено'}"
            )
            if not ok:
                speeds.pop(); yaws.pop(); shares.pop()
        if speeds:
            arr = np.array(speeds)
            median = float(np.median(arr))
            mad = float(np.median(np.abs(arr - median)))
            print()
            print(f"  принято окон: {len(arr)} из {max(1, len(range(0, len(images) - args.window, max(1, args.step))))}")
            print(f"  МЕДИАНА СКОРОСТИ: {median:.2f} м/с = {median * 3.6:.1f} км/ч")
            print(f"  разброс (MAD): ±{mad:.2f} м/с, диапазон {arr.min():.2f}..{arr.max():.2f} м/с")
            if len(arr) >= 4:
                print(f"  доверительный интервал грубо: "
                      f"{max(0, median - 2 * mad):.1f}..{median + 2 * mad:.1f} м/с")
            print("  Если медиана и отдельные окна расходятся сильно — доверять медиане,")
            print("  но помнить: фокусное расстояние задано неточно, оно масштабирует ответ.")
            windows_report = {
                "count": len(arr),
                "median_v": median,
                "mad_v": mad,
                "min_v": float(arr.min()),
                "max_v": float(arr.max()),
                "speeds": [float(x) for x in arr],
                "yaws": [float(x) for x in yaws],
            }
            windows_report = {"count": 0}
        else:
            windows_report = {"count": 0}
            print("  ни одно окно не прошло проверку — попробуйте --window поменьше")
            print("  или участок, где катер идёт без разворота")

    # 5. калибровка фокуса, если известны высота и строка уреза
    if args.height and fit["slope"] and not focal:
        implied_f = args.distance * (shore_row - fit["horizon"]) / args.height
        plausible = 0 < fit["horizon"] < shore_row and 200 < implied_f < 5000
        if not plausible:
            print()
            print(f"  Калибровка по высоте не сошлась: аппроксимация даёт горизонт на строке "
                  f"{fit['horizon']:.0f} (должен быть между 0 и {shore_row}).")
            print("  Значит наклон потока воды не описывается моделью плоскости — оценка 1 недостоверна.")
        elif implied_f > 0:
            implied_fov = math.degrees(2 * math.atan((w / 2.0) / implied_f))
            print()
            print(f"  Из высоты {args.height:g} м и строки уреза {shore_row} следует "
                  f"f = {implied_f:.0f} px (FOV {implied_fov:.1f}°) — проверьте правдоподобие")

    # 6. отчёт
    report = {
        "frames": {"dir": str(frames_dir), "start": args.start, "end": end, "fps": args.fps,
                   "duration_s": duration, "width": w, "height": h},
        "input": {"distance_m": args.distance, "height_m": args.height, "focal_px": focal,
                  "shore_row": shore_row},
        "water_plane": {**fit, **water},
        "shore_rate": shore,
        "essential": essential,
        "cells": cells[:20],
        "profile": profile,
        "windows": windows_report,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with (out_dir / "profile.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["row", "samples", "dx", "dy"])
        writer.writeheader()
        writer.writerows(profile)

    if not args.no_viz:
        p1 = draw_profile(profile, fit, out_dir / "profile.png", water_rows)
        if len(src):
            p2 = draw_map(images[0], cells, out_dir / "long_baseline.png", 12, 8)
            print(f"\nграфики    : {p1}\n             {p2}")
        else:
            print(f"\nграфики    : {p1}")
    print(f"отчёт      : {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
