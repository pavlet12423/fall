#!/usr/bin/env python3
"""Локальный инструмент: пипетка по видео + графики скорости точки.

    pip install numpy opencv-python-headless
    python probe.py --frames data/frames --fps 25

Откроется http://127.0.0.1:8765 — в браузере кадры, пипетка (с лупой для точного
попадания), траектория точки и графики скорости.

Как считается скорость:
  * точка трассируется Лукасом-Канаде от кадра, по которому кликнули, вперёд и
    назад; есть проверка «туда-обратно», потерянные кадры помечаются;
  * смещения в пикселях переводятся в м/с одним из двух способов:
      «водная поверхность»: v = v_px * h / (y - y_горизонт)   (фокус не нужен)
      «известное расстояние»: v = v_px * D / f                (нужен фокус/FOV)
    где y — строка точки, h — высота камеры над водой, D — расстояние до объекта.

Всё считается локально, наружу ничего не уходит.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Скорость точки на видео</title>
<style>
  :root { --bg:#14171c; --panel:#1d2128; --line:#2c323c; --fg:#e6e9ef; --dim:#8b95a5;
          --accent:#4da3ff; --good:#4ad07a; --bad:#ff6b6b; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line);
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header b { font-weight:600; }
  .wrap { display:flex; gap:12px; padding:12px; align-items:flex-start; flex-wrap:wrap; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  #stage { position:relative; }
  #video { display:block; background:#000; border-radius:4px; cursor:crosshair; }
  #loupe { position:absolute; width:120px; height:120px; border:2px solid var(--accent);
           border-radius:4px; pointer-events:none; display:none; background:#000; }
  .row { display:flex; gap:8px; align-items:center; margin:6px 0; flex-wrap:wrap; }
  label { color:var(--dim); font-size:12px; }
  input, select, button { background:#262b33; color:var(--fg); border:1px solid var(--line);
                          border-radius:5px; padding:5px 8px; font:inherit; }
  input[type=number] { width:74px; }
  button { cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.on { background:var(--accent); color:#08111c; border-color:var(--accent); font-weight:600; }
  #slider { width:100%; }
  .stats { font-variant-numeric:tabular-nums; }
  .stats td { padding:1px 10px 1px 0; }
  .stats td:first-child { color:var(--dim); }
  canvas.chart { display:block; margin-top:8px; background:#101318; border-radius:4px; }
  .hint { color:var(--dim); font-size:12px; max-width:560px; }
  .tag { font-size:11px; padding:1px 6px; border-radius:10px; background:#2b313b; color:var(--dim); }
  .ok { color:var(--good); } .err { color:var(--bad); }
</style>
</head>
<body>
<header>
  <b>Скорость точки на видео</b>
  <span class="tag" id="meta">загрузка…</span>
  <span class="hint">Включите пипетку, наведите курсор (лупа покажет увеличение) и кликните по
  объекту. Точка трассируется по всем кадрам, справа появятся графики.</span>
</header>

<div class="wrap">
  <div class="card" id="stage">
    <canvas id="video" width="1280" height="720"></canvas>
    <canvas id="loupe" width="120" height="120"></canvas>
    <input type="range" id="slider" min="1" value="1">
    <div class="row">
      <button id="play">▶ Играть</button>
      <button id="pipette" class="on">Пипетка</button>
      <button id="clear">Очистить</button>
      <span id="frameLabel" class="tag">кадр 1</span>
    </div>
  </div>

  <div class="card" style="min-width:330px">
    <div class="row">
      <label>fps</label><input type="number" id="fps" value="25" step="1">
      <label>кадр/с для просмотра</label><input type="number" id="playFps" value="10" step="1">
    </div>
    <div class="row">
      <label>перевод в м/с</label>
      <select id="mode">
        <option value="water">водная поверхность</option>
        <option value="distance">известное расстояние</option>
      </select>
    </div>
    <div id="waterInputs">
      <div class="row">
        <label>высота камеры над водой, м</label><input type="number" id="height" value="1.5" step="0.1">
      </div>
      <div class="row">
        <label>строка горизонта</label><input type="number" id="horizon" value="330" step="1">
        <button id="findHorizon">найти</button>
      </div>
    </div>
    <div id="distInputs" style="display:none">
      <div class="row">
        <label>расстояние до точки, м</label><input type="number" id="distance" value="30" step="1">
      </div>
      <div class="row">
        <label>угол обзора, град</label><input type="number" id="fov" value="70" step="1">
      </div>
    </div>
    <hr style="border:0;border-top:1px solid var(--line)">
    <table class="stats" id="stats"></table>
  </div>

  <div class="card">
    <div style="color:var(--dim);font-size:12px">скорость точки (синяя — м/с, серая — px/с)</div>
    <canvas class="chart" id="chartSpeed" width="760" height="240"></canvas>
    <div style="color:var(--dim);font-size:12px;margin-top:10px">положение точки: x (синий) и y (зелёный), px</div>
    <canvas class="chart" id="chartPos" width="760" height="200"></canvas>
    <div style="color:var(--dim);font-size:12px;margin-top:10px">пройденное расстояние, м</div>
    <canvas class="chart" id="chartDist" width="760" height="180"></canvas>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const video = $('video'), ctx = video.getContext('2d');
const loupe = $('loupe'), lctx = loupe.getContext('2d');
let meta = null, cache = new Map(), playing = false, probe = null;
let showFrame = 1, cursor = null;

function api(url, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {};
  return fetch(url, opts).then(r => r.json());
}

function loadFrame(n) {
  if (cache.has(n)) return Promise.resolve(cache.get(n));
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => { cache.set(n, img); res(img); };
    img.onerror = rej;
    img.src = '/frame/' + n;
  });
}

async function draw(n) {
  showFrame = n;
  const img = await loadFrame(n);
  video.width = img.width; video.height = img.height;
  ctx.drawImage(img, 0, 0);
  $('frameLabel').textContent = 'кадр ' + n;
  $('slider').value = n;
  if (probe) drawTrack(n);
}

function drawTrack(n) {
  const pts = probe.points;
  // trajectory
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(255,210,0,0.85)';
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < pts.length; i++) {
    if (!pts[i]) { started = false; continue; }
    if (!started) { ctx.moveTo(pts[i][0], pts[i][1]); started = true; }
    else ctx.lineTo(pts[i][0], pts[i][1]);
  }
  ctx.stroke();
  // current point
  const p = pts[n - 1];
  if (p) {
    ctx.strokeStyle = p[2] ? '#4ad07a' : '#ff6b6b';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p[0]-12, p[1]); ctx.lineTo(p[0]+12, p[1]);
    ctx.moveTo(p[0], p[1]-12); ctx.lineTo(p[0], p[1]+12); ctx.stroke();
  }
}

function drawLoupe() {
  if (!cursor || !$('pipette').classList.contains('on')) { loupe.style.display='none'; return; }
  const img = cache.get(showFrame);
  if (!img) { loupe.style.display='none'; return; }
  const zoom = 8, size = 120;
  loupe.style.display = 'block';
  loupe.style.left = Math.min(video.width - size, cursor.x + 16) + 'px';
  loupe.style.top = Math.max(0, cursor.y - size - 8) + 'px';
  lctx.imageSmoothingEnabled = false;
  lctx.clearRect(0,0,size,size);
  lctx.drawImage(img, cursor.x - size/(2*zoom), cursor.y - size/(2*zoom),
                 size/zoom, size/zoom, 0, 0, size, size);
  lctx.strokeStyle = '#4da3ff'; lctx.lineWidth = 1;
  lctx.beginPath();
  lctx.moveTo(size/2, size/2-10); lctx.lineTo(size/2, size/2+10);
  lctx.moveTo(size/2-10, size/2); lctx.lineTo(size/2+10, size/2);
  lctx.stroke();
}

// ------------------------------------------------------------- charts
function chart(cvs, series, opts={}) {
  const c = cvs.getContext('2d'), W = cvs.width, H = cvs.height;
  const pad = {l:52, r:12, t:12, b:22};
  c.clearRect(0,0,W,H);
  c.fillStyle = '#101318'; c.fillRect(0,0,W,H);
  const vals = series.flatMap(s => s.data.filter(v => v !== null && isFinite(v)));
  if (!vals.length) return;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (opts.zero) lo = Math.min(0, lo);
  if (hi - lo < 1e-9) hi = lo + 1;
  const n = Math.max(...series.map(s => s.data.length));
  const X = i => pad.l + (n <= 1 ? 0 : i * (W - pad.l - pad.r) / (n - 1));
  const Y = v => H - pad.b - (v - lo) / (hi - lo) * (H - pad.t - pad.b);
  // grid
  c.strokeStyle = '#232a33'; c.fillStyle = '#7b8595'; c.font = '11px sans-serif';
  for (let g = 0; g <= 4; g++) {
    const v = lo + (hi - lo) * g / 4, y = Y(v);
    c.beginPath(); c.moveTo(pad.l, y); c.lineTo(W - pad.r, y); c.stroke();
    c.fillText(v.toFixed(opts.dec === undefined ? 2 : opts.dec), 6, y + 4);
  }
  c.fillText('0', pad.l - 4, H - 8);
  c.fillText(String(n), W - pad.r - 12, H - 8);
  for (const s of series) {
    if (!s.data.length) continue;
    c.strokeStyle = s.color; c.lineWidth = s.width || 1.6;
    c.beginPath();
    let started = false;
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (v === null || !isFinite(v)) { started = false; continue; }
      if (!started) { c.moveTo(X(i), Y(v)); started = true; } else c.lineTo(X(i), Y(v));
    }
    c.stroke();
  }
  // marker for the current frame
  if (showFrame <= n) {
    c.strokeStyle = 'rgba(255,255,255,0.25)';
    c.beginPath(); c.moveTo(X(showFrame-1), pad.t); c.lineTo(X(showFrame-1), H-pad.b); c.stroke();
  }
}

function renderCharts() {
  if (!probe) return;
  chart($('chartSpeed'), [
    {data: probe.speeds_m_s, color:'#4da3ff', width:2},
    {data: probe.speeds_px_s.map(v => v === null ? null : v / 10), color:'#5a6472'}
  ], {zero:true});
  chart($('chartPos'), [
    {data: probe.pos_x, color:'#4da3ff'},
    {data: probe.pos_y, color:'#4ad07a'}
  ]);
  chart($('chartDist'), [{data: probe.cumulative_m, color:'#ffd24a', width:2}], {zero:true});
}

function renderStats() {
  const s = probe.stats;
  const rows = [
    ['кадр постановки', probe.frame],
    ['прослежено кадров', s.tracked + ' из ' + probe.points.length],
    ['потеря трека', s.lost ? '<span class="err">да, на кадре ' + s.lost_at + '</span>'
                            : '<span class="ok">нет</span>'],
    ['средняя скорость', fmt(s.mean_m_s) + ' м/с · ' + fmt(s.mean_px_s) + ' px/с'],
    ['медианная скорость', fmt(s.median_m_s) + ' м/с · ' + fmt(s.median_px_s) + ' px/с'],
    ['максимум', fmt(s.max_m_s) + ' м/с'],
    ['путь за отрезок', fmt(s.path_m) + ' м'],
    ['смещение по кадру', fmt(s.displacement_px) + ' px'],
  ];
  let html = rows.map(r => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('');
  for (const w of (probe.warnings || []))
    html += '<tr><td colspan="2" class="err" style="padding-top:6px">' + w + '</td></tr>';
  $('stats').innerHTML = html;
}
const fmt = v => (v === null || v === undefined || !isFinite(v)) ? '—' : v.toFixed(2);

async function probeAt(x, y) {
  const body = {
    x, y, frame: showFrame,
    fps: +$('fps').value,
    mode: $('mode').value,
    height: +$('height').value,
    horizon: +$('horizon').value,
    distance: +$('distance').value,
    fov: +$('fov').value,
  };
  $('stats').innerHTML = '<tr><td>считаю…</td><td></td></tr>';
  probe = await api('/api/probe', body);
  renderStats(); renderCharts(); draw(showFrame);
}

// ------------------------------------------------------------- events
video.addEventListener('mousemove', e => {
  const r = video.getBoundingClientRect();
  cursor = {x: (e.clientX - r.left) * video.width / r.width,
            y: (e.clientY - r.top) * video.height / r.height};
  drawLoupe();
});
video.addEventListener('mouseleave', () => { loupe.style.display = 'none'; cursor = null; });
video.addEventListener('click', e => {
  if (!$('pipette').classList.contains('on') || !cursor) return;
  probeAt(Math.round(cursor.x), Math.round(cursor.y));
});
$('pipette').onclick = () => {
  $('pipette').classList.toggle('on');
  if (!$('pipette').classList.contains('on')) loupe.style.display = 'none';
};
$('clear').onclick = () => { probe = null; $('stats').innerHTML = '';
  [$('chartSpeed'),$('chartPos'),$('chartDist')].forEach(c => c.getContext('2d').clearRect(0,0,c.width,c.height));
  draw(showFrame); };
$('slider').oninput = e => draw(+e.target.value);
$('play').onclick = async () => {
  playing = !playing;
  $('play').textContent = playing ? '⏸ Пауза' : '▶ Играть';
  const delay = 1000 / Math.max(1, +$('playFps').value);
  while (playing) {
    const next = showFrame >= meta.count ? 1 : showFrame + 1;
    await draw(next);
    await new Promise(r => setTimeout(r, delay));
  }
};
$('mode').onchange = () => {
  const w = $('mode').value === 'water';
  $('waterInputs').style.display = w ? '' : 'none';
  $('distInputs').style.display = w ? 'none' : '';
  if (probe) probeAt(Math.round(probe.points[probe.frame-1][0]), Math.round(probe.points[probe.frame-1][1]));
};
['fps','height','horizon','distance','fov'].forEach(id => {
  $(id).onchange = () => {
    if (probe && probe.points[probe.frame-1])
      probeAt(Math.round(probe.points[probe.frame-1][0]), Math.round(probe.points[probe.frame-1][1]));
  };
});
$('findHorizon').onclick = async () => {
  const r = await api('/api/horizon');
  $('horizon').value = r.row;
  if (probe && probe.points[probe.frame-1])
    probeAt(Math.round(probe.points[probe.frame-1][0]), Math.round(probe.points[probe.frame-1][1]));
};

(async function init() {
  meta = await api('/api/meta');
  $('meta').textContent = meta.count + ' кадров · ' + meta.width + '×' + meta.height + ' · ' + meta.fps + ' fps';
  $('fps').value = meta.fps;
  $('slider').max = meta.count;
  if (meta.horizon) $('horizon').value = meta.horizon;
  draw(1);
  // прогрев кэша кадров, чтобы воспроизведение не тормозило
  for (let n = 2; n <= meta.count; n++) { await loadFrame(n); if (n % 20 === 0) await new Promise(r=>setTimeout(r)); }
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------- Tracking


class Sequence:
    def __init__(self, frames_dir: Path, max_frames: int = 0):
        self.paths = sorted(
            [p for p in frames_dir.iterdir() if p.suffix.lower() in
             (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")],
            key=lambda p: p.name,
        )
        if max_frames:
            self.paths = self.paths[:max_frames]
        if not self.paths:
            raise SystemExit(f"в {frames_dir} нет кадров")
        self.frames = [self._read(p) for p in self.paths]
        self.height, self.width = self.frames[0].shape

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"не читается {path}")
        return image

    def jpeg(self, index: int) -> bytes:
        ok, buffer = cv2.imencode(".jpg", self.frames[index - 1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buffer.tobytes() if ok else b""

    def horizon_row(self) -> int:
        profile = np.mean([f.mean(axis=1) for f in self.frames[:: max(1, len(self.frames) // 10)]], axis=0)
        smoothed = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 31), 0).ravel()
        return int(np.argmax(np.abs(np.gradient(smoothed))))

    def track(self, x: float, y: float, start: int, fb_threshold: float = 2.0) -> dict:
        """Track one point forwards and backwards from frame *start* (1-based)."""
        n = len(self.frames)
        start = int(min(max(start, 1), n))
        point = np.array([[[x, y]]], np.float32)

        def step(src_idx: int, dst_idx: int, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nxt, status, err = cv2.calcOpticalFlowPyrLK(
                self.frames[src_idx],
                self.frames[dst_idx],
                pts,
                None,
                winSize=(41, 41),
                maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
            )
            return nxt, status.reshape(-1).astype(bool)

        forward: dict[int, np.ndarray] = {start: point.reshape(2)}
        pts = point
        for i in range(start - 1, n - 1):
            nxt, ok = step(i, i + 1, pts)
            if not ok[0]:
                break
            pts = nxt
            forward[i + 2] = nxt.reshape(2)

        backward: dict[int, np.ndarray] = {start: point.reshape(2)}
        pts = point
        for i in range(start - 1, 0, -1):
            nxt, ok = step(i, i - 1, pts)
            if not ok[0]:
                break
            pts = nxt
            backward[i] = nxt.reshape(2)

        points: list[list | None] = []
        for i in range(1, n + 1):
            if i in forward and i in backward and i != start:
                p = (forward[i] + backward[i]) / 2
            elif i in forward:
                p = forward[i]
            elif i in backward:
                p = backward[i]
            else:
                points.append(None)
                continue
            points.append([float(p[0]), float(p[1]), 1])

        # forward-backward consistency: re-track each consecutive pair backwards
        valid = [p is not None for p in points]
        for i in range(n - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            back, ok = step(i + 1, i, np.array([[points[i + 1][:2]]], np.float32))
            if ok[0]:
                error = float(np.linalg.norm(back.reshape(2) - np.array(points[i][:2])))
                if error > fb_threshold:
                    points[i + 1][2] = 0
        return {"points": points, "start": start}


def speeds_from_track(
    points: list, fps: float, mode: str, height: float, horizon: float,
    distance: float, fov: float, width: int
) -> dict:
    n = len(points)
    speeds_px = [None] * n
    speeds_ms = [None] * n
    cumulative_m = [None] * n
    pos_x = [None] * n
    pos_y = [None] * n
    total_px = 0.0
    total_m = 0.0

    focal = (width / 2.0) / math.tan(math.radians(fov) / 2.0) if fov else 0.0

    for i in range(1, n):
        a, b = points[i - 1], points[i]
        pos_x[i - 1] = a[0] if a else None
        pos_y[i - 1] = a[1] if a else None
        if not a or not b or not b[2]:
            continue
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        px_per_frame = math.hypot(dx, dy)
        px_per_s = px_per_frame * fps
        speeds_px[i] = px_per_s
        if mode == "water" and height:
            # the model only holds for points on the water, i.e. below the
            # horizon row; above it the point is shore or sky and the
            # conversion is meaningless rather than merely imprecise
            denom = b[1] - horizon
            speeds_ms[i] = px_per_s * height / denom if denom > 1.0 else None
        elif mode == "distance" and distance and focal:
            speeds_ms[i] = px_per_s * distance / focal
        total_px += px_per_frame
        if speeds_ms[i] is not None:
            total_m += speeds_ms[i] / fps
        cumulative_m[i] = total_m

    if points[-1]:
        pos_x[-1] = points[-1][0]
        pos_y[-1] = points[-1][1]

    good_px = [v for v in speeds_px if v is not None]
    good_ms = [v for v in speeds_ms if v is not None]
    tracked = sum(1 for p in points if p)
    lost_at = next((i + 1 for i, p in enumerate(points) if p is None), 0)
    start_idx = next((i for i, p in enumerate(points) if p), 0)
    end_idx = next((i for i in range(n - 1, -1, -1) if points[i]), 0)
    first, last = points[start_idx], points[end_idx]

    rows_ok = [p[1] for p in points if p]
    mean_row = float(np.mean(rows_ok)) if rows_ok else 0.0
    warnings = []
    if mode == "water":
        if mean_row <= horizon + 1:
            warnings.append(
                "точка на уровне горизонта или выше: модель водной поверхности к ней "
                "не применима — переключитесь на режим «известное расстояние»"
            )
        elif not height:
            warnings.append("не задана высота камеры — перевода в м/с нет")
    if mode == "distance" and not (distance and focal):
        warnings.append("задайте расстояние до точки и угол обзора — иначе перевода в м/с нет")
    if tracked < 0.25 * n:
        warnings.append(
            f"трек живёт всего {tracked} кадров из {n}: точка на воде или в размытии, "
            "скорость ориентировочная"
        )
    if lost_at and lost_at > start_idx + 1 and end_idx - start_idx < 5:
        warnings.append("трек потерян почти сразу — попробуйте точку с текстурой (берег, объект)")

    return {
        "warnings": warnings,
        "points": points,
        "speeds_px_s": speeds_px,
        "speeds_m_s": speeds_ms,
        "cumulative_m": cumulative_m,
        "pos_x": pos_x,
        "pos_y": pos_y,
        "stats": {
            "tracked": tracked,
            "lost": lost_at > 0,
            "lost_at": lost_at,
            "mean_px_s": float(np.mean(good_px)) if good_px else None,
            "median_px_s": float(np.median(good_px)) if good_px else None,
            "mean_m_s": float(np.mean(good_ms)) if good_ms else None,
            "median_m_s": float(np.median(good_ms)) if good_ms else None,
            "max_m_s": float(np.max(good_ms)) if good_ms else None,
            "path_m": float(total_m),
            "displacement_px": float(math.hypot(last[0] - first[0], last[1] - first[1])),
        },
    }


# ------------------------------------------------------------------ Server


def make_handler(seq: Sequence, fps: float) -> type:
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # тише
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, payload: dict) -> None:
            self._send(200, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif path.startswith("/frame/"):
                try:
                    index = int(path.rsplit("/", 1)[1])
                except ValueError:
                    self._send(404, b"bad frame", "text/plain")
                    return
                if not 1 <= index <= len(seq.frames):
                    self._send(404, b"no such frame", "text/plain")
                    return
                self._send(200, seq.jpeg(index), "image/jpeg")
            elif path == "/api/meta":
                try:
                    horizon = seq.horizon_row()
                except Exception:
                    horizon = 0
                self._json({
                    "count": len(seq.frames),
                    "width": seq.width,
                    "height": seq.height,
                    "fps": fps,
                    "horizon": horizon,
                })
            elif path == "/api/horizon":
                self._json({"row": seq.horizon_row()})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/probe":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"})
                return
            result = seq.track(float(body.get("x", 0)), float(body.get("y", 0)), int(body.get("frame", 1)))
            speeds = speeds_from_track(
                result["points"],
                float(body.get("fps", fps)) or fps,
                body.get("mode", "water"),
                float(body.get("height", 0) or 0),
                float(body.get("horizon", 0) or 0),
                float(body.get("distance", 0) or 0),
                float(body.get("fov", 0) or 0),
                seq.width,
            )
            speeds["frame"] = result["start"]
            self._json(speeds)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Пипетка и графики скорости точки на видео.")
    parser.add_argument("--frames", default="data/frames", help="папка с кадрами")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int, default=0, help="ограничить число кадров")
    args = parser.parse_args()

    seq = Sequence(Path(args.frames), args.max_frames)
    print(f"кадров: {len(seq.frames)} ({seq.width}x{seq.height}), {args.fps:g} fps")
    print(f"горизонт (авто): строка {seq.horizon_row()}")
    print(f"откройте http://{args.host}:{args.port}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(seq, args.fps))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
