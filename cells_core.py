"""Single-cell analysis for SURF (aimed at 40x recordings) and dF/F video.

Pipeline
  1. detection movie   frames in the SAME space as the grid measurement (rotation, grey mode),
                       spatially binned so a soma is ~12 px, temporally averaged to <= ~2000 frames
  2. cell ROIs         Suite2p (activity based), Cellpose (shape based, on the mean / max image),
                       a FIJI RoiSet.zip, or circles added by hand; every ROI can be accepted / rejected
  3. traces            F = F_cell - r * F_neuropil (ring around the cell, other cells excluded);
                       measured with SURF's fast sparse-matrix reader
  4. events            the rule of the "Caimg spike detection - Wave" notebooks, in % of baseline:
                       F0 = minimum of the raw trace in the baseline window, trace smoothed
                       (Gaussian), a peak must
                         * exceed the minimum of the next `drop_window_s` by `threshold_pct`
                         * (optionally) exceed the mean of +-`around_window_s` by `threshold_pct`
                         * be higher than the trace `local_max_s` before and after
                         * be >= the mean of the next / previous `short_window_s` + `eps_pct`
                         * be >= the mean of the next or previous `long_window_s` + `eps_pct`
                       candidates closer than `min_sep_s` are merged (the highest kept);
                       per event: amplitude (peak - minimum of the next 1 s) and 1/e decay time
                       (find_spike_data). Cells dimmer than `min_f0` grey are not analysed.
                       Alternative: scipy find_peaks prominence (as the grid Events tab).
  5. waves             an event belongs to a wave (long track of the grid analysis) if it falls
                       inside the wave's time window ('time', as the notebook) and, optionally,
                       the wave centre at that moment is within `link_radius_um` ('time+distance').
  6. metrics           per cell, per wave (recruitment, latency, cell-based speed) and summary,
                       with a chance level from circularly shifted event trains.

dF/F video (deltaF_F_prepare / viewer notebooks): F0 = per-pixel temporal mean after skipping the
start, 2x2 spatial and 2-frame temporal binning, contrast -3 sigma .. +7 sigma of dF/F, background
mask = pixels with F0 below a percentile, ImageJ "16 colors" look-up table.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import wave_core as wc

_p, _c, Cancelled = wc._p, wc._c, wc.Cancelled

import sys as _sys
for _n in ("stdout", "stderr", "__stdout__", "__stderr__"):   # no console (pythonw): progress output
    if getattr(_sys, _n, None) is None:
        try:
            setattr(_sys, _n, open(os.devnull, "w", encoding="utf-8"))
        except Exception:
            pass


# =============================================================================
# frames in the measured space
# =============================================================================
def frame_source(path, rotation: Optional[dict], gray_mode="mean", frame_range=None, step=1,
                 cancel=None):
    """Yield (index, grey float32 frame) in the measured space (rotation applied like the
    grid measurement). Decoding runs in a background thread."""
    info = wc.video_info(path)
    W, H = info["width"], info["height"]
    cal = wc.RotationCalib.from_dict(rotation) if rotation else None
    M = cal.matrix(W, H) if cal is not None else None
    start, stop = (0, info["n_frames"]) if frame_range is None else frame_range
    q: queue.Queue = queue.Queue(maxsize=32)
    stop_flag = threading.Event()

    def reader():
        cap = cv2.VideoCapture(path)
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while i < stop and not stop_flag.is_set():
            if (i - start) % step == 0:
                ok, fr = cap.read()
                if not ok:
                    break
                q.put((i, fr))
            elif not cap.grab():
                break
            i += 1
        cap.release()
        q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            i, fr = item
            g = wc.to_gray_f32(fr, gray_mode)
            if M is not None:
                g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            yield i, g
            if _c(cancel):
                raise Cancelled()
    finally:
        stop_flag.set()
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                time.sleep(0.01)


def detection_movie(path, rotation, gray_mode, frame_range, cell_diam_px, max_frames=1200,
                    progress=None, cancel=None, skip_dark=True, tissue_only=True, bad=None):
    """Binned movie for ROI detection: soma ~12 px, <= max_frames averaged frames, int16 (grey x 100).
    Returns dict(movie (T,Ly,Lx) int16, bin, tbin, fs, mean, max, std (binned), W, H)."""
    info = wc.video_info(path)
    W, H = info["width"], info["height"]
    start, stop = frame_range
    use = np.ones(stop - start, bool)
    if skip_dark and bad is not None:
        b = np.asarray(bad, bool)[start:stop]
        use[:len(b)] = ~b
    n = max(int(use.sum()), 1)
    sbin = max(1, int(round(cell_diam_px / 12.0)))
    tbin = max(1, int(math.ceil(n / max_frames)))
    Ly, Lx = H // sbin, W // sbin
    T = n // tbin
    mov = np.zeros((T, Ly, Lx), np.int16)
    acc = np.zeros((Ly, Lx), np.float32)
    s1 = np.zeros((Ly, Lx), np.float64)
    s2 = np.zeros((Ly, Lx), np.float64)
    mx = np.zeros((Ly, Lx), np.float32)
    k = c = 0
    for i, g in frame_source(path, rotation, gray_mode, (start, stop), cancel=cancel):
        if k >= T:
            break
        if not use[i - start]:
            continue                                   # light off / switching: left out
        acc += cv2.resize(g[:Ly * sbin, :Lx * sbin], (Lx, Ly), interpolation=cv2.INTER_AREA)
        c += 1
        if c == tbin:
            fr = acc / tbin
            mov[k] = np.clip(fr * 100.0, 0, 32000).astype(np.int16)
            s1 += fr
            s2 += fr.astype(np.float64) ** 2
            np.maximum(mx, fr, out=mx)
            acc[:] = 0
            c = 0
            k += 1
            if k % 50 == 0:
                _p(progress, k, T, "reading frames for cell detection")
    mov = mov[:k]
    # dark start (light switched on during the recording): every pixel 'jumps' once, which
    # Suite2p reads as activity everywhere -> leave those frames out
    fmean = mov.reshape(len(mov), -1).mean(1).astype(np.float32) if len(mov) else np.zeros(0)
    skip = 0
    if skip_dark and bad is None and len(fmean) > 10:
        ref = float(np.median(fmean))
        bright = np.nonzero(fmean >= 0.8 * ref)[0]
        skip = int(bright[0]) if len(bright) else 0
        if skip:
            skip = min(len(fmean) - 10, skip + 2)          # plus the rise itself
            mov = mov[skip:]
    s1 = np.zeros((Ly, Lx), np.float64)
    s2 = np.zeros((Ly, Lx), np.float64)
    mx = np.zeros((Ly, Lx), np.float32)
    for c0 in range(0, len(mov), 100):                 # chunked: no float copy of the whole movie
        ch = mov[c0:c0 + 100].astype(np.float32) / 100.0
        s1 += ch.sum(0)
        s2 += (ch.astype(np.float64) ** 2).sum(0)
        np.maximum(mx, ch.max(0), out=mx)
    k = max(len(mov), 1)
    mean = (s1 / k).astype(np.float32)
    sd = np.sqrt(np.maximum(s2 / k - (s1 / k) ** 2, 0)).astype(np.float32)
    mask = tissue_mask(mean, cell_diam_px / sbin) if tissue_only else np.ones((Ly, Lx), bool)
    _p(progress, T, T, "detection movie ready")
    return dict(movie=mov, bin=sbin, tbin=tbin, fs=info["fps"] / tbin, mean=mean, max=mx, std=sd,
                W=W, H=H, Ly=Ly, Lx=Lx, mask=mask, skipped_s=(start + skip * tbin) / info["fps"],
                frame_means=fmean, n_left_out=int((~use).sum()))


dark_frames = wc.dark_frames          # shared with the grid event detection
bridge = wc.bridge


def tissue_mask(mean, diam=12.0):
    """Where the retina is: Otsu threshold of the smoothed mean image (rotation padding and the
    dark surround excluded), holes filled, shrunk by half a cell."""
    g = cv2.GaussianBlur(mean.astype(np.float32), (0, 0), max(1.0, diam / 2))
    valid = mean > 0
    if not valid.any():
        return np.zeros_like(mean, bool)
    lo, hi = np.percentile(g[valid], (1, 99))
    u8 = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (u8 > t) & valid
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:                                          # keep the big pieces only
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.nonzero(sizes >= 0.05 * sizes.max())[0] + 1
        m = np.isin(lab, keep)
    r = max(1, int(round(diam / 2)))
    m = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return m if m.mean() > 0.02 else np.ones_like(mean, bool)      # a failed mask never blocks detection


def _size_ok(n_px_full, diam_px):
    a = math.pi * (diam_px / 2) ** 2
    return 0.25 * a <= n_px_full <= 4.0 * a


# =============================================================================
# cell ROIs
# =============================================================================
@dataclass
class Cell:
    ypix: np.ndarray            # full-resolution pixel coordinates (measured space)
    xpix: np.ndarray
    source: str = "manual"      # suite2p | cellpose | fiji | manual
    prob: float = 1.0           # classifier / cellpose probability (1 for manual)
    accepted: bool = True

    @property
    def area(self):
        return len(self.xpix)

    @property
    def centre(self):
        return float(self.xpix.mean()) + 0.5, float(self.ypix.mean()) + 0.5


def _mask_to_cell(m, source, prob=1.0):
    yy, xx = np.nonzero(m)
    return Cell(yy.astype(np.int32), xx.astype(np.int32), source, float(prob))


def _upscale(ypix, xpix, b, W, H):
    """Binned pixel list -> full-resolution pixel list (each binned px -> b x b block)."""
    if b == 1:
        return ypix.astype(np.int32), xpix.astype(np.int32)
    oy, ox = np.meshgrid(np.arange(b), np.arange(b), indexing="ij")
    yy = (ypix[:, None] * b + oy.ravel()[None, :]).ravel()
    xx = (xpix[:, None] * b + ox.ravel()[None, :]).ravel()
    ok = (yy < H) & (xx < W)
    return yy[ok].astype(np.int32), xx[ok].astype(np.int32)


def have_suite2p():
    try:
        import suite2p  # noqa: F401
        return True
    except Exception:
        return False


def have_cellpose():
    try:
        import cellpose  # noqa: F401
        return True
    except Exception:
        return False


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_suite2p(dm, threshold_scaling=1.0, tau=1.0, max_overlap=0.75, progress=None):
    """Suite2p (1.x) activity-based detection on the binned movie; classifier probability kept."""
    from suite2p.detection import detection_wrapper
    from suite2p import default_settings
    try:
        from suite2p import classify
    except ImportError:
        classify = None
    st = default_settings()
    det = dict(st["detection"])
    det.update(threshold_scaling=float(threshold_scaling), max_overlap=float(max_overlap),
               algorithm="sparsery")
    _p(progress, 0, 1, "Suite2p: detecting cells (this can take a few minutes)")
    d = 12.0
    mov = dm["movie"]
    mask = dm.get("mask")
    if mask is not None and not mask.all():
        # outside the tissue: a constant (no activity, no ROIs); crop to the tissue box
        mov = mov.copy()
        mov[:, ~mask] = np.int16(np.median(mov[:, mask]) if mask.any() else 0)
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mov = np.ascontiguousarray(mov[:, y0:y1, x0:x1])
    else:
        y0 = x0 = 0
    new, stat, _ = detection_wrapper(mov, diameter=[d, d], tau=tau, fs=dm["fs"],
                                     settings=det, device=_device())
    probs = np.full(len(stat), np.nan)
    if classify is not None and len(stat):
        try:
            from suite2p.classification import builtin_classfile
            ic = classify(stat=stat, classfile=builtin_classfile)
            probs = np.asarray(ic)[:, 1]
        except Exception:
            pass
    cells = []
    diam_full = d * dm["bin"]
    for s, pr in zip(stat, probs):
        yb, xb = np.asarray(s["ypix"]) + y0, np.asarray(s["xpix"]) + x0
        med = np.asarray(s.get("med", (np.median(yb) - y0, np.median(xb) - x0)), float) + [y0, x0]
        if mask is not None and not mask[int(np.clip(med[0], 0, mask.shape[0] - 1)),
                                         int(np.clip(med[1], 0, mask.shape[1] - 1))]:
            continue
        yy, xx = _upscale(yb, xb, dm["bin"], dm["W"], dm["H"])
        if len(xx):
            ok_p = (not np.isfinite(pr)) or pr >= 0.5
            cells.append(Cell(yy, xx, "suite2p", float(pr) if np.isfinite(pr) else 1.0,
                              bool(ok_p and _size_ok(len(xx), diam_full))))
    _p(progress, 1, 1, f"Suite2p: {len(cells)} ROIs")
    return cells


def detect_cellpose(dm, image="mean", flow_threshold=0.4, cellprob_threshold=0.0, progress=None):
    """Cellpose (4.x, cpsam) on the mean / max / std image of the binned movie."""
    from cellpose import models
    import torch
    img = {"mean": dm["mean"], "max": dm["max"], "std": dm["std"]}.get(image, dm["mean"])
    lo, hi = np.percentile(img, (1, 99.8))
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    _p(progress, 0, 1, "Cellpose: segmenting (the first run downloads the model)")
    gpu = torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    try:
        model = models.CellposeModel(gpu=gpu)          # NVIDIA GPU, or Apple-Silicon GPU (MPS)
    except Exception:
        model = models.CellposeModel(gpu=False)
    kw = dict(flow_threshold=float(flow_threshold), cellprob_threshold=float(cellprob_threshold))
    try:
        out = model.eval(img, **kw)
    except Exception:
        if not gpu:
            raise
        out = models.CellposeModel(gpu=False).eval(img, **kw)   # GPU path failed (e.g. MPS op missing)
    labels = out[0]
    cells = []
    for lab in range(1, int(labels.max()) + 1):
        yy, xx = np.nonzero(labels == lab)
        if len(xx) < 4:
            continue
        mask = dm.get("mask")
        if mask is not None and not mask[int(np.median(yy)), int(np.median(xx))]:
            continue
        yy, xx = _upscale(yy, xx, dm["bin"], dm["W"], dm["H"])
        cells.append(Cell(yy, xx, "cellpose", 1.0, bool(_size_ok(len(xx), 12.0 * dm["bin"]))))
    _p(progress, 1, 1, f"Cellpose: {len(cells)} cells")
    return cells


def cells_from_imagej_zip(path, W, H):
    import roifile
    rois = roifile.ImagejRoi.fromfile(path)
    if not isinstance(rois, list):
        rois = [rois]
    cells = []
    for r in rois:
        m = np.zeros((H, W), np.uint8)
        if r.roitype == roifile.ROI_TYPE.RECT:
            m[max(r.top, 0):r.bottom, max(r.left, 0):r.right] = 1
        elif r.roitype == roifile.ROI_TYPE.OVAL:
            cx, cy = (r.left + r.right) / 2.0, (r.top + r.bottom) / 2.0
            cv2.ellipse(m, (int(round(cx)), int(round(cy))),
                        (int(round((r.right - r.left) / 2)), int(round((r.bottom - r.top) / 2))), 0, 0, 360, 1, -1)
        else:
            pts = np.asarray(r.coordinates(), float)
            cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if m.any():
            cells.append(_mask_to_cell(m.astype(bool), "fiji"))
    return cells


def circle_cell(x, y, diam_px, W, H):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (int(round(x)), int(round(y))), max(1, int(round(diam_px / 2))), 1, -1)
    return _mask_to_cell(m.astype(bool), "manual")


def polygon_cell(pts, W, H, source="manual"):
    """Cell from polygon vertices [(x, y), ...] in frame coordinates."""
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float)).astype(np.int32)], 1)
    return _mask_to_cell(m.astype(bool), source) if m.any() else None


def polygon_of(cell, max_vertices=40):
    """Simplified outline vertices of a cell (for editing)."""
    x, y = outline(cell)
    if len(x) < 4:
        return list(zip(x, y))
    c = np.c_[x[:-1], y[:-1]].astype(np.float32).reshape(-1, 1, 2)
    eps = 0.5
    ap = cv2.approxPolyDP(c, eps, True)
    while len(ap) > max_vertices:
        eps *= 1.5
        ap = cv2.approxPolyDP(c, eps, True)
    return [tuple(map(float, q[0])) for q in ap]


def save_cells(path, cells, W, H, extra=None):
    off = np.cumsum([0] + [c.area for c in cells])
    np.savez_compressed(
        path, ypix=np.concatenate([c.ypix for c in cells]) if cells else np.zeros(0, np.int32),
        xpix=np.concatenate([c.xpix for c in cells]) if cells else np.zeros(0, np.int32),
        offsets=off, source=np.array([c.source for c in cells]), prob=np.array([c.prob for c in cells]),
        accepted=np.array([c.accepted for c in cells]), W=W, H=H, extra=json.dumps(extra or {}))


def load_cells(path):
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    cells = [Cell(z["ypix"][off[i]:off[i + 1]], z["xpix"][off[i]:off[i + 1]], str(z["source"][i]),
                  float(z["prob"][i]), bool(z["accepted"][i])) for i in range(len(off) - 1)]
    return cells, int(z["W"]), int(z["H"]), json.loads(str(z["extra"]))


def outline(cell, b=1):
    """Contour (x, y arrays) of a cell for drawing."""
    x0, y0 = cell.xpix.min(), cell.ypix.min()
    m = np.zeros((cell.ypix.max() - y0 + 3, cell.xpix.max() - x0 + 3), np.uint8)
    m[cell.ypix - y0 + 1, cell.xpix - x0 + 1] = 1
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return np.array([]), np.array([])
    c = max(cs, key=len)[:, 0, :].astype(float)
    c = np.vstack([c, c[:1]])
    return c[:, 0] + x0 - 1 + 0.5, c[:, 1] + y0 - 1 + 0.5


# =============================================================================
# traces
# =============================================================================
def neuropil_masks(cells, W, H, inner_px=2, outer_px=None, diam_px=12):
    """Ring around every cell (other cells excluded)."""
    outer_px = outer_px or max(6, int(round(diam_px)))
    allm = np.zeros((H, W), np.uint8)
    for c in cells:
        allm[c.ypix, c.xpix] = 1
    excl = cv2.dilate(allm, np.ones((2 * inner_px + 1, 2 * inner_px + 1), np.uint8)) > 0
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_px + 1, 2 * outer_px + 1))
    for c in cells:
        x0, x1 = max(c.xpix.min() - outer_px - 1, 0), min(c.xpix.max() + outer_px + 2, W)
        y0, y1 = max(c.ypix.min() - outer_px - 1, 0), min(c.ypix.max() + outer_px + 2, H)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        m[c.ypix - y0, c.xpix - x0] = 1
        ring = (cv2.dilate(m, k) > 0) & ~excl[y0:y1, x0:x1]
        yy, xx = np.nonzero(ring)
        if len(xx) < 5:                       # crowded: fall back to a bigger ring
            ring = (cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * outer_px + 1,) * 2)) > 0) \
                & ~excl[y0:y1, x0:x1]
            yy, xx = np.nonzero(ring)
        out.append((yy + y0, xx + x0))
    return out


def measure_cells(path, cells, rotation, gray_mode, diam_px, progress=None, cancel=None, frame_range=None):
    """Raw cell and neuropil traces (T, N) with SURF's fast reader."""
    info = wc.video_info(path)
    W, H = info["width"], info["height"]
    npil = neuropil_masks(cells, W, H, diam_px=diam_px)
    # RoiSet.from_masks wants boolean frames; build the sparse matrix directly instead
    from scipy import sparse
    ys = [c.ypix for c in cells] + [p[0] for p in npil]
    xs = [c.xpix for c in cells] + [p[1] for p in npil]
    allx, ally = np.concatenate(xs), np.concatenate(ys)
    bx0, by0 = int(allx.min()), int(ally.min())
    bw, bh = int(allx.max()) - bx0 + 1, int(ally.max()) - by0 + 1
    rows, cols, vals = [], [], []
    for k, (yy, xx) in enumerate(zip(ys, xs)):
        n = max(len(xx), 1)
        rows.append(np.full(len(xx), k))
        cols.append((yy - by0) * bw + (xx - bx0))
        vals.append(np.full(len(xx), 1.0 / n))
    Wm = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(len(ys), bw * bh), dtype=np.float32)
    names = [f"cell{i + 1}" for i in range(len(cells))] + [f"npil{i + 1}" for i in range(len(cells))]
    cx = np.array([c.centre[0] for c in cells] * 2)
    cy = np.array([c.centre[1] for c in cells] * 2)
    rs = wc.RoiSet("mask", names, cx, cy, np.array([len(x) for x in xs]), (bx0, by0, bw, bh), W=Wm)
    rot = wc.RotationCalib.from_dict(rotation) if rotation else None
    m = wc.measure_video(path, rs, rot, gray_mode, progress, cancel, frame_range)
    N = len(cells)
    return m.mean[:, :N].copy(), m.mean[:, N:].copy(), m.fps


# =============================================================================
# events
# =============================================================================
@dataclass
class CellParams:
    neuropil_r: float = 0.7          # F = F_cell - r * F_neuropil
    baseline_t0: float = 6.0         # baseline window (s): F0 = minimum of the raw trace in it
    baseline_t1: float = 20.0
    min_f0: float = 5.0              # cells with F0 below this grey value are not analysed
    t_start: float = 5.0             # analysis window (s)
    t_end: float = -1.0              # -1 = end
    method: int = 0                  # 0 notebook rule, 1 prominence (scipy find_peaks), 2 grid rule (tab 3)
    sigma: float = 4.0               # Gaussian smoothing (frames)
    threshold_pct: float = 2.0       # % of F0 (notebook: thresh / baselineCorr, e.g. 0.8 / 40)
    drop_window_s: float = 2.0       # the trace must fall by threshold within this time after a peak
    use_around_mean: int = 1         # also exceed the mean of +-around_window_s by the threshold
    around_window_s: float = 3.0
    local_max_s: float = 0.02        # higher than the trace this long before and after
    short_window_s: float = 0.1
    long_window_s: float = 0.5
    eps_pct: float = 0.025           # notebook 0.01 at baselineCorr 40
    min_sep_s: float = 0.2
    prominence_pct: float = 3.0      # method 1
    diameter_um: float = 12.0        # expected soma diameter (detection, neuropil ring, manual circles)
    link_mode: int = 1               # 0 time only (notebook), 1 time + distance to the wave centre
    link_radius_um: float = 200.0
    link_pad_s: float = 0.2          # extend each wave window by this at both ends
    link_tracks: int = 0             # waves = 0 long tracks only (tab 4 'long track'), 1 all final tracks
    # method 2 = the grid peak rule of tab 3 (copied from the Events tab when the analysis runs)
    grid_prom_unit: int = 0          # 0 prominence as % dF/F (grey levels / typical grid F0), 1 grey levels
    grid_sigma: float = 5.0          # frames
    grid_prominence: float = 1.5     # grey levels (tab 3)
    grid_distance: int = 50          # frames
    grid_exclude_s: float = 10.0
    grid_f0: float = 0.0             # typical grid-square brightness used for the % conversion (0 = unknown)


CELL_PRESETS = {
    "one-cell notebook": {},          # the defaults above
    "Grid rule (tab 3 settings)": dict(method=2),
    "40x wave notebook": dict(threshold_pct=3.33, use_around_mean=0, local_max_s=0.5, short_window_s=0.2,
                              eps_pct=0.17, baseline_t0=12.0, baseline_t1=20.0, min_f0=3.0, sigma=4.0),
}


def grid_prom_pct(p: CellParams):
    """Grid prominence (grey levels) as % dF/F at the typical grid-square brightness."""
    return 100.0 * p.grid_prominence / p.grid_f0 if p.grid_f0 > 0 else p.prominence_pct


def grid_f0(meas_mean):
    """Typical grid-square brightness: median over squares of each square's median grey level."""
    m = np.asarray(meas_mean, np.float32)
    return float(np.nanmedian(np.nanmedian(m, axis=0))) if m.size else 0.0


def baseline_f0(F, fps, p: CellParams):
    a = int(max(0, round(p.baseline_t0 * fps)))
    b = int(min(len(F), max(a + 1, round(p.baseline_t1 * fps))))
    return np.nanmin(F[a:b], axis=0)


def _window_stat(s, a, b, fn):
    """fn over s[t+a .. t+b] (frame offsets, inclusive), edges clamped."""
    T = len(s)
    out = np.empty(T, np.float32)
    idx = np.arange(T)
    lo = np.clip(idx + a, 0, T - 1)
    hi = np.clip(idx + b, 0, T - 1)
    cs = np.concatenate([[0.0], np.cumsum(s, dtype=np.float64)])
    out[:] = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
    return out


def detect_cell_events(F, fps, p: CellParams, bad=None):
    """F: (T,) neuropil-corrected trace. Returns (dff_pct smoothed, event frames, amp_pct, tau_s, f0).
    `bad` frames (light off) are bridged by interpolation and no event is accepted near them."""
    T = len(F)
    if bad is not None and np.any(bad):
        bad = np.asarray(bad, bool)[:T]
        F = bridge(F, bad)
    else:
        bad = None
    f0 = float(baseline_f0(F[:, None], fps, p)[0])
    if not np.isfinite(f0) or f0 < p.min_f0:
        return None, np.zeros(0, int), np.zeros(0), np.zeros(0), f0
    d = (F / f0 - 1.0) * 100.0
    sig = p.grid_sigma if p.method == 2 else p.sigma
    s = ndimage.gaussian_filter1d(d.astype(np.float32), sig) if sig > 0 else d.astype(np.float32)
    t0 = int(max(0, round((p.grid_exclude_s if p.method == 2 else p.t_start) * fps)))
    t1 = T if p.t_end is None or p.t_end < 0 else int(min(T, round(p.t_end * fps)))
    if p.method == 2:
        from scipy.signal import find_peaks
        dist = max(1, int(p.grid_distance))
        if p.grid_prom_unit == 1:                 # literally the grid rule: grey levels on the cell's F
            g = ndimage.gaussian_filter1d(np.asarray(F, np.float32), sig) if sig > 0 else np.asarray(F, np.float32)
            pk, _ = find_peaks(g, prominence=p.grid_prominence, distance=dist)
        else:                                     # same rise relative to brightness, per cell
            pk, _ = find_peaks(s, prominence=grid_prom_pct(p), distance=dist)
    elif p.method == 1:
        from scipy.signal import find_peaks
        pk, _ = find_peaks(s, prominence=p.prominence_pct, distance=max(1, int(round(p.min_sep_s * fps))))
    else:
        fr = lambda sec: max(1, int(round(sec * fps)))
        # minimum of the next drop_window (notebook samples t, t+0.5, ... < t+2 s)
        w = fr(p.drop_window_s)
        fwd_min = _fwd_min_fast(s, w)          # minimum of s[t .. t+w]
        c = s > fwd_min + p.threshold_pct
        if p.use_around_mean:
            a = fr(p.around_window_s)
            c &= s > _window_stat(s, -a, a, "mean") + p.threshold_pct
        h = fr(p.local_max_s)
        prev = np.concatenate([np.full(h, np.inf), s[:-h]])
        nxt = np.concatenate([s[h:], np.full(h, np.inf)])
        c &= (s > prev) & (s > nxt)
        sw, lw = fr(p.short_window_s), fr(p.long_window_s)
        c &= (s >= _window_stat(s, 0, sw, "mean") + p.eps_pct) & (s >= _window_stat(s, -sw, 0, "mean") + p.eps_pct)
        c &= (s >= _window_stat(s, 0, lw, "mean") + p.eps_pct) | (s >= _window_stat(s, -lw, 0, "mean") + p.eps_pct)
        cand = np.nonzero(c)[0]
        # merge candidates closer than min_sep (keep the highest of each group)
        pk = []
        sep = fr(p.min_sep_s)
        for i in cand:
            if pk and i - pk[-1] <= sep:
                if s[i] > s[pk[-1]]:
                    pk[-1] = i
            else:
                pk.append(i)
        pk = np.asarray(pk, int)
    pk = pk[(pk >= t0) & (pk < t1)]
    if bad is not None and len(pk):
        m = max(1, int(round(2 * sig)))
        near = ndimage.binary_dilation(bad, iterations=m)
        pk = pk[~near[pk]]
    # amplitude and 1/e decay (find_spike_data)
    w1 = max(1, int(round(1.0 * fps)))
    amp = np.zeros(len(pk))
    tau = np.full(len(pk), np.nan)
    for k, i in enumerate(pk):
        seg = s[i:min(T, i + w1)]
        lo = float(seg.min())
        amp[k] = float(s[i]) - lo
        thr = lo + amp[k] / math.e
        below = np.nonzero(seg < thr)[0]
        if len(below):
            tau[k] = below[0] / fps
    return s, pk, amp, tau, f0


def _fwd_min_fast(s, w):
    """min(s[t .. t+w]) for every t (end clamped)."""
    from scipy.ndimage import minimum_filter1d
    T = len(s)
    padded = np.concatenate([s, np.full(w, s[-1], s.dtype)])
    return minimum_filter1d(padded, w + 1, mode="nearest", origin=-((w + 1) // 2))[:T]


# =============================================================================
# waves <-> cells
# =============================================================================
def wave_windows(res: "wc.WaveResults", fps, all_tracks=False):
    """Tracks of the grid analysis used as waves: list of dict(id, f0, f1, frames, pos).
    Long tracks only (default) or every final track (all_tracks)."""
    tt = res.track_table
    if tt is None or not len(tt):
        return []
    long_ids = set(tt["track"].astype(int)) if (all_tracks or "long" not in tt) else \
        set(tt.loc[tt["long"].astype(bool), "track"].astype(int))
    out = []
    for tr in res.tracks:
        if int(tr.id) not in long_ids or not len(tr.frames):
            continue
        fr = np.asarray(tr.frames, int)
        pos = np.asarray(tr.cur, float)
        out.append(dict(id=int(tr.id), f0=int(fr.min()), f1=int(fr.max()), frames=fr, pos=pos))
    out.sort(key=lambda w: w["f0"])
    return out


def _wave_pos(w, f):
    i = int(np.clip(np.searchsorted(w["frames"], f), 0, len(w["frames"]) - 1))
    return w["pos"][i]


def link_events(events, cx, cy, waves, fps, p: CellParams, um_per_px=None):
    """events: list (per cell) of frame arrays. Returns per-cell list of wave ids (or -1)."""
    pad = int(round(p.link_pad_s * fps))
    R = p.link_radius_um / (um_per_px or 1.0)
    out = []
    for k, ev in enumerate(events):
        lab = np.full(len(ev), -1, int)
        for j, f in enumerate(ev):
            best, bd = -1, np.inf
            for w in waves:
                if w["f0"] - pad <= f <= w["f1"] + pad:
                    q = _wave_pos(w, f)
                    d = math.hypot(q[0] - cx[k], q[1] - cy[k])
                    if p.link_mode == 1 and d > R:
                        continue
                    if d < bd:
                        best, bd = w["id"], d
            lab[j] = best
        out.append(lab)
    return out


def _passed(waves, cx, cy, p, um_per_px):
    """Which waves 'passed' each cell (time mode: all; distance mode: came within the radius)."""
    R = p.link_radius_um / (um_per_px or 1.0)
    P = np.zeros((len(cx), len(waves)), bool)
    for j, w in enumerate(waves):
        if p.link_mode == 0:
            P[:, j] = True
        else:
            d = np.hypot(w["pos"][:, 0][None, :] - cx[:, None], w["pos"][:, 1][None, :] - cy[:, None])
            P[:, j] = d.min(1) <= R
    return P


@dataclass
class CellResults:
    cells: pd.DataFrame          # one row per analysed cell
    events: pd.DataFrame         # one row per event
    waves: pd.DataFrame          # one row per wave
    summary: dict
    dff: np.ndarray              # (T, N) smoothed dF/F % (NaN for skipped cells)
    params: dict = field(default_factory=dict)
    copart: Optional[dict] = None   # co-participation analysis (see coparticipation())


def analyse_cells(Fc, Fn, fps, cells, p: CellParams, res=None, um_per_px=None, n_shuffle=200,
                  progress=None, bad=None):
    T, N = Fc.shape
    F = Fc - p.neuropil_r * Fn
    cx = np.array([c.centre[0] for c in cells])
    cy = np.array([c.centre[1] for c in cells])
    area = np.array([c.area for c in cells], float)
    s_um = um_per_px or 1.0
    dff = np.full((T, N), np.nan, np.float32)
    events, amps, taus, f0s = [], [], [], []
    for k in range(N):
        s, pk, a, t, f0 = detect_cell_events(F[:, k], fps, p, bad)
        if s is not None:
            dff[:, k] = s
        events.append(pk)
        amps.append(a)
        taus.append(t)
        f0s.append(f0)
        if k % 50 == 0:
            _p(progress, k, N, "cell events")
    analysed = np.array([np.isfinite(dff[0, k]) for k in range(N)])
    t0 = p.grid_exclude_s if p.method == 2 else p.t_start          # rate window = detection window
    t1 = (T / fps) if p.t_end is None or p.t_end < 0 else min(p.t_end, T / fps)
    dur_min = max(t1 - t0, 1e-9) / 60.0
    if bad is not None and np.any(bad):              # only the time with the light on counts
        a_, b_ = int(t0 * fps), int(t1 * fps)
        dur_min = max(float((~np.asarray(bad, bool)[a_:b_]).sum()) / fps, 1e-9) / 60.0
    waves = wave_windows(res, fps, all_tracks=bool(getattr(p, "link_tracks", 0))) if res is not None else []
    waves = [w for w in waves if w["f1"] >= t0 * fps and w["f0"] <= t1 * fps]
    lab = link_events(events, cx, cy, waves, fps, p, um_per_px) if waves else [np.full(len(e), -1) for e in events]
    passed = _passed(waves, cx, cy, p, um_per_px) if waves else np.zeros((N, 0), bool)
    wid = {w["id"]: j for j, w in enumerate(waves)}

    # per-event table
    rows = []
    for k in range(N):
        for f, a, t, l in zip(events[k], amps[k], taus[k], lab[k]):
            rows.append(dict(cell=k + 1, time_s=f / fps, amplitude_pct=a, decay_tau_s=t, wave=int(l) if l >= 0 else None))
    ev_df = pd.DataFrame(rows, columns=["cell", "time_s", "amplitude_pct", "decay_tau_s", "wave"])

    # per-cell table
    joined = np.zeros((N, len(waves)), bool)
    for k in range(N):
        for l in lab[k]:
            if l >= 0:
                joined[k, wid[l]] = True
    crow = []
    for k in range(N):
        ev = events[k] / fps
        isi = np.diff(ev)
        n_pass = int(passed[k].sum())
        crow.append(dict(
            cell=k + 1, x_px=cx[k], y_px=cy[k], x_um=cx[k] * s_um, y_um=cy[k] * s_um,
            area_px=area[k], area_um2=area[k] * s_um ** 2, source=cells[k].source, prob=cells[k].prob,
            f0=f0s[k], analysed=bool(analysed[k]), n_events=len(ev), rate_per_min=len(ev) / dur_min,
            mean_isi_s=float(isi.mean()) if len(isi) else np.nan,
            mean_amplitude_pct=float(np.mean(amps[k])) if len(amps[k]) else np.nan,
            mean_decay_tau_s=float(np.nanmean(taus[k])) if len(taus[k]) and np.isfinite(taus[k]).any() else np.nan,
            events_in_waves=int((lab[k] >= 0).sum()), events_outside_waves=int((lab[k] < 0).sum()),
            fraction_events_in_waves=float((lab[k] >= 0).mean()) if len(lab[k]) else np.nan,
            waves_passed=n_pass, waves_joined=int(joined[k].sum()),
            participation=float(joined[k].sum() / n_pass) if n_pass else np.nan))
    cell_df = pd.DataFrame(crow)

    # chance participation: circularly shifted event trains (same events, random timing)
    rng = np.random.default_rng(0)
    chance = np.nan
    obs = float(np.nanmean(cell_df["participation"])) if len(cell_df) and cell_df["participation"].notna().any() else np.nan
    p_shuffle = np.nan
    if waves and n_shuffle and np.isfinite(obs):
        a0, a1 = int(t0 * fps), int(t1 * fps)
        span = max(a1 - a0, 1)
        sims = []
        for _ in range(n_shuffle):
            sh = [((e - a0 + rng.integers(0, span)) % span) + a0 for e in events]
            lb = link_events(sh, cx, cy, waves, fps, p, um_per_px)
            j2 = np.zeros_like(joined)
            for k in range(N):
                for l in lb[k]:
                    if l >= 0:
                        j2[k, wid[l]] = True
            with np.errstate(invalid="ignore", divide="ignore"):
                part = j2.sum(1) / passed.sum(1)
            sims.append(np.nanmean(np.where(passed.sum(1) > 0, part, np.nan)))
        sims = np.asarray(sims)
        chance = float(np.nanmean(sims))
        p_shuffle = float((np.sum(sims >= obs) + 1) / (len(sims) + 1))

    # per-wave table: recruitment, latency, cell-based speed
    wrow = []
    for j, w in enumerate(waves):
        members = []
        for k in range(N):
            for f, l in zip(events[k], lab[k]):
                if l == w["id"]:
                    members.append((k, f))
        n_pass = int(passed[:, j].sum())
        lat = np.array([(f - w["f0"]) / fps for _, f in members])
        speed = r2 = np.nan
        if len(members) >= 4:
            v = w["pos"][-1] - w["pos"][0]
            nv = np.hypot(*v)
            if nv > 0:
                u = v / nv
                proj = np.array([(cx[k] - w["pos"][0][0]) * u[0] + (cy[k] - w["pos"][0][1]) * u[1] for k, _ in members])
                tt = np.array([f / fps for _, f in members])
                if np.ptp(tt) > 0 and np.ptp(proj) > 0:
                    A = np.vstack([tt, np.ones_like(tt)]).T
                    slope, icpt = np.linalg.lstsq(A, proj, rcond=None)[0]
                    pred = A @ [slope, icpt]
                    r2 = 1 - np.sum((proj - pred) ** 2) / max(np.sum((proj - proj.mean()) ** 2), 1e-12)
                    speed = slope * s_um
        wrow.append(dict(wave=w["id"], start_s=w["f0"] / fps, end_s=w["f1"] / fps,
                         duration_s=(w["f1"] - w["f0"]) / fps, cells_passed=n_pass,
                         cells_recruited=len({k for k, _ in members}),
                         fraction_recruited=len({k for k, _ in members}) / n_pass if n_pass else np.nan,
                         first_latency_s=float(lat.min()) if len(lat) else np.nan,
                         median_latency_s=float(np.median(lat)) if len(lat) else np.nan,
                         cell_speed=speed, cell_speed_r2=r2))
    wave_df = pd.DataFrame(wrow)
    act = cell_df[cell_df.analysed] if len(cell_df) else cell_df
    unit = "um" if um_per_px else "px"
    summ = {
        "cells.n_cells": float(N), "cells.n_analysed": float(analysed.sum()),
        "cells.fraction_active": float((act.n_events > 0).mean()) if len(act) else np.nan,
        "cells.mean_rate_per_min": float(act.rate_per_min.mean()) if len(act) else np.nan,
        "cells.mean_isi_s": float(act.mean_isi_s.mean()) if len(act) else np.nan,
        "cells.mean_amplitude_pct": float(act.mean_amplitude_pct.mean()) if len(act) else np.nan,
        "cells.mean_decay_tau_s": float(act.mean_decay_tau_s.mean()) if len(act) else np.nan,
        "cells.fraction_events_in_waves": float(act.fraction_events_in_waves.mean()) if len(act) else np.nan,
        "cells.mean_participation": obs, "cells.chance_participation": chance,
        "cells.participation_p_shuffle": p_shuffle,
        "cells.n_waves": float(len(waves)),
        "cells.mean_fraction_recruited": float(wave_df.fraction_recruited.mean()) if len(wave_df) else np.nan,
        "cells.mean_cells_recruited": float(wave_df.cells_recruited.mean()) if len(wave_df) else np.nan,
        f"cells.mean_cell_speed_{unit}_s": float(wave_df.cell_speed.mean()) if len(wave_df) else np.nan,
        "cells.link_mode": "time + distance" if p.link_mode == 1 else "time only",
    }
    cp = None
    if len(waves) >= 3 and N >= 3:
        _p(progress, N, N, "co-participation")
        cp = coparticipation(joined, passed, cx, cy, s_um, [w["id"] for w in waves])
        summ.update(cp["summary"])
    _p(progress, N, N, "cells analysed")
    return CellResults(cell_df, ev_df, wave_df, summ, dff, asdict(p), cp)


def _bh(p):
    """Benjamini-Hochberg adjusted p values."""
    p = np.asarray(p, float)
    n = len(p)
    if not n:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def coparticipation(joined, passed, cx, cy, um=1.0, wave_ids=None, n_shuffle=1000, min_shared=3, seed=0):
    """Do the same cells take part in the same waves more often than chance?
    joined (cells x waves): cell had an event linked to the wave; passed: the wave reached the cell.
    Null model: for every wave the same NUMBER of cells is recruited, drawn at random from the cells
    the wave reached (so wave size and coverage are kept, only the identity of the cells is shuffled).
    Returns per pair: shared waves, observed co-participation, expected, z, p (one-sided, more than
    chance), BH-adjusted q; wave x wave Jaccard similarity; distance dependence; summary numbers."""
    J = np.asarray(joined, bool)
    E = np.asarray(passed, bool) | J                         # eligible: wave reached the cell
    N, Wn = J.shape
    rng = np.random.default_rng(seed)
    Jf = J.astype(np.float32)
    obs = Jf @ Jf.T                                          # waves joined together
    shared = E.astype(np.float32) @ E.astype(np.float32).T  # waves that reached both
    r = J.sum(0)
    elig = [np.nonzero(E[:, w])[0] for w in range(Wn)]
    s1 = np.zeros((N, N), np.float64)
    s2 = np.zeros((N, N), np.float64)
    ge = np.zeros((N, N), np.int32)
    wsim_null = []

    def jacc(M):
        inter = M.T @ M
        sz = M.sum(0)
        uni = sz[:, None] + sz[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            jj = np.where(uni > 0, inter / uni, np.nan)
        iu = np.triu_indices(M.shape[1], 1)
        return jj, float(np.nanmean(jj[iu])) if len(iu[0]) else np.nan

    for _ in range(n_shuffle):
        S = np.zeros((N, Wn), np.float32)
        for w in range(Wn):
            if r[w] and len(elig[w]):
                S[rng.choice(elig[w], size=min(r[w], len(elig[w])), replace=False), w] = 1
        c = S @ S.T
        s1 += c
        s2 += c * c
        ge += c >= obs - 1e-6
        wsim_null.append(jacc(S)[1])
    mu = s1 / n_shuffle
    sd = np.sqrt(np.maximum(s2 / n_shuffle - mu ** 2, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (obs - mu) / sd, 0.0)
    from scipy.stats import norm
    pperm = (ge + 1) / (n_shuffle + 1)
    # p from the shuffle distribution's mean / sd (normal approx.) – the permutation p alone cannot go
    # below 1/(n_shuffle+1), too coarse for thousands of pairs; never smaller than the data allow
    pval = np.where(sd > 0, norm.sf(z), pperm)
    iu = np.triu_indices(N, 1)
    ok = shared[iu] >= min_shared
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :]) * um
    pairs = pd.DataFrame(dict(cell_a=iu[0] + 1, cell_b=iu[1] + 1, distance=d[iu], waves_reaching_both=shared[iu],
                              waves_together=obs[iu], expected=mu[iu], z=z[iu], p=pval[iu],
                              p_perm=pperm[iu]))
    pairs = pairs[ok].reset_index(drop=True)
    pairs["q"] = _bh(pairs.p.to_numpy())
    pairs = pairs.sort_values("z", ascending=False).reset_index(drop=True)
    wsim, wmean = jacc(Jf)
    wnull = np.asarray(wsim_null, float)
    wp = float((np.sum(wnull >= wmean) + 1) / (len(wnull) + 1)) if np.isfinite(wmean) else np.nan
    # distance dependence (observed vs chance, per distance bin)
    dist_rows = []
    if len(pairs):
        edges = np.unique(np.percentile(pairs.distance, np.linspace(0, 100, 9)))
        for a, b in zip(edges[:-1], edges[1:]):
            m = (pairs.distance >= a) & (pairs.distance <= b)
            if m.sum() >= 5:
                dist_rows.append(dict(d_from=a, d_to=b, n_pairs=int(m.sum()),
                                      observed=float(pairs.waves_together[m].mean()),
                                      expected=float(pairs.expected[m].mean()), mean_z=float(pairs.z[m].mean())))
    dist_df = pd.DataFrame(dist_rows)
    # nearby cells are naturally in the same waves (a wave covers an area): z relative to pairs at a
    # similar distance tells whether a pair is together MORE than its distance explains
    if len(pairs) >= 20:
        nb = min(12, max(3, len(pairs) // 50))
        e2 = np.unique(np.percentile(pairs.distance, np.linspace(0, 100, nb + 1)))
        bi = np.clip(np.searchsorted(e2, pairs.distance, side="right") - 1, 0, len(e2) - 2)
        zz_ = pairs.z.to_numpy()
        mu_b = np.array([zz_[bi == j].mean() if np.any(bi == j) else 0 for j in range(len(e2) - 1)])
        sd_b = np.array([zz_[bi == j].std() if np.sum(bi == j) > 2 else 1 for j in range(len(e2) - 1)])
        sdv = np.where(sd_b > 0, sd_b, 1.0)
        pairs["z_vs_distance"] = (zz_ - mu_b[bi]) / sdv[bi]
    else:
        pairs["z_vs_distance"] = np.nan
    # cell order for the matrix plot (hierarchical clustering of the z matrix)
    order = np.arange(N)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        zz = np.nan_to_num(z, nan=0.0)
        zz = (zz + zz.T) / 2
        dm = zz.max() - zz
        np.fill_diagonal(dm, 0)
        if N > 2:
            order = leaves_list(linkage(squareform(dm, checks=False), "average"))
    except Exception:
        pass
    summ = {
        "cells.copart_pairs_tested": float(len(pairs)),
        "cells.copart_fraction_pairs_sig": float((pairs.q < 0.05).mean()) if len(pairs) else np.nan,
        "cells.copart_mean_z": float(pairs.z.mean()) if len(pairs) else np.nan,
        "cells.copart_fraction_beyond_distance": float((pairs.z_vs_distance > 3).mean()) if len(pairs) else np.nan,
        "cells.wave_similarity_jaccard": wmean,
        "cells.wave_similarity_chance": float(np.nanmean(wnull)) if len(wnull) else np.nan,
        "cells.wave_similarity_p": wp,
    }
    return dict(pairs=pairs, z=z.astype(np.float32), obs=obs.astype(np.float32), order=order, wave_sim=wsim,
                wave_ids=list(wave_ids) if wave_ids is not None else list(range(Wn)), dist=dist_df,
                wave_null=wnull, summary=summ, joined=J)


def save_cell_results(xlsx_path, cr: CellResults):
    """Add the cell sheets to the wave-results workbook (and cells.* rows to its summary)."""
    from openpyxl import load_workbook  # noqa: F401  (engine check)
    if os.path.exists(xlsx_path):
        xl = pd.read_excel(xlsx_path, sheet_name=None)
        summ = xl.get("summary", pd.DataFrame(columns=["metric", "value"]))
        summ = summ[~summ["metric"].astype(str).str.startswith("cells.")]
        add = pd.DataFrame(list(cr.summary.items()), columns=["metric", "value"])
        xl["summary"] = pd.concat([summ, add], ignore_index=True)
    else:
        xl = {"summary": pd.DataFrame(list(cr.summary.items()), columns=["metric", "value"])}
    xl["cells_single"] = cr.cells
    xl["cell_events"] = cr.events
    xl["cell_waves"] = cr.waves
    xl["cell_parameters"] = pd.DataFrame(list(cr.params.items()), columns=["parameter", "value"])
    cp = getattr(cr, "copart", None)
    for k in ("cell_pairs", "wave_similarity", "copart_distance"):
        xl.pop(k, None)
    if cp is not None:
        xl["cell_pairs"] = cp["pairs"]
        ids = cp["wave_ids"]
        xl["wave_similarity"] = pd.DataFrame(cp["wave_sim"], index=[f"wave {i}" for i in ids],
                                             columns=[f"wave {i}" for i in ids]).reset_index(names="")
        xl["copart_distance"] = cp["dist"]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        for name, df in xl.items():
            df.to_excel(xw, sheet_name=name[:31], index=False if name != "parameters" else False)


# =============================================================================
# dF/F video
# =============================================================================
def lut_imagej16():
    pts = [(0, (0, 0, 0)), (1, (0, 0, 1)), (2, (0, 1, 1)), (3, (0, 1, 0)), (4, (1, 1, 0)), (5, (1, 0, 0)), (6, (1, 1, 1))]
    x = np.linspace(0, 6, 256)
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(x, [p[0] for p in pts], [p[1][c] for p in pts])
    return (lut * 255).astype(np.uint8)          # RGB


def luminance(frame_bgr):
    f = frame_bgr.astype(np.float32)
    return 0.0722 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.2126 * f[:, :, 2]


def compute_f0(path, skip_frames=50, end_frame=None, sbin=2, tbin=2, progress=None, cancel=None):
    """Notebook 'prepare': per-pixel temporal mean of binned frames + dF/F sigma."""
    info = wc.video_info(path)
    W, H = info["width"], info["height"]
    n_end = info["n_frames"] if not end_frame or end_frame <= 0 else min(end_frame, info["n_frames"])
    Wo, Ho = W // sbin, H // sbin
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    acc = np.zeros((Ho, Wo), np.float64)
    nb = (n_end - skip_frames) // tbin
    samples = []
    count = 0
    frames_for_std = []
    for b in range(nb):
        g = None
        for _ in range(tbin):
            ok, fr = cap.read()
            if not ok:
                break
            l = luminance(fr)
            g = l if g is None else g + l
        if g is None:
            break
        g = cv2.resize((g / tbin)[:Ho * sbin, :Wo * sbin], (Wo, Ho), interpolation=cv2.INTER_AREA)
        acc += g
        count += 1
        if b % 30 == 0:
            frames_for_std.append(g.astype(np.float32))
        if b % 100 == 0:
            _p(progress, b, nb, "F0: averaging frames")
            if _c(cancel):
                cap.release()
                raise Cancelled()
    cap.release()
    F0 = (acc / max(count, 1)).astype(np.float32)
    safe = np.where(F0 < 1.0, 1.0, F0)
    for g in frames_for_std:
        d = (g - F0) / safe
        samples.append(d[F0 > 1.0][::4])
    std = float(np.std(np.concatenate(samples))) if samples else 0.1
    _p(progress, nb, nb, "F0 ready")
    return dict(F0=F0, std=std, skip=int(skip_frames), end=int(n_end), sbin=int(sbin), tbin=int(tbin),
                fps=float(info["fps"]), W=W, H=H)


def dff_rgb(frame_bgr, f0d, low_sigma=3.0, high_sigma=7.0, mask_pct=30.0, lut=None, f0=None, mask=None):
    """One frame -> RGB dF/F picture at the F0 resolution."""
    F0 = f0d["F0"] if f0 is None else f0
    Ho, Wo = F0.shape
    g = luminance(frame_bgr)
    if g.shape != (Ho, Wo):
        g = cv2.resize(g, (Wo, Ho), interpolation=cv2.INTER_AREA)
    safe = np.where(F0 < 1.0, 1.0, F0)
    d = (g - F0) / safe
    vmin, vmax = -low_sigma * f0d["std"], high_sigma * f0d["std"]
    idx = np.clip((d - vmin) / max(vmax - vmin, 1e-9) * 255, 0, 255).astype(np.uint8)
    rgb = (lut if lut is not None else lut_imagej16())[idx]
    if mask is None:
        mask = F0 < np.percentile(F0, mask_pct)
    rgb[mask] = 0
    return rgb


def save_f0(path, f0d):
    np.savez_compressed(path, **{k: v for k, v in f0d.items()})


def load_f0(path):
    z = np.load(path)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def render_dff_video(out_path, src, f0d, low_sigma, high_sigma, mask_pct, speed=4.0, t0=None, t1=None,
                     progress=None, cancel=None):
    """Notebook 'viewer' cell 3: binned dF/F movie at (fps / tbin) * speed."""
    tb = f0d["tbin"]
    fps_out = f0d["fps"] / tb * speed
    F0 = f0d["F0"]
    Ho, Wo = F0.shape
    lut = lut_imagej16()
    mask = F0 < np.percentile(F0, mask_pct)
    a = f0d["skip"] if t0 is None else int(round(t0 * f0d["fps"]))
    b = f0d["end"] if t1 is None or t1 <= 0 else int(round(t1 * f0d["fps"]))
    vw = None
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps_out, (Wo, Ho))
        if vw.isOpened():
            break
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    n = (b - a) // tb
    for i in range(n):
        acc = None
        for _ in range(tb):
            ok, fr = cap.read()
            if not ok:
                break
            f = fr.astype(np.float32)
            acc = f if acc is None else acc + f
        if acc is None:
            break
        rgb = dff_rgb((acc / tb), f0d, low_sigma, high_sigma, mask_pct, lut, mask=mask)
        vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if i % 100 == 0:
            _p(progress, i, n, "rendering dF/F video")
            if _c(cancel):
                break
    cap.release()
    vw.release()
    _p(progress, n, n, "dF/F video written")
    return out_path


# =============================================================================
# Cell-activity movie: video + cells flashing at their events + grid waves
# =============================================================================
EV_IN, EV_OUT = (0, 165, 255), (255, 220, 0)          # BGR: event in a wave (orange), outside (cyan)


def render_cell_movie(out_path, video_path, rotation, gray_mode, fps, cells, cr: "CellResults", f0, f1,
                      speed=1.0, colour="events", flash_s=0.6, res=None, active=None, grid_xy=None,
                      grid_size=None, show_grid=True, show_tracks=True, show_outlines=True,
                      levels_pct=(0.5, 99.7), gamma=1.0, dff_max=None, time_label=True, adapt=True,
                      max_side=1200, progress=None, cancel=None):
    """MP4 of the recording with every analysed cell drawn on it.
    colour 'events': a cell lights up at each detected event and fades over `flash_s`
                     (orange = event linked to a wave, cyan = outside waves);
    colour 'dff':    cells filled with their smoothed dF/F (inferno, 0 .. dff_max %).
    Optional: active grid squares (faint red), wave centres (blue) and tracks (orange line)."""
    info = wc.video_info(video_path)
    W, H = info["width"], info["height"]
    T = cr.dff.shape[0] if cr is not None else info["n_frames"]
    f0, f1 = max(0, int(f0)), min(int(f1), T, info["n_frames"])
    if f1 <= f0:
        raise ValueError("empty time range")
    speed = float(speed) if speed and speed > 0 else 1.0
    step = max(1, int(round(speed))) if speed >= 1 else 1
    fps_out = fps * (speed / step if speed >= 1 else speed)
    N = len(cells)
    # label image (last cell wins where they overlap) and outlines
    lab = np.full((H, W), -1, np.int32)
    for k, c in enumerate(cells):
        ok = (c.ypix < H) & (c.xpix < W)
        lab[c.ypix[ok], c.xpix[ok]] = k
    edge = np.zeros((H, W), np.uint8)
    if show_outlines and N:
        er = ndimage.grey_erosion(lab, size=3)
        di = ndimage.grey_dilation(lab, size=3)
        edge = ((lab >= 0) & ((er != lab) | (di != lab))).astype(np.uint8)
    # crop to the cells / grid (+ margin)
    ys, xs = np.nonzero(lab >= 0)
    if grid_xy is not None and len(grid_xy[0]):
        xs = np.r_[xs, grid_xy[0]]
        ys = np.r_[ys, grid_xy[1]]
    pad = 30
    if len(xs):
        x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad)
        y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad)
    else:
        x0, y0, x1, y1 = 0, 0, W, H
    cw, ch = x1 - x0, y1 - y0
    sc = min(1.0, max_side / max(cw, ch))
    ow, oh = int(cw * sc) // 2 * 2, int(ch * sc) // 2 * 2
    labc = lab[y0:y1, x0:x1]
    inside = labc >= 0
    labi = np.where(inside, labc, 0)
    edgec = edge[y0:y1, x0:x1].astype(bool)
    # per-cell event frames (+ wave link)
    ev_f = [np.zeros(0, int) for _ in range(N)]
    ev_w = [np.zeros(0, bool) for _ in range(N)]
    if cr is not None and len(cr.events):
        for k, g in cr.events.groupby("cell"):
            if 1 <= k <= N:
                ev_f[k - 1] = np.round(g.time_s.to_numpy() * fps).astype(int)
                ev_w[k - 1] = g.wave.notna().to_numpy()
    fl = max(1, int(round(flash_s * fps)))
    pre = max(1, int(round(0.1 * fps)))
    if colour == "dff" and cr is not None:
        d = cr.dff
        if dff_max is None:
            v = d[np.isfinite(d)]
            dff_max = float(np.percentile(v, 99)) if len(v) else 50.0
        dff_max = max(dff_max, 1e-3)
    import matplotlib
    cmap = matplotlib.colormaps["inferno"]
    lut_rgb = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)[:, ::-1]   # BGR
    g_lut = (np.power(np.linspace(0, 1, 256), 1.0 / max(gamma, 0.05)) * 255).astype(np.float32)
    # waves (grid results)
    win = []
    if cr is not None and len(cr.waves):
        win = [(int(r.wave), int(round(r.start_s * fps)), int(round(r.end_s * fps))) for r in cr.waves.itertuples()]
    tracks = list(res.tracks) if (res is not None and show_tracks) else []
    centres = res.centres if (res is not None and show_tracks) else {}
    half = (grid_size or 0) / 2.0
    levels = None
    vw = None
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps_out, (ow, oh))
        if vw.isOpened():
            break
    n_out = len(range(f0, f1, step))
    j = 0
    try:
        for i, g in frame_source(video_path, rotation, gray_mode, (f0, f1), step=step, cancel=cancel):
            g = g[y0:y1, x0:x1]
            if levels is None or adapt:
                lo, hi = np.percentile(g[::3, ::3], levels_pct)
                hi = max(hi, lo + 1)
                if levels is None:
                    levels = (float(lo), float(hi))
                else:                          # follow illumination changes smoothly (~0.3 s)
                    a_ = min(1.0, step / max(1.0, 0.3 * fps))
                    levels = (levels[0] + a_ * (lo - levels[0]), levels[1] + a_ * (hi - levels[1]))
            gi = np.clip((g - levels[0]) / (levels[1] - levels[0]) * 255, 0, 255).astype(np.uint8)
            img = np.repeat(g_lut[gi][..., None], 3, axis=2)
            # grid squares that are active now
            if show_grid and active is not None and grid_xy is not None and half > 0 and i < len(active):
                ov = img.copy()
                for k in np.nonzero(active[i])[0]:
                    cx_, cy_ = grid_xy[0][k] - x0, grid_xy[1][k] - y0
                    cv2.rectangle(ov, (int(cx_ - half), int(cy_ - half)), (int(cx_ + half), int(cy_ + half)),
                                  (40, 40, 230), -1)
                img = img * 0.75 + ov * 0.25
            # cell colours for this frame
            if N:
                col = np.zeros((N, 3), np.float32)
                al = np.zeros(N, np.float32)
                if colour == "dff" and cr is not None:
                    v = cr.dff[min(i, T - 1)]
                    okv = np.isfinite(v)
                    idx = np.clip((np.where(okv, v, 0) / dff_max) * 255, 0, 255).astype(int)
                    col = lut_rgb[idx]
                    al = np.where(okv, 0.25 + 0.6 * idx / 255.0, 0).astype(np.float32)
                else:
                    for k in range(N):
                        e = ev_f[k]
                        if not len(e):
                            continue
                        dt = i - e
                        m = (dt >= -pre) & (dt <= fl)
                        if m.any():
                            jj = np.nonzero(m)[0][np.argmin(np.abs(dt[m]))]
                            a = 0.85 if dt[jj] <= 0 else 0.85 * (1 - dt[jj] / fl)
                            al[k] = a
                            col[k] = EV_IN if ev_w[k][jj] else EV_OUT
                a_img = np.where(inside, al[labi], 0)[..., None]
                img = img * (1 - a_img) + col[labi] * a_img
                img[edgec] = img[edgec] * 0.4 + np.array([200, 200, 200], np.float32) * 0.6
            out = np.clip(img, 0, 255).astype(np.uint8)
            for t in tracks:
                if t.frames[0] <= i <= t.frames[-1]:
                    m = t.frames <= i
                    pts = np.vstack([t.prev[m][:1], t.cur[m]]) - [x0, y0]
                    cv2.polylines(out, [np.round(pts).astype(np.int32)], False, (0, 165, 255), 2)
            if i in centres:
                for cx_, cy_ in centres[i]:
                    cv2.circle(out, (int(cx_ - x0), int(cy_ - y0)), 6, (255, 80, 0), -1)
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
            fsc = max(0.5, ow / 1000.0)                 # text size follows the movie size
            if time_label:
                wn = [w for w, a, b in win if a <= i <= b]
                wc._label(out, f"{i / fps:.2f} s" + (f"  wave {wn[0]}" if wn else ""), "left", 0.7 * fsc)
                wc._label(out, f"{speed:g}x", "right", 0.7 * fsc)
            if colour != "dff":
                r_, th_ = int(8 * fsc), max(1, int(round(1.5 * fsc)))
                h_ = out.shape[0] - int(16 * fsc)
                x_ = int(14 * fsc)
                for colr, txt in ((EV_IN, "event in wave"), (EV_OUT, "outside waves")):
                    cv2.circle(out, (x_, h_), r_, colr, -1)
                    cv2.putText(out, txt, (x_ + r_ + 6, h_ + r_ // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * fsc,
                                (255, 255, 255), th_, cv2.LINE_AA)
                    x_ += r_ + 12 + cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * fsc, th_)[0][0] + int(20 * fsc)
            vw.write(out)
            j += 1
            if j % 50 == 0:
                _p(progress, j, n_out, "rendering cell movie")
    finally:
        vw.release()
    _p(progress, n_out, n_out, "cell movie written")
    return out_path
