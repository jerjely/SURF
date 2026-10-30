"""
wave_core.py - computational core for the retinal-wave calcium-imaging GUI.

Port of Yonehara_video_rotator.ipynb + Yonehara_wave_analysis.ipynb, plus a fast
replacement for the FIJI "Multi Measure" step.

Pipeline
    1. RotationCalib      temporal / nasal / centre + eye -> affine (nasal left, dorsal up)
    2. measure_video      grid (or ImageJ ROI zip) mean grey value per frame
                          -> Measurement (.npz) and a FIJI-format CSV
    3. detect_events      gaussian smoothing + scipy find_peaks per ROI
    4. build_rasters      'a' / -4..-1 / +1..+3 state code per ROI per frame
    5. filter_active      drop border ROIs and isolated active ROIs (8-neighbour rule)
    6. compute_centers    per-frame K-means wavefront centres (parallel)
    7. link_tracks        frame-to-frame nearest-neighbour linking -> tracks
    8. merge/drop tracks  VectorMerger equivalent
    9. analyse_waves      directions, segments, grid vector map, wave events, tables

Nothing in here imports Qt, so it can also be used from a notebook.
"""
from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, convolve
from scipy.signal import find_peaks

ProgressFn = Optional[Callable[[int, int, str], None]]
CancelFn = Optional[Callable[[], bool]]


def _p(progress, i, n, msg=""):
    if progress is not None:
        progress(int(i), int(n), msg)


def _c(cancel):
    return cancel is not None and cancel()


class Cancelled(Exception):
    pass


# =============================================================================
# Video helpers
# =============================================================================
def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = dict(
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 0.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def read_frame(path: str, index: int) -> Optional[np.ndarray]:
    """Return BGR frame `index` (0-based) or None."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def read_thumbnail(thm_path: str) -> Optional[np.ndarray]:
    """Sony .THM files are small JPEG thumbnails; decode by content, not extension."""
    try:
        data = np.fromfile(thm_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


GRAY_MODES = {
    "mean": "(R+G+B)/3  - FIJI default",
    "weighted": "weighted luminance (0.299R+0.587G+0.114B)",
    "green": "green channel only (GCaMP)",
}


def to_gray_f32(frame: np.ndarray, mode: str = "mean") -> np.ndarray:
    """BGR uint8 -> float32 single channel."""
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if mode == "green":
        return frame[:, :, 1].astype(np.float32)
    f = frame.astype(np.float32)
    if mode == "weighted":
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return cv2.transform(f, np.array([[1 / 3, 1 / 3, 1 / 3]], np.float32))


# =============================================================================
# 1. Rotation calibration  (Yonehara_video_rotator.ipynb)
# =============================================================================
def calculate_angle(p1, p2) -> float:
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


EYES = ("right", "left")


@dataclass
class RotationCalib:
    """Retina orientation from three clicks.

    version 3 (current): rotation only, no mirroring. Viewed from the ganglion-cell side with
    dorsal up, a RIGHT retina has nasal on the RIGHT (temporal left) and a LEFT retina has
    nasal on the LEFT (temporal right). Each retina is rotated to that view, so both end up
    DORSAL UP and are never mirrored.

    version 2 (21 Oct 2026, briefly): nasal-left for every eye with a top-to-bottom flip for
    left retinas (mirrored). Kept so measurements made with it still display.

    version 1 (legacy, notebook): rotation by the temporal->nasal angle (nasal ends up on the
    right) plus an optional cv2.flip "mirror". Kept so older measurements still display."""
    temporal: tuple
    nasal: tuple
    center: tuple
    eye: str = "right"
    mirror: bool = False         # legacy (version 1) only
    mirror_way: int = 1          # legacy: cv2.flip code 1 = horizontal, 0 = vertical
    version: int = 3

    @property
    def angle(self) -> float:
        """Rotation applied (degrees, cv2 convention)."""
        if self.version >= 3 and self.eye == "right":
            return calculate_angle(self.temporal, self.nasal)     # nasal ends up on the right
        if self.version >= 2:
            return calculate_angle(self.nasal, self.temporal)     # nasal ends up on the left
        return calculate_angle(self.temporal, self.nasal)

    @property
    def flips(self) -> bool:
        """True if the transform contains a reflection (changes handedness)."""
        if self.version >= 3:
            return False
        if self.version == 2:
            return self.eye == "left"
        return bool(self.mirror)

    def matrix(self, width: int, height: int) -> np.ndarray:
        cx, cy = float(self.center[0]), float(self.center[1])
        M = cv2.getRotationMatrix2D((cx, cy), self.angle, 1.0)
        M[0, 2] += (width // 2) - cx
        M[1, 2] += (height // 2) - cy
        F = None
        if self.version == 2:
            if self.eye == "left":                  # (v2) ventral was up -> flip top/bottom
                F = np.eye(3)
                F[1, 1], F[1, 2] = -1, height - 1
        elif self.mirror:
            F = np.eye(3)
            if int(self.mirror_way) == 1:
                F[0, 0], F[0, 2] = -1, width - 1
            else:
                F[1, 1], F[1, 2] = -1, height - 1
        if F is not None:
            M = (F @ np.vstack([M, [0, 0, 1]]))[:2]
        return M

    def apply(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        return cv2.warpAffine(gray, self.matrix(w, h), (w, h))

    def transform_points(self, pts, width, height) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        M = self.matrix(width, height)
        return pts @ M[:, :2].T + M[:, 2]

    # --- persistence -------------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d["temporal"] = [float(v) for v in self.temporal]
        d["nasal"] = [float(v) for v in self.nasal]
        d["center"] = [float(v) for v in self.center]
        return d

    @classmethod
    def from_dict(cls, d):
        version = int(d.get("version", 2 if "eye" in d else 1))   # stored explicitly since v2
        return cls(tuple(d["temporal"]), tuple(d["nasal"]), tuple(d["center"]),
                   str(d.get("eye", "right")), bool(d.get("mirror", False)),
                   int(d.get("mirror_way", 1)), version)

    def upgraded(self, eye=None):
        """Same clicks in the current (version 2) convention."""
        return RotationCalib(self.temporal, self.nasal, self.center, eye or self.eye, version=3)

    def save_pkl(self, path):
        """Same layout as the notebook's '<file>rotation_calibPoints.pkl'."""
        pd.DataFrame({"temporal": list(self.temporal), "nasal": list(self.nasal),
                      "centerPoint": list(self.center)}).to_pickle(path)

    @classmethod
    def load_pkl(cls, path, eye="right"):
        """Points from a '<file>rotation_calibPoints.pkl' (notebook or this app)."""
        df = pd.read_pickle(path)
        return cls((float(df["temporal"][0]), float(df["temporal"][1])),
                   (float(df["nasal"][0]), float(df["nasal"][1])),
                   (float(df["centerPoint"][0]), float(df["centerPoint"][1])), eye)


def export_rotated_video(src: str, dst: str, calib: RotationCalib,
                         progress: ProgressFn = None, cancel: CancelFn = None,
                         gray_mode: str = "mean"):
    """Rotated, grey, optionally mirrored copy (mp4v), like the notebook."""
    info = video_info(src)
    w, h = info["width"], info["height"]
    M = calib.matrix(w, h)
    cap = cv2.VideoCapture(src)
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (w, h), isColor=True)
    n = info["n_frames"]
    i = 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            g = to_gray_f32(fr, gray_mode)
            r = cv2.warpAffine(g, M, (w, h))
            r8 = np.clip(r + 0.5, 0, 255).astype(np.uint8)
            out.write(cv2.cvtColor(r8, cv2.COLOR_GRAY2BGR))
            i += 1
            if i % 50 == 0:
                _p(progress, i, n, "rotating")
                if _c(cancel):
                    raise Cancelled()
    finally:
        cap.release()
        out.release()
    _p(progress, n, n, "done")
    return dst


# =============================================================================
# 2. ROIs and measurement  (replaces FIJI Multi Measure)
# =============================================================================
@dataclass
class GridSpec:
    x0: int
    y0: int
    size: int
    rows: int
    cols: int

    @property
    def width(self):
        return self.size * self.cols

    @property
    def height(self):
        return self.size * self.rows

    def names(self):
        return [f"R{r + 1}C{c + 1}" for r in range(self.rows) for c in range(self.cols)]

    def centroids(self):
        c, r = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        x = self.x0 + c.ravel() * self.size + self.size / 2.0
        y = self.y0 + r.ravel() * self.size + self.size / 2.0
        return x, y


class RoiSet:
    """Either a regular grid (fast block-mean path) or arbitrary masks (sparse matmul)."""

    def __init__(self, kind, names, x, y, area, bbox, grid: GridSpec = None, W=None):
        self.kind = kind            # 'grid' | 'mask'
        self.names = list(names)
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.area = np.asarray(area, float)
        self.bbox = bbox            # (x0, y0, w, h) region that has to be sampled
        self.grid = grid
        self.W = W                  # sparse (N, w*h) averaging matrix for 'mask'

    def __len__(self):
        return len(self.names)

    @classmethod
    def from_grid(cls, g: GridSpec):
        x, y = g.centroids()
        return cls("grid", g.names(), x, y, np.full(g.rows * g.cols, g.size * g.size),
                   (g.x0, g.y0, g.width, g.height), grid=g)

    @classmethod
    def from_masks(cls, masks: Sequence[np.ndarray], names):
        """masks: list of boolean full-frame arrays."""
        ys, xs = [], []
        for m in masks:
            yy, xx = np.nonzero(m)
            ys.append(yy)
            xs.append(xx)
        allx = np.concatenate(xs)
        ally = np.concatenate(ys)
        bx0, by0 = int(allx.min()), int(ally.min())
        bw, bh = int(allx.max()) - bx0 + 1, int(ally.max()) - by0 + 1
        rows, cols, vals, area, cx, cy = [], [], [], [], [], []
        for k, (yy, xx) in enumerate(zip(ys, xs)):
            n = len(xx)
            rows.append(np.full(n, k))
            cols.append((yy - by0) * bw + (xx - bx0))
            vals.append(np.full(n, 1.0 / max(n, 1)))
            area.append(n)
            cx.append(xx.mean() + 0.5)
            cy.append(yy.mean() + 0.5)
        W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(len(masks), bw * bh), dtype=np.float32)
        return cls("mask", names, cx, cy, area, (bx0, by0, bw, bh), W=W)

    @classmethod
    def from_imagej_zip(cls, path, width, height):
        """Read a FIJI RoiSet.zip (rectangles, ovals, polygons). Needs `roifile`."""
        import roifile
        rois = roifile.ImagejRoi.fromfile(path)
        if not isinstance(rois, list):
            rois = [rois]
        masks, names = [], []
        for k, r in enumerate(rois):
            m = np.zeros((height, width), np.uint8)
            rt = r.roitype
            if rt == roifile.ROI_TYPE.RECT:
                m[max(r.top, 0):r.bottom, max(r.left, 0):r.right] = 1
            elif rt == roifile.ROI_TYPE.OVAL:
                cx, cy = (r.left + r.right) / 2.0, (r.top + r.bottom) / 2.0
                ax, ay = (r.right - r.left) / 2.0, (r.bottom - r.top) / 2.0
                cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(round(ax)), int(round(ay))),
                            0, 0, 360, 1, -1)
            else:
                pts = np.asarray(r.coordinates(), float)
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
            masks.append(m.astype(bool))
            names.append(r.name or f"ROI{k + 1}")
        g = _grid_from_rect_rois(rois)
        if g is not None:          # a FIJI grid -> fast exact block path, R{r}C{c} names
            return cls.from_grid(g)
        return cls.from_masks(masks, names)


def _grid_from_rect_rois(rois):
    """If the ROI zip is a regular, row-major grid of equal squares, return its GridSpec."""
    try:
        import roifile
        if not all(r.roitype == roifile.ROI_TYPE.RECT for r in rois):
            return None
        L = np.array([r.left for r in rois]); T = np.array([r.top for r in rois])
        Wd = np.array([r.right - r.left for r in rois]); Ht = np.array([r.bottom - r.top for r in rois])
        if not (np.all(Wd == Wd[0]) and np.all(Ht == Ht[0]) and Wd[0] == Ht[0]):
            return None
        s = int(Wd[0])
        xs, ys = np.unique(L), np.unique(T)
        if len(xs) * len(ys) != len(rois):
            return None
        g = GridSpec(int(xs[0]), int(ys[0]), s, len(ys), len(xs))
        exp_x, exp_y = g.centroids()
        got_x, got_y = L + s / 2.0, T + s / 2.0
        if np.allclose(exp_x, got_x) and np.allclose(exp_y, got_y):
            return g
    except Exception:
        pass
    return None


@dataclass
class Measurement:
    mean: np.ndarray            # (T, N) float32 mean grey value
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    names: list
    fps: float
    frame_size: tuple           # (w, h) of the measured (possibly rotated) space
    source: str = ""
    rotation: Optional[dict] = None
    grid: Optional[dict] = None
    gray_mode: str = "mean"
    elapsed_s: float = 0.0

    @property
    def n_frames(self):
        return self.mean.shape[0]

    @property
    def duration(self):
        return self.n_frames / self.fps

    def save_npz(self, path):
        np.savez_compressed(
            path, mean=self.mean, x=self.x, y=self.y, area=self.area,
            names=np.array(self.names), fps=self.fps, frame_size=np.array(self.frame_size),
            meta=json.dumps(dict(source=self.source, rotation=self.rotation, grid=self.grid,
                                 gray_mode=self.gray_mode, elapsed_s=self.elapsed_s)))

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(z["mean"], z["x"], z["y"], z["area"], [str(s) for s in z["names"]],
                   float(z["fps"]), tuple(int(v) for v in z["frame_size"]),
                   meta.get("source", ""), meta.get("rotation"), meta.get("grid"),
                   meta.get("gray_mode", "mean"), meta.get("elapsed_s", 0.0))

    # ---- FIJI-compatible CSV ---------------------------------------------
    def save_fiji_csv(self, path, progress: ProgressFn = None):
        """' ,Area(R1C1),Mean(R1C1),X(R1C1),Y(R1C1),...' exactly like Multi Measure,
        so the original notebook can read it unchanged."""
        hdr = [" "]
        consts = []
        for k, n in enumerate(self.names):
            hdr += [f"Area({n})", f"Mean({n})", f"X({n})", f"Y({n})"]
            consts.append((f"{int(round(self.area[k]))}", f"{self.x[k]:.3f}", f"{self.y[k]:.3f}"))
        fmt = "".join(f",{a},%.3f,{xx},{yy}" for a, xx, yy in consts) + "\n"
        T = self.n_frames
        with open(path, "w", newline="") as f:
            f.write(",".join(hdr) + "\n")
            for t in range(T):
                f.write(str(t + 1) + (fmt % tuple(self.mean[t].tolist())))
                if t % 500 == 0:
                    _p(progress, t, T, "writing CSV")
        _p(progress, T, T, "CSV written")

    @classmethod
    def load_fiji_csv(cls, path, fps, frame_size=(0, 0)):
        head = pd.read_csv(path, nrows=1)
        pat = re.compile(r"^(\w+)\((.+)\)$")
        cols = {}
        for c in head.columns:
            m = pat.match(c.strip())
            if m:
                cols.setdefault(m.group(2), {})[m.group(1)] = c
        names = [n for n in cols if "Mean" in cols[n]]
        mean_cols = [cols[n]["Mean"] for n in names]
        mean = pd.read_csv(path, usecols=mean_cols)[mean_cols].to_numpy(np.float32)
        mean = np.nan_to_num(mean)
        def first(key, default):
            return np.array([float(head[cols[n][key]].iloc[0]) if key in cols[n] else default
                             for n in names])
        return cls(mean, first("X", np.nan), first("Y", np.nan), first("Area", np.nan),
                   names, float(fps), tuple(frame_size), source=path)


def measure_video(path: str, rois: RoiSet, rotation: Optional[RotationCalib] = None,
                  gray_mode: str = "mean", progress: ProgressFn = None,
                  cancel: CancelFn = None, frame_range=None) -> Measurement:
    """Mean grey value of every ROI in every frame.

    Speed tricks (why this is not slow like a per-ROI loop):
      * frames are decoded in a background thread while the previous one is processed
      * rotation (+mirror) is folded into ONE warpAffine that only renders the ROI
        bounding box, not the whole frame
      * grid: cv2.resize(INTER_AREA) on an integer factor = exact block mean (SIMD)
      * masks: a single sparse matrix x pixel-vector product per frame
    """
    info = video_info(path)
    W, H = info["width"], info["height"]
    fps = info["fps"]
    bx, by, bw, bh = rois.bbox
    if rotation is not None:
        M = rotation.matrix(W, H).copy()
        M[0, 2] -= bx
        M[1, 2] -= by
        # only the source region that maps into the ROI box needs grey conversion
        Minv = cv2.invertAffineTransform(M)
        corners = np.array([[0, 0], [bw, 0], [0, bh], [bw, bh]], float)
        src = corners @ Minv[:, :2].T + Minv[:, 2]
        sx0 = int(max(0, np.floor(src[:, 0].min()) - 2)); sy0 = int(max(0, np.floor(src[:, 1].min()) - 2))
        sx1 = int(min(W, np.ceil(src[:, 0].max()) + 3)); sy1 = int(min(H, np.ceil(src[:, 1].max()) + 3))
        M[:, 2] += M[:, :2] @ np.array([sx0, sy0], float)
    else:
        if bx < 0 or by < 0 or bx + bw > W or by + bh > H:
            raise ValueError("ROI region extends outside the frame")
        M = None
    start, stop = (0, info["n_frames"]) if frame_range is None else frame_range
    n_est = max(stop - start, 1)
    N = len(rois)
    out = np.zeros((n_est + 64, N), np.float32)

    q: queue.Queue = queue.Queue(maxsize=64)
    stop_flag = threading.Event()

    def reader():
        cap = cv2.VideoCapture(path)
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while i < stop and not stop_flag.is_set():
            ok, fr = cap.read()
            if not ok:
                break
            q.put(fr)
            i += 1
        cap.release()
        q.put(None)

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    t0 = time.time()
    k = 0
    g = rois.grid
    try:
        while True:
            fr = q.get()
            if fr is None:
                break
            if M is not None:
                gray = to_gray_f32(fr[sy0:sy1, sx0:sx1], gray_mode)
                crop = cv2.warpAffine(gray, M, (bw, bh), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            else:
                crop = to_gray_f32(fr[by:by + bh, bx:bx + bw], gray_mode)
            if not crop.flags.c_contiguous:
                crop = np.ascontiguousarray(crop)
            if rois.kind == "grid":
                vals = cv2.resize(crop, (g.cols, g.rows), interpolation=cv2.INTER_AREA).ravel()
            else:
                vals = rois.W @ crop.ravel()
            if k >= out.shape[0]:
                out = np.vstack([out, np.zeros((1024, N), np.float32)])
            out[k] = vals
            k += 1
            if k % 100 == 0:
                el = time.time() - t0
                _p(progress, k, n_est, f"{k / el:.0f} frames/s")
                if _c(cancel):
                    raise Cancelled()
    finally:
        stop_flag.set()
        while th.is_alive():          # drain so the reader can exit
            try:
                q.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
    el = time.time() - t0
    _p(progress, k, k, f"done: {k} frames in {el:.1f} s ({k / max(el, 1e-9):.0f} fps)")
    return Measurement(out[:k], rois.x, rois.y, rois.area, rois.names, fps, (W, H), path,
                       rotation.to_dict() if rotation else None,
                       asdict(g) if g is not None else None, gray_mode, el)


# =============================================================================
# 3. Event detection  (notebook cell 16)
# =============================================================================
@dataclass
class EventParams:
    sigma: float = 5.0          # gaussian_filter1d sigma (frames)
    prominence: float = 1.5     # find_peaks prominence (grey levels)
    distance: int = 50          # find_peaks min distance (frames)
    exclude_s: float = 10.0     # ignore peaks before this time (onset artefact)
    skip_dark: int = 1          # light-off frames (detected automatically) bridged, no events near them


def dark_frames(brightness, fps, after_s=0.2, before_s=0.04):
    """Frames with the light off (brightness < half of the bright level) plus a margin after each
    light-on and before each light-off (the switching transients). All False for normal videos."""
    b = np.asarray(brightness, float)
    if len(b) < 5:
        return np.zeros(len(b), bool)
    hi = np.percentile(b, 75)
    dark = b < 0.5 * hi
    if not dark.any():
        return dark
    a, bf = max(0, int(round(after_s * fps))), max(0, int(round(before_s * fps)))
    k = np.ones(a + bf + 1, bool)
    # dilate: each dark frame marks [i - bf, i + a]
    out = np.convolve(dark.astype(int), k.astype(int), mode="full")[bf:bf + len(b)] > 0
    return out | dark


def bridge(F, bad):
    """Linear interpolation over bad frames ((T,) or (T, N))."""
    F = np.array(F, np.float32, copy=True)
    if bad is None or not np.any(bad):
        return F
    bad = np.asarray(bad, bool)[:len(F)]
    good = np.nonzero(~bad)[0]
    if len(good) < 2:
        return F
    t = np.arange(len(F))
    if F.ndim == 1:
        F[bad] = np.interp(t[bad], good, F[good])
    else:
        for k in range(F.shape[1]):
            F[bad, k] = np.interp(t[bad], good, F[good, k])
    return F


def meas_dark_frames(meas, p=None):
    if p is not None and not getattr(p, "skip_dark", 1):
        return None
    bad = dark_frames(meas.mean.mean(axis=1), meas.fps)
    return bad if bad.any() else None


def _near_dark(bad, sigma):
    from scipy.ndimage import binary_dilation
    return None if bad is None else binary_dilation(bad, iterations=max(1, int(round(2 * sigma))))


def _drop_near(idx, bad, sigma, near=None):
    if bad is None or not len(idx):
        return idx
    near = _near_dark(bad, sigma) if near is None else near
    return idx[~near[np.clip(idx, 0, len(near) - 1)]]


def smooth_traces(mean: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(mean.astype(np.float32), sigma, axis=0)


def detect_events(meas: Measurement, p: EventParams, progress: ProgressFn = None,
                  cancel: CancelFn = None):
    """Returns list (per ROI) of peak frame indices (0-based)."""
    bad = meas_dark_frames(meas, p)
    sm = smooth_traces(bridge(meas.mean, bad) if bad is not None else meas.mean, p.sigma)
    near = _near_dark(bad, p.sigma)
    first = int(math.ceil(p.exclude_s * meas.fps))
    peaks = []
    N = sm.shape[1]
    for k in range(N):
        idx, _ = find_peaks(sm[:, k], prominence=p.prominence, distance=p.distance)
        peaks.append(_drop_near(idx[idx > first], bad, p.sigma, near).astype(np.int64))
        if k % 200 == 0:
            _p(progress, k, N, "detecting peaks")
            if _c(cancel):
                raise Cancelled()
    _p(progress, N, N, "peaks done")
    return peaks


def detect_events_single(trace: np.ndarray, fps: float, p: EventParams, bad=None) -> np.ndarray:
    """Same rule as detect_events, for one ROI (used by the live preview)."""
    tr = bridge(trace, bad) if (bad is not None and getattr(p, "skip_dark", 1)) else np.asarray(trace, np.float32)
    sm = gaussian_filter1d(tr, p.sigma)
    idx, _ = find_peaks(sm, prominence=p.prominence, distance=p.distance)
    idx = idx[idx > int(math.ceil(p.exclude_s * fps))]
    return _drop_near(idx, bad if getattr(p, "skip_dark", 1) else None, p.sigma).astype(np.int64)


def cell_rates(peaks, meas: Measurement, exclude_s: float) -> np.ndarray:
    """Events per minute (notebook used len-1; here the true count)."""
    dur = max(meas.duration - exclude_s, 1e-9)
    return np.array([len(pk) for pk in peaks]) / dur * 60.0


def peaks_dataframe(peaks, fps) -> pd.DataFrame:
    longest = max([len(p) for p in peaks] + [1])
    data = {f"cluster_{k}": np.r_[p / fps, np.full(longest - len(p), np.nan)]
            for k, p in enumerate(peaks)}
    return pd.DataFrame(data)


# =============================================================================
# 4. Raster states  (notebook cell 17, first half)
# =============================================================================
I, A = 0, 10        # inactive / active; others: -4..-1 before, +1..+3 after
RASTER_BINS = [(-40, -35, -4), (-35, -20, -3), (-20, -10, -2), (-10, -6, -1),
               (-6, 6, A), (6, 10, 1), (10, 20, 2), (20, 35, 3)]


def build_rasters(peaks, n_frames, bins=RASTER_BINS) -> np.ndarray:
    """(T, N) int8 state codes; later peaks overwrite earlier ones (as in the notebook)."""
    T, N = n_frames, len(peaks)
    codes = np.zeros((T, N), np.int8)
    for k, pk in enumerate(peaks):
        col = codes[:, k]
        for p in pk:
            if p >= T - 50:
                continue
            for a, b, v in bins:
                lo, hi = max(p + a, 0), min(p + b, T)
                if hi > lo:
                    col[lo:hi] = v
    return codes


# =============================================================================
# 5. Spatial filter  (notebook cell 17, second half)
# =============================================================================
def infer_grid(x, y):
    """Map ROIs onto a (rows, cols) lattice if their centroids form one."""
    ux, uy = np.unique(np.round(x, 3)), np.unique(np.round(y, 3))
    if len(ux) * len(uy) > 4 * len(x) or len(ux) < 3 or len(uy) < 3:
        return None
    ci = np.searchsorted(ux, np.round(x, 3))
    ri = np.searchsorted(uy, np.round(y, 3))
    if len(set(zip(ri.tolist(), ci.tolist()))) != len(x):
        return None
    return dict(row=ri, col=ci, rows=len(uy), cols=len(ux), xs=ux, ys=uy)


def filter_active(codes: np.ndarray, x, y) -> np.ndarray:
    """Active = 'a', '-1', '-2' (notebook replaces -1/-2 by 'a'), then
       * ROIs on the outer border of the grid are switched off
       * an active ROI with no active 8-neighbour in the same frame is switched off.
    Returns (T, N) bool."""
    act = (codes == A) | (codes == -1) | (codes == -2)
    gi = infer_grid(x, y)
    T, N = act.shape
    if gi is not None:
        R, C = gi["rows"], gi["cols"]
        full = np.zeros((T, R, C), np.uint8)
        full[:, gi["row"], gi["col"]] = act
        full[:, 0, :] = 0
        full[:, -1, :] = 0
        full[:, :, 0] = 0
        full[:, :, -1] = 0
        k = np.ones((1, 3, 3), np.uint8)
        k[0, 1, 1] = 0
        nb = convolve(full, k, mode="constant", cval=0)
        keep = (full > 0) & (nb > 0)
        return keep[:, gi["row"], gi["col"]]
    # irregular ROIs: neighbours = within 1.5x median nearest-neighbour spacing
    from scipy.spatial import cKDTree
    pts = np.c_[x, y]
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    r = 1.5 * np.median(d[:, 1])
    pairs = tree.query_pairs(r, output_type="ndarray")
    adj = sparse.coo_matrix((np.ones(len(pairs) * 2), (np.r_[pairs[:, 0], pairs[:, 1]],
                                                        np.r_[pairs[:, 1], pairs[:, 0]])),
                            shape=(N, N)).tocsr()
    nb = (adj @ act.T.astype(np.float32)).T
    return act & (nb > 0)


# =============================================================================
# 6. Wavefront centres per frame  (notebook cell 21, K-means part)
# =============================================================================
@dataclass
class WaveParams:
    t_start: float = 10.0
    t_end: float = -1.0          # -1 = till the end
    min_pts_kmeans: int = 4      # fewer active ROIs -> single centroid
    min_sep: float = 100.0       # largest K whose centres are all > min_sep apart
    max_k: int = 39
    n_init: int = 10
    link_dist: float = 50.0      # frame-to-frame linking radius (px)
    merge_min_steps: int = 20
    merge_gap_frames: int = 5
    merge_dist: float = 200.0
    drop_min_steps: int = 5
    drop_min_disp: float = 10.0
    beg_end_min_disp: float = 50.0
    path_min_len: float = 500.0
    smooth_window: int = 30
    segment_len: float = 10.0
    long_track_steps: int = 50
    wave_isi_cutoff: float = 4.0
    grid_binning: int = 3
    workers: int = max(1, (os.cpu_count() or 2) - 1)


def frame_centres(pts: np.ndarray, min_pts=4, min_sep=100.0, max_k=39, n_init=10):
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2))
    if n < min_pts:
        return pts.mean(axis=0, keepdims=True)
    from sklearn.cluster import KMeans
    best = pts.mean(axis=0, keepdims=True)      # k = 1
    for k in range(2, min(max_k, n) + 1):
        km = KMeans(n_clusters=k, n_init=n_init, init="k-means++", random_state=42).fit(pts)
        lab = km.labels_
        cen = np.array([pts[lab == j].mean(axis=0) for j in range(k)])
        d = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(-1))
        d[np.diag_indices(k)] = np.inf
        if d.min() <= min_sep:
            break
        best = cen
    return best


def _centres_worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _centres_chunk(args):
    frames, idx_lists, xy, kw = args
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            return [(f, frame_centres(xy[ix], **kw)) for f, ix in zip(frames, idx_lists)]
    except ImportError:
        return [(f, frame_centres(xy[ix], **kw)) for f, ix in zip(frames, idx_lists)]


def compute_centres(active: np.ndarray, x, y, p: WaveParams, fps: float,
                    progress: ProgressFn = None, cancel: CancelFn = None) -> dict:
    """{frame: (k,2) centres} for frames with >=1 filtered-active ROI inside the window."""
    T = active.shape[0]
    f0 = int(round(p.t_start * fps))
    f1 = T if p.t_end is None or p.t_end < 0 else min(T, int(round(p.t_end * fps)))
    xy = np.c_[x, y].astype(float)
    frames = [f for f in range(f0, f1) if active[f].any()]
    idx = [np.nonzero(active[f])[0] for f in frames]
    kw = dict(min_pts=p.min_pts_kmeans, min_sep=p.min_sep, max_k=p.max_k, n_init=p.n_init)
    res = {}
    n = len(frames)
    chunk = 64
    jobs = [(frames[i:i + chunk], idx[i:i + chunk], xy, kw) for i in range(0, n, chunk)]
    if p.workers <= 1 or n < 256:
        for j, job in enumerate(jobs):
            for f, c in _centres_chunk(job):
                res[f] = c
            _p(progress, min((j + 1) * chunk, n), n, "wavefront centres")
            if _c(cancel):
                raise Cancelled()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=p.workers, initializer=_centres_worker_init) as ex:
            futs = [ex.submit(_centres_chunk, j) for j in jobs]
            done = 0
            for fu in as_completed(futs):
                for f, c in fu.result():
                    res[f] = c
                done += 1
                _p(progress, min(done * chunk, n), n, f"wavefront centres ({p.workers} processes)")
                if _c(cancel):
                    for f2 in futs:
                        f2.cancel()
                    raise Cancelled()
    _p(progress, n, n, "centres done")
    return res


# =============================================================================
# 7. Linking + 8. merging  (notebook cells 21, 22, 24)
# =============================================================================
@dataclass
class Track:
    id: int
    frames: np.ndarray          # frame index of each step (the 'current' frame)
    prev: np.ndarray            # (n,2) previous centre
    cur: np.ndarray             # (n,2) current centre
    dist: np.ndarray            # (n,) step length
    merged_from: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.frames)

    @property
    def start(self):
        return self.prev[0]

    @property
    def end(self):
        return self.cur[-1]

    @property
    def displacement(self):
        return float(np.hypot(*(self.end - self.start)))

    @property
    def path_length(self):
        return float(self.dist.sum())


def link_tracks(centres: dict, link_dist=50.0):
    """Greedy nearest-neighbour linking to the previous frame's centres.
    Returns tracks (list[Track], only those with >=1 step) and per-frame id arrays."""
    next_id = 0
    ids_at = {}
    pos = {}
    for f in sorted(centres):
        cur = centres[f]
        k = len(cur)
        ids = -np.ones(k, int)
        if (f - 1) in centres and len(centres[f - 1]) and k:
            pc = centres[f - 1]
            pid = ids_at[f - 1]
            d = np.sqrt(((cur[:, None, :] - pc[None, :, :]) ** 2).sum(-1))
            order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
            used_p = set()
            for ci, pj in order:
                if d[ci, pj] >= link_dist:
                    break
                if ids[ci] < 0 and pj not in used_p:
                    ids[ci] = pid[pj]
                    used_p.add(pj)
        for ci in range(k):
            if ids[ci] < 0:
                ids[ci] = next_id
                next_id += 1
        ids_at[f] = ids
        for ci in range(k):
            pos.setdefault(int(ids[ci]), []).append((f, cur[ci]))
    tracks = []
    for tid, lst in pos.items():
        if len(lst) < 2:
            continue
        fr = np.array([a for a, _ in lst])
        xy = np.array([b for _, b in lst])
        tracks.append(Track(tid, fr[1:], xy[:-1], xy[1:],
                            np.hypot(*(xy[1:] - xy[:-1]).T)))
    return tracks, ids_at


def merge_and_drop(tracks, p: WaveParams):
    """VectorMerger + drop rules of cells 24. Track B continues track A when both
    have > merge_min_steps steps, B's 2nd step is within merge_gap_frames of A's last
    step and A's end lies within merge_dist of B's 2nd-step origin. Chains are joined."""
    n = len(tracks)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    long_ = [i for i, t in enumerate(tracks) if t.n > p.merge_min_steps]
    merged_any = set()
    for i in long_:
        a = tracks[i]
        for j in long_:
            if i == j:
                continue
            b = tracks[j]
            if abs(int(b.frames[1]) - int(a.frames[-1])) < p.merge_gap_frames and \
                    np.hypot(*(a.cur[-1] - b.prev[1])) < p.merge_dist:
                parent[find(j)] = find(i)
                merged_any.update((i, j))
    groups = {}
    for i in merged_any:
        groups.setdefault(find(i), []).append(i)
    final = []
    for i, t in enumerate(tracks):
        if i in merged_any:
            continue
        if t.n < p.drop_min_steps or t.displacement < p.drop_min_disp:
            continue
        final.append(t)
    for g in groups.values():
        g = sorted(g, key=lambda i: tracks[i].frames[0])
        seen = set()
        parts = []
        for i in g:
            t = tracks[i]
            m = np.array([f not in seen for f in t.frames.tolist()])
            seen.update(t.frames.tolist())
            parts.append((t.frames[m], t.prev[m], t.cur[m], t.dist[m]))
        fr = np.concatenate([q[0] for q in parts])
        o = np.argsort(fr, kind="stable")
        final.append(Track(tracks[g[0]].id, fr[o], np.concatenate([q[1] for q in parts])[o],
                           np.concatenate([q[2] for q in parts])[o],
                           np.concatenate([q[3] for q in parts])[o],
                           merged_from=[int(tracks[i].id) for i in g]))
    final.sort(key=lambda t: t.frames[0])
    return final


# =============================================================================
# 9. Direction analysis  (notebook cells 26-43)
# =============================================================================
def circ_mean_deg(a):
    a = np.deg2rad(np.asarray(a, float))
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def direction_deg(v):
    """Image-coordinate angle (y down) of a displacement vector."""
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


ANGLE_CONVENTION = "anatomical: 0=temporal, 90=dorsal, 180=nasal, 270=ventral (direction of travel)"


@dataclass
class Orientation:
    """Nasal / temporal points in the measured frame + which eye the frame shows.

    `eye` is the eye *as seen in the measured frame* (ganglion-cell side up): a retina that
    was mirrored by an older rotation looks like the other eye, so `eye` is swapped there.
    Dorsal is derived: viewed from the GCL with dorsal up, a right retina has nasal on the
    right and a left retina has nasal on the left."""
    temporal: Optional[tuple] = None
    nasal: Optional[tuple] = None
    eye: str = "right"

    def axes(self):
        """Unit vectors (image coords, y down) pointing towards temporal and towards dorsal."""
        if self.temporal is None or self.nasal is None:
            u = np.array([1.0, 0.0])
        else:
            u = np.subtract(self.temporal, self.nasal).astype(float)
            n = np.hypot(*u)
            u = u / n if n > 0 else np.array([1.0, 0.0])
        # u points to temporal. Right retina: temporal left when dorsal up -> d = u rotated
        # +90 deg in screen terms; left retina the opposite.
        d = np.array([-u[1], u[0]]) if self.eye == "right" else np.array([u[1], -u[0]])
        return u, d

    @property
    def ref_deg(self):
        """Image angle of the nasal -> temporal axis."""
        u, _ = self.axes()
        return float(direction_deg(u))

    def anat(self, ang_img):
        """Image angle (deg, y down) -> anatomical angle (0 T, 90 D, 180 N, 270 V)."""
        a = np.deg2rad(np.asarray(ang_img, float))
        v = np.stack([np.cos(a), np.sin(a)], -1)
        u, d = self.axes()
        out = np.degrees(np.arctan2(v @ d, v @ u)) % 360
        return np.where(np.isclose(out, 360.0), 0.0, out)

    rel = anat      # backwards-compatible name

    def dv_points(self):
        """Dorsal / ventral label positions (for plots), centred on the N-T axis."""
        if self.temporal is None or self.nasal is None:
            return None, None
        mid = (np.asarray(self.temporal, float) + np.asarray(self.nasal, float)) / 2
        half = np.hypot(*np.subtract(self.temporal, self.nasal)) / 2
        _, d = self.axes()
        return tuple(mid + d * half), tuple(mid - d * half)

    def to_dict(self):
        return {"temporal": None if self.temporal is None else [float(v) for v in self.temporal],
                "nasal": None if self.nasal is None else [float(v) for v in self.nasal],
                "eye": self.eye}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(None if d.get("temporal") is None else tuple(d["temporal"]),
                   None if d.get("nasal") is None else tuple(d["nasal"]),
                   str(d.get("eye", "right")))

    @classmethod
    def from_rotation(cls, cal: "RotationCalib", width, height, eye=None):
        """Orientation in the rotated frame produced by `cal`. `eye` = the real eye (needed for
        legacy calibrations, which did not store it)."""
        tp = cal.transform_points([cal.temporal, cal.nasal], width, height)
        eye = cal.eye if cal.version >= 2 else (eye or "right")
        if cal.flips:
            eye = "left" if eye == "right" else "right"
        return cls(tuple(tp[0]), tuple(tp[1]), eye)


def moving_average_preserve_ends(data, window):
    sm = np.array(data, float, copy=True)
    h = window // 2
    for i in range(1, len(data) - 1):
        sm[i] = data[max(i - h, 0):min(i + h + 1, len(data))].mean(axis=0)
    return sm


@dataclass
class WaveResults:
    tracks: list
    centres: dict
    ids_at: dict
    beg_end: pd.DataFrame          # per track (displacement > beg_end_min_disp)
    segments: pd.DataFrame         # 'travelDetails': angle, distance per ~10 px chunk
    track_table: pd.DataFrame      # all final tracks
    smoothed_paths: list
    grid_vectors: pd.DataFrame     # binned lattice: X, Y, n, mean angle (abs + rel)
    wave_events: pd.DataFrame
    summary: dict
    orientation: Orientation


def analyse_waves(tracks, centres, ids_at, meas: Measurement, rates: np.ndarray,
                  p: WaveParams, ori: Orientation) -> WaveResults:
    fps = meas.fps
    # --- per-track table ---------------------------------------------------
    rows = []
    for t in tracks:
        ang = float(direction_deg(t.end - t.start))
        rows.append(dict(track=t.id, merged_from=",".join(map(str, t.merged_from)),
                         start_s=t.frames[0] / fps, end_s=t.frames[-1] / fps,
                         duration_s=t.n / fps, steps=t.n,
                         start_x=t.start[0], start_y=t.start[1], end_x=t.end[0], end_y=t.end[1],
                         start_end_length=t.displacement, path_length=t.path_length,
                         speed=t.path_length / (t.n / fps), net_speed=t.displacement / (t.n / fps),
                         angle_img_deg=ang % 360, angle_anat_deg=float(ori.anat(ang)),
                         long=t.n > p.long_track_steps))
    tt = pd.DataFrame(rows)
    # --- beginning-to-end directions (cell 27/28) ---------------------------
    be = tt[tt.start_end_length > p.beg_end_min_disp][["track", "angle_anat_deg", "start_end_length"]] \
        if len(tt) else pd.DataFrame(columns=["track", "angle_anat_deg", "start_end_length"])
    be = be.rename(columns={"angle_anat_deg": "angle", "start_end_length": "distance"})
    # --- smoothed paths + ~10 px segments (cells 29-31) ----------------------
    paths, seg_rows = [], []
    for t in tracks:
        if t.path_length <= p.path_min_len:
            continue
        sm = moving_average_preserve_ends(t.prev, p.smooth_window)
        paths.append((t.id, sm))
        acc_a, acc_d = [], 0.0
        for w, z in zip(sm[:-1], sm[1:]):
            acc_a.append(float(direction_deg(z - w)))
            acc_d += float(np.hypot(*(z - w)))
            if acc_d >= p.segment_len:
                seg_rows.append(dict(track=t.id, angle=float(ori.anat(circ_mean_deg(acc_a))),
                                     distance=acc_d))
                acc_a, acc_d = [], 0.0
    seg = pd.DataFrame(seg_rows, columns=["track", "angle", "distance"])
    # --- grid vector map (cells 39-43) --------------------------------------
    gi = infer_grid(meas.x, meas.y)
    gv = pd.DataFrame(columns=["X", "Y", "n", "angle_img_deg", "angle_anat_deg"])
    if gi is not None and paths:
        b = max(1, p.grid_binning)
        bx, byy = gi["xs"][b::b], gi["ys"][b::b]
        BX, BY = np.meshgrid(bx, byy)
        centres_b = np.c_[BX.ravel(), BY.ravel()]
        lists = [[] for _ in range(len(centres_b))]
        for _, sm in paths:
            if len(sm) <= p.long_track_steps:
                continue
            d2 = ((sm[:-1, None, :] - centres_b[None]) ** 2).sum(-1)
            near = d2.argmin(1)
            angs = direction_deg(sm[1:] - sm[:-1])
            for k, a in zip(near, angs):
                lists[k].append(a)
        rows = []
        for (X, Y), L in zip(centres_b, lists):
            m = circ_mean_deg(L) if L else np.nan
            rows.append(dict(X=X, Y=Y, n=len(L), angle_img_deg=m,
                             angle_anat_deg=float(ori.anat(m)) if L else np.nan))
        gv = pd.DataFrame(rows)
        gv.attrs["bin_width"] = float((bx[1] - bx[0]) if len(bx) > 1 else 1.0)
    # --- wave events: group track starts separated by > cutoff s (cells 36-38)
    ev_rows = []
    if len(tt):
        st = tt.sort_values("start_s").reset_index(drop=True)
        isi = st.start_s.diff()
        st["isi_s"] = isi
        new = isi.isna() | (isi > p.wave_isi_cutoff)
        st["wave"] = new.cumsum()
        for w, g in st.groupby("wave"):
            ev_rows.append(dict(wave=int(w), start_s=g.start_s.iloc[0], end_s=g.end_s.max(),
                                n_tracks=len(g), start_x=g.start_x.iloc[0], start_y=g.start_y.iloc[0]))
    ev = pd.DataFrame(ev_rows)
    if len(ev):
        ev["interval_s"] = ev.start_s.diff()
    long_t = tt[tt.long] if len(tt) else tt
    analysed_s = (min(meas.duration, p.t_end) if p.t_end and p.t_end > 0 else meas.duration) - p.t_start
    summary = dict(
        n_rois=len(meas.names), fps=fps, duration_s=meas.duration,
        analysed_window_s=f"{p.t_start} - {p.t_end if p.t_end and p.t_end > 0 else 'end'}",
        n_tracks=len(tt), n_long_tracks=int(len(long_t)),
        n_wave_events=len(ev),
        wave_frequency_per_min=(len(ev) / analysed_s * 60) if analysed_s > 0 else np.nan,
        mean_inter_wave_interval_s=float(ev.interval_s.mean()) if len(ev) > 1 else np.nan,
        mean_cell_rate_per_min=float(np.mean(rates)) if len(rates) else np.nan,
        active_cells_gt_0p2_per_min=int((rates > 0.2).sum()),
        mean_long_track_duration_s=float(long_t.duration_s.mean()) if len(long_t) else np.nan,
        mean_long_track_path_px=float(long_t.path_length.mean()) if len(long_t) else np.nan,
        mean_long_track_speed_px_s=float(long_t.speed.mean()) if len(long_t) else np.nan,
        mean_long_track_net_speed_px_s=float(long_t.net_speed.mean()) if len(long_t) else np.nan,
        segments_circular_mean_deg=circ_mean_deg(seg.angle) if len(seg) else np.nan,
        angle_convention=ANGLE_CONVENTION,
        eye_in_measured_frame=ori.eye,
        orientation_axis=("from nasal/temporal points" if ori.temporal is not None and ori.nasal is not None
                          else "not set - frame assumed nasal-left / temporal-right"),
    )
    return WaveResults(tracks, centres, ids_at, be, seg, tt, paths, gv, ev, summary, ori)


def run_wave_pipeline(meas, active, rates, p: WaveParams, ori: Orientation,
                      progress: ProgressFn = None, cancel: CancelFn = None) -> WaveResults:
    cen = compute_centres(active, meas.x, meas.y, p, meas.fps, progress, cancel)
    _p(progress, 0, 1, "linking tracks")
    tracks, ids_at = link_tracks(cen, p.link_dist)
    final = merge_and_drop(tracks, p)
    res = analyse_waves(final, cen, ids_at, meas, rates, p, ori)
    res.summary["n_raw_tracks"] = len(tracks)
    _p(progress, 1, 1, "waves done")
    return res


# length-like columns per table (stored in px internally); positions are also kept in px
_LEN_COLS = {
    "tracks": (["start_end_length", "path_length"], ["start_x", "start_y", "end_x", "end_y"]),
    "begToEnd": (["distance"], []),
    "travelDetails": (["distance"], []),
    "grid_vectors": ([], ["X", "Y"]),
    "wave_events": ([], ["start_x", "start_y"]),
    "cells": ([], ["X", "Y"]),
}
_SPEED_COLS = {"tracks": ["speed", "net_speed"]}


def _convert(df, name, s):
    """px -> um (s = um per px) or just add _px suffixes when s is None."""
    df = df.copy()
    unit, k = ("um", s) if s else ("px", 1.0)
    lens, poss = _LEN_COLS.get(name, ([], []))
    ren = {}
    for c in lens + poss:
        if c in df.columns:
            if s and c in poss:
                df[c + "_px"] = df[c]
            df[c] = df[c].astype(float) * k
            ren[c] = f"{c}_{unit}"
    for c in _SPEED_COLS.get(name, []):
        if c in df.columns:
            df[c] = df[c].astype(float) * k
            ren[c] = f"{c}_{unit}_s"
    if "area" in df.columns:
        df["area"] = df["area"].astype(float) * k * k
        ren["area"] = f"area_{unit}2"
    return df.rename(columns=ren)


def results_tables(res: WaveResults, meas: Measurement, peaks, rates, um_per_px=None) -> dict:
    """All result tables in display units (um when the scale is known, otherwise px)."""
    s = um_per_px if um_per_px and um_per_px > 0 else None
    summ = []
    for key, v in res.summary.items():
        if s and "_px" in key and isinstance(v, (int, float)) and not isinstance(v, bool):
            summ.append((key.replace("_px", "_um"), v * s))
        else:
            summ.append((key, v))
    summ.append(("scale_um_per_px", s if s else "not set (lengths in px)"))
    cells = pd.DataFrame(dict(roi=meas.names, X=meas.x, Y=meas.y, area=meas.area,
                              n_events=[len(p) for p in peaks], rate_per_min=rates))
    raw = {"tracks": res.track_table, "begToEnd": res.beg_end, "travelDetails": res.segments,
           "grid_vectors": res.grid_vectors, "wave_events": res.wave_events, "cells": cells}
    out = {"summary": pd.DataFrame(summ, columns=["metric", "value"])}
    out.update({k: _convert(v, k, s) for k, v in raw.items()})
    k = s or 1.0
    tt = res.track_table
    long_t = tt[tt.long] if len(tt) else tt
    out["result"] = pd.concat([       # same columns as the old _result.xlsx
        pd.Series(long_t.start_end_length.values * k if len(long_t) else [], name="startEnd length"),
        pd.Series(res.segments.angle.values, name="angles"),
        pd.Series(long_t.duration_s.values if len(long_t) else [], name="duration"),
        pd.Series(long_t.path_length.values * k if len(long_t) else [], name="total length"),
        pd.Series(rates[rates > 0.2], name="cell_rate")], axis=1)
    return out


def save_results_excel(path, res: WaveResults, meas: Measurement, peaks, rates,
                       ep: EventParams, wp: WaveParams, um_per_px=None, info=None):
    """One workbook with everything; the 'result' sheet mirrors the old _result.xlsx.
    `info` (animal, age, condition, objective, ...) is added to the summary as info.* rows."""
    tabs = results_tables(res, meas, peaks, rates, um_per_px)
    if info:
        extra = pd.DataFrame([(f"info.{k}", "" if v is None else str(v)) for k, v in info.items()],
                             columns=["metric", "value"])
        tabs["summary"] = pd.concat([tabs["summary"], extra], ignore_index=True)
    s = um_per_px if um_per_px and um_per_px > 0 else None
    pr = {**{f"event.{k}": v for k, v in asdict(ep).items()}}
    for k, v in asdict(wp).items():
        pr[f"wave.{k}"] = v
        if s and k in WAVE_LENGTH_PARAMS:
            pr[f"wave.{k}  (um)"] = v * s
    pr.update({f"orientation.{k}": str(v) for k, v in res.orientation.to_dict().items()})
    pr.update({"um_per_px": s, "gray_mode": meas.gray_mode, "source": meas.source})
    params = pd.DataFrame([pr]).T
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in ("summary", "tracks", "begToEnd", "travelDetails", "grid_vectors",
                     "wave_events", "cells", "result"):
            tabs[name].to_excel(xw, sheet_name=name, index=False)
        params.to_excel(xw, sheet_name="parameters", header=["value"])


# WaveParams fields that are distances (px internally)
WAVE_LENGTH_PARAMS = ("min_sep", "link_dist", "merge_dist", "drop_min_disp", "beg_end_min_disp",
                      "path_min_len", "segment_len")


# =============================================================================
# Wave movie (fast cv2 renderer; replaces the 1-hour seaborn loop)
# =============================================================================
CODE_STYLE = {   # code: (BGR, alpha)
    I: ((189, 189, 191), 0.2), A: ((0, 0, 255), 1.0),
    1: ((162, 147, 234), 0.9), 2: ((162, 147, 234), 0.6), 3: ((162, 147, 234), 0.3),
    -1: ((162, 147, 234), 0.5),
}


def _label(img, text, side="left", scale=0.8):
    """Black text on a white box in the top-left or top-right corner."""
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 2
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    x = 10 if side == "left" else img.shape[1] - tw - 10
    y = 10 + tht
    cv2.rectangle(img, (x - 5, y - tht - 5), (x + tw + 5, y + base + 3), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), th, cv2.LINE_AA)


def render_wave_movie(out_path, meas: Measurement, codes, active, res: Optional[WaveResults],
                      f0: int, f1: int, speed=1.0, mode="full", background=None,
                      time_label="video", show_speed=True,
                      video_path=None, rotation: Optional[RotationCalib] = None,
                      max_side=1000, progress: ProgressFn = None, cancel: CancelFn = None):
    """mode 'full' = all raster states + centres + tracks; 'filtered' = filtered active only.
    background: optional grey image of the measured space; or video_path to overlay
    the live (rotated) video.
    speed: playback speed vs real time. The movie keeps the recording's fps; speed > 1 keeps
    every `speed`-th frame, speed < 1 writes every frame at fps * speed. The speed is printed
    in the upper-right corner."""
    speed = float(speed) if speed and speed > 0 else 1.0
    fps_out = meas.fps * min(speed, 1.0)
    step = max(speed, 1.0)
    frames_out = np.unique(np.round(np.arange(f0, f1, step)).astype(int))
    frames_out = frames_out[(frames_out >= 0) & (frames_out < codes.shape[0])]
    gi = infer_grid(meas.x, meas.y)
    W, H = meas.frame_size
    pad = 50
    x0 = int(max(0, np.floor(meas.x.min() - pad))); y0 = int(max(0, np.floor(meas.y.min() - pad)))
    x1 = int(min(W, np.ceil(meas.x.max() + pad))); y1 = int(min(H, np.ceil(meas.y.max() + pad)))
    cw, ch = x1 - x0, y1 - y0
    sc = min(1.0, max_side / max(cw, ch))
    ow, oh = int(cw * sc) // 2 * 2, int(ch * sc) // 2 * 2
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (ow, oh))
    cap = None
    M = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        if rotation is not None:
            vi = video_info(video_path)
            M = rotation.matrix(vi["width"], vi["height"]).copy()
            M[0, 2] -= x0; M[1, 2] -= y0
    if background is not None:
        bg_static = cv2.cvtColor(np.clip(background[y0:y1, x0:x1], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        bg_static = np.full((ch, cw, 3), 255, np.uint8)
    # cell geometry
    if gi is not None:
        s = float(gi["xs"][1] - gi["xs"][0]) if len(gi["xs"]) > 1 else 10.0
        gx0 = gi["xs"][0] - s / 2 - x0; gy0 = gi["ys"][0] - s / 2 - y0
        R, C = gi["rows"], gi["cols"]
        gw, gh = int(round(C * s)), int(round(R * s))
    track_pts = {}
    if res is not None:
        for t in res.tracks:
            track_pts[t.id] = t
    n = len(frames_out)
    pos = f0                      # next frame the capture will return
    speed_txt = f"{speed:g}x"
    try:
        for j, i in enumerate(frames_out):
            if cap is not None:
                while pos < i:        # skipped frames: grab without decoding
                    cap.grab()
                    pos += 1
                ok, fr = cap.read()
                pos += 1
                if ok:
                    g = to_gray_f32(fr, meas.gray_mode)
                    if M is not None:
                        g = cv2.warpAffine(g, M, (cw, ch))
                    else:
                        g = g[y0:y1, x0:x1]
                    bg = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    bg = bg_static.copy()
            else:
                bg = bg_static.copy()
            img = bg.astype(np.float32)
            state = codes[i].copy() if mode == "full" else np.where(active[i], A, I).astype(np.int8)
            if gi is not None:
                col = np.zeros((R, C, 3), np.float32); al = np.zeros((R, C, 1), np.float32)
                for code, (bgr, a) in CODE_STYLE.items():
                    m = state == code
                    if m.any():
                        col[gi["row"][m], gi["col"][m]] = bgr
                        al[gi["row"][m], gi["col"][m]] = a
                colU = cv2.resize(col, (gw, gh), interpolation=cv2.INTER_NEAREST)
                alU = cv2.resize(al, (gw, gh), interpolation=cv2.INTER_NEAREST)[..., None]
                ix0, iy0 = int(round(gx0)), int(round(gy0))
                sx0, sy0 = max(0, -ix0), max(0, -iy0)
                ex, ey = min(gw, cw - ix0), min(gh, ch - iy0)
                sub = img[iy0 + sy0:iy0 + ey, ix0 + sx0:ix0 + ex]
                a_ = alU[sy0:ey, sx0:ex]
                sub[:] = sub * (1 - a_) + colU[sy0:ey, sx0:ex] * a_
            else:
                for code, (bgr, a) in CODE_STYLE.items():
                    if code == I:
                        continue
                    for k in np.nonzero(state == code)[0]:
                        cv2.circle(img, (int(meas.x[k] - x0), int(meas.y[k] - y0)), 5, bgr, -1)
            out = np.clip(img, 0, 255).astype(np.uint8)
            if res is not None and mode == "full":
                for t in track_pts.values():
                    if t.frames[0] <= i <= t.frames[-1]:
                        m = t.frames <= i
                        pts = np.vstack([t.prev[m][:1], t.cur[m]]) - [x0, y0]
                        cv2.polylines(out, [np.round(pts).astype(np.int32)], False, (0, 165, 255), 2)
                if i in res.centres:
                    for cx, cy in res.centres[i]:
                        cv2.circle(out, (int(cx - x0), int(cy - y0)), 7, (255, 0, 0), -1)
            out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
            if time_label == "video":          # time in the original recording
                _label(out, f"{i / meas.fps:.2f} s", "left")
            elif time_label == "movie":        # time in the rendered movie
                _label(out, f"{j / fps_out:.2f} s", "left")
            if show_speed:
                _label(out, speed_txt, "right")
            writer.write(out)
            if j % 100 == 0:
                _p(progress, j, n, "rendering movie")
                if _c(cancel):
                    raise Cancelled()
    finally:
        writer.release()
        if cap is not None:
            cap.release()
    _p(progress, n, n, "movie written")
    return out_path


# =============================================================================
# Session file (all settings for one video, human readable)
# =============================================================================
def session_path(video_path):
    stem, _ = os.path.splitext(video_path)
    return stem + "_wavegui.json"


def load_session(video_path) -> dict:
    p = session_path(video_path)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_session(video_path, d: dict):
    with open(session_path(video_path), "w") as f:
        json.dump(d, f, indent=2, default=float)
