"""
SURF - Spontaneous Undulation in Retina Finder
Retinal wave analysis: rotate -> measure -> events -> waves -> compare

    python wave_gui.py            (optionally:  python wave_gui.py <folder>)

Replaces Yonehara_video_rotator.ipynb, the FIJI Multi Measure step and
Yonehara_wave_analysis.ipynb.  All heavy work runs in background threads; the
K-means step uses several processes.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import time
import sys
import traceback
from dataclasses import asdict, fields


def _ensure_std_streams():
    """Started without a console (pythonw / the SURF shortcut on Windows) sys.stdout, sys.stderr and
    sys.__stdout__/__stderr__ are None, and libraries that print progress (Cellpose, Suite2p, PyTorch,
    tqdm) fail with "'NoneType' object has no attribute 'write'". Give them somewhere to write
    (a log file next to the settings, so their messages can still be read)."""
    log = None
    for name in ("stdout", "stderr", "__stdout__", "__stderr__"):
        if getattr(sys, name, None) is None:
            if log is None:
                try:
                    log = open(os.path.join(os.path.expanduser("~"), ".surf_console.log"), "a",
                               encoding="utf-8", buffering=1)
                except Exception:
                    log = open(os.devnull, "w", encoding="utf-8")
            try:
                setattr(sys, name, log)
            except Exception:
                pass


_ensure_std_streams()

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal

import pyqtgraph as pg
import matplotlib
matplotlib.use("QtAgg")
import figexport
figexport.setup_rc()                    # exported SVG / EPS / PDF keep their text as text
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

import wave_core as wc
import wave_plots as wpl

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

APP_NAME = "SURF"
APP_LONG = "Spontaneous Undulation in Retina Finder"
APP_VERSION = "1.0 (build 2026-10-29)"

VIDEO_EXT = (".mp4", ".avi", ".mov", ".m4v")
MARK_COLORS = {"temporal": (0, 200, 0), "nasal": (230, 0, 0), "center": (0, 160, 255),
               "dorsal": (0, 200, 0), "ventral": (230, 0, 0)}


# =============================================================================
# Background worker
# =============================================================================
class Worker(QtCore.QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @QtCore.Slot()
    def run(self):
        try:
            res = self.fn(*self.args, progress=lambda i, n, m: self.progress.emit(i, n, m),
                          cancel=lambda: self._cancel, **self.kwargs)
            self.finished.emit(res)
        except wc.Cancelled:
            self.failed.emit("Cancelled.")
        except Exception:
            self.failed.emit(traceback.format_exc())


# =============================================================================
# Widgets
# =============================================================================
class FrameReader:
    """Keeps videos open and reads forward when possible (fast playback); seeks otherwise.
    Skipped frames are grabbed without decoding."""

    def __init__(self):
        self.caps = {}          # path -> [cap, index of the next frame]

    def get(self, path, i):
        import cv2
        i = int(i)
        ent = self.caps.get(path)
        if ent is None:
            if len(self.caps) > 3:
                for c, _ in self.caps.values():
                    c.release()
                self.caps.clear()
            ent = self.caps[path] = [cv2.VideoCapture(path), 0]
        cap, nxt = ent
        gap = i - nxt
        if not (0 <= gap <= 60):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        else:
            for _ in range(gap):
                cap.grab()
        ok, fr = cap.read()
        ent[1] = i + 1 if ok else -10**9
        return fr if ok else None


class FrameView(QtWidgets.QWidget):
    """Video frame viewer with time slider and click reporting (pixel coordinates)."""
    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#202020")
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        self.markers = pg.ScatterPlotItem(size=14, pen=pg.mkPen("w", width=1.5))
        self.vb.addItem(self.markers)
        self.labels = []
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.time_lbl = QtWidgets.QLabel("–")
        self.time_spin = QtWidgets.QDoubleSpinBox()
        self.time_spin.setSuffix(" s")
        self.time_spin.setDecimals(2)
        self.time_spin.setKeyboardTracking(False)
        self.contrast = QtWidgets.QCheckBox("auto contrast")
        self.contrast.setChecked(True)
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_btn.setToolTip("Play / pause (space)")
        self.play_btn.setCheckable(True)
        self.play_btn.setFixedWidth(34)
        self.play_btn.toggled.connect(self._toggle_play)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "8×", "16×"])
        self.speed.setCurrentText("1×")
        self.speed.setToolTip("Playback speed relative to real time")
        self.speed.currentIndexChanged.connect(self._restart_clock)
        self._play_timer = QtCore.QTimer(self, interval=15)
        self._play_timer.timeout.connect(self._play_tick)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key_Space), self, activated=self.play_btn.toggle,
                        context=Qt.WidgetWithChildrenShortcut)
        self._playing = False
        self.stop_at = None             # frame where playback stops (interval play)
        self.fps_lbl = QtWidgets.QLabel("")
        self.fps_lbl.setStyleSheet("color: gray")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.play_btn)
        row.addWidget(self.speed)
        row.addWidget(self.slider, 1)
        row.addWidget(self.time_spin)
        row.addWidget(self.time_lbl)
        row.addWidget(self.contrast)
        row.addWidget(self.fps_lbl)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw, 1)
        lay.addLayout(row)
        self.provider = None
        self.fps = 1.0
        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.refresh)
        self.slider.valueChanged.connect(self._slider_moved)
        self.time_spin.valueChanged.connect(lambda v: self.slider.setValue(int(round(v * self.fps))))
        self.contrast.toggled.connect(self.refresh)
        self.current = None
        self._shape = None

    def set_source(self, provider, n_frames, fps, start_s=10.0):
        self.provider, self.fps = provider, max(fps, 1e-6)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(n_frames - 1, 0))
        self.slider.setValue(min(int(start_s * self.fps), max(n_frames - 1, 0)))
        self.slider.blockSignals(False)
        self.stop()
        self.time_spin.blockSignals(True)
        self.time_spin.setRange(0, n_frames / self.fps)
        self.time_spin.setValue(self.slider.value() / self.fps)
        self.time_spin.blockSignals(False)
        self.refresh(autorange=True)

    def frame_index(self):
        return self.slider.value()

    def _slider_moved(self, v):
        self.time_spin.blockSignals(True)
        self.time_spin.setValue(v / self.fps)
        self.time_spin.blockSignals(False)
        if self._playing:
            self.refresh()
        else:
            self._timer.start()

    # --- playback ---------------------------------------------------------
    def _rate(self):
        return float(self.speed.currentText().rstrip("×"))

    def _restart_clock(self, *_):
        self._t0 = time.perf_counter()
        self._f0 = self.slider.value()
        self._shown = 0

    def _toggle_play(self, on):
        if on and self.provider is None:
            self.play_btn.setChecked(False)
            return
        self._playing = on
        if not on:
            self.stop_at = None
        self.play_btn.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause if on else QtWidgets.QStyle.SP_MediaPlay))
        if on:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self._restart_clock()
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self.fps_lbl.setText("")

    def _play_tick(self):
        """Real-time clock: show the frame that should be on screen now (skipping if the
        decoder cannot keep up at high speeds)."""
        el = time.perf_counter() - self._t0
        target = self._f0 + int(el * self.fps * self._rate())
        end = self.slider.maximum() if self.stop_at is None else min(self.stop_at, self.slider.maximum())
        if target >= end:
            self.slider.setValue(end)
            self.play_btn.setChecked(False)
            self.stop_at = None
            return
        if target != self.slider.value():
            self.slider.setValue(target)
            self._shown += 1
            if el > 1:
                self.fps_lbl.setText(f"{self._shown / el:.0f} fps shown")

    def stop(self):
        try:
            self.play_btn.setChecked(False)
        except RuntimeError:          # widget already destroyed (app closing)
            pass

    def hideEvent(self, ev):
        self.stop()
        super().hideEvent(ev)

    def refresh(self, autorange=False):
        if self.provider is None:
            return
        i = self.slider.value()
        try:
            im = self.provider(i)
        except Exception as e:
            self.time_lbl.setText(f"error: {e}")
            return
        if im is None:
            return
        self.current = im
        if self.contrast.isChecked():
            lo, hi = np.percentile(im[::4, ::4], (0.5, 99.5))
            levels = (float(lo), float(max(hi, lo + 1)))
        else:
            levels = (0, 255)
        self.img.setImage(im, levels=levels, autoLevels=False)
        self.time_lbl.setText(f"#{i}")
        if autorange or self._shape != im.shape[:2]:
            self._shape = im.shape[:2]
            if self.isVisible() and self.glw.width() > 50:
                self.vb.autoRange(padding=0.02)
            else:
                self._needs_autorange = True

    def showEvent(self, ev):
        super().showEvent(ev)
        if getattr(self, "_needs_autorange", False):
            self._needs_autorange = False
            QtCore.QTimer.singleShot(0, lambda: self.vb.autoRange(padding=0.02))

    def set_markers(self, pts: dict):
        for t in self.labels:
            self.vb.removeItem(t)
        self.labels = []
        spots = []
        for name, (xy, color) in pts.items():
            if xy is None:
                continue
            spots.append(dict(pos=(xy[0], xy[1]), brush=pg.mkBrush(*color)))
            t = pg.TextItem(name, color=color, anchor=(0, 1))
            t.setPos(xy[0] + 6, xy[1] - 6)
            self.vb.addItem(t)
            self.labels.append(t)
        self.markers.setData(spots)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton or self.current is None:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        h, w = self.current.shape[:2]
        if 0 <= p.x() < w and 0 <= p.y() < h:
            self.clicked.emit(float(p.x()), float(p.y()))


class MplCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None, figsize=(6, 5)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tb = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(tb)
        lay.addWidget(self.canvas, 1)
        from mpl_nav import AxisNavigator, HINT
        self.nav = AxisNavigator(self.canvas, tb)       # drag / wheel on the axes
        self.canvas.setToolTip(HINT)

    def draw(self):
        self.canvas.draw_idle()


PARAM_META = {
    # EventParams
    "sigma": ("Smoothing σ (frames)", "gaussian_filter1d sigma applied before peak finding", 0.1, 100, 1, 2),
    "prominence": ("Peak prominence (grey)", "scipy find_peaks prominence on the smoothed trace", 0.01, 255, 0.1, 2),
    "distance": ("Min peak distance (frames)", "find_peaks distance", 1, 100000, 1, 0),
    "exclude_s": ("Ignore first (s)", "recording-onset artefact; also excluded from the rate", 0, 100000, 1, 2),
    "skip_dark": ("Ignore light-off frames (0/1)", "frames where the light is off (found automatically) are "
                  "bridged and no event is counted next to them; no effect on videos without dark frames",
                  0, 1, 1, 0),
    # WaveParams
    "t_start": ("Analyse from (s)", "", 0, 1e6, 1, 2),
    "t_end": ("Analyse till (s, -1 = end)", "", -1, 1e6, 1, 2),
    "min_pts_kmeans": ("K-means if ≥ n active ROIs", "fewer active ROIs -> one centroid", 1, 1000, 1, 0),
    "min_sep": ("Min centre separation (px)", "largest K whose centres are all further apart than this", 1, 1e5, 5, 1),
    "max_k": ("Max K", "", 1, 200, 1, 0),
    "n_init": ("K-means n_init", "10 = notebook; lower = faster", 1, 50, 1, 0),
    "link_dist": ("Link radius (px)", "centre-to-centre distance between consecutive frames", 1, 1e5, 5, 1),
    "merge_min_steps": ("Merge: min steps", "both tracks need more steps than this", 0, 1e5, 1, 0),
    "merge_gap_frames": ("Merge: max gap (frames)", "", 0, 1e5, 1, 0),
    "merge_dist": ("Merge: max distance (px)", "", 0, 1e5, 10, 1),
    "drop_min_steps": ("Drop: fewer steps than", "", 0, 1e5, 1, 0),
    "drop_min_disp": ("Drop: displacement < (px)", "", 0, 1e5, 1, 1),
    "beg_end_min_disp": ("Begin→end: displacement > (px)", "", 0, 1e5, 5, 1),
    "path_min_len": ("Smoothed paths: path > (px)", "", 0, 1e6, 10, 1),
    "smooth_window": ("Smoothing window (points)", "moving average, ends preserved", 1, 1000, 1, 0),
    "segment_len": ("Segment length (px)", "direction is averaged over this travelled distance", 0.1, 1e4, 1, 1),
    "long_track_steps": ("Long track: > steps", "used for durations / path lengths / grid map", 0, 1e5, 1, 0),
    "wave_isi_cutoff": ("New wave if start gap > (s)", "", 0, 1e4, 0.5, 2),
    "grid_binning": ("Grid map binning (ROIs)", "", 1, 100, 1, 0),
    "workers": ("Worker processes", "parallel K-means", 1, 128, 1, 0),
}


class ParamForm(QtWidgets.QWidget):
    """Form for a params dataclass. Distance fields (wc.WAVE_LENGTH_PARAMS) are stored in px
    but shown in um once a scale is set (set_scale)."""

    def __init__(self, dc_instance, parent=None):
        super().__init__(parent)
        self.cls = type(dc_instance)
        self.widgets = {}
        self.labels = {}
        self.scale = None
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight)
        for f in fields(self.cls):
            label, tip, lo, hi, step, dec = PARAM_META.get(f.name, (f.name, "", -1e9, 1e9, 1, 3))
            v = getattr(dc_instance, f.name)
            if dec == 0:
                w = QtWidgets.QSpinBox()
                w.setRange(int(lo), int(hi))
            else:
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(dec if f.name not in wc.WAVE_LENGTH_PARAMS else 2)
            w.setSingleStep(step)
            w.setValue(v)
            w.setToolTip(tip)
            lab = QtWidgets.QLabel(label)
            form.addRow(lab, w)
            self.widgets[f.name] = w
            self.labels[f.name] = (lab, label)

    def _k(self, name):
        return self.scale if (self.scale and name in wc.WAVE_LENGTH_PARAMS) else 1.0

    def value(self):
        out = {}
        for k, w in self.widgets.items():
            out[k] = int(w.value()) if isinstance(w, QtWidgets.QSpinBox) else float(w.value()) / self._k(k)
        return self.cls(**out)

    def set_value(self, d: dict):
        """d in px (canonical)."""
        for k, v in d.items():
            if k in self.widgets:
                self.widgets[k].setValue(v * self._k(k))

    def set_scale(self, um_per_px):
        px = asdict(self.value())
        self.scale = um_per_px if um_per_px and um_per_px > 0 else None
        for k, (lab, text) in self.labels.items():
            if k in wc.WAVE_LENGTH_PARAMS:
                lab.setText(text.replace("(px)", "(µm)") if self.scale else text)
        self.set_value({k: v for k, v in px.items() if k in wc.WAVE_LENGTH_PARAMS})


# ---------------------------------------------------------------------------
# Microscope scale: objective x turret -> video height in um (per-computer settings)
# ---------------------------------------------------------------------------
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".wavegui_settings.json")
DEFAULT_SETTINGS = {"objectives": {"4x": 0.0, "10x": 0.0, "40x": 0.0},
                    "turrets": [0.5, 1.0, 1.5, 2.0], "turret_factors": {}, "last_scale": None}
CUSTOM = "Custom height"


LAB_DEFAULTS_PATH = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
                                 "defaults", "surf_defaults.json")


def lab_defaults():
    """Presets + objective calibration shipped in defaults/surf_defaults.json ({} if absent)."""
    import json
    try:
        with open(LAB_DEFAULTS_PATH, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("_comment", None)
        return d
    except Exception:
        return {}


def merge_lab_defaults(d, overwrite=False):
    """Add the shipped presets / objectives / turret factors to settings `d`.
    Existing entries are kept unless overwrite=True. Returns the number of items added."""
    lab = lab_defaults()
    n = 0
    for key in ("objectives", "turret_factors", "objective_defaults", "presets"):
        src = lab.get(key) or {}
        dst = d.setdefault(key, {})
        for k, v in src.items():
            if overwrite or k not in dst or (key == "objectives" and not dst.get(k)):
                if dst.get(k) != v:
                    dst[k] = v
                    n += 1
    if lab.get("turrets"):
        t = sorted(set(map(float, d.get("turrets") or [])) | set(map(float, lab["turrets"])))
        d["turrets"] = t
    return n


def load_settings():
    import json
    d = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            d.update(json.load(f))
    except FileNotFoundError:
        merge_lab_defaults(d)            # first start: presets + calibration from defaults/
    except Exception:
        pass
    return d


def save_settings(d):
    import json
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


def turret_factor(label, settings: dict) -> float:
    """Real magnification step behind a turret label (measured factor if known, else the label)."""
    t = float(label or 1.0)
    return float((settings.get("turret_factors") or {}).get(f"{t:g}", t))


def parse_turrets(text):
    """'0.5, 1, 1.25=1.301, 1.5=1.593' -> ([labels], {label: factor})."""
    labels, facs = [], {}
    for part in text.replace(";", ",").split(","):
        if not part.strip():
            continue
        lab, _, fac = part.partition("=")
        try:
            t = float(lab)
            if t <= 0:
                continue
            labels.append(t)
            if fac.strip():
                f = float(fac)
                if f > 0 and abs(f - t) > 1e-9:
                    facs[f"{t:g}"] = f
        except ValueError:
            pass
    return sorted(set(labels)), facs


def format_turrets(settings):
    tf = settings.get("turret_factors") or {}
    return ", ".join(f"{t:g}={tf[f'{t:g}']:g}" if f"{t:g}" in tf else f"{t:g}" for t in settings["turrets"])


def scale_height_um(scale: dict, settings: dict):
    """Video height in um for {'objective','turret','custom_height_um'}; None if unknown."""
    if not scale:
        return None
    if scale.get("objective") == CUSTOM:
        h = float(scale.get("custom_height_um") or 0)
        return h if h > 0 else None
    base = float(settings["objectives"].get(scale.get("objective"), 0) or 0)
    t = turret_factor(scale.get("turret") or 1.0, settings)
    return base / t if base > 0 and t > 0 else None


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Microscope calibration")
        self.resize(460, 420)
        lay = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel(
            "For each objective, enter the <b>full video height in µm at turret 1×</b> "
            "(e.g. film a stage micrometer, or use Tools → Calibration tool). Other turret "
            "settings are derived as height ÷ turret factor.")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Objective", "Video height at 1× (µm)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        for name, h in settings["objectives"].items():
            self._add_row(name, h)
        lay.addWidget(self.table, 1)
        b_add = QtWidgets.QPushButton("Add objective")
        b_add.clicked.connect(lambda: self._add_row("", 0.0))
        b_del = QtWidgets.QPushButton("Remove selected")
        b_del.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        lay.addLayout(hrow(b_add, b_del, "stretch"))
        self.turrets = QtWidgets.QLineEdit(format_turrets(settings))
        lay.addWidget(QtWidgets.QLabel(
            "Turret settings, comma separated. Write <b>label=measured factor</b> when the real step "
            "differs from the label, e.g. <tt>0.5, 1, 1.25=1.30, 1.5=1.59, 2=2.14</tt>"))
        lay.addWidget(self.turrets)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _add_row(self, name, h):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(0, 1e6)
        sp.setDecimals(2)
        sp.setSuffix(" µm")
        sp.setSpecialValueText("not calibrated")
        sp.setValue(float(h or 0))
        self.table.setCellWidget(r, 1, sp)

    def result_settings(self, settings):
        objs = {}
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            name = it.text().strip() if it else ""
            if name and name != CUSTOM:
                objs[name] = self.table.cellWidget(r, 1).value()
        tur, facs = parse_turrets(self.turrets.text())
        d = dict(settings)
        d["objectives"] = objs or settings["objectives"]
        if tur:
            d["turrets"], d["turret_factors"] = tur, facs
        return d


def group(title, *widgets_or_layouts):
    g = QtWidgets.QGroupBox(title)
    lay = QtWidgets.QVBoxLayout(g)
    for w in widgets_or_layouts:
        if isinstance(w, QtWidgets.QLayout):
            lay.addLayout(w)
        else:
            lay.addWidget(w)
    return g


def hrow(*ws):
    l = QtWidgets.QHBoxLayout()
    for w in ws:
        if w == "stretch":
            l.addStretch(1)
        else:
            l.addWidget(w)
    return l


def scroll(widget):
    s = QtWidgets.QScrollArea()
    s.setWidgetResizable(True)
    s.setWidget(widget)
    s.setFrameShape(QtWidgets.QFrame.NoFrame)
    return s


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, folder=None):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - {APP_LONG}")
        self.resize(1500, 920)
        # ---- state ----
        self.folder = None
        self.video = None
        self.info = None
        self.session = {}
        self.settings = load_settings()
        self.reader = FrameReader()
        self.calib = None                 # RotationCalib being edited
        self.click_target = "temporal"
        self.rois = None                  # RoiSet for measurement
        self.roi_zip = None
        self.meas = None
        self.peaks = self.rates = self.codes = self.active = None
        self.results = None
        self.orientation = wc.Orientation()
        self.ori_target = "temporal"
        self._thread = None
        self._worker = None
        self._bg_cache = None
        # ---- UI ----
        self._build_sidebar()
        self.tabs = QtWidgets.QTabWidget()
        from info_tab import InfoTab
        self.info_tab = InfoTab(self)
        self.tabs.addTab(self.info_tab, "0  Info")
        self.tabs.addTab(self._build_rotate_tab(), "1  Rotate")
        self.tabs.addTab(self._build_measure_tab(), "2  Measure")
        self.tabs.addTab(self._build_events_tab(), "3  Events")
        self.tabs.addTab(self._build_waves_tab(), "4  Waves")
        from compare_tab import CompareTab
        self.compare_tab = CompareTab(self)
        from cells_tab import CellsTab
        self.cells_tab = CellsTab(self)
        self.tabs.addTab(self.cells_tab, "5  Cells")
        self.tabs.addTab(self.compare_tab, "6  Compare")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)
        self._build_statusbar()
        tm = self.menuBar().addMenu("Tools")
        tm.addAction("Calibration tool (measure µm/px)…", self._open_calib_tool)
        tm.addAction("Load lab defaults (presets + objective calibration)…", self._load_lab_defaults)
        hm = self.menuBar().addMenu("Help")
        hm.addAction("README / manual", self._open_readme)
        hm.addAction(f"About {APP_NAME}", self._about)
        if folder:
            self.open_folder(folder)

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        b = QtWidgets.QPushButton("Open folder…")
        b.clicked.connect(self._choose_folder)
        self.folder_lbl = QtWidgets.QLabel("no folder")
        self.folder_lbl.setWordWrap(True)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setIconSize(QtCore.QSize(80, 60))
        self.file_list.currentItemChanged.connect(self._file_selected)
        self.info_lbl = QtWidgets.QLabel("")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        b_show = QtWidgets.QPushButton("Show video in folder")
        b_show.setToolTip("Open Explorer with the selected video highlighted (right-click a video for more)")
        b_show.clicked.connect(lambda: self.video and self._reveal(self.video))
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self._file_menu)
        lay.addWidget(b)
        lay.addWidget(self.folder_lbl)
        lay.addWidget(self.file_list, 1)
        lay.addWidget(b_show)
        lay.addWidget(self.info_lbl)
        dock = QtWidgets.QDockWidget("Videos", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        title = QtWidgets.QLabel("  Videos")
        title.setStyleSheet("font-weight: bold; padding: 4px 0")
        dock.setTitleBarWidget(title)
        dock.setWidget(w)
        dock.setMinimumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.prog = QtWidgets.QProgressBar()
        self.prog.setMaximumWidth(320)
        self.prog.setVisible(False)
        self.prog_lbl = QtWidgets.QLabel("")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_task)
        sb.addPermanentWidget(self.prog_lbl)
        sb.addPermanentWidget(self.prog)
        sb.addPermanentWidget(self.cancel_btn)

    def _reveal(self, path):
        if not reveal_in_folder(path):
            self.error(f"Could not open the folder for\n{path}")

    def _file_menu(self, pos):
        it = self.file_list.itemAt(pos)
        if it is None or not it.data(Qt.UserRole):
            return
        path = it.data(Qt.UserRole)
        stem = os.path.splitext(path)[0]
        menu = QtWidgets.QMenu(self)
        menu.addAction("Show video in folder", lambda: self._reveal(path))
        for res in (stem + "_rotated_grid", stem + "_grid"):
            if os.path.isdir(res):
                menu.addAction(f"Open results folder ({os.path.basename(res)})", lambda r=res: self._reveal(r))
        menu.exec(self.file_list.mapToGlobal(pos))

    def _choose_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording folder", self.folder or "")
        if d:
            self.open_folder(d)

    def open_folder(self, d):
        self.folder = d
        self.folder_lbl.setText(d)
        self.file_list.clear()
        found = find_videos(d)
        if not found:
            try:
                entries = os.listdir(d)
            except OSError as e:
                entries = []
                self.error(f"Cannot read folder:\n{e}")
            exts = sorted({os.path.splitext(e)[1] for e in entries if os.path.splitext(e)[1]})
            msg = ("No videos found here or in subfolders\n(looked for "
                   + ", ".join(VIDEO_EXT) + ").")
            if exts:
                msg += "\nFile types in this folder: " + ", ".join(exts[:12])
            it = QtWidgets.QListWidgetItem(msg)
            it.setFlags(Qt.NoItemFlags)
            self.file_list.addItem(it)
            return
        for rel in found:
            full = os.path.join(d, rel)
            vd, f = os.path.split(full)
            stem = os.path.splitext(f)[0]
            it = QtWidgets.QListWidgetItem(rel.replace(os.sep, "/"))
            it.setData(Qt.UserRole, full)
            d_ = d
            d = vd
            thm = None
            for ext in (".THM", ".thm"):
                p = os.path.join(d, stem + ext)
                if os.path.exists(p):
                    thm = wc.read_thumbnail(p)
                    break
            if thm is not None:
                rgb = cv2_to_qimage(thm)
                it.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(rgb)))
            it.setData(Qt.UserRole + 1, rel.replace(os.sep, "/"))
            it.setText(self._list_label(full, rel.replace(os.sep, "/")))
            self.file_list.addItem(it)
            d = d_
        self.file_list.setCurrentRow(0)

    def _list_label(self, full, rel):
        d, f = os.path.split(full)
        stem = os.path.splitext(f)[0]
        lines = [rel]
        ses = {}
        sp = wc.session_path(full)
        if os.path.exists(sp):
            try:
                ses = wc.load_session(full)
            except Exception:
                ses = {}
        inf = ses.get("info") or {}
        obj = (ses.get("scale") or {}).get("objective")
        desc = [str(inf[k]) for k in ("animal", "age", "condition") if inf.get(k)] + ([obj] if obj else [])
        if desc:
            lines.append("  " + " · ".join(desc))
        tags = []
        if any(os.path.exists(os.path.join(d, n)) for n in
               (stem + "_grid.npz", stem + "_rotated_grid.npz", stem + "_grid.csv")):
            tags.append("measured")
        if ses.get("rotation") or os.path.exists(os.path.join(d, stem + "rotation_calibPoints.pkl")):
            tags.append("calib")
        if any(os.path.exists(os.path.join(d, stem + suf, "cells_rois.npz")) for suf in ("_grid", "_rotated_grid")):
            tags.append("cells")
        if tags:
            lines.append(f"  [{', '.join(tags)}]")
        return "\n".join(lines)

    def _listed_videos(self):
        out = []
        for i in range(self.file_list.count()):
            v = self.file_list.item(i).data(Qt.UserRole)
            if v:
                out.append(v)
        return out

    def _refresh_list_label(self, full):
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if it.data(Qt.UserRole) and os.path.abspath(it.data(Qt.UserRole)) == os.path.abspath(full):
                it.setText(self._list_label(full, it.data(Qt.UserRole + 1) or os.path.basename(full)))

    def _recording_info(self):
        """Info written into the result workbook."""
        d = {k: v for k, v in (self.session.get("info") or {}).items()}
        sc = self._scale_dict()
        st = self.preset_state()
        d.update(objective=sc.get("objective"), turret=sc.get("turret"), eye=self._eye(),
                 video=os.path.basename(self.video or ""),
                 preset=(st["name"] or "") + (" (modified)" if st["modified"] else ""))
        return d

    # --------------------------------------------------------------- selection
    def _file_selected(self, cur, _prev=None):
        if cur is None:
            return
        mode = getattr(self, "_pick", None)
        if mode and _prev is not None:
            # picking a source video: copy from it, stay on the current video
            src = cur.data(Qt.UserRole)
            self.file_list.blockSignals(True)
            self.file_list.setCurrentItem(_prev)
            self.file_list.blockSignals(False)
            if src:
                {"grid": self._copy_grid_from, "calib": self._copy_calib_from,
                 "info": self.info_tab.copy_from, "cells": self.cells_tab.copy_from}[mode](src)
            self._pick_buttons()[mode].setChecked(False)
            return
        path = cur.data(Qt.UserRole)
        try:
            self.info = wc.video_info(path)
        except Exception as e:
            self.error(str(e))
            return
        self.video = path
        self.session = wc.load_session(path)
        self.meas = None
        self.peaks = self.rates = self.codes = self.active = None
        self.results = None
        self._bg_cache = None
        i = self.info
        self.info_lbl.setText(f"<b>{os.path.basename(path)}</b><br>{i['width']}×{i['height']} px, "
                              f"{i['fps']:.2f} fps<br>{i['n_frames']} frames "
                              f"({i['n_frames'] / max(i['fps'], 1e-9) / 60:.1f} min)")
        # rotation calibration: session -> notebook pkl -> none
        self.calib = None
        eye = self.session.get("eye", "right")
        if self.session.get("rotation"):
            self.calib = wc.RotationCalib.from_dict(self.session["rotation"])
            if self.calib.version < 3:        # older convention (mirror / flip)
                self.calib = self.calib.upgraded(eye)
                self.log("Older rotation calibration: same points, now rotated dorsal-up without "
                         "mirroring. Check the eye and press Save calibration.")
        else:
            pkl = os.path.splitext(path)[0] + "rotation_calibPoints.pkl"
            if os.path.exists(pkl):
                try:
                    self.calib = wc.RotationCalib.load_pkl(pkl, eye)
                except Exception:
                    pass
        self._sync_rotate_widgets()
        self.rot_view.set_source(self._raw_or_rotated_frame, i["n_frames"], i["fps"])
        self._update_rot_markers()
        # measure tab
        self.apply_rot_chk.setChecked(bool(self.session.get("apply_rotation", self.calib is not None
                                                            and "_rotated" not in path)))
        g = self.session.get("grid")
        if g:
            self._set_grid_spins(wc.GridSpec(**g))
        self.gray_combo.setCurrentIndex(list(wc.GRAY_MODES).index(self.session.get("gray_mode", "mean")))
        self.roi_zip = self.session.get("roi_zip")
        if self.roi_zip and not os.path.exists(self.roi_zip):
            self.roi_zip = None
        self._refresh_measure_view(autorange=True)
        # scale (per video, else the last one used on this computer)
        sc = self.session.get("scale") or self.settings.get("last_scale")
        self._set_scale_widgets(sc, from_last=not self.session.get("scale"))
        # params
        if self.session.get("event_params"):
            self.event_form.set_value(self.session["event_params"])
        if self.session.get("wave_params"):
            d = dict(self.session["wave_params"])
            d.pop("workers", None)
            self.wave_form.set_value(d)
        self._last_obj = self.obj_combo.currentText()
        self._presets_changed()
        new_video = not (self.session.get("event_params") or self.session.get("wave_params"))
        self.orientation = wc.Orientation.from_dict(self.session["orientation"]) \
            if self.session.get("orientation") else wc.Orientation()
        # existing measurement?
        for cand in self._meas_candidates():
            if os.path.exists(cand):
                try:
                    self._set_measurement(wc.Measurement.load_npz(cand), cand)
                    break
                except Exception as e:
                    self.log(f"could not load {cand}: {e}")
        else:
            self.meas_info.setText("No measurement yet for this video.")
            self._clear_downstream()
            if hasattr(self, "cells_tab"):
                self.cells_tab.reset_for_video()
        if new_video:
            self._apply_objective_default()
        for b in getattr(self, "preset_bars", []):
            b.sync_video()
        self._update_preset_state()
        self.info_tab.load_video()
        self.ori_view.set_source(self._meas_space_frame, i["n_frames"], i["fps"])
        self._update_ori_markers()

    def _meas_candidates(self):
        stem = os.path.splitext(self.video)[0]
        return [stem + "_rotated_grid.npz", stem + "_grid.npz"]

    def _open_readme(self):
        here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(here, "README.md")
        if os.path.exists(p):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p))
        else:
            self.error("README.md not found next to the program.")

    def _open_calib_tool(self):
        from surf_calibrate import CalibWindow
        path = getattr(self, "video", None)
        self._calib_win = CalibWindow(path if path and os.path.exists(str(path)) else None, parent=self)
        self._calib_win.setWindowFlag(QtCore.Qt.Window, True)
        self._calib_win.show()

    def _about(self):
        here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        box = QtWidgets.QMessageBox(self)
        logo = os.path.join(here, "surf.png")
        if os.path.exists(logo):
            box.setIconPixmap(QtGui.QPixmap(logo).scaled(112, 112, Qt.KeepAspectRatio,
                                                         Qt.SmoothTransformation))
        box.setWindowTitle(f"About {APP_NAME}")
        box.setText(f"<h2>{APP_NAME} {APP_VERSION}</h2><p><i>{APP_LONG}</i></p>"
                    "<p>Calcium-imaging analysis of developmental retinal waves: rotation to a "
                    "dorsal-up, unmirrored view, fast grid measurement (replaces FIJI Multi Measure), "
                    "event detection, wave tracking with anatomical directions, and group statistics.</p>"
                    "<p>Based on the Yonehara-lab wave notebooks.</p>")
        box.exec()

    def _load_lab_defaults(self):
        lab = lab_defaults()
        if not lab:
            self.error(f"No lab defaults found ({LAB_DEFAULTS_PATH}).")
            return
        names = ", ".join(lab.get("presets", {})) or "none"
        objs = ", ".join(f"{k} = {v:g} µm" for k, v in (lab.get("objectives") or {}).items())
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Load lab defaults")
        box.setText(f"<b>Presets:</b> {names}<br><b>Objectives (frame height):</b> {objs}<br><br>"
                    "These are calibrated for the Völgyi-lab microscope and camera. On another setup, "
                    "calibrate your objectives with Tools → Calibration tool.")
        add = box.addButton("Add missing only", QtWidgets.QMessageBox.AcceptRole)
        rep = box.addButton("Replace mine with these", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() not in (add, rep):
            return
        n = merge_lab_defaults(self.settings, overwrite=box.clickedButton() is rep)
        save_settings(self.settings)
        try:
            self._fill_scale_combos()
        except Exception:
            pass
        self._presets_changed()
        self.log(f"Lab defaults loaded ({n} item(s) added or updated).")

    def save_app_settings(self):
        save_settings(self.settings)

    # ---------------------------------------------------------------- presets
    def _presets_changed(self):
        for b in getattr(self, "preset_bars", []):
            b.refresh()
        self._update_preset_state()

    @staticmethod
    def _param_snapshot(ep, wp):
        from presets import SKIP_WAVE
        d = {f"event.{k}": float(v) for k, v in asdict(ep).items()}
        d.update({f"wave.{k}": float(v) for k, v in asdict(wp).items() if k not in SKIP_WAVE})
        return d

    def preset_state(self):
        ses = self.session or {}
        name = ses.get("preset")
        modified = False
        snap = ses.get("preset_params")
        if name and snap and hasattr(self, "wave_form"):
            cur = self._param_snapshot(self.event_form.value(), self.wave_form.value())
            modified = any(not np.isclose(cur.get(k, np.nan), v, rtol=1e-3, atol=1e-6)
                           for k, v in snap.items())
        legacy = not name and bool(ses.get("wave_params") or ses.get("event_params"))
        return dict(name=name, modified=modified, legacy=legacy, ok=bool(name) or legacy)

    def _watch_form(self, form):
        """Refresh the 'modified' note when a setting is edited."""
        if not hasattr(self, "_preset_timer"):
            self._preset_timer = QtCore.QTimer(self, singleShot=True, interval=250)
            self._preset_timer.timeout.connect(self._update_preset_state)
        for w in form.widgets.values():
            w.valueChanged.connect(lambda *_: self._preset_timer.start())

    def _update_preset_state(self, *_):
        st = self.preset_state()
        for b in getattr(self, "preset_bars", []):
            b.update_status()
        tip = "" if st["ok"] else "Apply an analysis preset first (top of this tab)"
        for b in (getattr(self, "run_ev_btn", None), getattr(self, "run_wave_btn", None)):
            if b is not None:
                b.setEnabled(st["ok"] or not self.video)
                b.setToolTip(tip)

    def _require_preset(self):
        if self.preset_state()["ok"]:
            return True
        QtWidgets.QMessageBox.information(
            self, "Analysis preset",
            "Choose an analysis preset for this video and press Apply (top of the Events or Waves "
            "tab). The default for this objective is pre-selected; 'Notebook defaults' gives the "
            "original notebook values.")
        return False

    def apply_preset(self, p, quiet=False):
        from presets import apply_preset
        notes = apply_preset(self, p, allow_grid=not quiet or self.meas is None)
        self.session["event_params"] = asdict(self.event_form.value())
        wp = asdict(self.wave_form.value())
        wp.pop("workers", None)
        self.session["wave_params"] = wp
        self.session["preset"] = p.get("name")
        self.session["preset_params"] = self._param_snapshot(self.event_form.value(), self.wave_form.value())
        self.save_session()
        self._update_preset_state()
        if self.live_chk.isChecked():
            self._draw_trace()
        if quiet:
            extra = ""
            if any(n.startswith("Grid size not changed") for n in notes):
                extra = " - grid kept, this video is already measured"
            elif any(n.startswith("Grid square size set") for n in notes):
                extra = f" - grid squares now {self._grid_spec().size} px"
            self.log(f"Preset '{p.get('name')}' applied (default for {self.obj_combo.currentText()}){extra}")
            for b in getattr(self, "preset_bars", []):
                i = b.combo.findData(p.get("name"))
                if i >= 0:
                    b.combo.setCurrentIndex(i)
        else:
            QtWidgets.QMessageBox.information(self, "Preset", f"Applied '{p.get('name')}'.\n\n" + "\n".join(notes))

    def _apply_objective_default(self):
        from presets import default_for
        p = default_for(self.settings, self.obj_combo.currentText())
        if p:
            self.apply_preset(p, quiet=True)

    def save_session(self):
        if self.video:
            self.session["scale"] = self._scale_dict()
            try:
                wc.save_session(self.video, self.session)
            except Exception as e:
                self.log(f"session not saved: {e}")

    # ------------------------------------------------------------------ tasks
    def run_task(self, fn, *args, on_done=None, label="Working", **kwargs):
        if self._thread is not None:
            self.error("Another task is still running.")
            return
        self._thread = QtCore.QThread()
        self._worker = Worker(fn, *args, **kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._on_done = on_done
        self._worker.finished.connect(self._task_done)
        self._worker.failed.connect(self._task_failed)
        self.prog.setVisible(True)
        self.prog.setValue(0)
        self.cancel_btn.setVisible(True)
        self.prog_lbl.setText(label + "…")
        self.tabs.setEnabled(False)
        self._thread.start()

    @QtCore.Slot(int, int, str)
    def _on_progress(self, i, n, msg):
        self.prog.setMaximum(max(n, 1))
        self.prog.setValue(min(i, max(n, 1)))
        self.prog_lbl.setText(msg)

    def _finish_thread(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        self.prog.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.tabs.setEnabled(True)

    @QtCore.Slot(object)
    def _task_done(self, res):
        on_done = self._on_done
        self._finish_thread()
        self.prog_lbl.setText("Done.")
        if on_done:
            try:
                on_done(res)
            except Exception:
                self.error(traceback.format_exc())

    @QtCore.Slot(str)
    def _task_failed(self, msg):
        self._finish_thread()
        self.prog_lbl.setText("")
        if msg.strip() == "Cancelled.":
            self.log("Cancelled.")
        else:
            self.error(msg)

    def _cancel_task(self):
        if self._worker:
            self._worker.cancel()

    def error(self, msg):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle(f"{APP_NAME} - problem")
        box.setText(msg.strip().splitlines()[-1] if msg.strip() else "Error")
        box.setDetailedText(msg)
        box.exec()

    def log(self, msg):
        self.statusBar().showMessage(msg, 8000)

    # =====================================================================
    # Tab 1 - Rotate
    # =====================================================================
    # ---------------------------------------------------------------- scale
    def _build_scale_group(self):
        self.obj_combo = QtWidgets.QComboBox()
        self.tur_combo = QtWidgets.QComboBox()
        self.custom_h = QtWidgets.QDoubleSpinBox()
        self.custom_h.setRange(0, 1e6)
        self.custom_h.setDecimals(1)
        self.custom_h.setSuffix(" µm")
        self.custom_h.setSpecialValueText("enter video height")
        self.custom_h.setKeyboardTracking(False)
        b_cal = QtWidgets.QPushButton("Edit calibration…")
        b_cal.clicked.connect(self._edit_calibration)
        self.scale_lbl = QtWidgets.QLabel("")
        self.scale_lbl.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        form.addRow("Objective", self.obj_combo)
        form.addRow("Turret", self.tur_combo)
        form.addRow("Height", self.custom_h)
        self._custom_row_label = form.labelForField(self.custom_h)
        self._fill_scale_combos()
        for w in (self.obj_combo, self.tur_combo):
            w.currentIndexChanged.connect(self._scale_changed)
        self.custom_h.valueChanged.connect(self._scale_changed)
        return group("Scale (µm)", form, self.scale_lbl, b_cal)

    def _fill_scale_combos(self):
        for w in (self.obj_combo, self.tur_combo):
            w.blockSignals(True)
        cur_o, cur_t = self.obj_combo.currentText(), self.tur_combo.currentText()
        self.obj_combo.clear()
        self.obj_combo.addItems(list(self.settings["objectives"]) + [CUSTOM])
        self.tur_combo.clear()
        self.tur_combo.addItems([f"{t:g}×" for t in self.settings["turrets"]])
        if self.obj_combo.findText(cur_o) >= 0:
            self.obj_combo.setCurrentText(cur_o)
        if self.tur_combo.findText(cur_t) >= 0:
            self.tur_combo.setCurrentText(cur_t)
        elif self.tur_combo.findText("1×") >= 0:
            self.tur_combo.setCurrentText("1×")
        for w in (self.obj_combo, self.tur_combo):
            w.blockSignals(False)

    def _scale_dict(self):
        t = self.tur_combo.currentText().rstrip("×")
        return {"objective": self.obj_combo.currentText(), "turret": float(t) if t else 1.0,
                "custom_height_um": self.custom_h.value()}

    def _set_scale_widgets(self, d, from_last=False):
        ws = (self.obj_combo, self.tur_combo, self.custom_h)
        for w in ws:
            w.blockSignals(True)
        if d:
            if self.obj_combo.findText(d.get("objective", "")) < 0 and d.get("objective") not in (None, CUSTOM):
                # objective no longer in the calibration table -> add it back as uncalibrated
                self.settings["objectives"][d["objective"]] = 0.0
                self._fill_scale_combos()
            self.obj_combo.setCurrentText(d.get("objective", self.obj_combo.currentText()))
            tt = f"{float(d.get('turret', 1.0)):g}×"
            if self.tur_combo.findText(tt) < 0:
                self.settings["turrets"].append(float(d.get("turret", 1.0)))
                self._fill_scale_combos()
            self.tur_combo.setCurrentText(tt)
            self.custom_h.setValue(float(d.get("custom_height_um") or 0))
        for w in ws:
            w.blockSignals(False)
        self._scale_from_last = from_last
        self._apply_scale()

    def _scale_changed(self, *_):
        self._presets_changed()
        self._scale_from_last = False
        obj_changed = self.obj_combo.currentText() != getattr(self, "_last_obj", None)
        self._last_obj = self.obj_combo.currentText()
        self.settings["last_scale"] = self._scale_dict()
        save_settings(self.settings)
        self.save_session()
        self._apply_scale()
        if obj_changed and self.sender() is self.obj_combo:
            self._apply_objective_default()
        if self.video:
            self._refresh_list_label(self.video)

    def _edit_calibration(self):
        dlg = CalibrationDialog(self.settings, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            keep = self._scale_dict()
            self.settings = dlg.result_settings(self.settings)
            save_settings(self.settings)
            self._fill_scale_combos()
            self._set_scale_widgets(keep)

    def um_per_px(self):
        if not self.info:
            return None
        h = scale_height_um(self._scale_dict(), self.settings)
        H = self.meas.frame_size[1] if (self.meas is not None and self.meas.frame_size[1]) else self.info["height"]
        return h / H if h and H else None

    def _apply_scale(self):
        custom = self.obj_combo.currentText() == CUSTOM
        self.custom_h.setVisible(custom)
        self._custom_row_label.setVisible(custom)
        self.tur_combo.setEnabled(not custom)
        s = self.um_per_px()
        sd = self._scale_dict()
        f = turret_factor(sd["turret"], self.settings)
        ftxt = (f" <span style='color:gray'>(turret {sd['turret']:g} = measured ×{f:.3f})</span>"
                if abs(f - sd["turret"]) > 1e-9 and not custom else "")
        h_cal = scale_height_um(sd, self.settings)
        if s and self.info:
            W, H = self.info["width"], self.info["height"]
            txt = (f"Video: {W * s:.0f} × {H * s:.0f} µm<br><b>{s:.4f} µm/px</b> "
                   f"({1 / s:.3f} px/µm){ftxt}")
            if getattr(self, "_scale_from_last", False):
                txt += "<br><span style='color:gray'>(same as the last video - change if needed)</span>"
        elif not custom and self.info:
            txt = ("<span style='color:#d33'>This objective is not calibrated yet - click "
                   "<b>Edit calibration…</b>. Until then lengths are in px.</span>")
        elif not self.info and h_cal:
            txt = f"Calibrated: video height {h_cal:.0f} µm at this setting{ftxt}."
        elif not self.info and not custom:
            txt = ("<span style='color:#d33'>This objective is not calibrated yet - click "
                   "<b>Edit calibration…</b>.</span>")
        else:
            txt = "<span style='color:#d33'>Enter the video height. Until then lengths are in px.</span>"
        self.scale_lbl.setText(txt)
        if hasattr(self, "info_tab"):
            self.info_tab.sync_from_main()
        if hasattr(self, "wave_form"):
            self.wave_form.set_scale(s)
        if hasattr(self, "g_spin"):
            self._set_grid_units(s)
        if hasattr(self, "grid_lbl"):
            self._update_roi_display()
        if getattr(self, "peaks", None) is not None and getattr(self, "meas", None) is not None:
            self._draw_events()
        if getattr(self, "results", None) is not None:
            self._draw_waves()

    def _build_rotate_tab(self):
        self.rot_view = FrameView()
        self.rot_view.clicked.connect(self._rot_clicked)
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        intro = QtWidgets.QLabel("Click the <b>temporal pole</b>, the <b>nasal pole</b> and the "
                                 "<b>centre</b> on the raw frame and choose the eye. The video is "
                                 "rotated (never mirrored) so the retina is <b>dorsal up</b> with the "
                                 "centre in the middle: a right retina nasal-right, a left retina "
                                 "nasal-left.")
        intro.setWordWrap(True)
        self.rot_radio = {}
        rb_lay = QtWidgets.QVBoxLayout()
        for k, txt in (("temporal", "1  Temporal pole"), ("nasal", "2  Nasal pole"), ("center", "3  Centre")):
            rb = QtWidgets.QRadioButton(txt)
            rb.toggled.connect(lambda on, k=k: on and setattr(self, "click_target", k))
            self.rot_radio[k] = rb
            rb_lay.addWidget(rb)
        self.rot_radio["temporal"].setChecked(True)
        self.rot_pts_lbl = QtWidgets.QLabel("")
        self.eye_right = QtWidgets.QRadioButton("Right retina")
        self.eye_left = QtWidgets.QRadioButton("Left retina")
        self.eye_right.setChecked(True)
        self.eye_right.toggled.connect(self._eye_changed)
        eye_help = QtWidgets.QLabel("Viewed from the ganglion-cell side with dorsal up: a <b>right</b> "
                                    "retina has nasal on the right (temporal left), a <b>left</b> "
                                    "retina nasal on the left (temporal right).")
        eye_help.setWordWrap(True)
        eye_help.setStyleSheet("color: gray")
        self.preview_chk = QtWidgets.QCheckBox("Show rotated preview")
        self.preview_chk.toggled.connect(lambda _: (self.rot_view.refresh(), self._update_rot_markers()))
        b_save = QtWidgets.QPushButton("Save calibration")
        b_save.setToolTip("Writes <video>rotation_calibPoints.pkl (notebook format) + session file")
        b_save.clicked.connect(self._save_calib)
        b_load = QtWidgets.QPushButton("Load calibration .pkl…")
        b_load.clicked.connect(self._load_calib)
        self.copy_calib_btn = QtWidgets.QPushButton("Copy calibration from another video…")
        self.copy_calib_btn.setCheckable(True)
        self.copy_calib_btn.setToolTip("Click this, then click another video in the list: its rotation "
                                       "points and the eye are copied to the current video.")
        self.copy_calib_btn.toggled.connect(lambda on: self._pick_toggled("calib", on))
        b_clear = QtWidgets.QPushButton("Clear points")
        b_clear.clicked.connect(self._clear_calib)
        b_export = QtWidgets.QPushButton("Export rotated video…")
        b_export.setToolTip("Optional: the Measure tab can apply the rotation on the fly, which is "
                            "faster and avoids re-compressing the video.")
        b_export.clicked.connect(self._export_rotated)
        pl.addWidget(intro)
        pl.addWidget(self._build_scale_group())
        pl.addWidget(group("Click target", rb_lay))
        pl.addWidget(self.rot_pts_lbl)
        pl.addWidget(group("Eye", hrow(self.eye_right, self.eye_left, "stretch"), eye_help))
        pl.addWidget(self.preview_chk)
        pl.addWidget(b_save)
        pl.addWidget(b_load)
        pl.addWidget(self.copy_calib_btn)
        pl.addWidget(b_clear)
        pl.addSpacing(12)
        pl.addWidget(b_export)
        note = QtWidgets.QLabel("Exporting is optional - measurement can rotate on the fly.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        pl.addWidget(note)
        pl.addStretch(1)
        return self._split(self.rot_view, panel)

    def _split(self, left, right, right_w=330):
        sp = QtWidgets.QSplitter()
        sp.addWidget(left)
        right = scroll(right) if not isinstance(right, QtWidgets.QScrollArea) else right
        right.setMinimumWidth(right_w)
        sp.addWidget(right)
        sp.setStretchFactor(0, 1)
        sp.setSizes([1100, right_w])
        return sp

    def _gray_mode(self):
        return list(wc.GRAY_MODES)[self.gray_combo.currentIndex()]

    def _raw_or_rotated_frame(self, i):
        fr = self.reader.get(self.video, i)
        if fr is None:
            return None
        g = wc.to_gray_f32(fr, self._gray_mode())
        if self.preview_chk.isChecked() and self._calib_complete():
            g = self.calib.apply(g)
        return g

    def _calib_complete(self):
        c = self.calib
        return c is not None and None not in (c.temporal, c.nasal, c.center)

    def _rot_clicked(self, x, y):
        if self.preview_chk.isChecked() and self._calib_complete():
            # the view shows the rotated frame: switch back to raw, then click again
            self.preview_chk.setChecked(False)
            self.log("Preview switched off - click the point again on the raw frame.")
            return
        pt = (round(x, 1), round(y, 1))
        c = self.calib or wc.RotationCalib(None, None, None, self._eye())
        if self.click_target == "temporal":
            c.temporal = pt
            nxt = "nasal"
        elif self.click_target == "nasal":
            c.nasal = pt
            nxt = "center"
        else:
            c.center = pt
            nxt = "center"
        c.eye, c.version = self._eye(), 3
        self.calib = c
        self.rot_radio[nxt].setChecked(True)
        self._rot_settings_changed()

    def _eye(self):
        return "left" if self.eye_left.isChecked() else "right"

    def _eye_changed(self, *_):
        if hasattr(self, "info_tab"):
            self.info_tab.sync_from_main()
        self.session["eye"] = self._eye()
        if self.video:
            self.save_session()
        self._rot_settings_changed()
        if hasattr(self, "ori_lbl"):
            self._update_ori_markers()

    def _rot_settings_changed(self, *_):
        if self.calib is not None:
            if self.calib.version < 3:
                self.calib = self.calib.upgraded(self._eye())
            self.calib.eye = self._eye()
        self._update_rot_markers()
        if self.preview_chk.isChecked():
            self.rot_view.refresh()

    def _sync_rotate_widgets(self):
        eye = self.session.get("eye") or (self.calib.eye if self.calib and self.calib.version >= 2 else "right")
        for w in (self.eye_right, self.eye_left):
            w.blockSignals(True)
        (self.eye_left if eye == "left" else self.eye_right).setChecked(True)
        for w in (self.eye_right, self.eye_left):
            w.blockSignals(False)

    def _update_rot_markers(self):
        c = self.calib
        if c is None:
            self.rot_view.set_markers({})
            self.rot_pts_lbl.setText("No points yet.")
            return
        pts = {"temporal": c.temporal, "nasal": c.nasal, "center": c.center}
        if self.preview_chk.isChecked() and self._calib_complete():
            tp = c.transform_points([c.temporal, c.nasal, c.center], self.info["width"], self.info["height"])
            pts = dict(zip(pts, map(tuple, tp)))
        self.rot_view.set_markers({k: (v, MARK_COLORS[k]) for k, v in pts.items()})
        txt = "<br>".join(f"{k}: {'–' if v is None else f'({v[0]:.0f}, {v[1]:.0f})'}"
                          for k, v in (("temporal", c.temporal), ("nasal", c.nasal), ("centre", c.center)))
        if c.temporal is not None and c.nasal is not None:
            txt += (f"<br><b>rotation angle: {c.angle:.2f}°</b>, {c.eye} retina → dorsal up, "
                    f"nasal {'right' if c.eye == 'right' else 'left'}")
        self.rot_pts_lbl.setText(txt)

    def _save_calib(self):
        if not self._calib_complete():
            self.error("Click all three points first.")
            return
        self._rot_settings_changed()
        self.calib.save_pkl(os.path.splitext(self.video)[0] + "rotation_calibPoints.pkl")
        self.session["rotation"] = self.calib.to_dict()
        self.save_session()
        self.apply_rot_chk.setChecked(True)
        self._refresh_measure_view(autorange=True)
        self.log("Calibration saved.")

    def _load_calib(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Calibration", self.folder or "", "pickle (*.pkl)")
        if p:
            try:
                self.calib = wc.RotationCalib.load_pkl(p, self._eye())
            except Exception as e:
                self.error(f"Not a rotation calibration file:\n{e}")
                return
            self._update_rot_markers()
            self.rot_view.refresh()

    def _clear_calib(self):
        self.calib = None
        self.session.pop("rotation", None)
        self.save_session()
        self.rot_radio["temporal"].setChecked(True)
        self._update_rot_markers()
        self.rot_view.refresh()

    def _export_rotated(self):
        if not self._calib_complete():
            self.error("Define the rotation first.")
            return
        self._rot_settings_changed()
        dst = os.path.splitext(self.video)[0] + "_rotated.MP4"
        dst, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Rotated video", dst, "MP4 (*.MP4 *.mp4)")
        if not dst:
            return
        calib = wc.RotationCalib.from_dict(self.calib.to_dict())
        src = self.video
        info = self.info

        def done(_):
            # orientation for the rotated copy: T/N transformed into its frame space
            tp = calib.transform_points([calib.temporal, calib.nasal], info["width"], info["height"])
            wc.save_session(dst, {"derived_from": src, "rotation_applied": calib.to_dict(),
                                  "orientation": wc.Orientation.from_rotation(
                                      calib, info["width"], info["height"]).to_dict(),
                                  "eye": wc.Orientation.from_rotation(calib, info["width"], info["height"]).eye,
                                  "scale": self._scale_dict()})
            self.log(f"Saved {dst}")
            self.open_folder(self.folder)
        self.run_task(wc.export_rotated_video, src, dst, calib, gray_mode=self._gray_mode(),
                      on_done=done, label="Exporting rotated video")

    # =====================================================================
    # Tab 2 - Measure
    # =====================================================================
    def _build_measure_tab(self):
        self.meas_view = FrameView()
        vb = self.meas_view.vb
        self.grid_lines = pg.PlotCurveItem(pen=pg.mkPen((255, 200, 0, 140), width=1), connect="pairs")
        vb.addItem(self.grid_lines)
        self.grid_roi = pg.RectROI([300, 66], [1250, 1000], pen=pg.mkPen((255, 140, 0), width=2),
                                   snapSize=1, translateSnap=True, scaleSnap=True)
        vb.addItem(self.grid_roi)
        self.grid_roi.sigRegionChanged.connect(self._grid_roi_moved)
        self.grid_roi.sigRegionChangeFinished.connect(self._grid_roi_finished)
        self.roi_dots = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(255, 200, 0, 180), pen=None)
        vb.addItem(self.roi_dots)

        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        self.apply_rot_chk = QtWidgets.QCheckBox("Apply rotation calibration on the fly")
        self.apply_rot_chk.setToolTip("Measure the original video in rotated coordinates - no need "
                                      "to export a rotated copy first.")
        self.apply_rot_chk.toggled.connect(lambda _: self._refresh_measure_view(autorange=True))
        self.gray_combo = QtWidgets.QComboBox()
        self.gray_combo.addItems(list(wc.GRAY_MODES.values()))
        self.gray_combo.currentIndexChanged.connect(lambda _: self.meas_view.refresh())
        # grid spins
        self.g_spin = {}
        self.g_lbl = {}
        self._gk = 1.0                 # display units per px (µm/px once calibrated)
        gform = QtWidgets.QFormLayout()
        for k, lab, lo, hi in (("x0", "Left x", -5000, 10000), ("y0", "Top y", -5000, 10000),
                               ("size", "Square size", 1, 2000), ("cols", "Columns", 1, 2000),
                               ("rows", "Rows", 1, 2000)):
            if k in ("x0", "y0", "size"):        # lengths: px or µm
                s = QtWidgets.QDoubleSpinBox()
                s.setDecimals(0)
                s.setRange(lo, hi)
                s.setSuffix(" px")
                s.setKeyboardTracking(False)
                s.editingFinished.connect(lambda k=k: self._snap_grid_field(k))
            else:
                s = QtWidgets.QSpinBox()
                s.setRange(lo, hi)
            s.valueChanged.connect(self._grid_spins_changed)
            lb = QtWidgets.QLabel(lab)
            gform.addRow(lb, s)
            self.g_spin[k] = s
            self.g_lbl[k] = lb
        self.grid_lbl = QtWidgets.QLabel("")
        b_default = QtWidgets.QPushButton("FIJI default grid (300, 66, 25 px, 50×40)")
        b_default.clicked.connect(lambda: self._set_grid_spins(wc.GridSpec(300, 66, 25, 40, 50)))
        b_fit = QtWidgets.QPushButton("Fill frame")
        b_fit.clicked.connect(self._grid_fill_frame)
        b_zip = QtWidgets.QPushButton("Use ImageJ ROI zip…")
        b_zip.setToolTip("e.g. the *_ROI.zip you used in FIJI. A regular grid of squares is "
                         "detected and measured exactly like the grid; other shapes are supported too.")
        b_zip.clicked.connect(self._choose_roi_zip)
        self.zip_lbl = QtWidgets.QLabel("")
        self.zip_lbl.setWordWrap(True)
        b_zip_clear = QtWidgets.QPushButton("Back to drawn grid")
        b_zip_clear.clicked.connect(self._clear_roi_zip)
        self.csv_chk = QtWidgets.QCheckBox("Also write FIJI-format CSV (for the old notebook)")
        self.csv_chk.setChecked(True)
        self.btn_measure = QtWidgets.QPushButton("▶  Measure all frames")
        self.btn_measure.setStyleSheet("font-weight: bold; padding: 6px")
        self.btn_measure.clicked.connect(self._run_measure)
        b_load = QtWidgets.QPushButton("Load existing measurement (.npz or FIJI .csv)…")
        b_load.clicked.connect(self._load_measurement)
        self.meas_info = QtWidgets.QLabel("")
        self.meas_info.setWordWrap(True)
        self.meas_info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        pl.addWidget(group("Frame space", self.apply_rot_chk, QtWidgets.QLabel("Grey value"), self.gray_combo))
        pl.addWidget(group("Grid (drag / resize the orange box)", gform, self.grid_lbl, b_default, b_fit))
        pl.addWidget(group("…or ROIs from FIJI", b_zip, self.zip_lbl, b_zip_clear))
        self.copy_grid_btn = QtWidgets.QPushButton("Copy grid from another video…")
        self.copy_grid_btn.setCheckable(True)
        self.copy_grid_btn.setToolTip("Click this, then click another video in the list on the left: "
                                      "its grid (or ROI zip) is copied to the current video.")
        self.copy_grid_btn.toggled.connect(lambda on: self._pick_toggled("grid", on))
        pl.addWidget(self.copy_grid_btn)
        pl.addWidget(self.csv_chk)
        pl.addWidget(self.btn_measure)
        pl.addWidget(b_load)
        pl.addWidget(self.meas_info)
        pl.addStretch(1)
        self._set_grid_spins(wc.GridSpec(300, 66, 25, 40, 50))
        return self._split(self.meas_view, panel, 360)

    def _meas_space_frame(self, i, use_meas=False):
        """Grey frame in the coordinate space the measurement uses."""
        video = self.video
        rot = self.calib if (self.apply_rot_chk.isChecked() and self._calib_complete()) else None
        mode = self._gray_mode()
        if use_meas and self.meas is not None:
            if self.meas.source and os.path.exists(self.meas.source) and \
                    self.meas.source.lower().endswith(VIDEO_EXT):
                video = self.meas.source
            rot = wc.RotationCalib.from_dict(self.meas.rotation) if self.meas.rotation else None
            mode = self.meas.gray_mode
        fr = self.reader.get(video, i)
        if fr is None:
            return None
        g = wc.to_gray_f32(fr, mode)
        return rot.apply(g) if rot is not None else g

    def _refresh_measure_view(self, autorange=False):
        if not self.video:
            return
        self.meas_view.set_source(self._meas_space_frame, self.info["n_frames"], self.info["fps"])
        self._update_roi_display()

    def _pick_buttons(self):
        return {"grid": self.copy_grid_btn, "calib": self.copy_calib_btn, "info": self.info_tab.copy_btn,
                "cells": self.cells_tab.copy_btn}

    def _pick_toggled(self, mode, on):
        texts = {"grid": "Copy grid from another video…", "calib": "Copy calibration from another video…",
                 "info": "Copy info from another video…", "cells": "Copy cells from another video…"}
        btns = self._pick_buttons()
        if on:
            for other, b_ in btns.items():
                if other != mode and b_.isChecked():
                    b_.setChecked(False)
            self._pick = mode
            btns[mode].setText("Now click the source video in the list ←  (click here to cancel)")
            btns[mode].setStyleSheet("background: #ffe08a; font-weight: bold")
            self.log("Click the video to copy from.")
        else:
            if getattr(self, "_pick", None) == mode:
                self._pick = None
            btns[mode].setText(texts[mode])
            btns[mode].setStyleSheet("")

    def _copy_calib_from(self, src):
        if os.path.abspath(src) == os.path.abspath(self.video or ""):
            self.log("That is the current video - pick another one.")
            return
        ses = wc.load_session(src)
        cal = None
        src_eye = ses.get("eye", "right")
        if ses.get("rotation"):
            cal = wc.RotationCalib.from_dict(ses["rotation"])
            if cal.version < 3:
                cal = cal.upgraded(src_eye)
        else:
            pkl = os.path.splitext(src)[0] + "rotation_calibPoints.pkl"
            if os.path.exists(pkl):
                try:
                    cal = wc.RotationCalib.load_pkl(pkl, src_eye)
                except Exception:
                    cal = None
        if cal is None:
            self.error(f"{os.path.basename(src)} has no rotation calibration yet.")
            return
        si = wc.video_info(src)
        if (si["width"], si["height"]) != (self.info["width"], self.info["height"]):
            self.error(f"{os.path.basename(src)} is {si['width']}×{si['height']} px but this video is "
                       f"{self.info['width']}×{self.info['height']} px - points may not fit.")
        self.calib = cal
        self.session["eye"] = cal.eye
        self._sync_rotate_widgets()
        self._update_rot_markers()
        self.rot_view.refresh()
        self._save_calib()
        self.log(f"Copied rotation calibration ({cal.angle:.1f}°, {cal.eye} retina) "
                 f"from {os.path.basename(src)} - check it with 'Show rotated preview'.")

    def _copy_grid_from(self, src):
        if os.path.abspath(src) == os.path.abspath(self.video or ""):
            self.log("That is the current video - pick another one.")
            return
        ses = wc.load_session(src)
        grid, zip_, rot_used = ses.get("grid"), ses.get("roi_zip"), ses.get("apply_rotation")
        if not grid and not zip_:          # no session: look at its measurement
            stem = os.path.splitext(src)[0]
            for cand in (stem + "_rotated_grid.npz", stem + "_grid.npz"):
                if os.path.exists(cand):
                    try:
                        m = wc.Measurement.load_npz(cand)
                        grid, rot_used = m.grid, bool(m.rotation)
                    except Exception:
                        pass
                    break
        if not grid and not zip_:
            self.error(f"{os.path.basename(src)} has no saved grid or measurement yet.")
            return
        if zip_ and os.path.exists(zip_):
            self.roi_zip = zip_
            what = f"ROI zip {os.path.basename(zip_)}"
        else:
            self.roi_zip = None
            self.zip_lbl.setText("")
            self._set_grid_spins(wc.GridSpec(**grid))
            g_ = wc.GridSpec(**grid)
            what = f"grid {g_.cols}×{g_.rows}, {g_.size} px at ({g_.x0}, {g_.y0})"
        self._update_roi_display()
        self.session.update(grid=asdict(self._grid_spec()), roi_zip=self.roi_zip)
        self.save_session()
        msg = f"Copied {what} from {os.path.basename(src)}."
        here_rot = self.apply_rot_chk.isChecked() and self._calib_complete()
        if bool(rot_used) != bool(here_rot):
            msg += (" Note: the source was measured " + ("with" if rot_used else "without") +
                    " rotation, this video " + ("with" if here_rot else "without") +
                    " - check that the grid still sits on the retina.")
            self.error(msg)
        self.log(msg)

    _GRID_LEN = ("x0", "y0", "size")

    def _gpx(self, k):
        """Grid field in px (lengths are shown in µm once the scale is known)."""
        v = self.g_spin[k].value()
        if k in self._GRID_LEN:
            return max(1, int(round(v / self._gk))) if k == "size" else int(round(v / self._gk))
        return int(v)

    def _gset(self, k, px):
        self.g_spin[k].setValue(px * self._gk if k in self._GRID_LEN else px)

    def _snap_grid_field(self, k):
        """After typing a µm value show the exact length of the whole number of px used."""
        px = self._gpx(k)
        self.g_spin[k].blockSignals(True)
        self._gset(k, px)
        self.g_spin[k].blockSignals(False)
        self.g_spin[k].setToolTip(f"{px} px")

    def _set_grid_units(self, um_per_px):
        """Switch the grid length fields between px and µm without changing the grid."""
        g = self._grid_spec()
        self._gk = um_per_px if um_per_px else 1.0
        unit = " µm" if um_per_px else " px"
        for k in self._GRID_LEN:
            sp = self.g_spin[k]
            sp.blockSignals(True)
            sp.setDecimals(2 if um_per_px else 0)
            lo, hi = (1, 2000) if k == "size" else (-5000, 10000)
            sp.setRange(lo * self._gk, hi * self._gk)
            sp.setSingleStep(self._gk)
            sp.setSuffix(unit)
            sp.blockSignals(False)
        self._set_grid_spins(g)

    def _grid_spec(self):
        return wc.GridSpec(self._gpx("x0"), self._gpx("y0"), self._gpx("size"),
                           self._gpx("rows"), self._gpx("cols"))

    def _set_grid_spins(self, g: wc.GridSpec):
        for k in self.g_spin:
            self.g_spin[k].blockSignals(True)
            self._gset(k, getattr(g, k))
            self.g_spin[k].setToolTip(f"{getattr(g, k)} px" if k in self._GRID_LEN else "")
            self.g_spin[k].blockSignals(False)
        self._grid_spins_changed()

    def _grid_spins_changed(self, *_):
        g = self._grid_spec()
        self.grid_roi.blockSignals(True)
        self.grid_roi.setPos([g.x0, g.y0], update=False)
        self.grid_roi.setSize([g.width, g.height])
        self.grid_roi.blockSignals(False)
        self._update_roi_display()

    def _grid_roi_moved(self):
        p, s = self.grid_roi.pos(), self.grid_roi.size()
        size = max(self._gpx("size"), 1)
        vals = dict(x0=int(round(p.x())), y0=int(round(p.y())),
                    cols=max(1, int(round(s.x() / size))), rows=max(1, int(round(s.y() / size))))
        for k, v in vals.items():
            self.g_spin[k].blockSignals(True)
            self._gset(k, v)
            self.g_spin[k].blockSignals(False)
        self._update_roi_display(lines_only=True)

    def _grid_roi_finished(self):
        self._grid_spins_changed()   # snap box to whole squares

    def _grid_fill_frame(self):
        if not self.info:
            return
        s = self._gpx("size")
        w, h = self.info["width"], self.info["height"]
        c, r = w // s, h // s
        self._set_grid_spins(wc.GridSpec((w - c * s) // 2, (h - r * s) // 2, s, r, c))

    def _update_roi_display(self, lines_only=False):
        if self.roi_zip:
            self.grid_roi.setVisible(False)
            self.grid_lines.setData([], [])
            rs = self._load_zip_rois()
            if rs is not None:
                if rs.kind == "grid":
                    self._draw_grid_lines(rs.grid)
                    self.roi_dots.setData([], [])
                else:
                    self.roi_dots.setData(rs.x, rs.y)
            return
        self.grid_roi.setVisible(True)
        self.roi_dots.setData([], [])
        g = self._grid_spec()
        self._draw_grid_lines(g)
        warn = ""
        if self.info:
            W, H = self.info["width"], self.info["height"]
            if g.x0 < 0 or g.y0 < 0 or g.x0 + g.width > W or g.y0 + g.height > H:
                warn = "  <span style='color:#d33'>(outside the frame - those pixels read 0)</span>"
        s = self.um_per_px() if hasattr(self, "obj_combo") else None
        um = f"<br>square {g.size * s:.1f} µm, grid {g.width * s:.0f}×{g.height * s:.0f} µm" if s else ""
        self.grid_lbl.setText(f"{g.rows * g.cols} ROIs, {g.width}×{g.height} px{um}{warn}")

    def _draw_grid_lines(self, g):
        xs, ys = [], []
        for c in range(g.cols + 1):
            x = g.x0 + c * g.size
            xs += [x, x]
            ys += [g.y0, g.y0 + g.height]
        for r in range(g.rows + 1):
            y = g.y0 + r * g.size
            xs += [g.x0, g.x0 + g.width]
            ys += [y, y]
        self.grid_lines.setData(np.array(xs, float), np.array(ys, float))

    def _load_zip_rois(self):
        if not self.roi_zip or not self.info:
            return None
        key = (self.roi_zip, self.info["width"], self.info["height"])
        if getattr(self, "_zip_cache", (None, None))[0] != key:
            try:
                rs = wc.RoiSet.from_imagej_zip(self.roi_zip, self.info["width"], self.info["height"])
            except ImportError:
                self.error("Reading ROI zips needs the 'roifile' package:\n  pip install roifile")
                self.roi_zip = None
                return None
            except Exception as e:
                self.error(f"Could not read ROI zip:\n{e}")
                self.roi_zip = None
                return None
            self._zip_cache = (key, rs)
            txt = f"{os.path.basename(self.roi_zip)}: {len(rs)} ROIs"
            if rs.kind == "grid":
                g = rs.grid
                txt += f" - regular grid {g.cols}×{g.rows}, {g.size} px at ({g.x0}, {g.y0})"
            self.zip_lbl.setText(txt)
        return self._zip_cache[1]

    def _choose_roi_zip(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "ImageJ ROI set", self.folder or "",
                                                     "ROI zip (*.zip *.roi)")
        if p:
            self.roi_zip = p
            self._update_roi_display()

    def _clear_roi_zip(self):
        self.roi_zip = None
        self.zip_lbl.setText("")
        self._update_roi_display()

    def _run_measure(self):
        if not self.video:
            return
        rot = self.calib if (self.apply_rot_chk.isChecked() and self._calib_complete()) else None
        if self.apply_rot_chk.isChecked() and rot is None:
            self.error("'Apply rotation' is ticked but the calibration is incomplete (tab 1).")
            return
        rois = self._load_zip_rois() if self.roi_zip else wc.RoiSet.from_grid(self._grid_spec())
        if rois is None:
            return
        stem = os.path.splitext(self.video)[0] + ("_rotated" if rot is not None else "")
        npz, csv = stem + "_grid.npz", stem + "_grid.csv"
        self.session.update(apply_rotation=rot is not None, gray_mode=self._gray_mode(),
                            grid=asdict(self._grid_spec()), roi_zip=self.roi_zip)
        if rot is not None:
            self.session["rotation"] = rot.to_dict()
        self.save_session()
        write_csv = self.csv_chk.isChecked()
        rot_copy = wc.RotationCalib.from_dict(rot.to_dict()) if rot else None
        gm = self._gray_mode()
        video = self.video

        def job(progress=None, cancel=None):
            m = wc.measure_video(video, rois, rotation=rot_copy, gray_mode=gm,
                                 progress=progress, cancel=cancel)
            m.save_npz(npz)
            if write_csv:
                m.save_fiji_csv(csv, progress=progress)
            return m

        self.run_task(job, on_done=lambda m: self._set_measurement(m, npz, fresh=True),
                      label="Measuring")

    def _load_measurement(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Measurement", self.folder or "",
                                                     "Measurements (*.npz *.csv)")
        if not p:
            return
        if p.lower().endswith(".npz"):
            self._set_measurement(wc.Measurement.load_npz(p), p)
            return
        fps, ok = QtWidgets.QInputDialog.getDouble(self, "Frame rate", "fps of the measured video:",
                                                   self.info["fps"] if self.info else 50.0, 1, 10000, 3)
        if not ok:
            return
        size = (self.info["width"], self.info["height"]) if self.info else (0, 0)
        vid = self.video

        def job(progress=None, cancel=None):
            m = wc.Measurement.load_fiji_csv(p, fps, size)
            m.source = vid    # frames for display come from the selected video
            return m
        self.run_task(job, on_done=lambda m: self._set_measurement(m, p), label="Reading CSV")

    def _set_measurement(self, m: wc.Measurement, path, fresh=False):
        self.meas = m
        self.meas_path = path
        self._bg_cache = None
        speed = f"<br>measured in {m.elapsed_s:.0f} s ({m.n_frames / m.elapsed_s:.0f} frames/s)" \
            if fresh and m.elapsed_s else ""
        rot = "rotated on the fly" if m.rotation else "as recorded"
        self.meas_info.setText(f"<b>Measurement:</b> {os.path.basename(path)}<br>{m.n_frames} frames × "
                               f"{len(m.names)} ROIs, {m.fps:.2f} fps, {m.duration:.1f} s, {rot}, "
                               f"grey: {m.gray_mode}{speed}")
        self._clear_downstream()
        if m.rotation and (fresh or self.orientation.temporal is None
                           or self.session.get("orientation_source") != "manual"):
            self._derive_orientation()
        self.ori_view.set_source(lambda i: self._meas_space_frame(i, use_meas=True),
                                 m.n_frames, m.fps)
        self._update_ori_markers()
        self.ev_roi_spin.setRange(0, len(m.names) - 1)
        self._setup_ev_view()
        self.events_status.setText("Measurement loaded - run event detection.")
        self._update_movie_info()
        self.mov_view.set_source(lambda i: self._meas_space_frame(i, use_meas=True), m.n_frames, m.fps)
        self.mov_to.setMaximum(m.duration)
        self.mov_from.setMaximum(m.duration)
        self._update_interval_lbl()
        if not fresh:
            self._restore_saved_analysis()
        else:
            self.log("Measurement saved: " + path)
        if hasattr(self, "cells_tab"):
            self.cells_tab.reset_for_video()

    def _clear_downstream(self):
        self.peaks = self.rates = self.codes = self.active = None
        self.results = None
        for c in getattr(self, "ev_canvases", {}).values():
            c.fig.clear()
            c.draw()
        for c in getattr(self, "wave_canvases", {}).values():
            c.fig.clear()
            c.draw()
        if hasattr(self, "summary_table"):
            self.summary_table.setRowCount(0)

    def results_dir(self):
        stem = os.path.splitext(self.meas_path)[0]
        os.makedirs(stem, exist_ok=True)
        return stem

    def background(self):
        if self._bg_cache is None and self.meas is not None:
            idx = min(int(10 * self.meas.fps), self.meas.n_frames - 1)
            try:
                self._bg_cache = self._meas_space_frame(idx, use_meas=True)
            except Exception:
                self._bg_cache = None
        return self._bg_cache

    # =====================================================================
    # Tab 3 - Events
    # =====================================================================
    def _build_events_tab(self):
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        self.event_form = ParamForm(wc.EventParams())
        b = self.run_ev_btn = QtWidgets.QPushButton("▶  Detect events")
        b.setStyleSheet("font-weight: bold; padding: 6px")
        b.clicked.connect(self._run_events)
        self.events_status = QtWidgets.QLabel("Load or run a measurement first.")
        self.events_status.setWordWrap(True)
        self.ev_roi_spin = QtWidgets.QSpinBox()
        self.ev_roi_spin.valueChanged.connect(self._draw_trace)
        hint = QtWidgets.QLabel("Tip: click a square on the activity map to see its trace.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        b_tr = QtWidgets.QPushButton("Re-plot trace with current σ")
        b_tr.clicked.connect(self._draw_trace)
        from presets import PresetBar
        self.preset_bars = getattr(self, "preset_bars", [])
        bar = PresetBar(self)
        self.preset_bars.append(bar)
        pl.addWidget(group("Analysis preset (objective / wave stage)", bar))
        pl.addWidget(group("Peak detection (notebook cell 16)", self.event_form))
        pl.addWidget(b)
        pl.addWidget(self.events_status)
        self.live_chk = QtWidgets.QCheckBox("Live preview on the selected ROI")
        self.live_chk.setToolTip("Re-detect events on this ROI whenever a setting changes. "
                                 "'Detect events' then applies the settings to all ROIs.")
        self.live_lbl = QtWidgets.QLabel("")
        self.live_lbl.setWordWrap(True)
        self._live_timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._live_timer.timeout.connect(self._draw_trace)
        self.live_chk.toggled.connect(lambda _: self._draw_trace())
        self._watch_form(self.event_form)
        for w in self.event_form.widgets.values():
            w.valueChanged.connect(lambda *_: self.live_chk.isChecked() and self._live_timer.start())
        pl.addWidget(group("Inspect ROI", hrow(QtWidgets.QLabel("ROI index"), self.ev_roi_spin),
                           self.live_chk, self.live_lbl, b_tr, hint))
        pl.addStretch(1)
        self.ev_tabs = QtWidgets.QTabWidget()
        self.ev_canvases = {}
        for k, t in (("trace", "Trace"), ("raster", "Raster"), ("map", "Activity map"), ("hist", "Rate histogram")):
            c = MplCanvas()
            self.ev_canvases[k] = c
            self.ev_tabs.addTab(self._map_panel(c) if k == "map" else c, t)
        self.ev_canvases["map"].canvas.mpl_connect("button_release_event", self._map_clicked)
        # video frame with the ROI grid: click a square -> its trace
        self.ev_view = FrameView()
        self.ev_view.clicked.connect(self._ev_frame_clicked)
        self.ev_view.slider.valueChanged.connect(self._ev_frame_changed)
        vb = self.ev_view.vb
        self.ev_grid = pg.PlotCurveItem(pen=pg.mkPen((255, 200, 0, 70), width=1), connect="pairs")
        self.ev_active = pg.ScatterPlotItem(pxMode=False, symbol="s", pen=None,
                                            brush=pg.mkBrush(255, 40, 40, 110))
        self.ev_sel = pg.ScatterPlotItem(pxMode=False, symbol="s", brush=None,
                                         pen=pg.mkPen((0, 255, 255), width=2.5))
        for it in (self.ev_grid, self.ev_active, self.ev_sel):
            vb.addItem(it)
        hint_v = QtWidgets.QLabel("Click a square to show its trace · red = active ('a') at this frame")
        hint_v.setStyleSheet("color: gray")
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.ev_view, 1)
        tl.addWidget(hint_v)
        vsplit = QtWidgets.QSplitter(Qt.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(self.ev_tabs)
        vsplit.setSizes([480, 420])
        return self._split(vsplit, panel, 330)

    # --- video viewer on the Events tab -----------------------------------
    def _roi_size_px(self):
        m = self.meas
        if m is None or not len(m.area):
            return 10.0
        return float(np.sqrt(np.nanmedian(m.area)))

    def _setup_ev_view(self):
        m = self.meas
        self.ev_view.set_source(lambda i: self._meas_space_frame(i, use_meas=True), m.n_frames, m.fps)
        if m.grid:
            g = wc.GridSpec(**m.grid)
            xs, ys = [], []
            for c in range(g.cols + 1):
                x = g.x0 + c * g.size
                xs += [x, x]; ys += [g.y0, g.y0 + g.height]
            for r in range(g.rows + 1):
                y = g.y0 + r * g.size
                xs += [g.x0, g.x0 + g.width]; ys += [y, y]
            self.ev_grid.setData(np.array(xs, float), np.array(ys, float))
        else:
            self.ev_grid.setData([], [])
        self.ev_active.setData([], [])
        self._update_ev_selection()

    def _ev_frame_clicked(self, x, y):
        m = self.meas
        if m is None:
            return
        d = np.hypot(m.x - x, m.y - y)
        k = int(np.argmin(d))
        if d[k] > self._roi_size_px():        # clicked outside the ROIs
            return
        self.ev_roi_spin.setValue(k)
        self.ev_tabs.setCurrentIndex(0)

    def _update_ev_selection(self):
        m = self.meas
        if m is None:
            return
        k = min(self.ev_roi_spin.value(), len(m.names) - 1)
        sz = self._roi_size_px()
        self.ev_sel.setData([m.x[k]], [m.y[k]], size=sz)

    def _ev_frame_changed(self, i):
        m = self.meas
        if m is None:
            return
        i = min(max(i, 0), m.n_frames - 1)
        if self.codes is not None:
            on = self.codes[i] == wc.A
            self.ev_active.setData(m.x[on], m.y[on], size=self._roi_size_px())
        else:
            self.ev_active.setData([], [])
        ax = self.ev_canvases["trace"].fig.axes
        now = time.perf_counter()
        if self.ev_view._playing and now - getattr(self, "_last_mark", 0) < 0.3:
            return                      # don't redraw the plot on every played frame
        self._last_mark = now
        if ax and hasattr(ax[0], "_tmark"):
            t = i / m.fps
            ax[0]._tmark.set_xdata([t, t])
            self.ev_canvases["trace"].draw()

    def _run_events(self, then=None):
        if self.meas is None:
            self.error("No measurement loaded.")
            return
        if not self._require_preset():
            return
        ep = self.event_form.value()
        self.session["event_params"] = asdict(ep)
        self.save_session()
        m = self.meas

        def job(progress=None, cancel=None):
            pk = wc.detect_events(m, ep, progress, cancel)
            rates = wc.cell_rates(pk, m, ep.exclude_s)
            _ = progress and progress(0, 1, "building rasters")
            codes = wc.build_rasters(pk, m.n_frames)
            act = wc.filter_active(codes, m.x, m.y)
            return pk, rates, codes, act

        def done(r):
            self.peaks, self.rates, self.codes, self.active = r
            self.results = None
            out = self.results_dir()
            base = os.path.basename(out)
            wc.peaks_dataframe(self.peaks, m.fps).to_csv(os.path.join(out, f"peaks_{base}.csv"))
            np.savez_compressed(os.path.join(out, "rasters.npz"), codes=self.codes, active=self.active)
            self._save_events_state(ep)
            n = sum(len(p) for p in self.peaks)
            self.events_status.setText(f"{n} events in {len(self.peaks)} ROIs; "
                                       f"{int((self.rates > 0.2).sum())} ROIs > 0.2 events/min; "
                                       f"mean {self.rates.mean():.2f}/min.<br>Saved to {base}/")
            if self.ev_roi_spin.value() == 0 and len(self.rates):
                self.ev_roi_spin.setValue(int(np.argmax(self.rates)))   # start on an active ROI
            self._draw_events()
            if callable(then):
                then()
        self.run_task(job, on_done=done, label="Detecting events")

    # ---------------------------------------------------------- activity-map colours
    def _map_panel(self, canvas):
        st = self.settings.setdefault("map_colors", {})
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.map_cmap = QtWidgets.QComboBox()
        self.map_cmap.addItems(wpl.MAP_CMAPS)
        self.map_cmap.setCurrentText(st.get("cmap", "viridis"))
        self.map_min, self.map_max = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.map_min, st.get("vmin", 0.2)), (self.map_max, st.get("vmax", 3.0))):
            sp.setRange(0, 1000)
            sp.setDecimals(2)
            sp.setSingleStep(0.1)
            sp.setSuffix(" /min")
            sp.setValue(v)
        self.map_auto = QtWidgets.QCheckBox("auto range")
        self.map_auto.setToolTip("Colour range from the 2nd–98th percentile of the active squares' rates")
        self.map_auto.setChecked(st.get("auto", True))
        self.map_alpha = QtWidgets.QSpinBox()
        self.map_alpha.setRange(5, 100)
        self.map_alpha.setSuffix(" % opaque")
        self.map_alpha.setValue(int(st.get("alpha", 100)))
        self.map_bg = QtWidgets.QCheckBox("image behind")
        self.map_bg.setChecked(st.get("bg", True))
        self.map_hide = QtWidgets.QCheckBox("hide squares below min")
        self.map_hide.setChecked(st.get("hide", False))
        for wd in (self.map_cmap,):
            wd.currentTextChanged.connect(self._map_changed)
        for wd in (self.map_min, self.map_max, self.map_alpha):
            wd.valueChanged.connect(self._map_changed)
        for wd in (self.map_auto, self.map_bg, self.map_hide):
            wd.toggled.connect(self._map_changed)
        row = QtWidgets.QHBoxLayout()
        for x in (QtWidgets.QLabel("Colours"), self.map_cmap, QtWidgets.QLabel("min"), self.map_min,
                  QtWidgets.QLabel("max"), self.map_max, self.map_auto, self.map_alpha, self.map_bg, self.map_hide):
            row.addWidget(x)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(canvas, 1)
        self._map_sync_enabled()
        return w

    def _map_sync_enabled(self):
        self.map_min.setEnabled(not self.map_auto.isChecked())
        self.map_max.setEnabled(not self.map_auto.isChecked())

    def _map_changed(self, *_):
        if getattr(self, "_map_busy", False):
            return
        self._map_sync_enabled()
        self.settings["map_colors"] = dict(cmap=self.map_cmap.currentText(), vmin=self.map_min.value(),
                                           vmax=self.map_max.value(), auto=self.map_auto.isChecked(),
                                           alpha=self.map_alpha.value(), bg=self.map_bg.isChecked(),
                                           hide=self.map_hide.isChecked())
        if self.meas is not None and self.rates is not None:
            self._draw_map()

    def _draw_map(self):
        if self.map_auto.isChecked():
            lo, hi = wpl.map_limits(self.rates)
            self._map_busy = True
            self.map_min.setValue(lo)
            self.map_max.setValue(hi)
            self._map_busy = False
        c = self.ev_canvases["map"]
        wpl.activity_map(c.fig, self.meas, self.rates, self.background(), s=self.um_per_px(),
                         vmin=self.map_min.value(), vmax=max(self.map_max.value(), self.map_min.value() + 1e-3),
                         cmap=self.map_cmap.currentText(), alpha=self.map_alpha.value() / 100.0,
                         show_bg=self.map_bg.isChecked(), hide_below=self.map_hide.isChecked())
        c.draw()

    def _draw_events(self):
        m = self.meas
        c = self.ev_canvases
        wpl.raster(c["raster"].fig, self.peaks, m.fps, m.duration)
        self._draw_map()
        wpl.rate_hist(c["hist"].fig, self.rates)
        for k in ("raster", "hist"):
            c[k].draw()
        self._draw_trace()
        self._ev_frame_changed(self.ev_view.frame_index())

    def _draw_trace(self, *_):
        if self.meas is None:
            return
        k = self.ev_roi_spin.value()
        ep = self.event_form.value()
        live = self.live_chk.isChecked()
        if live:
            if getattr(self, "_bad_sig", None) != id(self.meas):
                self._bad, self._bad_sig = wc.meas_dark_frames(self.meas), id(self.meas)
            pk = wc.detect_events_single(self.meas.mean[:, k], self.meas.fps, ep, self._bad)
            peaks = {k: pk}
            ref = f" (last full run: {len(self.peaks[k])})" if self.peaks is not None else ""
            self.live_lbl.setText(f"<b>{len(pk)} events</b> in {self.meas.names[k]} with these settings{ref}")
        else:
            peaks = self.peaks
            self.live_lbl.setText("")
        fig = self.ev_canvases["trace"].fig
        keep = None                     # same ROI redrawn (e.g. live preview): keep the zoom
        if fig.axes and getattr(self, "_trace_roi", None) == k:
            a0 = fig.axes[0]
            keep = (a0.get_xlim(), a0.get_ylim(), getattr(a0, "_surf_home", None))
        wpl.trace(fig, self.meas, k, peaks, ep.sigma, s=self.um_per_px(),
                  t_mark=self.ev_view.frame_index() / self.meas.fps, preview=live)
        self._trace_roi = k
        if keep and fig.axes:
            a1 = fig.axes[0]
            if keep[2] is not None:
                a1._surf_home = keep[2]
            a1.set_xlim(*keep[0])
            a1.set_ylim(*keep[1])
        self.ev_canvases["trace"].draw()
        self._update_ev_selection()

    def _map_clicked(self, ev):
        if ev.inaxes is None or self.meas is None or ev.xdata is None:
            return
        if self.ev_canvases["map"].canvas.toolbar.mode or ev.button != 1:
            return
        if getattr(self.ev_canvases["map"], "nav", None) and self.ev_canvases["map"].nav.last_moved:
            return                       # that was a pan, not a click on a square
        k = wpl._u(self.um_per_px())[0]          # map axes are in µm once the scale is known
        d = (self.meas.x * k - ev.xdata) ** 2 + (self.meas.y * k - ev.ydata) ** 2
        self.ev_roi_spin.setValue(int(np.argmin(d)))
        self.ev_tabs.setCurrentIndex(0)

    # =====================================================================
    # Tab 4 - Waves
    # =====================================================================
    def _build_waves_tab(self):
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        # orientation
        self.ori_radio = {}
        ol = QtWidgets.QGridLayout()
        for n, k in enumerate(("temporal", "nasal")):
            rb = QtWidgets.QRadioButton(k.capitalize())
            rb.toggled.connect(lambda on, k=k: on and setattr(self, "ori_target", k))
            self.ori_radio[k] = rb
            ol.addWidget(rb, n // 2, n % 2)
        self.ori_radio["temporal"].setChecked(True)
        self.ori_lbl = QtWidgets.QLabel("")
        self.ori_lbl.setWordWrap(True)
        b_ori_clear = QtWidgets.QPushButton("Clear")
        b_ori_clear.clicked.connect(self._clear_orientation)
        ori_hint = QtWidgets.QLabel("Taken from the rotation calibration; without rotation, click "
                                    "temporal and nasal on the <b>Orientation</b> view. Dorsal / "
                                    "ventral follow from the eye (Rotate tab). Directions: "
                                    "<b>0° temporal, 90° dorsal, 180° nasal, 270° ventral</b>.")
        ori_hint.setWordWrap(True)
        pl.addWidget(group("Retinal orientation", ori_hint, ol, self.ori_lbl, b_ori_clear))
        self.wave_form = ParamForm(wc.WaveParams())
        self._watch_form(self.wave_form)
        from presets import PresetBar
        self.preset_bars = getattr(self, "preset_bars", [])
        bar = PresetBar(self)
        self.preset_bars.append(bar)
        pl.addWidget(group("Analysis preset (objective / wave stage)", bar))
        pl.addWidget(group("Wave tracking (cells 21-43)", self.wave_form))
        b = self.run_wave_btn = QtWidgets.QPushButton("▶  Run wave analysis")
        b.setStyleSheet("font-weight: bold; padding: 6px")
        b.clicked.connect(self._run_waves)
        pl.addWidget(b)
        b_save = QtWidgets.QPushButton("Save results (xlsx + figures)")
        b_save.clicked.connect(self._save_wave_results)
        pl.addWidget(b_save)
        # movie
        self.mov_mode = QtWidgets.QComboBox()
        self.mov_mode.addItems(["full (raster states + centres + tracks)", "filtered active only"])
        self.mov_from = QtWidgets.QDoubleSpinBox(); self.mov_from.setRange(0, 1e6); self.mov_from.setValue(10)
        self.mov_to = QtWidgets.QDoubleSpinBox(); self.mov_to.setRange(0, 1e6); self.mov_to.setValue(60)
        self.mov_speed = QtWidgets.QComboBox()
        self.mov_speed.setEditable(True)
        self.mov_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "5×", "10×", "20×"])
        self.mov_speed.setCurrentText("1×")
        self.mov_speed.setToolTip("Movie uses the recording's fps; faster speeds keep every n-th frame. "
                                  "You can type another value, e.g. 3.")
        self.mov_speed_lbl = QtWidgets.QLabel("")
        self.mov_speed_lbl.setStyleSheet("color: gray")
        for w_ in (self.mov_from, self.mov_to):
            w_.valueChanged.connect(self._update_movie_info)
        self.mov_speed.currentTextChanged.connect(self._update_movie_info)
        self.mov_speed.currentTextChanged.connect(lambda *_: hasattr(self, "mov_iv_lbl") and self._update_interval_lbl())
        self.mov_overlay = QtWidgets.QCheckBox("Overlay on the video")
        self.mov_overlay.setChecked(True)
        mf = QtWidgets.QFormLayout()
        mf.addRow("Mode", self.mov_mode)
        mf.addRow("From (s)", self.mov_from)
        mf.addRow("To (s)", self.mov_to)
        mf.addRow("Speed", self.mov_speed)
        mf.addRow("", self.mov_speed_lbl)
        self.mov_time = QtWidgets.QComboBox()
        self.mov_time.addItems(["Real video time", "Movie time", "None"])
        self.mov_time.setToolTip("Upper-left label: time in the original recording, time in the "
                                 "rendered movie, or no label")
        self.mov_show_speed = QtWidgets.QCheckBox("Show speed (upper right)")
        self.mov_show_speed.setChecked(True)
        mf.addRow("Time label", self.mov_time)
        mf.addRow("", self.mov_show_speed)
        b_iv = QtWidgets.QPushButton("Choose interval on the video…")
        b_iv.setToolTip("Opens the Movie view: play the video and set start / end there")
        b_iv.clicked.connect(lambda: self.wave_tabs.setCurrentWidget(self.mov_page))
        bm = QtWidgets.QPushButton("Render movie…")
        bm.clicked.connect(self._render_movie)
        self.mov_show_btn = QtWidgets.QPushButton("Show movie in folder")
        self.mov_show_btn.setEnabled(False)
        self.mov_show_btn.clicked.connect(lambda: self._reveal(self._last_movie))
        self._last_movie = None
        pl.addWidget(group("Movie", mf, b_iv, self.mov_overlay, bm, self.mov_show_btn))
        pl.addStretch(1)

        self.wave_tabs = QtWidgets.QTabWidget()
        self.ori_view = FrameView()
        self.ori_view.clicked.connect(self._ori_clicked)
        self.wave_tabs.addTab(self.ori_view, "Orientation")
        self.summary_table = QtWidgets.QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["metric", "value"])
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.verticalHeader().setVisible(False)
        self.wave_tabs.addTab(self.summary_table, "Summary")
        self.wave_tabs.addTab(self._build_movie_page(), "Movie")
        self.wave_canvases = {}
        for k, t in (("tracks", "Tracks"), ("paths", "Smoothed paths"), ("arrows", "Direction map"),
                     ("p_be", "Polar: begin→end"), ("p_seg", "Polar: segments"),
                     ("p_grid", "Polar: grid"), ("isi", "Start intervals")):
            c = MplCanvas()
            self.wave_canvases[k] = c
            self.wave_tabs.addTab(c, t)
        return self._split(self.wave_tabs, panel, 380)

    def _ori_clicked(self, x, y):
        setattr(self.orientation, self.ori_target, (round(x, 1), round(y, 1)))
        self.ori_radio["nasal" if self.ori_target == "temporal" else "temporal"].setChecked(True)
        self.session["orientation"] = self.orientation.to_dict()
        self.session["orientation_source"] = "manual"
        self.save_session()
        self._update_ori_markers()

    def _clear_orientation(self):
        self.orientation = wc.Orientation()
        self.session.pop("orientation", None)
        self.session.pop("orientation_source", None)
        self.save_session()
        self.ori_radio["temporal"].setChecked(True)
        if self.meas is not None and self.meas.rotation:
            self._derive_orientation()
        self._update_ori_markers()

    def _frame_eye(self):
        """Which eye the *measured frame* shows (a flipped left retina looks like a right one)."""
        m = self.meas
        eye = self._eye()
        if m is not None and m.rotation:
            cal = wc.RotationCalib.from_dict(m.rotation)
            base = cal.eye if cal.version >= 2 else eye
            return ("left" if base == "right" else "right") if cal.flips else base
        return eye

    def _effective_orientation(self):
        o = self.orientation
        return wc.Orientation(o.temporal, o.nasal, self._frame_eye())

    def _derive_orientation(self):
        m = self.meas
        cal = wc.RotationCalib.from_dict(m.rotation)
        o = wc.Orientation.from_rotation(cal, *m.frame_size, eye=self._eye())
        self.orientation = wc.Orientation(o.temporal, o.nasal, o.eye)
        self.session["orientation"] = self.orientation.to_dict()
        self.session["orientation_source"] = "rotation"

    def _update_ori_markers(self):
        o = self._effective_orientation()
        D, V = o.dv_points()
        self.ori_view.set_markers({"temporal": (o.temporal, MARK_COLORS["temporal"]),
                                   "nasal": (o.nasal, MARK_COLORS["nasal"]),
                                   "dorsal": (D, (255, 160, 0)), "ventral": (V, (255, 160, 0))})
        if o.temporal is None or o.nasal is None:
            self.ori_lbl.setText("<span style='color:#d33'>Temporal and nasal not set: the frame is "
                                 "assumed to be nasal-left / temporal-right already. Click them on "
                                 "the Orientation view if not.</span>")
            return
        txt = (f"{self._eye()} retina; in the measured frame dorsal is "
               f"{'up' if o.axes()[1][1] < 0 else 'down'} (nasal→temporal axis at {o.ref_deg:.0f}°).")
        m = self.meas
        if m is not None and m.rotation:
            cal = wc.RotationCalib.from_dict(m.rotation)
            if cal.version >= 2 and cal.eye != self._eye():
                txt += ("<br><span style='color:#d33'>This measurement was made as a "
                        f"<b>{cal.eye}</b> retina. Re-measure after changing the eye.</span>")
            elif cal.version < 3:
                txt += ("<br><span style='color:gray'>Measured with an older rotation (mirrored / "
                        "flipped). Directions are still correct if the eye is right; re-measure "
                        "for the unmirrored view.</span>")
        self.ori_lbl.setText(txt)

    def _run_waves(self):
        if self.meas is None:
            self.error("No measurement loaded.")
            return
        if not self._require_preset():
            return
        if self.active is None:
            self._run_events(then=self._run_waves)
            return
        wp = self.wave_form.value()
        self.session["wave_params"] = asdict(wp)
        self.save_session()
        ori = self._effective_orientation()
        if ori.temporal is None or ori.nasal is None:
            self.log("No nasal-temporal axis set: assuming the frame is already nasal-left / "
                     f"temporal-right ({self._frame_eye()} retina).")
        self.run_task(wc.run_wave_pipeline, self.meas, self.active, self.rates, wp, ori,
                      on_done=self._waves_done, label="Tracking waves")

    def _waves_done(self, res):
        res.summary["eye"] = self._eye()          # the real eye (for the Compare tab)
        self.results = res
        if getattr(self, "cells_tab", None) is not None and self.cells_tab.result is not None:
            self.cells_tab.res_lbl.setText("<span style='color:#b36b00'>The waves changed – run "
                                           "'Detect events + link to waves' again on the Cells tab.</span>")
        self._draw_waves()
        self.wave_tabs.setCurrentIndex(1)
        self._save_wave_results(silent=True)
        self._save_waves_state()

    # --- saved analysis state (restored when the video is opened again) -----
    def _meas_signature(self):
        m = self.meas
        try:
            mt = os.path.getmtime(self.meas_path)
        except OSError:
            mt = 0
        return f"{m.n_frames}x{len(m.names)}@{mt:.0f}"

    def _save_events_state(self, ep):
        try:
            lens = np.array([len(p) for p in self.peaks], np.int64)
            cat = np.concatenate(self.peaks) if len(self.peaks) else np.zeros(0, np.int64)
            np.savez_compressed(os.path.join(self.results_dir(), "events_state.npz"),
                                codes=self.codes, active=self.active, peaks=cat, lens=lens,
                                sig=self._meas_signature(), params=json.dumps(asdict(ep)))
        except Exception as e:
            self.log(f"events not saved for reopening: {e}")

    def _save_waves_state(self):
        try:
            with open(os.path.join(self.results_dir(), "wave_state.pkl"), "wb") as f:
                pickle.dump(dict(sig=self._meas_signature(), results=self.results,
                                 wave_params=asdict(self.wave_form.value())), f)
        except Exception as e:
            self.log(f"wave analysis not saved for reopening: {e}")

    def _restore_saved_analysis(self):
        """Load events / waves saved for this measurement (skipped if it was re-measured)."""
        d = os.path.splitext(self.meas_path)[0]
        sig = self._meas_signature()
        ev = os.path.join(d, "events_state.npz")
        if not os.path.exists(ev):
            return
        try:
            z = np.load(ev, allow_pickle=False)
            if str(z["sig"]) != sig:
                self.events_status.setText("Saved events belong to an older measurement - run detection again.")
                return
            lens, cat = z["lens"], z["peaks"]
            self.peaks = np.split(cat, np.cumsum(lens)[:-1]) if len(lens) else []
            ep = wc.EventParams(**json.loads(str(z["params"])))
            self.event_form.set_value(asdict(ep))
            self.rates = wc.cell_rates(self.peaks, self.meas, ep.exclude_s)
            self.codes, self.active = z["codes"], z["active"]
        except Exception as e:
            self.log(f"could not restore events: {e}")
            self.peaks = self.rates = self.codes = self.active = None
            return
        n = sum(len(p) for p in self.peaks)
        self.events_status.setText(f"Restored previous detection: {n} events in {len(self.peaks)} ROIs; "
                                   f"{int((self.rates > 0.2).sum())} ROIs > 0.2 events/min.")
        if len(self.rates):
            self.ev_roi_spin.setValue(int(np.argmax(self.rates)))
        self._draw_events()
        wv = os.path.join(d, "wave_state.pkl")
        if os.path.exists(wv):
            try:
                with open(wv, "rb") as f:
                    st = pickle.load(f)
                res = st.get("results")
                if st.get("sig") == sig and "angle_convention" not in getattr(res, "summary", {}):
                    self.log("The saved wave analysis uses the old angle convention - run Waves again.")
                elif st.get("sig") == sig:
                    self.results = res
                    wp = dict(st.get("wave_params", {}))
                    wp.pop("workers", None)
                    self.wave_form.set_value(wp)
                    self._draw_waves()
                    self.log("Restored the previous event detection and wave analysis.")
            except Exception as e:
                self.log(f"could not restore wave analysis: {e}")
                self.results = None

    def _draw_waves(self):
        res, m, bg = self.results, self.meas, self.background()
        c = self.wave_canvases
        wp = self.wave_form.value()
        sc = self.um_per_px()
        wpl.tracks_map(c["tracks"].fig, m, res, bg, s=sc)
        wpl.smoothed_paths(c["paths"].fig, m, res, bg, s=sc)
        wpl.arrow_map(c["arrows"].fig, m, res, bg, s=sc)
        # draw the polar plots like the image of this retina (temporal on its side)
        try:
            tl = bool(res.orientation.axes()[0][0] < 0)
        except Exception:
            tl = True
        wpl.polar_hist(c["p_be"].fig, res.beg_end.angle, "Beginning → end direction", temporal_left=tl)
        wpl.polar_hist(c["p_seg"].fig, res.segments.angle, "Segment directions (travelDetails)",
                       temporal_left=tl)
        wpl.polar_hist(c["p_grid"].fig, res.grid_vectors.get("angle_anat_deg", []), "Grid mean directions",
                       temporal_left=tl)
        wpl.log_isi(c["isi"].fig, res, wp.wave_isi_cutoff)
        for k in c:
            c[k].draw()
        tab = wc.results_tables(res, m, self.peaks, self.rates, self.um_per_px())["summary"]
        self.summary_table.setRowCount(len(tab))
        for r, (k, v) in enumerate(zip(tab.metric, tab.value)):
            self.summary_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k)))
            vv = f"{v:.3f}" if isinstance(v, float) else str(v)
            self.summary_table.setItem(r, 1, QtWidgets.QTableWidgetItem(vv))
        self.summary_table.resizeColumnToContents(0)

    def _save_wave_results(self, silent=False):
        if self.results is None:
            if not silent:
                self.error("Run the wave analysis first.")
            return
        out = self.results_dir()
        base = os.path.basename(out)
        try:
            wc.save_results_excel(os.path.join(out, f"{base}_wave_results.xlsx"), self.results, self.meas,
                                  self.peaks, self.rates, self.event_form.value(), self.wave_form.value(),
                                  um_per_px=self.um_per_px(), info=self._recording_info())
            figs = dict(self.wave_canvases)
            figs.update({f"ev_{k}": v for k, v in self.ev_canvases.items()})
            for k, cnv in figs.items():
                if cnv.fig.axes:
                    figexport.save_publication(cnv.fig, os.path.join(out, f"{base}_{k}"))
        except PermissionError as e:
            self.error(f"Could not write (file open in Excel?):\n{e}")
            return
        self.log(f"Results saved to {out}")

    # --- Movie page: original video + interval selection ----------------------
    def _build_movie_page(self):
        self.mov_page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.mov_page)
        lay.setContentsMargins(0, 0, 0, 0)
        self.mov_view = FrameView()
        vb = self.mov_view.vb
        self.mov_act = pg.ScatterPlotItem(pxMode=False, symbol="s", pen=None, brush=pg.mkBrush(255, 40, 40, 100))
        self.mov_cen = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(40, 120, 255), pen=pg.mkPen("w"))
        vb.addItem(self.mov_act)
        vb.addItem(self.mov_cen)
        self.mov_view.slider.valueChanged.connect(self._mov_frame_changed)
        b_start = QtWidgets.QPushButton("⟦  Set start here")
        b_end = QtWidgets.QPushButton("Set end here  ⟧")
        b_start.clicked.connect(lambda: self._set_interval("start"))
        b_end.clicked.connect(lambda: self._set_interval("end"))
        b_go_s = QtWidgets.QToolButton(text="⇤ start")
        b_go_s.setToolTip("Jump to the interval start")
        b_go_s.clicked.connect(lambda: self.mov_view.slider.setValue(self._iv_frames()[0]))
        b_go_e = QtWidgets.QToolButton(text="end ⇥")
        b_go_e.setToolTip("Jump to the interval end")
        b_go_e.clicked.connect(lambda: self.mov_view.slider.setValue(self._iv_frames()[1]))
        b_play = QtWidgets.QPushButton("▶  Play interval")
        b_play.clicked.connect(self._play_interval)
        self.mov_show_overlay = QtWidgets.QCheckBox("Show detected wave")
        self.mov_show_overlay.setToolTip("Active squares (red) and wave centres (blue); needs events / waves")
        self.mov_show_overlay.toggled.connect(lambda _: self._mov_frame_changed(self.mov_view.frame_index()))
        self.mov_iv_lbl = QtWidgets.QLabel("")
        row = hrow(b_start, b_end, b_go_s, b_go_e, b_play, self.mov_show_overlay, "stretch", self.mov_iv_lbl)
        lay.addWidget(self.mov_view, 1)
        lay.addLayout(row)
        for w_ in (self.mov_from, self.mov_to):
            w_.valueChanged.connect(self._update_interval_lbl)
        return self.mov_page

    def _iv_frames(self):
        fps = self.meas.fps if self.meas is not None else (self.info["fps"] if self.info else 1)
        return int(round(self.mov_from.value() * fps)), int(round(self.mov_to.value() * fps))

    def _set_interval(self, which):
        fps = self.mov_view.fps
        t = self.mov_view.frame_index() / fps
        if which == "start":
            self.mov_from.setValue(t)
            if self.mov_to.value() <= t:
                self.mov_to.setValue(min(t + 10, self.mov_view.slider.maximum() / fps))
        else:
            if t <= self.mov_from.value():
                self.error("The end must be after the start.")
                return
            self.mov_to.setValue(t)
        self._update_interval_lbl()

    def _update_interval_lbl(self, *_):
        a, b = self.mov_from.value(), self.mov_to.value()
        sp = self._movie_speed()
        self.mov_iv_lbl.setText(f"Interval <b>{a:.2f} – {b:.2f} s</b> ({max(b - a, 0):.2f} s) "
                                f"→ movie {max(b - a, 0) / sp:.1f} s at {sp:g}×")

    def _play_interval(self):
        f0, f1 = self._iv_frames()
        if f1 <= f0:
            self.error("Set an end after the start first.")
            return
        v = self.mov_view
        v.stop()
        v.slider.setValue(f0)
        v.stop_at = f1
        v.play_btn.setChecked(True)
        v.stop_at = f1          # (toggling on resets the clock, keep the stop frame)

    def _mov_frame_changed(self, i):
        m = self.meas
        if m is None or not self.mov_show_overlay.isChecked():
            self.mov_act.setData([], [])
            self.mov_cen.setData([], [])
            return
        i = min(max(int(i), 0), m.n_frames - 1)
        if self.codes is not None:
            on = self.codes[i] == wc.A
            self.mov_act.setData(m.x[on], m.y[on], size=self._roi_size_px())
        if self.results is not None and i in self.results.centres:
            c = self.results.centres[i]
            self.mov_cen.setData(c[:, 0], c[:, 1])
        else:
            self.mov_cen.setData([], [])

    def _movie_done(self, p):
        self._last_movie = p
        self.mov_show_btn.setEnabled(True)
        self.log(f"Movie saved: {p}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Movie saved")
        box.setText(f"Movie saved:\n{p}")
        b_open = box.addButton("Show in folder", QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Close", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_open:
            self._reveal(p)

    def _movie_speed(self):
        try:
            v = float(self.mov_speed.currentText().replace("×", "").replace("x", "").strip())
            return v if v > 0 else 1.0
        except ValueError:
            return 1.0

    def _update_movie_info(self, *_):
        m = getattr(self, "meas", None)
        fps = m.fps if m is not None else (self.info["fps"] if self.info else None)
        if not fps:
            self.mov_speed_lbl.setText("")
            return
        sp = self._movie_speed()
        dur = max(self.mov_to.value() - self.mov_from.value(), 0)
        how = f"1 of every {sp:g} frames" if sp > 1 else ("every frame" if sp == 1 else "every frame, slowed")
        self.mov_speed_lbl.setText(f"{fps * min(sp, 1):g} fps, {how}; movie length {dur / sp:.1f} s")

    def _render_movie(self):
        if self.results is None or self.codes is None:
            self.error("Run the wave analysis first.")
            return
        out = os.path.join(self.results_dir(), "wave_movie.mp4")
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Movie", out, "MP4 (*.mp4)")
        if not out:
            return
        m = self.meas
        f0 = int(self.mov_from.value() * m.fps)
        f1 = min(m.n_frames, int(self.mov_to.value() * m.fps))
        if f1 <= f0:
            self.error("'To' must be after 'From'.")
            return
        mode = "full" if self.mov_mode.currentIndex() == 0 else "filtered"
        vid = rot = None
        bg = None
        if self.mov_overlay.isChecked():
            vid = m.source if (m.source and os.path.exists(m.source) and
                               m.source.lower().endswith(VIDEO_EXT)) else self.video
            rot = wc.RotationCalib.from_dict(m.rotation) if m.rotation else None
        else:
            bg = self.background()
        self.run_task(wc.render_wave_movie, out, m, self.codes, self.active, self.results, f0, f1,
                      speed=self._movie_speed(), mode=mode, background=bg, video_path=vid,
                      time_label=["video", "movie", "none"][self.mov_time.currentIndex()],
                      show_speed=self.mov_show_speed.isChecked(),
                      rotation=rot, on_done=self._movie_done,
                      label="Rendering movie")

    # ------------------------------------------------------------------
    def _tab_changed(self, i):
        if i == 2:
            self._update_roi_display()

    def closeEvent(self, e):
        try:
            self.info_tab.player.release()
            self.save_app_settings()          # remembers the player's brightness settings
        except Exception:
            pass
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(e)


def reveal_in_folder(path):
    """Open Explorer / Finder with `path` selected (or the folder itself)."""
    import subprocess
    path = os.path.abspath(path)
    if not os.path.exists(path):
        path = os.path.dirname(path)
        if not os.path.exists(path):
            return False
    try:
        if sys.platform.startswith("win"):
            if os.path.isfile(path):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            else:
                os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path] if os.path.isfile(path) else ["open", path])
        else:
            subprocess.Popen(["xdg-open", path if os.path.isdir(path) else os.path.dirname(path)])
        return True
    except Exception:
        return False


def find_videos(root, max_depth=3):
    """Videos in `root` and up to `max_depth` levels of subfolders (relative paths), skipping
    hidden folders and the app's own results folders (<name>_grid)."""
    out = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep) + (dirpath != root)
        dirnames[:] = sorted(n for n in dirnames if not n.startswith((".", "_"))
                             and not n.endswith("_grid") and depth < max_depth)
        for f in sorted(filenames):
            if f.lower().endswith(VIDEO_EXT) and not f.startswith("._"):
                out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return out


def cv2_to_qimage(bgr):
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


def _install_crash_handler():
    """Started from a shortcut there is no console: log uncaught errors and show them."""
    log = os.path.join(os.path.expanduser("~"), "surf_error.log")

    def hook(etype, value, tb):
        txt = "".join(traceback.format_exception(etype, value, tb))
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(txt + "\n")
        except Exception:
            pass
        if QtWidgets.QApplication.instance() is not None:
            QtWidgets.QMessageBox.critical(None, "SURF - error",
                                           f"{value}\n\nDetails were written to\n{log}")
    sys.excepthook = hook




def main():
    multiprocessing.freeze_support()
    _install_crash_handler()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    for name in ("surf.ico",):
        ico = os.path.join(here, name)
        if os.path.exists(ico):
            app.setWindowIcon(QtGui.QIcon(ico))
            break
    w = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    if os.environ.get("WAVEGUI_SELFTEST"):
        _selftest(w, app)
    sys.exit(app.exec())


def _selftest(w, app):
    """WAVEGUI_SELFTEST=1 python wave_gui.py <folder>: measure -> events -> waves on the first
    video (using its saved session), write selftest.txt into the folder and quit."""
    steps = iter([w._run_measure, w._run_events, w._run_waves])
    out = os.path.join(w.folder or ".", "selftest.txt")
    errors = []
    w.error = lambda msg: errors.append(msg)

    def tick():
        if w._thread is not None:
            return
        if errors:
            finish("FAILED\n" + "\n".join(errors))
            return
        try:
            next(steps)()
        except StopIteration:
            try:     # comparison stack (statsmodels is imported lazily)
                import pandas as pd
                import wave_stats as ws_
                df = pd.DataFrame(dict(animal=list("aabbccdd") * 2, condition=["C", "D"] * 8,
                                       recording=[str(i) for i in range(16)], unit="um", path="",
                                       wave_frequency_per_min=np.r_[np.arange(16) % 3 + 1.0]))
                for k_, _, _ in ws_.METRICS:
                    if k_ not in df:
                        df[k_] = np.nan
                df[ws_.CIRC] = np.nan
                t = ws_.run_stats(df, ws_.StatsOptions(unit="recording", method="mixed", reference="C"))[2]
                mixed = f"mixed model p = {t.p.iloc[0]:.3g}"
            except Exception as e:
                finish(f"FAILED (stats): {e}")
                return
            finish("OK\n" + mixed + "\n" + "\n".join(f"{k}: {v}" for k, v in w.results.summary.items()))

    def finish(txt):
        timer.stop()
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
        app.quit()
    timer = QtCore.QTimer(w, interval=300)
    timer.timeout.connect(tick)
    timer.start()


if __name__ == "__main__":
    main()
