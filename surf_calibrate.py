"""SURF calibration tool - measure a structure of known size in a video (or photo) and get um/px.

Workflow
  1. Open the calibration video (or a photo), choose objective and turret.
  2. Pause on a frame where the structure is sharp ("Go to" a time, or use the slider / play).
  3. Click points on the rim of a circle (3 or more; a circle is fitted live), or two points
     across it (Line mode). Points can be dragged; Backspace removes the last one.
  4. "Add as REFERENCE" with the known diameter (e.g. 3000 um)  ->  um/px for this objective/turret.
     "Add as MEASUREMENT"  ->  the structure's size in um, using the reference for the same
     objective/turret. A measured structure can then be used as the known size at higher
     magnification ("Use row as known size").
  5. The summary shows um/px, video height and turret / objective factors.  "Send to SURF"
     writes the heights and turret factors into SURF's Edit calibration settings.

Every row keeps its frame and clicked points: select it to jump back and check what was marked.
Run standalone:  python surf_calibrate.py [video]     or from SURF:  Tools > Calibration tool.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".wavegui_settings.json")
NOMINAL_TURRETS = [0.5, 1.0, 1.25, 1.5, 2.0]
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
CSV_COLS = ["#", "Structure", "Role", "Time (s)", "Frame", "Objective", "Turret", "Method",
        "Diameter (px)", "Fit RMS (px)", "Known (µm)", "µm/px", "Measured (µm)", "Video height (µm)"]
COLS = ["#", "Structure", "Role", "Size (µm)", "Diameter (px)", "µm/px", "Objective", "Turret",
        "Time (s)", "Frame", "Method", "Fit RMS (px)", "Video height (µm)"]


# ----------------------------------------------------------------------------- geometry
def fit_circle(pts):
    """Least-squares circle through >=3 points -> (cx, cy, r, rms, arc_deg)."""
    p = np.asarray(pts, float)
    x, y = p[:, 0], p[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0))
    try:                                         # geometric refinement
        from scipy.optimize import least_squares
        res = least_squares(lambda q: np.hypot(x - q[0], y - q[1]) - q[2], [cx, cy, r])
        cx, cy, r = res.x
    except Exception:
        pass
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    ang = np.sort(np.mod(np.arctan2(y - cy, x - cx), 2 * np.pi))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    arc = float(np.degrees(2 * np.pi - gaps.max()))      # angular span covered by the points
    return float(cx), float(cy), float(abs(r)), rms, arc


# ----------------------------------------------------------------------------- settings
def load_surf_settings():
    d = {"objectives": {"4x": 0.0, "10x": 0.0, "40x": 0.0}, "turrets": [0.5, 1.0, 1.5, 2.0],
         "turret_factors": {}, "last_scale": None}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            d.update(json.load(f))
    except Exception:
        pass
    return d


def save_surf_settings(d):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


# ----------------------------------------------------------------------------- source
class Source:
    """A video (random access via OpenCV) or a single still image."""

    def __init__(self, path):
        self.path = path
        self.image = None
        if path.lower().endswith(IMG_EXT):
            img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Cannot read {path}")
            self.image, self.n, self.fps = img, 1, 1.0
            self.h, self.w = img.shape[:2]
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = -1

    def read(self, i):
        if self.image is not None:
            return self.image
        i = int(np.clip(i, 0, self.n - 1))
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if not ok:                                   # some codecs overshoot the frame count
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(i - 1, 0))
            ok, fr = self.cap.read()
            if not ok:
                return None
        self.pos = i
        return fr

    def close(self):
        if self.image is None:
            self.cap.release()


def enhance(fr, mode):
    if mode == "Original":
        return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mode == "Auto contrast":
        lo, hi = np.percentile(g, (0.5, 99.5))
        g = (g - lo) / max(hi - lo, 1) * 255
    elif mode == "CLAHE":
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g.astype(np.uint8)).astype(np.float32)
    elif mode == "Edges (high-pass)":
        s = max(fr.shape[:2]) / 150
        hp = g - cv2.GaussianBlur(g, (0, 0), s)
        lo, hi = np.percentile(hp, (1, 99))
        g = (hp - lo) / max(hi - lo, 1e-6) * 255
    return np.clip(g, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- window
class CalibWindow(QtWidgets.QMainWindow):
    def __init__(self, path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SURF – calibration tool")
        self.resize(1500, 900)
        self.src = None
        self.frame_i = 0
        self.raw = None
        self.points = []          # TargetItems
        self.rows = []            # measurement dicts
        self.settings = load_surf_settings()
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        if path:
            self.open(path)

    # ------------------------------------------------------------------ UI
    def _build(self):
        split = QtWidgets.QSplitter()
        # left: image + transport
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        self.circle_item = QtWidgets.QGraphicsEllipseItem()
        self.circle_item.setPen(pg.mkPen((0, 255, 0), width=2, cosmetic=True))
        self.vb.addItem(self.circle_item)
        self.centre_item = pg.ScatterPlotItem(symbol="+", size=14, pen=pg.mkPen((0, 255, 0), width=2))
        self.vb.addItem(self.centre_item)
        self.line_item = pg.PlotCurveItem(pen=pg.mkPen((255, 200, 0), width=2))
        self.vb.addItem(self.line_item)
        # overlay of the selected table row (read-only, cyan)
        self.ov_circle = QtWidgets.QGraphicsEllipseItem()
        self.ov_circle.setPen(pg.mkPen((0, 220, 255), width=2, style=QtCore.Qt.DashLine, cosmetic=True))
        self.ov_circle.setVisible(False)
        self.vb.addItem(self.ov_circle)
        self.ov_pts = pg.ScatterPlotItem(symbol="x", size=10, pen=pg.mkPen((0, 220, 255), width=2), brush=None)
        self.vb.addItem(self.ov_pts)
        self.ov_line = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=2, style=QtCore.Qt.DashLine))
        self.vb.addItem(self.ov_line)
        self.glw.scene().sigMouseClicked.connect(self._clicked)
        ll.addWidget(self.glw, 1)

        tr = QtWidgets.QHBoxLayout()
        st = self.style()
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self._toggle_play)
        tr.addWidget(self.play_btn)
        for txt, d in (("◀◀ 1 s", -1.0), ("◀ frame", None), ("frame ▶", None), ("1 s ▶▶", 1.0)):
            b = QtWidgets.QToolButton()
            b.setText(txt)
            step = (-1 if "◀" in txt else 1) if d is None else d
            b.clicked.connect(lambda _=False, s=step, sec=d is not None: self._step(s, sec))
            tr.addWidget(b)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentText("1×")
        self.speed.currentTextChanged.connect(self._restart_timer)
        tr.addWidget(self.speed)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._seek)
        tr.addWidget(self.slider, 1)
        self.time_lbl = QtWidgets.QLabel("–")
        self.time_lbl.setMinimumWidth(170)
        tr.addWidget(self.time_lbl)
        tr.addWidget(QtWidgets.QLabel("Go to (s):"))
        self.goto = QtWidgets.QDoubleSpinBox()
        self.goto.setDecimals(2)
        self.goto.setMaximum(1e6)
        self.goto.setKeyboardTracking(False)
        self.goto.valueChanged.connect(lambda t: self.slider.setValue(int(round(t * self._fps()))))
        tr.addWidget(self.goto)
        ll.addLayout(tr)
        hint = QtWidgets.QLabel("Left-click: add point · drag a point to move it · Backspace: remove last "
                                "point · wheel: zoom · drag background: pan · Space: play/pause")
        hint.setStyleSheet("color: gray")
        ll.addWidget(hint)
        split.addWidget(left)

        # right: controls + tables
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        b = QtWidgets.QPushButton("Open video / photo…")
        b.clicked.connect(self._choose)
        rl.addWidget(b)
        self.file_lbl = QtWidgets.QLabel("no file")
        self.file_lbl.setWordWrap(True)
        rl.addWidget(self.file_lbl)

        g = QtWidgets.QGroupBox("Microscope")
        f = QtWidgets.QFormLayout(g)
        self.obj = QtWidgets.QComboBox()
        self.obj.setEditable(True)
        self.obj.addItems(list(self.settings["objectives"]) or ["4x", "10x", "40x"])
        b_obj = QtWidgets.QPushButton("Add objective…")
        b_obj.setToolTip("Add another objective (e.g. 20x, 60x). It is added to SURF's calibration "
                         "when you press 'Send to SURF calibration'.")
        b_obj.clicked.connect(self._add_objective)
        oh = QtWidgets.QHBoxLayout()
        oh.addWidget(self.obj, 1)
        oh.addWidget(b_obj)
        f.addRow("Objective", oh)
        self.tur = QtWidgets.QComboBox()
        self.tur.setEditable(True)
        self.tur.addItems([f"{t:g}" for t in NOMINAL_TURRETS])
        self.tur.setCurrentText("1")
        f.addRow("Turret (label)", self.tur)
        self.view_mode = QtWidgets.QComboBox()
        self.view_mode.addItems(["Original", "Auto contrast", "CLAHE", "Edges (high-pass)"])
        self.view_mode.currentTextChanged.connect(lambda _: self._show())
        f.addRow("Display", self.view_mode)
        rl.addWidget(g)

        g = QtWidgets.QGroupBox("Mark")
        v = QtWidgets.QVBoxLayout(g)
        h = QtWidgets.QHBoxLayout()
        self.m_circle = QtWidgets.QRadioButton("Circle (≥3 rim points)")
        self.m_line = QtWidgets.QRadioButton("Line (2 points across)")
        self.m_circle.setChecked(True)
        self.m_circle.toggled.connect(lambda _: self._update_fit())
        h.addWidget(self.m_circle)
        h.addWidget(self.m_line)
        v.addLayout(h)
        h = QtWidgets.QHBoxLayout()
        for txt, fn in (("Undo point", self._undo_point), ("Clear points", self._clear_points),
                        ("Reset zoom", self._reset_zoom)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            h.addWidget(b)
        v.addLayout(h)
        self.fit_lbl = QtWidgets.QLabel("no points")
        self.fit_lbl.setWordWrap(True)
        self.fit_lbl.setStyleSheet("font-weight: bold")
        v.addWidget(self.fit_lbl)
        f = QtWidgets.QFormLayout()
        self.struct = QtWidgets.QLineEdit("3 mm circle")
        f.addRow("Structure name", self.struct)
        self.known = QtWidgets.QDoubleSpinBox()
        self.known.setRange(0, 1e6)
        self.known.setDecimals(2)
        self.known.setValue(3000)
        self.known.setSuffix(" µm")
        f.addRow("Known diameter", self.known)
        v.addLayout(f)
        h = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Add as REFERENCE (known size)")
        b.setStyleSheet("background:#2e7d32; color:white; padding:5px")
        b.clicked.connect(lambda: self._add("reference"))
        h.addWidget(b)
        b = QtWidgets.QPushButton("Add as MEASUREMENT")
        b.setStyleSheet("background:#1565c0; color:white; padding:5px")
        b.clicked.connect(lambda: self._add("measurement"))
        h.addWidget(b)
        v.addLayout(h)
        self.live_lbl = QtWidgets.QLabel("")
        self.live_lbl.setWordWrap(True)
        v.addWidget(self.live_lbl)
        rl.addWidget(g)

        rl.addWidget(QtWidgets.QLabel("Marked structures – select a row to jump back and see the points. "
                                     "To reuse a measured size, select it and press <b>Use row as known size</b>."))
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._row_selected)
        rl.addWidget(self.table, 2)
        h = QtWidgets.QHBoxLayout()
        for txt, fn in (("Use row as known size", self._use_as_known), ("Delete row", self._delete_row),
                        ("Export CSV…", self._export)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            h.addWidget(b)
        rl.addLayout(h)

        rl.addWidget(QtWidgets.QLabel("Summary (references averaged per objective × turret):"))
        self.summary = QtWidgets.QTableWidget(0, 7)
        self.summary.setHorizontalHeaderLabels(["Objective", "Turret", "n ref", "µm/px", "Video height (µm)",
                                                "Turret factor vs 1×", "Objective factor vs 4× (1×)"])
        self.summary.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rl.addWidget(self.summary, 1)
        b = QtWidgets.QPushButton("Send to SURF calibration…")
        b.clicked.connect(self._send_to_surf)
        rl.addWidget(b)
        split.addWidget(right)
        split.setSizes([950, 550])
        self.setCentralWidget(split)

        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, self._toggle_play)
        QtGui.QShortcut(QtGui.QKeySequence("Backspace"), self, self._undo_point)
        QtGui.QShortcut(QtGui.QKeySequence("Left"), self, lambda: self._step(-1, False))
        QtGui.QShortcut(QtGui.QKeySequence("Right"), self, lambda: self._step(1, False))

    # ------------------------------------------------------------------ file / frames
    def _choose(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open calibration video or photo", "",
            "Video / image (*.mp4 *.MP4 *.avi *.mov *.tif *.tiff *.jpg *.jpeg *.png *.bmp);;All files (*)")
        if p:
            self.open(p)

    def open(self, path):
        self._save_session()
        self.timer.stop()
        if self.src:
            self.src.close()
        try:
            self.src = Source(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open", str(e))
            self.src = None
            return
        s = self.src
        self.file_lbl.setText(f"{os.path.basename(path)}\n{s.w}×{s.h} px · {s.n} frames · {s.fps:.2f} fps "
                              f"· {s.n / s.fps:.1f} s")
        self.slider.blockSignals(True)
        self.slider.setRange(0, s.n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.goto.setMaximum(s.n / s.fps)
        self.rows = []
        self._clear_points()
        self._load_session()
        self._seek(0)
        self._reset_zoom()
        self._refresh_tables()

    def _fps(self):
        return self.src.fps if self.src else 25.0

    def _seek(self, i):
        if not self.src:
            return
        self.frame_i = int(i)
        fr = self.src.read(self.frame_i)
        if fr is None:
            return
        self.raw = fr
        self._show()
        t = self.frame_i / self._fps()
        self.time_lbl.setText(f"{t:7.2f} s  ·  frame {self.frame_i}/{self.src.n - 1}")

    def _show(self):
        if self.raw is not None:
            self.img.setImage(enhance(self.raw, self.view_mode.currentText()), autoLevels=False,
                              levels=(0, 255))

    def _reset_zoom(self):
        if self.src:
            self.vb.setRange(QtCore.QRectF(0, 0, self.src.w, self.src.h), padding=0.02)

    def _step(self, s, seconds):
        if self.src:
            n = int(round(s * self._fps())) if seconds else int(s)
            self.slider.setValue(int(np.clip(self.frame_i + n, 0, self.src.n - 1)))

    def _toggle_play(self):
        if not self.src or self.src.n < 2:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        else:
            self._restart_timer()
            self.play_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))

    def _restart_timer(self, *_):
        if self.src and (self.timer.isActive() or self.sender() is not self.speed):
            sp = float(self.speed.currentText().rstrip("×"))
            self.timer.start(max(1, int(1000 / (self._fps() * sp))))

    def _tick(self):
        if self.frame_i >= self.src.n - 1:
            self._toggle_play()
            return
        self.slider.setValue(self.frame_i + 1)

    # ------------------------------------------------------------------ points
    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or not self.src:
            return
        if any(isinstance(it, pg.TargetItem) for it in self.glw.scene().items(ev.scenePos())):
            return                                           # clicked on an existing point
        p = self.vb.mapSceneToView(ev.scenePos())
        self._show_overlay(None)
        if self.m_line.isChecked() and len(self.points) >= 2:
            self._clear_points()
        self._add_point(p.x(), p.y())

    def _add_point(self, x, y):
        t = pg.TargetItem(pos=(x, y), size=12, symbol="+", movable=True,
                          pen=pg.mkPen((255, 60, 60), width=2), hoverPen=pg.mkPen((255, 255, 0), width=2))
        t.sigPositionChanged.connect(self._update_fit)
        self.vb.addItem(t)
        self.points.append(t)
        self._update_fit()

    def _undo_point(self):
        if self.points:
            self.vb.removeItem(self.points.pop())
            self._update_fit()

    def _clear_points(self):
        for t in self.points:
            self.vb.removeItem(t)
        self.points = []
        self._update_fit()

    def _pts(self):
        return [(t.pos().x(), t.pos().y()) for t in self.points]

    def _current(self):
        """Current marking -> dict or None."""
        pts = self._pts()
        if self.m_line.isChecked():
            if len(pts) < 2:
                return None
            (x0, y0), (x1, y1) = pts[:2]
            return {"method": "line", "diam_px": float(np.hypot(x1 - x0, y1 - y0)), "rms": None,
                    "arc": None, "points": pts[:2]}
        if len(pts) < 3:
            return None
        cx, cy, r, rms, arc = fit_circle(pts)
        return {"method": "circle", "diam_px": 2 * r, "rms": rms, "arc": arc, "cx": cx, "cy": cy,
                "points": pts}

    def _update_fit(self, *_):
        self.circle_item.setVisible(False)
        self.centre_item.setData([], [])
        self.line_item.setData([], [])
        c = self._current()
        n = len(self.points)
        if c is None:
            need = 2 if self.m_line.isChecked() else 3
            self.fit_lbl.setText(f"{n} point(s) – need {need}" if n else "no points")
            self.live_lbl.setText("")
            return
        if c["method"] == "circle":
            r = c["diam_px"] / 2
            self.circle_item.setRect(c["cx"] - r, c["cy"] - r, 2 * r, 2 * r)
            self.circle_item.setVisible(True)
            self.centre_item.setData([c["cx"]], [c["cy"]])
            warn = ""
            if c["arc"] < 180:
                warn = "  ⚠ points cover <180° of the rim – spread them around"
            elif n < 6:
                warn = "  (6–12 points spread around the rim are more accurate)"
            self.fit_lbl.setText(f"Circle: diameter {c['diam_px']:.1f} px · {n} pts · fit RMS {c['rms']:.2f} px "
                                 f"· arc {c['arc']:.0f}°{warn}")
        else:
            (x0, y0), (x1, y1) = c["points"]
            self.line_item.setData([x0, x1], [y0, y1])
            self.fit_lbl.setText(f"Line: {c['diam_px']:.1f} px")
        sc = self._scale_for(self.obj.currentText(), self._tur())
        txt = []
        if self.known.value() > 0:
            u = self.known.value() / c["diam_px"]
            txt.append(f"as reference: {u:.4f} µm/px → video height {u * self.src.h:.0f} µm")
        if sc:
            txt.append(f"as measurement: {c['diam_px'] * sc[0]:.1f} µm (using {sc[0]:.4f} µm/px)")
        self.live_lbl.setText("\n".join(txt))

    # ------------------------------------------------------------------ rows
    def _add_objective(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add objective", "Objective name (e.g. 20x):")
        name = name.strip()
        if ok and name:
            if self.obj.findText(name) < 0:
                self.obj.addItem(name)
            self.obj.setCurrentText(name)

    def _tur(self):
        try:
            return float(self.tur.currentText().replace("×", "").replace(",", "."))
        except ValueError:
            return 1.0

    def _scale_for(self, obj, tur):
        v = [r["um_per_px"] for r in self.rows if r["role"] == "reference" and r["objective"] == obj
             and abs(r["turret"] - tur) < 1e-6]
        return (float(np.mean(v)), len(v)) if v else None

    def _add(self, role):
        c = self._current()
        if c is None or not self.src:
            QtWidgets.QMessageBox.information(self, "Add", "Mark the structure first.")
            return
        obj, tur = self.obj.currentText(), self._tur()
        row = dict(c, role=role, structure=self.struct.text().strip() or "structure",
                   frame=self.frame_i, time=self.frame_i / self._fps(), objective=obj, turret=tur,
                   file=os.path.basename(self.src.path), H=self.src.h)
        if role == "reference":
            if self.known.value() <= 0:
                QtWidgets.QMessageBox.information(self, "Add", "Enter the known diameter.")
                return
            row["known_um"] = self.known.value()
            row["um_per_px"] = row["known_um"] / row["diam_px"]
        else:
            sc = self._scale_for(obj, tur)
            if not sc:
                QtWidgets.QMessageBox.information(
                    self, "Add", f"No reference yet for {obj} at turret {tur:g}.\n"
                                 "Add a reference of known size at this setting first.")
                return
            row["um_per_px"] = sc[0]
            row["measured_um"] = row["diam_px"] * sc[0]
        self.rows.append(row)
        self._clear_points()
        self._refresh_tables()
        self.table.blockSignals(True)
        self.table.selectRow(len(self.rows) - 1)
        self.table.blockSignals(False)
        self._show_overlay(row)
        self._save_session()

    def _recompute(self):
        """Measurements follow the (averaged) reference of their setting."""
        for r in self.rows:
            if r["role"] == "measurement":
                sc = self._scale_for(r["objective"], r["turret"])
                r["um_per_px"] = sc[0] if sc else None
                r["measured_um"] = r["diam_px"] * sc[0] if sc else None

    def _refresh_tables(self):
        self._recompute()
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            u = r.get("um_per_px")
            if r["role"] == "reference":
                size = f"{r['known_um']:g} (known)"
            else:
                size = "" if r.get("measured_um") is None else f"{r['measured_um']:.1f}"
            rms = "" if r.get("rms") is None else f"{r['rms']:.2f}"
            if r["method"] == "circle" and len(r["points"]) == 3:
                rms = "– (3 pts)"
            vals = [i + 1, r["structure"], r["role"], size, f"{r['diam_px']:.1f}",
                    "" if u is None else f"{u:.4f}", r["objective"], f"{r['turret']:g}",
                    f"{r['time']:.2f}", r["frame"], r["method"], rms,
                    "" if u is None else f"{u * r['H']:.0f}"]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if r["role"] == "reference":
                    it.setBackground(QtGui.QColor(46, 125, 50, 50))
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)

        keys = sorted({(r["objective"], r["turret"]) for r in self.rows if r["role"] == "reference"},
                      key=lambda k: (float("".join(ch for ch in k[0] if ch.isdigit() or ch == ".") or 0), k[1]))
        H = self.src.h if self.src else 1080
        base4 = self._scale_for("4x", 1.0)
        self.summary.setRowCount(len(keys))
        for i, (o, t) in enumerate(keys):
            u, n = self._scale_for(o, t)
            one = self._scale_for(o, 1.0)
            vals = [o, f"{t:g}", n, f"{u:.4f}", f"{u * H:.0f}",
                    f"{one[0] / u:.3f}" if one else "",
                    f"{base4[0] / one[0]:.3f}" if (base4 and one and abs(t - 1) < 1e-6) else ""]
            for j, v in enumerate(vals):
                self.summary.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))
        self.summary.resizeColumnsToContents()
        self._update_fit()

    def _sel(self):
        r = self.table.currentRow()
        return r if 0 <= r < len(self.rows) else None

    def _row_selected(self):
        i = self._sel()
        if i is None:
            return
        r = self.rows[i]
        self.timer.stop()
        self._clear_points()
        self.slider.setValue(r["frame"])
        self.obj.setCurrentText(r["objective"])
        self.tur.setCurrentText(f"{r['turret']:g}")
        self._show_overlay(r)
        self.fit_lbl.setText(f"Showing row {i + 1} ({r['structure']}, cyan): {r['diam_px']:.1f} px. "
                             "Click to start a new marking.")

    def _show_overlay(self, r=None):
        self.ov_circle.setVisible(False)
        self.ov_pts.setData([], [])
        self.ov_line.setData([], [])
        if not r:
            return
        xs, ys = zip(*r["points"])
        self.ov_pts.setData(list(xs), list(ys))
        if r["method"] == "circle":
            d = r["diam_px"]
            self.ov_circle.setRect(r["cx"] - d / 2, r["cy"] - d / 2, d, d)
            self.ov_circle.setVisible(True)
        else:
            self.ov_line.setData(list(xs), list(ys))

    def _use_as_known(self):
        i = self._sel()
        if i is None:
            return
        r = self.rows[i]
        um = r.get("measured_um") or r.get("known_um")
        if um:
            self.known.setValue(um)
            self.struct.setText(r["structure"])
            self._clear_points()

    def _delete_row(self):
        i = self._sel()
        if i is not None:
            del self.rows[i]
            self._clear_points()
            self._refresh_tables()
            self._save_session()

    # ------------------------------------------------------------------ persistence / export
    def _session_path(self):
        return os.path.splitext(self.src.path)[0] + "_calib.json" if self.src else None

    def _save_session(self):
        p = self._session_path()
        if p and self.rows:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"rows": self.rows}, f, indent=1)
            except Exception:
                pass

    def _load_session(self):
        p = self._session_path()
        if p and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    self.rows = json.load(f).get("rows", [])
                if self.rows:
                    self.obj.setCurrentText(self.rows[-1]["objective"])
                    self.tur.setCurrentText(f"{self.rows[-1]['turret']:g}")
            except Exception:
                self.rows = []

    def _export(self):
        if not self.rows:
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export", os.path.splitext(self.src.path)[0] + "_calib.csv", "CSV (*.csv)")
        if not p:
            return
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file"] + CSV_COLS[1:] + ["points (x;y)"])
            for r in self.rows:
                u = r.get("um_per_px")
                w.writerow([r["file"], r["structure"], r["role"], f"{r['time']:.3f}", r["frame"], r["objective"],
                            r["turret"], r["method"], f"{r['diam_px']:.2f}", r.get("rms") or "",
                            r.get("known_um") or "", u or "", r.get("measured_um") or "",
                            u * r["H"] if u else "", " ".join(f"{x:.1f};{y:.1f}" for x, y in r["points"])])

    def _send_to_surf(self):
        H = self.src.h if self.src else 1080
        objs = {o for o, _ in {(r["objective"], r["turret"]) for r in self.rows if r["role"] == "reference"}}
        heights = {o: self._scale_for(o, 1.0)[0] * H for o in objs if self._scale_for(o, 1.0)}
        facs = {}                                     # nominal turret -> factors from every objective
        for o in objs:
            one = self._scale_for(o, 1.0)
            for t in NOMINAL_TURRETS:
                s = self._scale_for(o, t)
                if one and s and abs(t - 1) > 1e-6:
                    facs.setdefault(t, []).append(one[0] / s[0])
        if not heights and not facs:
            QtWidgets.QMessageBox.information(self, "Send to SURF",
                                              "Need at least one reference at turret 1×.")
            return
        cur = load_surf_settings()
        par = self.parent()
        if par is not None and hasattr(par, "settings"):
            cur = dict(par.settings)                     # the running SURF holds the live settings
        mean_f = {t: round(float(np.mean(v)), 3) for t, v in facs.items()}
        # turret labels stay the nominal values; the measured step is stored behind each label.
        # Labels that are not nominal values (e.g. a factor typed in earlier) are kept as they are.
        new_t = sorted(set(cur["turrets"]) | {1.0} | set(mean_f))
        new_f = dict(cur.get("turret_factors") or {})
        new_f.update({f"{t:g}": f for t, f in mean_f.items()})
        msg = ["This will update SURF's Edit calibration:\n"]
        for o, h in heights.items():
            msg.append(f"  {o}: video height at turret 1× = {h:.0f} µm  (was {cur['objectives'].get(o, 0):g})")
        for t, v in sorted(facs.items()):
            msg.append(f"  turret {t:g} → measured factor {mean_f[t]:.3f}  (from {len(v)} objective(s))")
        msg.append("\nThe turret menu keeps its labels (1×, 1.25×, …); SURF uses the measured factor "
                   "behind each label, also for videos you set up earlier.")
        if QtWidgets.QMessageBox.question(self, "Send to SURF", "\n".join(msg)) != QtWidgets.QMessageBox.Yes:
            return
        cur["objectives"] = dict(cur["objectives"])
        cur["objectives"].update({o: round(h, 1) for o, h in heights.items()})
        cur["turrets"], cur["turret_factors"] = new_t, new_f
        try:
            save_surf_settings(cur)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Send to SURF", f"Could not write settings:\n{e}")
            return
        if par is not None and hasattr(par, "_fill_scale_combos"):
            keep = par._scale_dict()
            par.settings = cur
            par._fill_scale_combos()
            par._set_scale_widgets(keep)
        QtWidgets.QMessageBox.information(self, "Send to SURF", "SURF calibration updated.")

    def closeEvent(self, ev):
        self._save_session()
        self.timer.stop()
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major")
    w = CalibWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    w.show()
    if os.environ.get("SURF_CALIB_SELFTEST"):
        QtCore.QTimer.singleShot(1500, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
