"""Tab 5 - single cells (aimed at 40x): find somata (Suite2p / Cellpose / FIJI / by hand), review them,
measure neuropil-corrected traces, detect events with the notebook rule and link them to the
waves of the grid analysis (time only, or time + distance to the wave centre)."""
from __future__ import annotations

import os
import pickle
import sys
import traceback
from dataclasses import asdict

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

import cells_core as cc
import wave_core as wc

PARAM_META_CELLS = {
    "neuropil_r": ("Neuropil factor r", "F = F_cell − r · F_neuropil (Suite2p standard 0.7; 0 = off)", 0, 2, 0.05, 2),
    "baseline_t0": ("Baseline from (s)", "F0 = minimum of the raw trace in the baseline window", 0, 1e5, 1, 2),
    "baseline_t1": ("Baseline to (s)", "", 0, 1e5, 1, 2),
    "min_f0": ("Skip cells dimmer than (grey)", "notebook 'cutoff_threshold'", 0, 255, 0.5, 2),
    "method": ("Peak rule (0 notebook, 1 prominence, 2 grid)", "2 = the grid rule of the Events tab (tab 3): "
               "Gaussian σ, prominence, min. distance and start exclusion are taken from there", 0, 2, 1, 0),
    "grid_prom_unit": ("Grid rule prominence (0 % ΔF/F, 1 grey)", "0 = the grid's grey-level prominence "
                       "converted to % ΔF/F at the typical grid-square brightness, so each cell is judged against "
                       "its own brightness (recommended); 1 = exactly the grid number in grey levels", 0, 1, 1, 0),
    "threshold_pct": ("Threshold (% of F0)", "notebook thresh / baselineCorr (0.8 / 40 = 2 %)", 0, 1000, 0.1, 3),
    "drop_window_s": ("…trace must fall within (s)", "peak above the minimum of the following window", 0.05, 30, 0.1, 2),
    "use_around_mean": ("Also above mean of ±window (0/1)", "one-cell notebook: on; 40x wave notebook: off", 0, 1, 1, 0),
    "around_window_s": ("±window for the mean (s)", "", 0.1, 60, 0.5, 2),
    "local_max_s": ("Local maximum over ± (s)", "one-cell 0.02 s, 40x wave notebook 0.5 s", 0.0, 10, 0.01, 3),
    "short_window_s": ("Short window (s)", "≥ mean of next / previous short window + ε", 0.01, 10, 0.01, 3),
    "long_window_s": ("Long window (s)", "≥ mean of next or previous long window + ε", 0.01, 10, 0.05, 3),
    "eps_pct": ("ε (% of F0)", "", 0, 100, 0.01, 3),
    "min_sep_s": ("Merge peaks closer than (s)", "", 0.0, 60, 0.05, 3),
    "prominence_pct": ("Prominence (% of F0, rule 1)", "", 0, 1000, 0.5, 2),
    "diameter_um": ("Cell diameter (µm)", "detection scale, neuropil ring, circles added by hand", 1, 500, 1, 1),
    "link_mode": ("Link to waves (0 time, 1 time+distance)", "0 = notebook: inside the wave's time window; "
                  "1 = also within the radius of the wave centre", 0, 1, 1, 0),
    "link_radius_um": ("Link radius (µm)", "", 1, 1e5, 10, 1),
    "link_tracks": ("Waves = (0 long tracks, 1 all tracks)", "0 = only the tracks longer than tab 4's "
                    "'Long track: > steps' count as waves; 1 = every track that survived tab 4's filters (short "
                    "wave patches too – useful at 40×, where many waves cross the field quickly)", 0, 1, 1, 0),
    "link_pad_s": ("Wave window padding (s)", "", 0, 60, 0.1, 2),
}


ADDON_DEPS = ["torch", "torchvision", "numba", "natsort", "tifffile", "tqdm", "fastremap", "imagecodecs",
              "roifile", "fill-voids", "segment_anything", "importlib-metadata"]


def _install_deps(parent, on_done):
    """Install Suite2p + Cellpose (+ PyTorch) into SURF's environment without touching OpenCV.
    Windows: in a separate window after SURF closes (loaded packages are locked)."""
    if getattr(sys, "frozen", False):
        QtWidgets.QMessageBox.information(parent, "Install", "The standalone .exe cannot install packages. "
                                          "Use the normal installer (install_windows.bat) for cell detection.")
        return
    if sys.platform.startswith("win"):
        script = os.path.join(os.environ.get("LOCALAPPDATA", ""), "WaveAnalysis", "install_cells_addon.ps1")
        if not os.path.exists(script):
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "install", "install_cells_addon.ps1")
            script = here if os.path.exists(here) else ""
        if not script:
            QtWidgets.QMessageBox.information(parent, "Install", "Run install_cells_addon.bat from the SURF "
                                              "folder (with SURF closed).")
            return
        if QtWidgets.QMessageBox.question(
                parent, "Install Suite2p + Cellpose",
                "SURF will close and an installer window opens (about 1–3 GB download, several minutes). "
                "Windows locks files that SURF is using, so the add-on is installed while SURF is closed; "
                "SURF starts again when it is done.\n\nSave your work first. Continue?") != QtWidgets.QMessageBox.Yes:
            return
        restart = sys.executable + "|" + os.path.abspath(sys.argv[0])
        exe = sys.executable
        args = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-WaitForPid", str(os.getpid())]
        # restart SURF through its own python (pythonw) + wave_gui.py
        launcher = os.path.join(os.path.dirname(script), "restart_surf.cmd")
        try:
            with open(launcher, "w") as fh:
                fh.write(f'@start "" "{exe}" "{os.path.abspath(sys.argv[0])}"\n')
            args += ["-RestartWith", launcher]
        except OSError:
            pass
        # a NEW CONSOLE so the installer window is visible (SURF runs without a console)
        import subprocess
        try:
            subprocess.Popen(["powershell.exe"] + args, creationflags=0x00000010 | 0x00000200,
                             close_fds=True, cwd=os.path.dirname(script))     # CREATE_NEW_CONSOLE | NEW_PROCESS_GROUP
            ok = True
        except Exception:
            ok = False
        if not ok:
            QtWidgets.QMessageBox.warning(parent, "Install", "Could not start the installer. Close SURF and run "
                                          "install_cells_addon.bat from the SURF folder.")
            return
        win = parent.window()
        main = getattr(parent, "main", None)
        try:
            if main is not None:
                main.save_session()
                main.save_app_settings()
        except Exception:
            pass
        QtWidgets.QApplication.instance().quit()
        return
    # macOS / Linux: no file locking, install in place
    import platform
    if sys.platform == "darwin" and platform.machine() == "x86_64" and sys.version_info >= (3, 13):
        QtWidgets.QMessageBox.warning(
            parent, "Install Suite2p + Cellpose",
            "This is an Intel Mac with Python %d.%d. PyTorch (needed by Suite2p and Cellpose) has no build for "
            "that combination – the last Intel-Mac PyTorch supports Python up to 3.12.\n\n"
            "Install Python 3.12 (python.org), delete ~/Library/Application Support/WaveAnalysis/venv and run "
            "install_mac.command again; SURF then uses 3.12 and the add-on installs." % sys.version_info[:2])
        return
    if QtWidgets.QMessageBox.question(
            parent, "Install Suite2p + Cellpose",
            "This downloads Suite2p, Cellpose and PyTorch into SURF's environment – about 1–3 GB, several "
            "minutes. Continue?") != QtWidgets.QMessageBox.Yes:
        return
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Installing Suite2p + Cellpose")
    dlg.resize(720, 420)
    lay = QtWidgets.QVBoxLayout(dlg)
    log = QtWidgets.QPlainTextEdit()
    log.setReadOnly(True)
    lay.addWidget(log)
    btn = QtWidgets.QPushButton("Close")
    btn.setEnabled(False)
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)
    py = sys.executable
    steps = [[py, "-m", "pip", "install", "--disable-pip-version-check"] + ADDON_DEPS,
             [py, "-m", "pip", "install", "--disable-pip-version-check", "--no-deps",
              "suite2p>=1.0,<2", "cellpose>=4,<5"]]
    proc = QtCore.QProcess(dlg)
    proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
    proc.readyRead.connect(lambda: log.appendPlainText(bytes(proc.readAll()).decode(errors="replace").rstrip()))

    def nxt(*_):
        if steps:
            cmd = steps.pop(0)
            log.appendPlainText("> " + " ".join(cmd))
            proc.start(cmd[0], cmd[1:])
        else:
            log.appendPlainText("\nDone. Restart SURF to use the new packages.")
            btn.setEnabled(True)
            on_done()

    proc.finished.connect(nxt)
    nxt()
    dlg.exec()


def _importable(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


class BrushViewBox(pg.ViewBox):
    """ViewBox that hands mouse drags to a callback while the brush tool is active (no panning)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.drag_cb = None
        self.space_pan = False          # Space held: normal panning even in brush mode

    def mouseDragEvent(self, ev, axis=None):
        if self.space_pan and ev.button() == Qt.LeftButton:
            ev.accept()                 # pan like the default left-drag
            dif = self.mapToView(ev.lastPos()) - self.mapToView(ev.pos())
            self.translateBy(x=dif.x(), y=dif.y())
            return
        if self.drag_cb is not None and ev.button() in (Qt.LeftButton, Qt.RightButton):
            ev.accept()
            self.drag_cb(ev)
            return
        super().mouseDragEvent(ev, axis)


class _CopyCellsDialog(QtWidgets.QDialog):
    def __init__(self, tab, name, cells, src_size, here_size, shift, note, have_cells):
        super().__init__(tab)
        self.setWindowTitle("Copy cells")
        self.tab, self.cells = tab, cells
        lay = QtWidgets.QVBoxLayout(self)
        acc = sum(c.accepted for c in cells)
        lay.addWidget(QtWidgets.QLabel(f"<b>{len(cells)} cells</b> ({acc} accepted) from <b>{name}</b>"))
        if tuple(src_size) != tuple(here_size):
            w = QtWidgets.QLabel(f"⚠ Frame size differs: source {src_size[0]}×{src_size[1]} px, this video "
                                 f"{here_size[0]}×{here_size[1]} px (different rotation/crop?). Cells outside "
                                 "the frame are dropped.")
            w.setWordWrap(True)
            w.setStyleSheet("color:#b36b00")
            lay.addWidget(w)
        self.replace = QtWidgets.QRadioButton("Replace the cells of this video")
        self.add = QtWidgets.QRadioButton("Add to the cells of this video")
        self.replace.setChecked(True)
        if have_cells:
            lay.addWidget(self.replace)
            lay.addWidget(self.add)
        self.keep_state = QtWidgets.QCheckBox("keep accepted / rejected as in the source")
        self.keep_state.setChecked(True)
        lay.addWidget(self.keep_state)
        box = QtWidgets.QGroupBox("Shift (the field of view may have moved between recordings)")
        g = QtWidgets.QGridLayout(box)
        self.align = QtWidgets.QCheckBox("Line up automatically with this recording's image")
        dx, dy, resp = shift
        self._auto = (dx, dy)
        self.dx = QtWidgets.QDoubleSpinBox(); self.dy = QtWidgets.QDoubleSpinBox()
        for s_ in (self.dx, self.dy):
            s_.setRange(-5000, 5000); s_.setDecimals(1); s_.setSuffix(" px")
        g.addWidget(self.align, 0, 0, 1, 4)
        g.addWidget(QtWidgets.QLabel("x"), 1, 0); g.addWidget(self.dx, 1, 1)
        g.addWidget(QtWidgets.QLabel("y"), 1, 2); g.addWidget(self.dy, 1, 3)
        if note:
            self.align.setEnabled(False)
            info = note
        else:
            good = resp > 0.05
            info = (f"Detected shift ({dx:+.1f}, {dy:+.1f}) px, match quality {resp:.2f}"
                    + ("" if good else " – low, probably not the same field; check by eye"))
            self.align.setChecked(good and (abs(dx) >= 0.5 or abs(dy) >= 0.5))
        il = QtWidgets.QLabel(info)
        il.setWordWrap(True)
        il.setStyleSheet("color: gray")
        g.addWidget(il, 2, 0, 1, 4)
        self.align.toggled.connect(self._align_toggled)
        self._align_toggled(self.align.isChecked())
        lay.addWidget(box)
        lay.addWidget(QtWidgets.QLabel("<span style='color:gray'>The shift can also be typed in (+x = right, "
                                       "+y = down).</span>"))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Copy cells")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _align_toggled(self, on):
        self.dx.setValue(self._auto[0] if on else 0.0)
        self.dy.setValue(self._auto[1] if on else 0.0)

    def shift(self):
        return self.dx.value(), self.dy.value()


class CellsTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.cells: list[cc.Cell] = []
        self.dm_images = {}            # mean / max / std of the detection movie (full-res coords)
        self.Fc = self.Fn = None
        self.trace_sig = None
        self.result: cc.CellResults | None = None
        self.sel = -1
        self.label_map = None
        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self):
        from wave_gui import ParamForm, PARAM_META, MplCanvas, group, hrow
        for k, v in PARAM_META_CELLS.items():      # shared names (sigma, t_start, t_end) keep their labels
            PARAM_META.setdefault(k, v)
        outer = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter()
        outer.addWidget(split)
        # ---- left: image + plots
        left = QtWidgets.QSplitter(Qt.Vertical)
        img_w = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(img_w)
        il.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#151515")
        self.vb = BrushViewBox(lockAspect=True, invertY=True)
        self.glw.addItem(self.vb)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        self.c_acc = pg.PlotCurveItem(pen=pg.mkPen((0, 230, 0), width=1.2))
        self.c_rej = pg.PlotCurveItem(pen=pg.mkPen((230, 60, 60), width=1, style=Qt.DashLine))
        self.c_sel = pg.PlotCurveItem(pen=pg.mkPen((255, 230, 0), width=2.5))
        self.c_poly = pg.PlotCurveItem(pen=pg.mkPen((0, 220, 255), width=1.5, style=Qt.DashLine))
        self.s_poly = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(0, 220, 255), pen=None)
        for c in (self.c_acc, self.c_rej, self.c_sel, self.c_poly, self.s_poly):
            self.vb.addItem(c)
        self.c_mask = pg.PlotCurveItem(pen=pg.mkPen((0, 200, 255), width=1, style=Qt.DotLine))
        self.vb.addItem(self.c_mask)
        self.brush_circle = QtWidgets.QGraphicsEllipseItem()
        self.brush_circle.setPen(pg.mkPen((255, 140, 0), width=1.5, cosmetic=True))
        self.brush_circle.setVisible(False)
        self.vb.addItem(self.brush_circle)
        self.c_brush = pg.PlotCurveItem(pen=pg.mkPen((255, 140, 0), width=2))
        self.vb.addItem(self.c_brush)
        self._stroke = None
        self.glw.scene().sigMouseMoved.connect(self._brush_cursor)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.poly_pts = []
        self.edit_roi = None
        self.selset = set()
        self.glw.scene().sigMouseClicked.connect(self._img_clicked)
        il.addWidget(self.glw, 1)
        self.bg_combo = QtWidgets.QComboBox()
        self.bg_combo.addItems(["Mean image", "Max projection", "Std (activity)", "Video frame"])
        self.bg_combo.currentIndexChanged.connect(self._show_bg)
        self.frame_spin = QtWidgets.QDoubleSpinBox()
        self.frame_spin.setRange(0, 1e6)
        self.frame_spin.setSuffix(" s")
        self.frame_spin.setValue(10)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self._show_bg)
        self.mode_sel = QtWidgets.QRadioButton("Select")
        self.mode_poly = QtWidgets.QRadioButton("Draw polygon")
        self.mode_add = QtWidgets.QRadioButton("Add circle")
        self.mode_brush = QtWidgets.QRadioButton("Brush")
        self.mode_brush.setToolTip("Like FIJI's selection brush: drag starting inside the selected cell adds, "
                                   "starting outside it erases; right-drag always erases; with no cell selected "
                                   "painting makes a new cell")
        self.brush_size = QtWidgets.QDoubleSpinBox()
        self.brush_size.setRange(0.2, 200)
        self.brush_size.setValue(3.0)
        self.brush_size.setSuffix(" µm")
        self.brush_size.setToolTip("Brush radius")
        self.brush_size.valueChanged.connect(lambda *_: self._brush_cursor(None))
        self.mode_brush.toggled.connect(self._brush_mode)
        self.brush_new = QtWidgets.QCheckBox("new cell from empty space")
        self.brush_new.setChecked(True)
        self.brush_new.setToolTip("On: a left-drag that starts outside the current cell paints a NEW cell (right-drag "
                                  "erases). Off: FIJI behaviour – starting outside the current cell erases from it.")
        self.brush_new.toggled.connect(lambda *_: self._brush_status())
        self.b_newcell = QtWidgets.QPushButton("New cell (Esc)")
        self.b_newcell.setToolTip("Let go of the current cell: the next stroke paints a new one")
        self.b_newcell.clicked.connect(self._brush_release)
        self.b_prevc = QtWidgets.QToolButton(); self.b_prevc.setText("◀ P")
        self.b_prevc.setToolTip("Previous cell (P) – selects it and centres the view on it")
        self.b_prevc.clicked.connect(lambda: self._step_cell(-1))
        self.b_nextc = QtWidgets.QToolButton(); self.b_nextc.setText("N ▶")
        self.b_nextc.setToolTip("Next cell (N) – selects it and centres the view on it")
        self.b_nextc.clicked.connect(lambda: self._step_cell(1))
        self.brush_lbl = QtWidgets.QLabel("")
        self.brush_lbl.setStyleSheet("color:#c06000")
        for w_ in (self.brush_new, self.b_newcell, self.brush_lbl):
            w_.setVisible(False)
        self.mode_sel.setChecked(True)
        for w in (self.mode_sel, self.mode_poly, self.mode_add, self.mode_brush):
            w.toggled.connect(lambda *_: self._cancel_poly())
        self.show_rej = QtWidgets.QCheckBox("show rejected")
        self.show_rej.setChecked(True)
        self.show_rej.toggled.connect(self._draw_rois)
        # brightness of the background image (percentile black / white points + gamma)
        def sl(lo, hi, v, tip):
            w_ = QtWidgets.QSlider(Qt.Horizontal)
            w_.setRange(lo, hi)
            w_.setValue(v)
            w_.setMinimumWidth(90)
            w_.setToolTip(tip)
            w_.valueChanged.connect(self._apply_levels)
            return w_
        self.lv_lo = sl(0, 999, 5, "Black point (percentile of the image)")
        self.lv_hi = sl(1, 1000, 997, "White point (percentile of the image) – lower = brighter")
        self.lv_gamma = sl(20, 300, 100, "Gamma – right = brighten the dark parts")
        b_lv = QtWidgets.QPushButton("Auto")
        b_lv.setToolTip("Reset brightness (0.5–99.7 %, gamma 1)")
        b_lv.clicked.connect(self._reset_levels)
        self.lv_lbl = QtWidgets.QLabel("")
        self.lv_lbl.setStyleSheet("color: gray")
        self.roi_alpha = sl(10, 100, 100, "Outline opacity")
        self.roi_alpha.valueChanged.disconnect(self._apply_levels)
        self.roi_alpha.valueChanged.connect(self._draw_rois)
        il.addLayout(hrow(QtWidgets.QLabel("Background"), self.bg_combo, self.frame_spin,
                          QtWidgets.QLabel("  black"), self.lv_lo, QtWidgets.QLabel("white"), self.lv_hi,
                          QtWidgets.QLabel("gamma"), self.lv_gamma, b_lv, self.lv_lbl,
                          QtWidgets.QLabel("  outlines"), self.roi_alpha, "stretch"))
        il.addLayout(hrow(QtWidgets.QLabel("Click:"), self.mode_sel, self.mode_poly, self.mode_add,
                          self.mode_brush, self.brush_size, self.brush_new, self.b_newcell,
                          self.b_prevc, self.b_nextc, self.brush_lbl,
                          self.show_rej, "stretch"))
        hint = QtWidgets.QLabel("Select: click a cell (Ctrl+click adds to the selection), double-click = accept ↔ "
                                "reject · Brush: drag on empty space = new cell, drag from inside a cell = add to it, "
                                "right-drag = erase, Esc = new cell · N / P = next / previous cell (any mode), D/Delete = delete the selected cell(s), hold Space to drag the image · "
                                "Draw polygon: click the corners, double-click or Enter closes, "
                                "Backspace removes the last corner, Esc cancels · Edit outline: drag the corners, "
                                "Enter applies · Delete key removes the selected cells · green = accepted, "
                                "red = rejected")
        hint.setStyleSheet("color: gray")
        hint.setWordWrap(True)
        il.addWidget(hint)
        left.addWidget(img_w)
        self.plots = QtWidgets.QTabWidget()
        self.cv = {}
        for k, t in (("trace", "Trace"), ("raster", "Raster"), ("map", "Cell map"), ("hist", "Histograms"),
                     ("waves", "Waves"), ("copart", "Same waves?")):
            self.cv[k] = MplCanvas(figsize=(7, 4))
            if k == "copart":
                cpw = QtWidgets.QWidget()
                cl = QtWidgets.QVBoxLayout(cpw)
                cl.setContentsMargins(0, 0, 0, 0)
                self.cp_panel = QtWidgets.QComboBox()
                self.cp_panel.addItems(["cell pairs (matrix)", "waves × waves", "distance", "map of pairs",
                                        "all four"])
                self.cp_panel.currentIndexChanged.connect(self._draw_copart)
                self.cp_map = QtWidgets.QComboBox()
                self.cp_map.addItems(["pairs together beyond distance (z vs distance > 3)",
                                      "all pairs more than chance (q < 0.05)"])
                self.cp_map.currentIndexChanged.connect(self._draw_copart)
                b_pop = QtWidgets.QPushButton("Open in window")
                b_pop.clicked.connect(self._copart_window)
                cl.addLayout(hrow(QtWidgets.QLabel("show"), self.cp_panel, QtWidgets.QLabel("map:"), self.cp_map,
                                  b_pop, "stretch"))
                cl.addWidget(self.cv[k], 1)
                self.plots.addTab(cpw, t)
                continue
            self.plots.addTab(self.cv[k], t)
        rast_opts = QtWidgets.QWidget()
        self.shuffle_chk = QtWidgets.QCheckBox("shuffled cell order (control)")
        self.shuffle_chk.toggled.connect(self._draw_raster)
        self.map_combo = QtWidgets.QComboBox()
        self.map_combo.addItems(["participation", "rate_per_min", "fraction_events_in_waves",
                                 "mean_amplitude_pct", "mean_decay_tau_s"])
        self.map_combo.currentIndexChanged.connect(self._draw_map)
        self.wave_spin = QtWidgets.QSpinBox()
        self.wave_spin.valueChanged.connect(self._draw_waves)
        ro = QtWidgets.QHBoxLayout(rast_opts)
        ro.setContentsMargins(0, 0, 0, 0)
        for w in (self.shuffle_chk, QtWidgets.QLabel("   map colour"), self.map_combo,
                  QtWidgets.QLabel("   wave #"), self.wave_spin):
            ro.addWidget(w)
        ro.addStretch(1)
        pw = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(pw)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(rast_opts)
        pl.addWidget(self.plots, 1)
        left.addWidget(pw)
        left.setSizes([560, 420])
        split.addWidget(left)

        # ---- right: controls
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        self.status = QtWidgets.QLabel("Measure the grid first (tab 2): cells use the same frame space, "
                                       "grey mode and rotation. The wave analysis (tab 4) is needed for "
                                       "linking events to waves.")
        self.status.setWordWrap(True)
        rl.addWidget(self.status)
        # 1 find
        self.det_method = QtWidgets.QComboBox()
        self.det_method.addItems(["Suite2p (activity)", "Cellpose (shape, mean image)",
                                  "FIJI RoiSet.zip", "Only cells added by hand"])
        self.det_t0 = QtWidgets.QDoubleSpinBox(); self.det_t0.setRange(0, 1e6); self.det_t0.setSuffix(" s")
        self.det_t1 = QtWidgets.QDoubleSpinBox(); self.det_t1.setRange(-1, 1e6); self.det_t1.setValue(-1)
        self.det_t1.setSuffix(" s"); self.det_t1.setSpecialValueText("end")
        self.s2p_thr = QtWidgets.QDoubleSpinBox(); self.s2p_thr.setRange(0.1, 10); self.s2p_thr.setValue(1.0)
        self.s2p_thr.setSingleStep(0.1)
        self.s2p_thr.setToolTip("Suite2p threshold_scaling: higher = fewer, stronger ROIs")
        self.cp_img = QtWidgets.QComboBox(); self.cp_img.addItems(["mean", "max", "std"])
        self.cp_flow = QtWidgets.QDoubleSpinBox(); self.cp_flow.setRange(0, 3); self.cp_flow.setValue(0.4)
        self.cp_flow.setSingleStep(0.1)
        self.cp_prob = QtWidgets.QDoubleSpinBox(); self.cp_prob.setRange(-6, 6); self.cp_prob.setValue(0.0)
        self.cp_prob.setSingleStep(0.5)
        self.keep_manual = QtWidgets.QCheckBox("keep cells added by hand")
        self.keep_manual.setChecked(True)
        self.skip_dark = QtWidgets.QCheckBox("leave out light-off frames (found automatically)")
        self.skip_dark.setChecked(True)
        self.skip_dark.setToolTip("Frames where the light is off (dark start, or light switched on and off during "
                                  "the recording) and the switching transients are left out of the detection, "
                                  "the traces' baseline and the events")
        self.skip_dark3 = QtWidgets.QCheckBox("leave out light-off / dim frames (grey bands)")
        self.skip_dark3.setChecked(True)
        self.skip_dark3.setToolTip("Untick if the dim part is real data: the trace is then used as it is, "
                                   "without bridging the grey bands")
        self.skip_dark.toggled.connect(self._skip_dark_changed)
        self.skip_dark3.toggled.connect(self._skip_dark_changed)
        self.tissue_only = QtWidgets.QCheckBox("only inside the tissue (automatic mask, cyan)")
        self.tissue_only.setChecked(True)
        self.tissue_only.setToolTip("No ROIs on the dark surround or the rotation borders")
        f = QtWidgets.QFormLayout()
        f.addRow("Method", self.det_method)
        f.addRow("Use frames from", self.det_t0)
        f.addRow("…to", self.det_t1)
        f.addRow("Suite2p threshold", self.s2p_thr)
        f.addRow("Cellpose image", self.cp_img)
        f.addRow("Cellpose flow threshold", self.cp_flow)
        f.addRow("Cellpose cell probability", self.cp_prob)
        b_find = QtWidgets.QPushButton("▶  Find cells")
        b_find.setStyleSheet("font-weight: bold; padding: 5px")
        b_find.clicked.connect(self._find_cells)
        self.inst_btn = QtWidgets.QPushButton("Install Suite2p + Cellpose…")
        self.inst_btn.clicked.connect(lambda: _install_deps(self, self._deps_changed))
        self.deps_lbl = QtWidgets.QLabel("")
        self.deps_lbl.setWordWrap(True)
        tip = QtWidgets.QLabel("Suite2p finds cells that are <b>active</b> in the recording; Cellpose finds "
                               "<b>bright round somata</b> on the mean image, also inactive ones.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray")
        self.copy_btn = QtWidgets.QPushButton("Copy cells from another video…")
        self.copy_btn.setCheckable(True)
        self.copy_btn.setToolTip("Click this, then click another video in the list on the left: its cells are "
                                 "copied here (optionally shifted to line up with this recording).")
        self.copy_btn.toggled.connect(lambda on: self.main._pick_toggled("cells", on))
        rl.addWidget(group("1  Find cells", f, self.skip_dark, self.tissue_only, self.keep_manual, b_find,
                           self.copy_btn, tip,
                           self.deps_lbl, self.inst_btn))
        self._deps_changed()
        # 2 review
        self.prob_spin = QtWidgets.QDoubleSpinBox(); self.prob_spin.setRange(0, 1); self.prob_spin.setValue(0.5)
        self.prob_spin.setSingleStep(0.05)
        b_prob = QtWidgets.QPushButton("Accept if probability ≥")
        b_prob.clicked.connect(self._accept_by_prob)
        self.min_area = QtWidgets.QDoubleSpinBox(); self.min_area.setRange(0, 1e5); self.min_area.setValue(0)
        self.min_area.setSuffix(" µm²")
        b_area = QtWidgets.QPushButton("Reject smaller than")
        b_area.clicked.connect(self._reject_small)
        self.max_area = QtWidgets.QDoubleSpinBox(); self.max_area.setRange(0, 1e6); self.max_area.setValue(400)
        self.max_area.setSuffix(" µm²")
        self.max_area.setToolTip("Suite2p can return large irregular patches of neuropil – reject them by size")
        b_big = QtWidgets.QPushButton("Reject larger than")
        b_big.clicked.connect(self._reject_big)
        b_all = QtWidgets.QPushButton("Accept all"); b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QtWidgets.QPushButton("Reject all"); b_none.clicked.connect(lambda: self._set_all(False))
        b_tog = QtWidgets.QPushButton("Accept / reject selected"); b_tog.clicked.connect(self._toggle_sel)
        b_del = QtWidgets.QPushButton("Delete selected"); b_del.clicked.connect(self._delete_sel)
        self.b_edit = QtWidgets.QPushButton("Edit outline"); self.b_edit.clicked.connect(self._start_edit)
        self.b_apply = QtWidgets.QPushButton("Apply outline"); self.b_apply.clicked.connect(self._apply_edit)
        self.b_cancel = QtWidgets.QPushButton("Cancel"); self.b_cancel.clicked.connect(self._cancel_edit)
        for b_ in (self.b_apply, self.b_cancel):
            b_.setVisible(False)
        self.count_lbl = QtWidgets.QLabel("no cells")
        self.table = QtWidgets.QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["#", "use", "source", "area µm²", "prob", "events", "particip."])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(220)
        self.table.itemSelectionChanged.connect(self._table_selected)
        self.table.itemChanged.connect(self._table_item_changed)
        self.table.setSortingEnabled(False)
        rl.addWidget(group("2  Review and edit ROIs", self.count_lbl, self.table,
                           hrow(b_tog, b_del), hrow(self.b_edit, self.b_apply, self.b_cancel),
                           hrow(b_all, b_none), hrow(b_prob, self.prob_spin), hrow(b_area, self.min_area),
                           hrow(b_big, self.max_area)))
        for key, fn in ((Qt.Key_Return, self._enter), (Qt.Key_Enter, self._enter), (Qt.Key_Escape, self._escape),
                        (Qt.Key_Backspace, self._undo_vertex), (Qt.Key_Delete, self._delete_sel),
                        (Qt.Key_N, lambda: self._step_cell(1)), (Qt.Key_P, lambda: self._step_cell(-1)), (Qt.Key_D, lambda: None if self.poly_pts else self._delete_sel())):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn, context=Qt.WidgetWithChildrenShortcut)
        # 3 traces + events
        self.form = ParamForm(cc.CellParams())
        for k, txt in (("grid_sigma", "Smoothing σ (frames)"),
                       ("grid_prominence", "Prominence (grey)"),
                       ("grid_distance", "Min. distance (frames)"),
                       ("grid_exclude_s", "Ignore peaks before (s)"),
                       ("grid_f0", "Typical grid square (grey)")):
            lab, _ = self.form.labels[k]           # start from tab 3, may be changed for the cells
            lab.setText(txt)
            self.form.labels[k] = (lab, txt)
            self.form.widgets[k].valueChanged.connect(lambda *_: self._params())
        self.form.widgets["grid_f0"].setEnabled(False)
        self.form.widgets["grid_f0"].setToolTip("Median brightness of the grid squares, used to turn the grey "
                                                "prominence into % ΔF/F (filled in automatically)")
        self.b_tab3 = QtWidgets.QPushButton("Copy grid settings from tab 3")
        self.b_tab3.setToolTip("Set σ, prominence, distance and start exclusion to the Events tab values")
        self.b_tab3.clicked.connect(lambda: self._copy_tab3())
        self.grid_lbl = QtWidgets.QLabel("")
        self.grid_lbl.setWordWrap(True)
        self.grid_lbl.setStyleSheet("color:#1f4fbf")
        self.form.widgets["method"].valueChanged.connect(lambda *_: (self._form_visibility(), self._params()))
        self.form.widgets["grid_prom_unit"].valueChanged.connect(lambda *_: (self._form_visibility(), self._params()))
        self.rule_combo = QtWidgets.QComboBox()
        self.rule_combo.addItems(list(cc.CELL_PRESETS))
        b_rule = QtWidgets.QPushButton("Reset values")
        b_rule.setToolTip("Load the chosen rule's values again (choosing a rule in the menu applies it)")
        b_rule.clicked.connect(self._apply_rule)
        self.rule_combo.activated.connect(lambda *_: self._apply_rule())
        self._form_visibility()
        self.live_chk = QtWidgets.QCheckBox("Live preview on the selected cell")
        self._live_timer = QtCore.QTimer(self, singleShot=True, interval=200)
        self._live_timer.timeout.connect(self._draw_trace)
        for w in self.form.widgets.values():
            w.valueChanged.connect(lambda *_: self.live_chk.isChecked() and self._live_timer.start())
        b_meas = QtWidgets.QPushButton("▶  Measure cell traces")
        b_meas.clicked.connect(self._measure)
        b_an = QtWidgets.QPushButton("▶  Detect events + link to waves")
        b_an.setStyleSheet("font-weight: bold; padding: 5px")
        b_an.clicked.connect(self._analyse)
        b_open = QtWidgets.QPushButton("Show results folder")
        b_open.clicked.connect(lambda: self.main.meas is not None and
                               __import__("wave_gui").reveal_in_folder(self.main.results_dir()))
        self.res_lbl = QtWidgets.QLabel("")
        self.res_lbl.setWordWrap(True)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(inner).addWidget(self.form)
        sc.setWidget(inner)
        sc.setMinimumHeight(260)
        rl.addWidget(group("3  Traces, events and waves", hrow(QtWidgets.QLabel("Rule"), self.rule_combo, b_rule),
                           self.grid_lbl, self.b_tab3, sc, self.skip_dark3, self.live_chk, b_meas, b_an, b_open, self.res_lbl))
        # 4 movie
        self.cm_from, self.cm_to = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for sp, v in ((self.cm_from, 0.0), (self.cm_to, -1.0)):
            sp.setRange(-1, 1e6)
            sp.setDecimals(1)
            sp.setSuffix(" s")
            sp.setValue(v)
        self.cm_to.setSpecialValueText("end")
        self.cm_speed = QtWidgets.QComboBox()
        self.cm_speed.setEditable(True)
        self.cm_speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "10×"])
        self.cm_speed.setCurrentText("1×")
        self.cm_colour = QtWidgets.QComboBox()
        self.cm_colour.addItems(["cells flash at events (orange = in wave, cyan = outside)", "cells coloured by ΔF/F"])
        self.cm_grid = QtWidgets.QCheckBox("active grid squares (red)")
        self.cm_grid.setChecked(True)
        self.cm_tracks = QtWidgets.QCheckBox("wave centres + tracks")
        self.cm_tracks.setChecked(True)
        self.cm_outl = QtWidgets.QCheckBox("cell outlines")
        self.cm_outl.setChecked(True)
        self.cm_adapt = QtWidgets.QCheckBox("follow brightness changes")
        self.cm_adapt.setChecked(True)
        self.cm_adapt.setToolTip("Re-adjust the brightness during the movie (e.g. when the illumination changes); "
                                 "off = fixed from the first frame")
        self.cm_flash = QtWidgets.QDoubleSpinBox()
        self.cm_flash.setRange(0.05, 10)
        self.cm_flash.setValue(0.6)
        self.cm_flash.setSuffix(" s")
        self.cm_flash.setToolTip("How long a cell stays lit after an event")
        b_mov = QtWidgets.QPushButton("▶  Render cell movie…")
        b_mov.clicked.connect(self._render_movie)
        rl.addWidget(group("4  Movie of cells and waves",
                           hrow(QtWidgets.QLabel("from"), self.cm_from, QtWidgets.QLabel("to"), self.cm_to,
                                QtWidgets.QLabel("speed"), self.cm_speed, "stretch"),
                           self.cm_colour, hrow(QtWidgets.QLabel("flash"), self.cm_flash, self.cm_outl, "stretch"),
                           hrow(self.cm_grid, self.cm_tracks, "stretch"), self.cm_adapt, b_mov))
        rl.addStretch(1)
        rs = QtWidgets.QScrollArea()
        rs.setWidgetResizable(True)
        rs.setWidget(right)
        rs.setMinimumWidth(400)
        split.addWidget(rs)
        split.setSizes([1000, 430])

    # ------------------------------------------------------------------ state
    def _um(self):
        return self.main.um_per_px() if self.main.meas is not None else None

    def _diam_px(self):
        return self.form.value().diameter_um / (self._um() or 1.0)

    def _files(self):
        d = self.main.results_dir()
        return dict(rois=os.path.join(d, "cells_rois.npz"), imgs=os.path.join(d, "cells_images.npz"),
                    traces=os.path.join(d, "cells_traces.npz"), res=os.path.join(d, "cells_results.pkl"),
                    xlsx=os.path.join(d, f"{os.path.basename(d)}_wave_results.xlsx"))

    def _sig(self):
        return f"{len(self.cells)}:{sum(int(c.xpix.sum()) for c in self.cells if c.accepted)}:" \
               f"{sum(c.accepted for c in self.cells)}"

    def reset_for_video(self):
        """Called when a video / measurement is loaded: restore saved cells, traces and results."""
        self.cells, self.dm_images, self.Fc, self.Fn, self.result, self.sel = [], {}, None, None, None, -1
        self.label_map = None
        m = self.main
        ses = m.session or {}
        if ses.get("cell_params"):
            self.form.set_value(ses["cell_params"])
        self._form_visibility()
        self.skip_dark.blockSignals(True)
        self.skip_dark3.blockSignals(True)
        self.skip_dark.setChecked(bool(ses.get("cell_skip_dark", True)))
        self.skip_dark3.setChecked(bool(ses.get("cell_skip_dark", True)))
        self.skip_dark.blockSignals(False)
        self.skip_dark3.blockSignals(False)
        if m.meas is None:
            self.status.setText("Measure the grid first (tab 2): cells use the same frame space.")
            self._draw_rois()
            self._clear_plots()
            return
        F = self._files()
        try:
            if os.path.exists(F["rois"]):
                self.cells, *_ = cc.load_cells(F["rois"])
            if os.path.exists(F["imgs"]):
                z = np.load(F["imgs"])
                self.dm_images = {k: z[k] for k in z.files}
            if os.path.exists(F["traces"]):
                z = np.load(F["traces"])
                if str(z["sig"]) == self._sig():
                    self.Fc, self.Fn, self.trace_sig = z["Fc"], z["Fn"], str(z["sig"])
            if os.path.exists(F["res"]) and self.Fc is not None:
                with open(F["res"], "rb") as fh:
                    self.result = pickle.load(fh)
        except Exception as e:
            m.log(f"could not restore cells: {e}")
        self._update_status()
        self._show_bg()
        self._draw_rois()
        self._draw_all()
        self._params()

    # ------------------------------------------------------------------ copy cells from another video
    @staticmethod
    def source_files(src):
        """cells_rois.npz / cells_images.npz of another video (newest of rotated / unrotated measurement)."""
        stem = os.path.splitext(src)[0]
        c = [os.path.join(stem + suf, "cells_rois.npz") for suf in ("_rotated_grid", "_grid")]
        c = [x for x in c if os.path.exists(x)]
        if not c:
            return None, None
        r = max(c, key=os.path.getmtime)
        im = os.path.join(os.path.dirname(r), "cells_images.npz")
        return r, (im if os.path.exists(im) else None)

    def _here_image(self):
        im = self.dm_images.get("mean")
        if im is None:
            im = self.main.background()
        return None if im is None else np.asarray(im, np.float32)

    @staticmethod
    def find_shift(a, b):
        """(dx, dy, response) that moves image a onto image b (phase correlation on high-passed images)."""
        import cv2
        from scipy import ndimage

        def prep(x):
            x = np.asarray(x, np.float32)
            if x.ndim == 3:
                x = x.mean(axis=2)
            x = x - ndimage.gaussian_filter(x, 25)
            return (x - x.mean()) / (x.std() + 1e-6)
        a, b = prep(a), prep(b)
        if a.shape != b.shape:
            return 0.0, 0.0, 0.0
        win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
        return float(dx), float(dy), float(resp)

    def copy_from(self, src):
        m = self.main
        if m.meas is None:
            m.error("Measure this video first (tab 2) – the cells are placed in its frame space.")
            return
        if os.path.abspath(src) == os.path.abspath(m.video or ""):
            m.log("That is the current video – pick another one.")
            return
        rois, imgs = self.source_files(src)
        if rois is None:
            m.error(f"{os.path.basename(src)} has no cells yet (find or draw them in its Cells tab first).")
            return
        cells, sW, sH, _ = cc.load_cells(rois)
        if not cells:
            m.error(f"{os.path.basename(src)} has no cells.")
            return
        W, H = m.meas.frame_size
        shift, note = (0.0, 0.0, 0.0), ""
        here = self._here_image()
        if imgs is not None and here is not None:
            try:
                there = np.load(imgs)["mean"]
                if there.shape == here.shape:
                    shift = self.find_shift(there, here)
                else:
                    note = "images differ in size – no automatic alignment"
            except Exception as e:
                note = f"alignment not possible ({e})"
        else:
            note = "the source has no mean image (cells drawn by hand) – no automatic alignment"
        dlg = _CopyCellsDialog(self, os.path.basename(src), cells, (sW, sH), (W, H), shift, note,
                               bool(self.cells))
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        dx, dy = dlg.shift()
        new = []
        for c in cells:
            x = np.round(c.xpix + dx).astype(np.int32)
            y = np.round(c.ypix + dy).astype(np.int32)
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if ok.sum() < 4:
                continue
            new.append(cc.Cell(y[ok], x[ok], c.source, c.prob, c.accepted if dlg.keep_state.isChecked() else True))
        lost = len(cells) - len(new)
        self.cells = (self.cells if dlg.add.isChecked() else []) + new
        self.sel = -1
        self.selset = set()
        self._cells_changed()
        self._update_status()
        m.log(f"Copied {len(new)} cells from {os.path.basename(src)}"
              + (f", shifted by ({dx:+.1f}, {dy:+.1f}) px" if (dx or dy) else "")
              + (f"; {lost} fell outside the frame" if lost else "")
              + ". Check them on the mean image, then measure the traces.")

    def _update_status(self):
        n, a = len(self.cells), sum(c.accepted for c in self.cells)
        self.count_lbl.setText(f"{n} ROIs, <b>{a} accepted</b>" if n else "no cells")
        parts = []
        if self.main.meas is not None:
            parts.append(f"Frame space of {os.path.basename(self.main.meas_path)}"
                         + (" (rotated)" if self.main.meas.rotation else ""))
        parts.append("traces: " + ("measured" if self.Fc is not None else "not yet"))
        parts.append("waves: " + ("available" if self.main.results is not None else "run tab 4 to link"))
        self.status.setText(" · ".join(parts))

    def _deps_changed(self):
        s2p, cp = _importable("suite2p"), _importable("cellpose")
        self.deps_lbl.setText(f"Suite2p: {'installed' if s2p else 'not installed'} · "
                              f"Cellpose: {'installed' if cp else 'not installed'}")
        self.inst_btn.setVisible(not (s2p and cp))

    def _save_rois(self):
        if self.main.meas is None:
            return
        f = self._files()["rois"]
        new = not os.path.exists(f)
        cc.save_cells(f, self.cells, *self.main.meas.frame_size)
        self.label_map = None
        if new and self.main.video:
            self.main._refresh_list_label(self.main.video)

    # ------------------------------------------------------------------ image + ROIs
    def _show_bg(self, *_):
        m = self.main
        if m.meas is None:
            self.img.clear()
            return
        k = self.bg_combo.currentIndex()
        im = None
        if k < 3:
            im = self.dm_images.get(["mean", "max", "std"][k])
        if im is None:
            i = int(self.frame_spin.value() * m.meas.fps)
            im = m._meas_space_frame(min(i, m.meas.n_frames - 1), use_meas=True)
        if im is None:
            return
        g = np.asarray(im, np.float32)
        g = g.mean(axis=2) if g.ndim == 3 else g
        self._lv_sample = np.sort(g[::4, ::4].ravel())
        self.img.setImage(g, autoLevels=False)
        self._apply_levels()
        W, H = m.meas.frame_size
        self.img.setRect(QtCore.QRectF(0, 0, W, H))
        mk = self.dm_images.get("mask")
        if mk is not None and mk.ndim == 2 and not mk.all():
            import cv2
            cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xs, ys = [], []
            for c in cs:
                c = c[:, 0, :].astype(float)
                c = np.vstack([c, c[:1]])
                xs += [c[:, 0], [np.nan]]
                ys += [c[:, 1], [np.nan]]
            self.c_mask.setData(np.concatenate(xs), np.concatenate(ys), connect="finite") if xs else self.c_mask.setData([], [])
        else:
            self.c_mask.setData([], [])

    def _apply_levels(self, *_):
        smp = getattr(self, "_lv_sample", None)
        if smp is None or not len(smp):
            return
        a, b = self.lv_lo.value(), self.lv_hi.value()
        if b <= a:
            b = a + 1
        lo = float(smp[min(len(smp) - 1, int(a / 1000 * (len(smp) - 1)))])
        hi = float(smp[min(len(smp) - 1, int(b / 1000 * (len(smp) - 1)))])
        hi = max(hi, lo + 1e-3)
        gm = self.lv_gamma.value() / 100.0
        x = np.linspace(0, 1, 256)
        lut = (np.power(x, 1.0 / gm) * 255 + 0.5).astype(np.uint8)
        self.img.setLookupTable(lut)
        self.img.setLevels((lo, hi))
        self.lv_lbl.setText(f"{lo:.0f}–{hi:.0f}, γ {gm:.2f}")

    def _reset_levels(self):
        for w_, v in ((self.lv_lo, 5), (self.lv_hi, 997), (self.lv_gamma, 100)):
            w_.blockSignals(True)
            w_.setValue(v)
            w_.blockSignals(False)
        self._apply_levels()

    def _draw_rois(self, *_):
        def curve(idx):
            xs, ys = [], []
            for i in idx:
                x, y = cc.outline(self.cells[i])
                if len(x):
                    xs += [x, [np.nan]]
                    ys += [y, [np.nan]]
            if not xs:
                return np.array([]), np.array([])
            return np.concatenate(xs), np.concatenate(ys)

        acc = [i for i, c in enumerate(self.cells) if c.accepted]
        rej = [i for i, c in enumerate(self.cells) if not c.accepted] if self.show_rej.isChecked() else []
        self.selset = {i for i in self.selset if 0 <= i < len(self.cells)}
        sel = sorted(self.selset | ({self.sel} if 0 <= self.sel < len(self.cells) else set()))
        for item, idx in ((self.c_acc, acc), (self.c_rej, rej), (self.c_sel, sel)):
            x, y = curve(idx)
            item.setData(x, y, connect="finite")
        if hasattr(self, "roi_alpha"):
            for item in (self.c_acc, self.c_rej):
                item.setOpacity(self.roi_alpha.value() / 100.0)
        self._update_status()
        self._refresh_table()

    # ------------------------------------------------------------------ cell list
    def _refresh_table(self):
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(self.cells))
        um2 = (self._um() or 1.0) ** 2
        res = self.result if (self.result is not None and self._sig() == self.trace_sig) else None
        rc = res.cells.set_index("cell") if res is not None and len(res.cells) else None
        for i, c in enumerate(self.cells):
            k = self._acc_index(i)
            ev = pa = ""
            if rc is not None and k >= 0 and (k + 1) in rc.index:
                r = rc.loc[k + 1]
                ev = str(int(r.n_events))
                pa = "" if not np.isfinite(r.participation) else f"{r.participation:.2f}"
            vals = [str(i + 1), "", c.source, f"{c.area * um2:.0f}", f"{c.prob:.2f}", ev, pa]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if j == 1:
                    it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    it.setCheckState(Qt.Checked if c.accepted else Qt.Unchecked)
                if not c.accepted:
                    it.setForeground(QtGui.QColor(190, 60, 60))
                t.setItem(i, j, it)
        t.clearSelection()
        for i in sorted(self.selset | ({self.sel} if 0 <= self.sel < len(self.cells) else set())):
            for j in range(t.columnCount()):
                t.item(i, j).setSelected(True)
        if 0 <= self.sel < len(self.cells):
            t.scrollToItem(t.item(self.sel, 0))
        t.resizeColumnsToContents()
        t.blockSignals(False)

    def _table_selected(self):
        rows = sorted({ix.row() for ix in self.table.selectedIndexes()})
        if not rows:
            return
        self.selset = set(rows)
        self.sel = rows[-1] if self.sel not in rows else self.sel
        if len(rows) == 1:
            self.sel = rows[0]
            x, y = self.cells[self.sel].centre
            r = self.vb.viewRect()
            self.vb.translateBy(x=x - r.center().x(), y=y - r.center().y())
        self._draw_sel_only()
        self._draw_trace()

    def _draw_sel_only(self):
        xs, ys = [], []
        for i in sorted(self.selset | ({self.sel} if 0 <= self.sel < len(self.cells) else set())):
            x, y = cc.outline(self.cells[i])
            xs += [x, [np.nan]]
            ys += [y, [np.nan]]
        self.c_sel.setData(np.concatenate(xs) if xs else np.array([]), np.concatenate(ys) if ys else np.array([]),
                           connect="finite")

    def _table_item_changed(self, it):
        if it.column() == 1 and 0 <= it.row() < len(self.cells):
            self.cells[it.row()].accepted = it.checkState() == Qt.Checked
            self._cells_changed()

    def _targets(self):
        rows = sorted(self.selset | ({self.sel} if 0 <= self.sel < len(self.cells) else set()))
        return [i for i in rows if 0 <= i < len(self.cells)]

    # ------------------------------------------------------------------ brush (FIJI selection brush)
    def _brush_mode(self, on):
        self.vb.drag_cb = self._brush_drag if on else None
        self.brush_circle.setVisible(on)
        for w_ in (self.brush_new, self.b_newcell, self.brush_lbl):
            w_.setVisible(on)
        if on:
            self._cancel_poly()
        self._brush_status()

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease) and ev.key() == Qt.Key_Space \
                and self.mode_brush.isChecked() and self.isVisible():
            if t == QtCore.QEvent.KeyPress and not self.glw.underMouse() and not self.vb.space_pan:
                return False                              # Space elsewhere (text fields, buttons) works as usual
            if not ev.isAutoRepeat():
                down = t == QtCore.QEvent.KeyPress
                self.vb.space_pan = down
                if down:
                    self.glw.setCursor(Qt.OpenHandCursor)
                else:
                    self.glw.unsetCursor()
                self.brush_circle.setVisible(not down)
            return True
        return False

    def _step_cell(self, d):
        """Select the next / previous cell in the list and centre the view on it (zoom kept)."""
        if not self.cells or self.poly_pts or self.edit_roi is not None:
            return
        n = len(self.cells)
        self.sel = (d if d < 0 else 0) % n if not (0 <= self.sel < n) else (self.sel + d) % n
        self.selset = set()
        cx, cy = self.cells[self.sel].centre
        (x0, x1), (y0, y1) = self.vb.viewRange()
        mx, my = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        if not (x0 + mx < cx < x1 - mx and y0 + my < cy < y1 - my):
            self.vb.translateBy(x=cx - (x0 + x1) / 2, y=cy - (y0 + y1) / 2)
        self._draw_rois()
        self._draw_trace()
        self._brush_status()

    def _brush_release(self):
        """Deselect, so the next brush stroke starts a new cell."""
        self.sel = -1
        self.selset = set()
        self._draw_rois()
        self._brush_status()

    def _brush_status(self):
        if not hasattr(self, "brush_lbl"):
            return
        if 0 <= self.sel < len(self.cells):
            self.brush_lbl.setText(f"painting on cell #{self.sel + 1}")
        else:
            self.brush_lbl.setText("next stroke = new cell")

    def _brush_r(self):
        return max(1.0, self.brush_size.value() / (self._um() or 1.0))

    def _brush_cursor(self, pos):
        if not self.mode_brush.isChecked():
            return
        if pos is not None:
            p = self.vb.mapSceneToView(pos)
            self._cur = (p.x(), p.y())
        x, y = getattr(self, "_cur", (0, 0))
        r = self._brush_r()
        self.brush_circle.setRect(x - r, y - r, 2 * r, 2 * r)

    def _brush_drag(self, ev):
        if self.main.meas is None:
            return
        import cv2
        W, H = self.main.meas.frame_size
        p = self.vb.mapSceneToView(ev.scenePos())
        x, y = p.x(), p.y()
        self._cur = (x, y)
        self._brush_cursor(None)
        r = int(round(self._brush_r()))
        if ev.isStart() or self._stroke is None:
            if ev.isStart():
                p0 = self.vb.mapSceneToView(ev.buttonDownScenePos()) if hasattr(ev, "buttonDownScenePos") else p
                x, y = p0.x(), p0.y()
            xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
            right = ev.button() == Qt.RightButton
            tgt = self.sel if 0 <= self.sel < len(self.cells) else -1
            under = int(self._labels()[yi, xi])
            inside = tgt >= 0 and under == tgt
            if not inside and under >= 0 and (right or self.brush_new.isChecked()):
                tgt, inside = under, True                 # stroke starts in another cell: edit that one
            elif not inside and not right and self.brush_new.isChecked():
                tgt = -1                                  # empty space: new cell
            m = np.zeros((H, W), np.uint8)
            if tgt >= 0:
                c = self.cells[tgt]
                m[c.ypix, c.xpix] = 1
            erase = right or (tgt >= 0 and not inside)
            if tgt < 0 and erase:
                return                                    # nothing to erase from
            self._stroke = dict(tgt=tgt, mask=m, erase=erase, last=(x, y))
        st = self._stroke
        val = 0 if st["erase"] else 1
        cv2.line(st["mask"], (int(round(st["last"][0])), int(round(st["last"][1]))),
                 (int(round(x)), int(round(y))), val, thickness=2 * r + 1)
        cv2.circle(st["mask"], (int(round(x)), int(round(y))), r, val, -1)
        st["last"] = (x, y)
        cs, _ = cv2.findContours(st["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xs, ys = [], []
        for c in cs:
            c = c[:, 0, :].astype(float) + 0.5
            c = np.vstack([c, c[:1]])
            xs += [c[:, 0], [np.nan]]
            ys += [c[:, 1], [np.nan]]
        self.c_brush.setData(np.concatenate(xs) if xs else np.array([]),
                             np.concatenate(ys) if ys else np.array([]), connect="finite")
        if ev.isFinish():
            self._finish_stroke()

    def _finish_stroke(self):
        st, self._stroke = self._stroke, None
        self.c_brush.setData([], [])
        if st is None:
            return
        yy, xx = np.nonzero(st["mask"])
        if st["tgt"] >= 0:
            old = self.cells[st["tgt"]]
            if len(xx) == 0:
                del self.cells[st["tgt"]]                 # everything erased
                self.sel = -1
            else:
                self.cells[st["tgt"]] = cc.Cell(yy.astype(np.int32), xx.astype(np.int32), old.source,
                                                old.prob, old.accepted)
                self.sel = st["tgt"]
        elif len(xx):
            self.cells.append(cc.Cell(yy.astype(np.int32), xx.astype(np.int32), "manual", 1.0, True))
            self.sel = len(self.cells) - 1
        self.selset = set()
        self._cells_changed()
        self._brush_status()

    # ------------------------------------------------------------------ polygon drawing / outline editing
    def _draw_poly(self):
        if self.poly_pts:
            xy = np.array(self.poly_pts)
            self.c_poly.setData(xy[:, 0], xy[:, 1])
            self.s_poly.setData(xy[:, 0], xy[:, 1])
        else:
            self.c_poly.setData([], [])
            self.s_poly.setData([], [])

    def _finish_poly(self):
        if len(self.poly_pts) >= 3 and self.main.meas is not None:
            W, H = self.main.meas.frame_size
            c = cc.polygon_cell(self.poly_pts, W, H)
            if c is not None:
                self.cells.append(c)
                self.sel = len(self.cells) - 1
                self.selset = set()
        self.poly_pts = []
        self._draw_poly()
        self._cells_changed()

    def _cancel_poly(self):
        self.poly_pts = []
        self._draw_poly()

    def _undo_vertex(self):
        if self.poly_pts:
            self.poly_pts.pop()
            self._draw_poly()

    def _enter(self):
        if self.edit_roi is not None:
            self._apply_edit()
        elif self.poly_pts:
            self._finish_poly()

    def _escape(self):
        if self.mode_brush.isChecked():
            self._brush_release()
        elif self.edit_roi is not None:
            self._cancel_edit()
        else:
            self._cancel_poly()

    def _start_edit(self):
        if not (0 <= self.sel < len(self.cells)):
            self.main.error("Select a cell first.")
            return
        self._cancel_edit()
        pts = cc.polygon_of(self.cells[self.sel])
        if len(pts) < 3:
            return
        self.edit_roi = pg.PolyLineROI(pts, closed=True, pen=pg.mkPen((255, 230, 0), width=2),
                                       handlePen=pg.mkPen((255, 230, 0)), movable=True)
        self.vb.addItem(self.edit_roi)
        self.edit_idx = self.sel
        for b_ in (self.b_apply, self.b_cancel):
            b_.setVisible(True)
        self.b_edit.setVisible(False)
        self.main.log("Drag the corners (click a side to add a corner, right-click a corner to remove it), "
                      "then Apply outline / Enter.")

    def _apply_edit(self):
        if self.edit_roi is None:
            return
        pts = [self.edit_roi.mapToParent(h["item"].pos()) for h in self.edit_roi.handles]
        pts = [(p.x(), p.y()) for p in pts]
        W, H = self.main.meas.frame_size
        c = cc.polygon_cell(pts, W, H, source=self.cells[self.edit_idx].source)
        if c is not None:
            c.accepted = self.cells[self.edit_idx].accepted
            c.prob = self.cells[self.edit_idx].prob
            self.cells[self.edit_idx] = c
        self._cancel_edit()
        self._cells_changed()

    def _cancel_edit(self):
        if self.edit_roi is not None:
            self.vb.removeItem(self.edit_roi)
            self.edit_roi = None
        for b_ in (self.b_apply, self.b_cancel):
            b_.setVisible(False)
        self.b_edit.setVisible(True)

    def _labels(self):
        if self.label_map is None and self.main.meas is not None:
            W, H = self.main.meas.frame_size
            lm = np.full((H, W), -1, np.int32)
            for i, c in enumerate(self.cells):
                lm[np.clip(c.ypix, 0, H - 1), np.clip(c.xpix, 0, W - 1)] = i
            self.label_map = lm
        return self.label_map

    def _img_clicked(self, ev):
        if ev.button() != Qt.LeftButton or self.main.meas is None:
            return
        p = self.vb.mapSceneToView(ev.scenePos())
        x, y = p.x(), p.y()
        W, H = self.main.meas.frame_size
        if not (0 <= x < W and 0 <= y < H):
            return
        if self.edit_roi is not None:
            return                                    # editing an outline: the ROI handles take the clicks
        if self.mode_brush.isChecked():
            lm = self._labels()
            i = int(lm[int(y), int(x)])
            self.sel = i                              # click = pick the cell to paint on (outside: none -> new)
            self.selset = set()
            self._draw_rois()
            self._draw_trace()
            self._brush_status()
            return
        if self.mode_poly.isChecked():
            if ev.double():
                self._finish_poly()
            else:
                self.poly_pts.append((x, y))
                self._draw_poly()
            return
        if self.mode_add.isChecked():
            self.cells.append(cc.circle_cell(x, y, self._diam_px(), W, H))
            self.sel = len(self.cells) - 1
            self._cells_changed()
            return
        lm = self._labels()
        i = int(lm[int(y), int(x)])
        if i < 0 and self.cells:                      # nearest centre within one diameter
            cen = np.array([c.centre for c in self.cells])
            d = np.hypot(cen[:, 0] - x, cen[:, 1] - y)
            i = int(d.argmin()) if d.min() < self._diam_px() else -1
        if i < 0:
            return
        if ev.double():
            self.cells[i].accepted = not self.cells[i].accepted
            self._cells_changed()
        if ev.modifiers() & Qt.ControlModifier:
            self.selset ^= {i}
            if 0 <= self.sel < len(self.cells):
                self.selset.add(self.sel)
                if i not in self.selset:
                    self.selset.discard(self.sel)
        else:
            self.selset = set()
        self.sel = i
        self._draw_rois()
        self._draw_trace()

    def _cells_changed(self):
        self._save_rois()
        self._draw_rois()
        if self.Fc is not None and self._sig() != self.trace_sig:
            self.res_lbl.setText("<span style='color:#b36b00'>Cells changed – measure the traces again.</span>")

    def _accept_by_prob(self):
        t = self.prob_spin.value()
        for c in self.cells:
            if c.source in ("suite2p", "cellpose"):
                c.accepted = c.prob >= t
        self._cells_changed()

    def _reject_small(self):
        a = self.min_area.value() / ((self._um() or 1.0) ** 2)
        for c in self.cells:
            if c.area < a:
                c.accepted = False
        self._cells_changed()

    def _reject_big(self):
        a = self.max_area.value() / ((self._um() or 1.0) ** 2)
        for c in self.cells:
            if c.area > a:
                c.accepted = False
        self._cells_changed()

    def _set_all(self, v):
        for c in self.cells:
            c.accepted = v
        self._cells_changed()

    def _toggle_sel(self):
        idx = self._targets()
        if idx:
            v = not all(self.cells[i].accepted for i in idx)
            for i in idx:
                self.cells[i].accepted = v
            self._cells_changed()

    def _delete_sel(self):
        if self.poly_pts:                  # Delete while drawing = undo a corner
            self._undo_vertex()
            return
        idx = self._targets()
        if not idx:
            return
        if len(idx) > 1 and QtWidgets.QMessageBox.question(
                self, "Delete", f"Delete {len(idx)} cells?") != QtWidgets.QMessageBox.Yes:
            return
        for i in sorted(idx, reverse=True):
            del self.cells[i]
        self.sel = -1
        self.selset = set()
        self._cancel_edit()
        self._cells_changed()
        self._brush_status()

    # ------------------------------------------------------------------ jobs
    def _need_meas(self):
        if self.main.meas is None:
            self.main.error("Measure the grid first (tab 2) – the cells use the same frame space.")
            return False
        return True

    def _src(self):
        m = self.main.meas
        src = m.source if (m.source and os.path.exists(m.source) and m.source.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv", ".m4v"))) else self.main.video
        return src, m.rotation, m.gray_mode

    def _find_cells(self):
        if not self._need_meas():
            return
        meth = self.det_method.currentIndex()
        W, H = self.main.meas.frame_size
        keep = [c for c in self.cells if c.source == "manual"] if self.keep_manual.isChecked() else []
        if meth == 2:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "FIJI RoiSet", os.path.dirname(self.main.video),
                                                         "ImageJ ROIs (*.zip *.roi)")
            if not p:
                return
            try:
                self.cells = keep + cc.cells_from_imagej_zip(p, W, H)
            except Exception as e:
                self.main.error(f"Could not read the ROIs:\n{e}")
                return
            self._after_find(None)
            return
        if meth == 3:
            self.cells = keep
            self.mode_poly.setChecked(True)
            self._after_find(None)
            return
        if meth == 0 and not _importable("suite2p") or meth == 1 and not _importable("cellpose"):
            self.main.error("This method needs Suite2p / Cellpose – press 'Install Suite2p + Cellpose…'.")
            return
        src, rot, gray = self._src()
        info = wc.video_info(src)
        fps = info["fps"]
        a = int(self.det_t0.value() * fps)
        b = info["n_frames"] if self.det_t1.value() < 0 else min(info["n_frames"], int(self.det_t1.value() * fps))
        diam = self._diam_px()
        thr, img, flow, cprob = self.s2p_thr.value(), self.cp_img.currentText(), self.cp_flow.value(), self.cp_prob.value()
        skip, tissue = self.skip_dark.isChecked(), self.tissue_only.isChecked()
        bad = self._bad()

        def job(progress=None, cancel=None):
            dm = cc.detection_movie(src, rot, gray, (a, b), diam, progress=progress, cancel=cancel,
                                    skip_dark=skip, tissue_only=tissue, bad=bad)
            if meth == 0:
                cells = cc.detect_suite2p(dm, threshold_scaling=thr, progress=progress)
            else:
                try:
                    cells = cc.detect_cellpose(dm, img, flow, cprob, progress=progress)
                except Exception as e:
                    if "urlopen" in str(e) or "URLError" in type(e).__name__ or "Connection" in str(e):
                        raise RuntimeError("Cellpose could not download its model (needed once, ~1 GB). "
                                           "Check the internet connection and try again.") from e
                    raise
            b_ = dm["bin"]
            imgs = {k: cv2_resize(dm[k], dm["W"], dm["H"]) for k in ("mean", "max", "std")}
            imgs["mask"] = cv2_resize(dm["mask"].astype(np.float32), dm["W"], dm["H"]) > 0.5
            imgs["skipped_s"] = np.float32(dm["skipped_s"])
            return cells, imgs

        def done(r):
            cells, imgs = r
            self.cells = keep + cells
            self.dm_images = imgs
            np.savez_compressed(self._files()["imgs"], **imgs)
            self._after_find(None)
            nb = 0 if bad is None else int(np.asarray(bad)[a:b].sum())
            self.main.log(f"{len(cells)} ROIs ({sum(c.accepted for c in cells)} accepted by probability and size)"
                          + (f"; {nb} light-off frames ({nb / max(b - a, 1) * 100:.0f}%) left out" if nb else ""))

        self.main.run_task(job, on_done=done, label="Finding cells")

    def _after_find(self, _):
        self.sel = -1
        self.Fc = self.Fn = None
        self.result = None
        self._save_rois()
        if self.bg_combo.currentIndex() == 3 and self.dm_images:
            self.bg_combo.setCurrentIndex(0)
        self._show_bg()
        self._draw_rois()
        self._clear_plots()
        self.main.log(f"{len(self.cells)} cells")

    def _measure(self):
        if not self._need_meas():
            return
        acc = [c for c in self.cells if c.accepted]
        if not acc:
            self.main.error("No accepted cells.")
            return
        src, rot, gray = self._src()
        diam = self._diam_px()
        sig = self._sig()

        def job(progress=None, cancel=None):
            return cc.measure_cells(src, acc, rot, gray, diam, progress, cancel)

        def done(r):
            self.Fc, self.Fn, _ = r
            self.trace_sig = sig
            np.savez_compressed(self._files()["traces"], Fc=self.Fc, Fn=self.Fn, sig=sig)
            self.result = None
            self.res_lbl.setText("Traces measured – now detect events.")
            self._update_status()
            self._draw_trace()

        self.main.run_task(job, on_done=done, label="Measuring cell traces")

    def _render_movie(self):
        m = self.main
        if m.meas is None:
            m.error("Measure the grid first (tab 2).")
            return
        if self.result is None or self._sig() != self.trace_sig:
            m.error("Run 'Detect events + link to waves' first.")
            return
        meas = m.meas
        vid = meas.source if (meas.source and os.path.exists(meas.source) and
                              meas.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".tif", ".tiff"))) else m.video
        out = os.path.join(m.results_dir(), "cells_movie.mp4")
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Cell movie", out, "MP4 (*.mp4)")
        if not out:
            return
        try:
            sp = float(self.cm_speed.currentText().replace("×", "").replace("x", "").strip())
        except ValueError:
            sp = 1.0
        fps = meas.fps
        f0 = int(max(0, self.cm_from.value()) * fps)
        t1 = self.cm_to.value()
        f1 = meas.n_frames if t1 < 0 else int(t1 * fps)
        if f1 <= f0:
            m.error("'to' must be after 'from'.")
            return
        a, b = self.lv_lo.value() / 10.0, self.lv_hi.value() / 10.0
        size = (meas.grid or {}).get("size") if isinstance(meas.grid, dict) else None
        self.main.run_task(
            cc.render_cell_movie, out, vid, meas.rotation, meas.gray_mode, fps, self._accepted(), self.result,
            f0, f1, speed=sp, colour="dff" if self.cm_colour.currentIndex() == 1 else "events",
            flash_s=self.cm_flash.value(), res=m.results, active=getattr(m, "active", None),
            grid_xy=(meas.x, meas.y), grid_size=size, show_grid=self.cm_grid.isChecked(),
            show_tracks=self.cm_tracks.isChecked(), show_outlines=self.cm_outl.isChecked(),
            levels_pct=(a, max(b, a + 0.1)), gamma=self.lv_gamma.value() / 100.0, adapt=self.cm_adapt.isChecked(),
            on_done=lambda p: (m.log(f"Cell movie written: {p}"),
                               __import__("wave_gui").reveal_in_folder(p)),
            label="Rendering cell movie")

    def _skip_dark_changed(self, on):
        for cb in (self.skip_dark, self.skip_dark3):
            if cb.isChecked() != on:
                cb.blockSignals(True)
                cb.setChecked(on)
                cb.blockSignals(False)
        ses = self.main.session
        if ses is not None and ses.get("cell_skip_dark", True) != on:
            ses["cell_skip_dark"] = on
            self.main.save_session()
            if self.result is not None:
                self.res_lbl.setText("<span style='color:#b36b00'>Light-off setting changed – run "
                                     "'Detect events + link to waves' again.</span>")
        self._draw_trace()

    def _bad(self):
        """Light-off frames of this recording (from the grid measurement), or None."""
        m = self.main.meas
        if m is None or not self.skip_dark.isChecked():
            return None
        if getattr(self, "_bad_for", None) is not m:
            self._bad_cache, self._bad_for = wc.meas_dark_frames(m), m
        return self._bad_cache

    def _accepted(self):
        return [c for c in self.cells if c.accepted]

    _TAB3 = (("grid_sigma", "sigma"), ("grid_prominence", "prominence"), ("grid_distance", "distance"),
             ("grid_exclude_s", "exclude_s"))

    def _copy_tab3(self, quiet=False):
        """Fill the grid-rule fields with the Events tab (tab 3) values."""
        ep = self.main.event_form.value()
        for k, e in self._TAB3:
            w_ = self.form.widgets[k]
            w_.blockSignals(True)
            w_.setValue(getattr(ep, e))
            w_.blockSignals(False)
        self._params()
        if self.live_chk.isChecked() and not quiet:
            self._draw_trace()

    def _params(self):
        """Cell parameters. Grid rule (method 2): σ / prominence / distance / exclusion start from tab 3
        and can be changed here; values that differ from tab 3 are highlighted orange."""
        p = self.form.value()
        if p.method != 2:
            self.grid_lbl.setText("")
            return p
        m = self.main
        ep = m.event_form.value()
        gf = cc.grid_f0(m.meas.mean) if m.meas is not None else 0.0
        w_ = self.form.widgets["grid_f0"]
        if gf > 0 and abs(w_.value() - gf) > 1e-3:
            w_.blockSignals(True)
            w_.setValue(gf)
            w_.blockSignals(False)
        p = self.form.value()
        diff = []
        for k, e in self._TAB3:
            same = abs(float(getattr(p, k)) - float(getattr(ep, e))) < 1e-6
            self.form.widgets[k].setStyleSheet(
                "background:#dce8ff" if same else "background:#ffd9a8; font-weight:bold")
            self.form.widgets[k].setToolTip("same as tab 3" if same else f"tab 3: {getattr(ep, e):g}")
            if not same:
                diff.append(self.form.labels[k][1].split(" (")[0].lower())
        fps = m.meas.fps if m.meas is not None else 0
        txt = (f"Grid rule: σ = {p.grid_sigma:g} frames, prominence {p.grid_prominence:g} grey, "
               f"min. distance {p.grid_distance:g} frames" + (f" ({p.grid_distance / fps:.1f} s)" if fps else "")
               + f", events after {p.grid_exclude_s:g} s. "
               + ("<span style='color:#c06000'>Changed from tab 3: " + ", ".join(diff) + ".</span> " if diff
                  else "(= tab 3) "))
        if p.grid_prom_unit == 1:
            txt += "Prominence used as grey levels on each cell's trace."
        elif p.grid_f0 > 0:
            txt += (f"Typical grid square {p.grid_f0:.1f} grey → prominence "
                    f"<b>{cc.grid_prom_pct(p):.2f} % ΔF/F</b> on each cell.")
        else:
            txt += "Measure the grid first for the % conversion."
        self.grid_lbl.setText(txt)
        return p

    def _analyse(self):
        if self.Fc is None or self._sig() != self.trace_sig:
            self.main.error("Measure the cell traces first (the cells changed since the last measurement).")
            return
        p = self._params()
        self.main.session["cell_params"] = asdict(p)
        self.main.save_session()
        acc = self._accepted()
        res = self.main.results
        um = self._um()
        fps = self.main.meas.fps
        Fc, Fn = self.Fc, self.Fn

        bad = self._bad()

        def job(progress=None, cancel=None):
            return cc.analyse_cells(Fc, Fn, fps, acc, p, res, um, progress=progress, bad=bad)

        def done(cr):
            self.result = cr
            F = self._files()
            with open(F["res"], "wb") as fh:
                pickle.dump(cr, fh)
            try:
                cc.save_cell_results(F["xlsx"], cr)
                saved = f"saved to {os.path.basename(F['xlsx'])}"
            except PermissionError:
                saved = "<span style='color:#d33'>workbook open in Excel – close it and run again to save</span>"
            s = cr.summary
            note = "" if res is not None else "<br><span style='color:#b36b00'>No wave analysis yet: events " \
                                              "are not linked to waves (run tab 4, then again).</span>"
            self.res_lbl.setText(
                f"{int(s['cells.n_analysed'])} cells analysed, {s['cells.fraction_active'] * 100:.0f}% active, "
                f"{s['cells.mean_rate_per_min']:.2f} events/min<br>"
                f"waves: {int(s['cells.n_waves'])}, participation {_pct(s['cells.mean_participation'])} "
                f"(chance {_pct(s['cells.chance_participation'])}, p = {_num(s['cells.participation_p_shuffle'])}), "
                f"{_pct(s['cells.fraction_events_in_waves'])} of events in waves ({s['cells.link_mode']})<br>"
                + (f"same waves: {s['cells.copart_fraction_pairs_sig'] * 100:.1f}% of cell pairs together more than "
                   f"chance ({s.get('cells.copart_fraction_beyond_distance', np.nan) * 100:.1f}% beyond distance); "
                   f"waves recruit the same cells: p = {_num(s['cells.wave_similarity_p'])}<br>"
                   if 'cells.copart_pairs_tested' in s else "")
                + f"{saved}"
                + note)
            self.wave_spin.setRange(0, max(len(cr.waves) - 1, 0))
            self._draw_all()

        self.main.run_task(job, on_done=done, label="Cell events and waves")

    # which form rows each peak rule uses (the rest is hidden)
    _RULE_FIELDS = {
        0: {"sigma", "threshold_pct", "drop_window_s", "use_around_mean", "around_window_s", "local_max_s",
            "short_window_s", "long_window_s", "eps_pct", "min_sep_s", "t_start"},
        1: {"sigma", "prominence_pct", "min_sep_s", "t_start"},
        2: {"grid_prom_unit", "grid_sigma", "grid_prominence", "grid_distance", "grid_exclude_s", "grid_f0"},
    }

    def _form_visibility(self):
        meth = int(self.form.widgets["method"].value())
        rule_only = set().union(*self._RULE_FIELDS.values())
        show = self._RULE_FIELDS.get(meth, set())
        for k, w_ in self.form.widgets.items():
            vis = (k not in rule_only) or (k in show)
            if k == "grid_f0" and meth == 2 and int(self.form.widgets["grid_prom_unit"].value()) == 1:
                vis = False                            # grey levels: no conversion needed
            w_.setVisible(vis)
            self.form.labels[k][0].setVisible(vis)
        if hasattr(self, "b_tab3"):
            self.b_tab3.setVisible(meth == 2)
        # keep the menu in step with the rule in use
        name = "Grid rule (tab 3 settings)" if meth == 2 else None
        if name and self.rule_combo.currentText() != name:
            self.rule_combo.blockSignals(True)
            self.rule_combo.setCurrentText(name)
            self.rule_combo.blockSignals(False)
        elif meth != 2 and self.rule_combo.currentText().startswith("Grid rule"):
            self.rule_combo.blockSignals(True)
            self.rule_combo.setCurrentIndex(0)
            self.rule_combo.blockSignals(False)

    def _apply_rule(self):
        d = cc.CELL_PRESETS.get(self.rule_combo.currentText(), {})
        if d.get("method") == 2:                   # grid rule: only switch the rule, keep the rest
            v = asdict(self.form.value())
            v["method"] = 2
            self.form.set_value(v)
            self._form_visibility()
            self._copy_tab3(quiet=True)
            if self.live_chk.isChecked():
                self._draw_trace()
            return
        base = asdict(cc.CellParams())
        keep = {k: v for k, v in asdict(self.form.value()).items()
                if k in ("diameter_um", "link_mode", "link_radius_um", "link_pad_s", "t_start", "t_end", "neuropil_r")}
        base.update(d)
        base.update(keep)
        self.form.set_value(base)
        self._form_visibility()
        self._params()
        if self.live_chk.isChecked():
            self._draw_trace()

    # ------------------------------------------------------------------ plots
    def _clear_plots(self):
        for c in self.cv.values():
            c.fig.clear()
            c.draw()

    def _draw_all(self):
        self._draw_trace()
        self._draw_raster()
        self._draw_map()
        self._draw_hist()
        self._draw_waves()
        self._draw_copart()

    def _acc_index(self, i):
        """Index of cell i among the accepted cells (traces / results are for accepted cells)."""
        if not (0 <= i < len(self.cells)) or not self.cells[i].accepted:
            return -1
        return sum(c.accepted for c in self.cells[:i])

    def _draw_trace(self, *_):
        fig = self.cv["trace"].fig
        fig.clear()
        k = self._acc_index(self.sel)
        if self.Fc is None or k < 0 or k >= self.Fc.shape[1] or self._sig() != self.trace_sig:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Select an accepted cell (after 'Measure cell traces')", ha="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            self.cv["trace"].draw()
            return
        fps = self.main.meas.fps
        p = self._params()
        F = self.Fc[:, k] - p.neuropil_r * self.Fn[:, k]
        bad = self._bad()
        s, pk, amp, tau, f0 = cc.detect_cell_events(F, fps, p, bad)
        t = np.arange(len(F)) / fps
        ax = fig.add_subplot(111)
        if s is None:
            ax.plot(t, F, color="0.5", lw=0.8)
            ax.set_title(f"cell {k + 1}: F0 = {f0:.1f} < {p.min_f0} – not analysed", fontsize=9)
        else:
            raw = (F / f0 - 1) * 100
            ax.plot(t, raw, color="0.75", lw=0.6, label="raw")
            ax.plot(t, s, color="#1f4fbf", lw=1.0,
                    label=f"smoothed (σ={p.grid_sigma if p.method == 2 else p.sigma:g})")
            ax.plot(t[pk], s[pk], "v", color="magenta", ms=6, label=f"{len(pk)} events")
            res = self.main.results
            if res is not None:
                for w in cc.wave_windows(res, fps, all_tracks=bool(p.link_tracks)):
                    ax.axvspan(w["f0"] / fps, w["f1"] / fps, color="orange", alpha=0.12, lw=0)
            ax.axvspan(p.baseline_t0, p.baseline_t1, color="green", alpha=0.07, lw=0)
            if bad is not None:
                raw = raw.copy()
                raw[bad[:len(raw)]] = np.nan              # light off: not drawn
                edges = np.flatnonzero(np.diff(np.r_[0, bad[:len(t)].astype(int), 0]))
                for a_, b_ in zip(edges[::2], edges[1::2]):
                    ax.axvspan(a_ / fps, b_ / fps, color="0.3", alpha=0.18, lw=0)
                ax.lines[0].set_ydata(raw)
            ax.set_ylabel("ΔF/F0 (%)")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"cell {k + 1}  (F0 = {f0:.1f}; orange = waves, green = baseline window)", fontsize=9)
        ax.set_xlabel("Time (s)")
        self.cv["trace"].draw()

    def _order(self):
        cen = np.array([c.centre for c in self._accepted()])
        if not len(cen):
            return np.array([], int)
        d = self._diam_px()
        row = np.round(cen[:, 1] / max(d, 1))
        o = np.lexsort((cen[:, 0], row))               # row by row, top -> bottom, left -> right
        if self.shuffle_chk.isChecked():
            o = np.random.default_rng(0).permutation(o)
        return o

    def _draw_raster(self, *_):
        fig = self.cv["raster"].fig
        fig.clear()
        cr = self.result
        if cr is None:
            self.cv["raster"].draw()
            return
        ax = fig.add_subplot(111)
        o = self._order()
        ev = [cr.events.loc[cr.events.cell == k + 1, "time_s"].to_numpy() for k in o]
        ax.eventplot(ev, colors="k", linelengths=0.8, linewidths=1)
        for _, w in cr.waves.iterrows():
            ax.axvline(w.start_s, color="green", lw=0.8, alpha=0.7)
            ax.axvline(w.end_s, color="red", lw=0.8, alpha=0.7)
        ax.set_ylabel("cells (" + ("shuffled" if self.shuffle_chk.isChecked() else "row by row") + ")")
        ax.set_xlabel("Time (s)")
        ax.invert_yaxis()
        ax.set_title("Cell events; green / red = wave start / end", fontsize=9)
        self.cv["raster"].draw()

    def _draw_map(self, *_):
        fig = self.cv["map"].fig
        fig.clear()
        cr = self.result
        if cr is None or not len(cr.cells):
            self.cv["map"].draw()
            return
        ax = fig.add_subplot(111)
        im = self.dm_images.get("mean")
        if im is not None:
            ax.imshow(im, cmap="gray")
        key = self.map_combo.currentText()
        c = cr.cells
        sc = ax.scatter(c.x_px, c.y_px, c=c[key], s=np.clip(c.area_px / max(c.area_px.max(), 1) * 120, 10, 200),
                        cmap="viridis", edgecolor="w", linewidth=0.4)
        fig.colorbar(sc, ax=ax, label=key)
        if im is None:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"cells coloured by {key}", fontsize=9)
        self.cv["map"].draw()

    def _draw_hist(self):
        fig = self.cv["hist"].fig
        fig.clear()
        cr = self.result
        if cr is None:
            self.cv["hist"].draw()
            return
        c = cr.cells[cr.cells.analysed] if len(cr.cells) else cr.cells
        items = [("rate_per_min", "events / min"), ("mean_isi_s", "mean ISI (s)"),
                 ("participation", "wave participation"), ("fraction_events_in_waves", "events in waves"),
                 ("mean_amplitude_pct", "amplitude (% F0)"), ("mean_decay_tau_s", "decay τ (s)")]
        for i, (k, lab) in enumerate(items):
            ax = fig.add_subplot(2, 3, i + 1)
            v = c[k].dropna().to_numpy(float) if k in c else np.array([])
            if len(v):
                ax.hist(v, bins=20, color="#4c72b0")
                ax.axvline(np.mean(v), color="r", lw=1)
            ax.set_title(f"{lab}\nmean {np.mean(v):.3g}" if len(v) else lab, fontsize=8)
            ax.tick_params(labelsize=7)
        self.cv["hist"].draw()

    def _draw_waves(self, *_):
        fig = self.cv["waves"].fig
        fig.clear()
        cr = self.result
        if cr is None or not len(cr.waves):
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No waves linked yet", ha="center", transform=ax.transAxes, color="gray")
            ax.axis("off")
            self.cv["waves"].draw()
            return
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(cr.waves)), cr.waves.fraction_recruited, color="#55a868")
        ax1.set_xlabel("wave #")
        ax1.set_ylabel("fraction of cells recruited")
        j = min(self.wave_spin.value(), len(cr.waves) - 1)
        ax1.bar([j], [cr.waves.fraction_recruited.iloc[j]], color="orange")
        w = cr.waves.iloc[j]
        ax2 = fig.add_subplot(122)
        sub = cr.events[cr.events.wave == w.wave]
        if len(sub):
            cx = cr.cells.set_index("cell").loc[sub.cell]
            s = self._um() or 1.0
            sc = ax2.scatter(cx.x_px * s, cx.y_px * s, c=sub.time_s - w.start_s, cmap="plasma", s=25)
            fig.colorbar(sc, ax=ax2, label="latency (s)")
            ax2.invert_yaxis()
            ax2.set_aspect("equal")
        ax2.set_title(f"wave {int(w.wave)}: {int(w.cells_recruited)}/{int(w.cells_passed)} cells, "
                      f"speed from cells {_num(w.cell_speed)} (R² {_num(w.cell_speed_r2)})", fontsize=8)
        self.cv["waves"].draw()

    def _copart_window(self):
        from wave_gui import MplCanvas
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Same waves? – co-participation")
        dlg.resize(1400, 1000)
        lay = QtWidgets.QVBoxLayout(dlg)
        c = MplCanvas(figsize=(14, 10))
        lay.addWidget(c)
        self._draw_copart(canvas=c, panel=4)
        dlg.show()
        self._cp_dlg = dlg

    def _draw_copart(self, *_, canvas=None, panel=None):
        cvs = canvas or self.cv["copart"]
        fig = cvs.fig
        fig.clear()
        cr = self.result
        cp = getattr(cr, "copart", None) if cr is not None else None
        if cp is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Needs at least 3 linked waves and 3 cells\n(run tab 4, then 'Detect events + link "
                    "to waves')", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")
            cvs.draw()
            return
        panel = self.cp_panel.currentIndex() if panel is None else panel
        sm = cp["summary"]
        um = self._um()
        unit = "µm" if um else "px"
        fs = 9
        axes = ([fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]
                if panel == 4 else [None] * 4)
        if panel != 4:
            axes[panel] = fig.add_subplot(111)

        if axes[0] is not None:              # cell x cell
            ax = axes[0]
            o = cp["order"]
            z = cp["z"][np.ix_(o, o)].copy()
            np.fill_diagonal(z, np.nan)
            im = ax.imshow(z, cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest")
            fig.colorbar(im, ax=ax, label="z vs chance", shrink=0.8)
            f1 = sm['cells.copart_fraction_pairs_sig'] * 100
            f2 = sm.get('cells.copart_fraction_beyond_distance', np.nan) * 100
            ax.set_title(f"Cell pairs in the same waves (cells clustered)\n{f1:.1f}% of "
                         f"{int(sm['cells.copart_pairs_tested'])} pairs more than chance, "
                         f"{f2:.1f}% beyond what distance explains", fontsize=fs)
            ax.set_xticks([])
            ax.set_yticks([])
        if axes[1] is not None:              # wave x wave
            ax = axes[1]
            ws = cp["wave_sim"].copy()
            np.fill_diagonal(ws, np.nan)
            vmax = max(0.05, float(np.nanpercentile(ws, 99))) if np.isfinite(ws).any() else 1
            im = ax.imshow(ws, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            fig.colorbar(im, ax=ax, label="Jaccard overlap", shrink=0.8)
            ids = cp["wave_ids"]
            n = len(ids)
            stp = max(1, int(np.ceil(n / (30 if panel != 4 else 12))))
            ax.set_xticks(range(0, n, stp))
            ax.set_xticklabels([str(ids[i]) for i in range(0, n, stp)], fontsize=7, rotation=90)
            ax.set_yticks(range(0, n, stp))
            ax.set_yticklabels([str(ids[i]) for i in range(0, n, stp)], fontsize=7)
            ax.set_xlabel("wave", fontsize=fs)
            ax.set_ylabel("wave", fontsize=fs)
            ax.set_title(f"Do waves recruit the same cells?\nmean overlap {_num(sm['cells.wave_similarity_jaccard'])}"
                         f" vs chance {_num(sm['cells.wave_similarity_chance'])}, "
                         f"p = {_num(sm['cells.wave_similarity_p'])}", fontsize=fs)
        if axes[2] is not None:              # distance
            ax = axes[2]
            dd = cp["dist"]
            if len(dd):
                mid = (dd.d_from + dd.d_to) / 2
                ax.plot(mid, dd.observed, "o-", color="#c44e52", label="observed")
                ax.plot(mid, dd.expected, "s--", color="0.5", label="chance")
                ax.legend(fontsize=8)
            ax.set_xlabel(f"distance between the two cells ({unit})", fontsize=fs)
            ax.set_ylabel("waves together per pair", fontsize=fs)
            ax.set_title("Does it depend on distance?", fontsize=fs)
        if axes[3] is not None:              # map
            ax = axes[3]
            cells = cr.cells
            k = um or 1.0
            ax.scatter(cells.x_px * k, cells.y_px * k, s=10, color="0.65", zorder=1)
            pr = cp["pairs"]
            beyond = self.cp_map.currentIndex() == 0 and "z_vs_distance" in pr
            sig = (pr[pr.z_vs_distance > 3].sort_values("z_vs_distance", ascending=False) if beyond
                   else pr[pr.q < 0.05])
            n_all = len(sig)
            sig = sig.head(300)
            if len(sig):
                from matplotlib.collections import LineCollection
                ci = cells.set_index("cell")
                a, b = ci.loc[sig.cell_a], ci.loc[sig.cell_b]
                segs = np.stack([np.c_[a.x_px * k, a.y_px * k], np.c_[b.x_px * k, b.y_px * k]], axis=1)
                val = (sig.z_vs_distance if beyond else sig.z).to_numpy()
                lc = LineCollection(segs, cmap="autumn_r", linewidths=1.0, alpha=0.8, zorder=2)
                lc.set_array(val)
                ax.add_collection(lc)
                fig.colorbar(lc, ax=ax, label="z vs distance" if beyond else "z", shrink=0.8)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel(f"x ({unit})", fontsize=fs)
            ax.set_ylabel(f"y ({unit})", fontsize=fs)
            ax.set_title(("Pairs together beyond distance" if beyond else "Pairs together more than chance")
                         + f" ({n_all}" + (", strongest 300 shown)" if n_all > 300 else ")"), fontsize=fs)
            ax.tick_params(labelsize=7)
        cvs.draw()

def cv2_resize(im, W, H):
    import cv2
    return cv2.resize(im.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def _pct(v):
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.0f}%"


def _num(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.3g}"
