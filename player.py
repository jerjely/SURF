"""Fast video player for the Info tab.

* A background thread decodes the video once (sequentially - the fast way for H.264) into a
  small in-memory preview (every n-th frame, ~320 px wide, <= ~350 MB). Scrubbing and fast
  playback show the preview instantly; when you stop, the exact full-resolution frame replaces it.
* Brightness / contrast / gamma / channel are applied through a 256-entry lookup table, so
  changing them is instant even while playing.
Keys: Space play/pause · ←/→ 1 frame · Shift+←/→ 1 s · PgUp/PgDn 10 s · Home/End.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

PREVIEW_W = 320
PREVIEW_BUDGET = 350e6          # bytes
CHANNELS = ["Colour", "Grey", "Green", "Red", "Blue"]


class PreviewCache:
    """Every `step`-th frame at PREVIEW_W wide, filled by a background thread."""

    def __init__(self, path, n, w, h):
        self.path = path
        sw = min(PREVIEW_W, w)
        sh = max(1, int(round(h * sw / max(w, 1))))
        per = sw * sh * 3
        self.m = max(1, min(n, int(PREVIEW_BUDGET // per)))
        self.step = max(1, int(np.ceil(n / self.m)))
        self.m = int(np.ceil(n / self.step))
        self.size = (sw, sh)
        self.frames = np.zeros((self.m, sh, sw, 3), np.uint8)
        self.have = np.zeros(self.m, bool)
        self.done = 0                     # frames decoded (for the progress label)
        self.n = n
        self._stop = False
        self.error = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            cap = cv2.VideoCapture(self.path)
            i = 0
            while not self._stop and i < self.n:
                if i % self.step == 0:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    self.frames[i // self.step] = cv2.resize(fr, self.size, interpolation=cv2.INTER_AREA)
                    self.have[i // self.step] = True
                else:
                    if not cap.grab():
                        break
                i += 1
                self.done = i
            cap.release()
        except Exception as e:           # never crash the GUI over the preview
            self.error = str(e)

    def stop(self):
        self._stop = True

    def get(self, i):
        """Nearest cached preview frame for frame i (None if not decoded yet)."""
        j = min(int(round(i / self.step)), self.m - 1)
        if self.have[j]:
            return self.frames[j]
        j = min(i // self.step, self.m - 1)
        return self.frames[j] if self.have[j] else None

    @property
    def progress(self):
        return self.done / max(self.n, 1)


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else {}
        self.path = None
        self.cache = None
        self.cap = None
        self.cap_next = -1
        self.n = 0
        self.fps = 25.0
        self.w = self.h = 0
        self.cur_full = None
        self._lut = None
        self._playing = False
        self._shown = 0

        # ---- image
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#151515")
        self.vb = self.glw.addViewBox(lockAspect=True, invertY=True)
        self.vb.setMenuEnabled(False)
        self.img = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.img)
        self.overlay = pg.TextItem("", color=(255, 255, 255), anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 140))
        self.vb.addItem(self.overlay)
        self.overlay.setVisible(False)

        # ---- transport
        st = self.style()
        self.play_btn = QtWidgets.QToolButton()
        self.play_btn.setCheckable(True)
        self.play_btn.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.play_btn.setToolTip("Play / pause (Space)")
        self.play_btn.toggled.connect(self._toggle_play)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×", "8×", "16×", "32×", "64×"])
        self.speed.setCurrentText("1×")
        self.speed.currentIndexChanged.connect(self._restart_clock)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._slider_changed)
        self.slider.sliderReleased.connect(self._load_full_soon)
        self.time_lbl = QtWidgets.QLabel("–")
        self.time_lbl.setMinimumWidth(190)
        f = self.time_lbl.font()
        f.setFamily("Consolas")
        self.time_lbl.setFont(f)
        self.goto = QtWidgets.QLineEdit()
        self.goto.setPlaceholderText("go to  m:ss / s")
        self.goto.setFixedWidth(110)
        self.goto.returnPressed.connect(self._goto)
        steps = []
        for txt, d, tip in (("⏮", "home", "start (Home)"), ("−10 s", -10.0, "PgUp"), ("−1 s", -1.0, "Shift+←"),
                            ("−1", -1, "previous frame (←)"), ("+1", 1, "next frame (→)"),
                            ("+1 s", 1.0, "Shift+→"), ("+10 s", 10.0, "PgDn"), ("⏭", "end", "end (End)")):
            b = QtWidgets.QToolButton()
            b.setText(txt)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, d=d: self._jump(d))
            steps.append(b)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.play_btn)
        row.addWidget(self.speed)
        for b in steps:
            row.addWidget(b)
        row.addWidget(self.time_lbl)
        row.addStretch(1)
        row.addWidget(self.goto)

        # ---- display adjustments
        ps = self.settings.setdefault("player", {})
        self.chan = QtWidgets.QComboBox()
        self.chan.addItems(CHANNELS)
        self.chan.setCurrentText(ps.get("channel", "Colour"))
        self.lo = self._slider(0, 254, ps.get("lo", 0))
        self.hi = self._slider(1, 255, ps.get("hi", 255))
        self.gamma = self._slider(20, 300, int(ps.get("gamma", 1.0) * 100))
        self.auto_chk = QtWidgets.QCheckBox("auto (each frame)")
        self.auto_chk.setChecked(ps.get("auto", False))
        b_auto = QtWidgets.QPushButton("Auto once")
        b_auto.setToolTip("Set black / white levels from the current frame (0.5-99.5 %)")
        b_auto.clicked.connect(self._auto_once)
        b_reset = QtWidgets.QPushButton("Reset")
        b_reset.clicked.connect(self._reset_display)
        b_fit = QtWidgets.QPushButton("Fit view")
        b_fit.clicked.connect(lambda: self.vb.autoRange(padding=0.01))
        self.lvl_lbl = QtWidgets.QLabel("")
        self.lvl_lbl.setStyleSheet("color: gray")
        for w in (self.lo, self.hi, self.gamma):
            w.valueChanged.connect(self._display_changed)
        self.chan.currentIndexChanged.connect(self._display_changed)
        self.auto_chk.toggled.connect(self._display_changed)
        adj = QtWidgets.QGridLayout()
        adj.addWidget(QtWidgets.QLabel("View"), 0, 0)
        adj.addWidget(self.chan, 0, 1)
        adj.addWidget(QtWidgets.QLabel("Black"), 0, 2)
        adj.addWidget(self.lo, 0, 3)
        adj.addWidget(QtWidgets.QLabel("White"), 0, 4)
        adj.addWidget(self.hi, 0, 5)
        adj.addWidget(QtWidgets.QLabel("Gamma"), 0, 6)
        adj.addWidget(self.gamma, 0, 7)
        adj.addWidget(self.auto_chk, 0, 8)
        adj.addWidget(b_auto, 0, 9)
        adj.addWidget(b_reset, 0, 10)
        adj.addWidget(b_fit, 0, 11)
        for c in (3, 5, 7):
            adj.setColumnStretch(c, 1)
        self.cache_lbl = QtWidgets.QLabel("")
        self.cache_lbl.setStyleSheet("color: gray")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.slider)
        lay.addLayout(row)
        lay.addLayout(adj)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.lvl_lbl)
        h.addStretch(1)
        h.addWidget(self.cache_lbl)
        lay.addLayout(h)

        self._full_timer = QtCore.QTimer(self, singleShot=True, interval=110)
        self._full_timer.timeout.connect(self._show_full)
        self._play_timer = QtCore.QTimer(self, interval=10)
        self._play_timer.timeout.connect(self._tick)
        self._cache_timer = QtCore.QTimer(self, interval=400)
        self._cache_timer.timeout.connect(self._cache_status)
        for key, fn in ((Qt.Key_Space, self.play_btn.toggle),
                        (Qt.Key_Left, lambda: self._jump(-1)), (Qt.Key_Right, lambda: self._jump(1)),
                        (QtGui.QKeySequence(Qt.SHIFT | Qt.Key_Left), lambda: self._jump(-1.0)),
                        (QtGui.QKeySequence(Qt.SHIFT | Qt.Key_Right), lambda: self._jump(1.0)),
                        (Qt.Key_PageUp, lambda: self._jump(-10.0)), (Qt.Key_PageDown, lambda: self._jump(10.0)),
                        (Qt.Key_Home, lambda: self._jump("home")), (Qt.Key_End, lambda: self._jump("end"))):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=fn, context=Qt.WidgetWithChildrenShortcut)
        self.setFocusPolicy(Qt.StrongFocus)
        self._build_lut()

    @staticmethod
    def _slider(lo, hi, v):
        s = QtWidgets.QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(int(v))
        return s

    # ------------------------------------------------------------------ source
    def set_video(self, path, n, fps, w, h):
        self.release()
        self.path, self.n, self.fps, self.w, self.h = path, max(int(n), 1), max(float(fps), 1e-6), w, h
        self.cap = cv2.VideoCapture(path)
        self.cap_next = 0
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.cache = None
        if self.isVisible():
            self._start_cache()
        self._show_full()
        QtCore.QTimer.singleShot(0, lambda: self.vb.autoRange(padding=0.01))

    def _start_cache(self):
        """Decode the preview only while the player is on screen (it competes for CPU)."""
        if self.cache is None and self.path:
            self.cache = PreviewCache(self.path, self.n, self.w, self.h)
            self._cache_timer.start()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._start_cache()

    def release(self):
        self.play_btn.setChecked(False)
        if self.cache:
            self.cache.stop()
            self.cache = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._cache_timer.stop()

    def _read_full(self, i):
        """Exact frame i at full resolution (sequential reads are cheap, seeks are not)."""
        if self.cap is None:
            return None
        gap = i - self.cap_next
        if not (0 <= gap <= 30):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        else:
            for _ in range(gap):
                self.cap.grab()
        ok, fr = self.cap.read()
        self.cap_next = i + 1 if ok else -10 ** 9
        return fr if ok else None

    # ------------------------------------------------------------------ display
    def _build_lut(self):
        lo, hi = self.lo.value(), max(self.hi.value(), self.lo.value() + 1)
        g = self.gamma.value() / 100.0
        x = np.clip((np.arange(256, dtype=np.float32) - lo) / (hi - lo), 0, 1)
        self._lut = (np.power(x, 1.0 / g) * 255 + 0.5).astype(np.uint8)
        self.lvl_lbl.setText(f"black {lo} · white {hi} · gamma {g:.2f}")

    def _display_changed(self, *_):
        ps = self.settings.setdefault("player", {})
        ps.update(channel=self.chan.currentText(), lo=self.lo.value(), hi=self.hi.value(),
                  gamma=self.gamma.value() / 100.0, auto=self.auto_chk.isChecked())
        self._build_lut()
        self._redraw()

    def _channel(self, fr):
        c = self.chan.currentText()
        if c == "Colour":
            return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        if c == "Grey":
            return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        return fr[:, :, {"Blue": 0, "Green": 1, "Red": 2}[c]]

    def _auto_levels(self, im):
        lo, hi = np.percentile(im[::4, ::4], (0.5, 99.5))
        return int(lo), int(max(hi, lo + 1))

    def _auto_once(self):
        if self._last is None:
            return
        lo, hi = self._auto_levels(self._channel(self._last))
        for w, v in ((self.lo, lo), (self.hi, hi)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._display_changed()

    def _reset_display(self):
        for w, v in ((self.lo, 0), (self.hi, 255), (self.gamma, 100)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.auto_chk.setChecked(False)
        self._display_changed()

    _last = None
    _last_is_full = False

    def _show(self, fr, full):
        self._last, self._last_is_full = fr, full
        self._redraw()

    dff = None          # dict(f0d, low, high, mask, maskarr, lut) when the dF/F view is on

    def set_dff(self, d):
        self.dff = d
        self._redraw()

    def _redraw(self):
        fr = self._last
        if fr is None:
            return
        if self.dff is not None:
            import cells_core as cc
            d = self.dff
            rgb = cc.dff_rgb(fr, d["f0d"], d["low"], d["high"], d["mask"], d["lut"], mask=d["maskarr"])
            self.img.setImage(rgb, autoLevels=False, levels=None)
            self.img.setRect(QtCore.QRectF(0, 0, self.w or rgb.shape[1], self.h or rgb.shape[0]))
            i = self.slider.value()
            self.time_lbl.setText(f"{int(i / self.fps // 60)}:{i / self.fps % 60:05.2f}  #{i}  ·dF/F"
                                  + ("" if self._last_is_full else " preview"))
            return
        im = self._channel(fr)
        if self.auto_chk.isChecked():
            lo, hi = self._auto_levels(im)
            x = np.clip((np.arange(256, dtype=np.float32) - lo) / max(hi - lo, 1), 0, 1)
            lut = (np.power(x, 100.0 / self.gamma.value()) * 255 + 0.5).astype(np.uint8)
        else:
            lut = self._lut
        im = cv2.LUT(im, lut)
        # preview frames are drawn at full-frame coordinates so zoom / pan stay put
        self.img.setImage(im, autoLevels=False, levels=None)     # uint8 straight to QImage (no rescale)
        self.img.setRect(QtCore.QRectF(0, 0, self.w or im.shape[1], self.h or im.shape[0]))
        i = self.slider.value()
        t = i / self.fps
        tot = self.n / self.fps
        self.time_lbl.setText(f"{int(t // 60)}:{t % 60:05.2f} / {int(tot // 60)}:{tot % 60:05.2f}  #{i}"
                              + ("" if self._last_is_full else "  ·preview"))

    # ------------------------------------------------------------------ navigation
    def _slider_changed(self, i):
        if self._playing:
            return
        fr = self.cache.get(i) if self.cache else None
        if fr is not None and self.slider.isSliderDown():
            self._show(fr, False)              # instant while dragging
        elif fr is not None:
            self._show(fr, False)
        self._load_full_soon()

    def _load_full_soon(self):
        if not self._playing:
            self._full_timer.start()

    def _show_full(self):
        fr = self._read_full(self.slider.value())
        if fr is not None:
            self._show(fr, True)

    def _jump(self, d):
        if not self.n:
            return
        i = self.slider.value()
        if d == "home":
            i = 0
        elif d == "end":
            i = self.n - 1
        elif isinstance(d, float):
            i += int(round(d * self.fps))
        else:
            i += d
        self.slider.setValue(int(np.clip(i, 0, self.n - 1)))

    def _goto(self):
        txt = self.goto.text().strip().replace(",", ".")
        try:
            if ":" in txt:
                m, s = txt.split(":", 1)
                t = int(m) * 60 + float(s)
            else:
                t = float(txt)
        except ValueError:
            return
        self.slider.setValue(int(np.clip(round(t * self.fps), 0, self.n - 1)))
        self.setFocus()

    # ------------------------------------------------------------------ playback
    def _rate(self):
        return float(self.speed.currentText().rstrip("×"))

    def _restart_clock(self, *_):
        self._t0 = time.perf_counter()
        self._f0 = self.slider.value()
        self._shown = 0

    def _toggle_play(self, on):
        if on and not self.path:
            self.play_btn.setChecked(False)
            return
        self._playing = on
        self.play_btn.setIcon(self.style().standardIcon(
            QtWidgets.QStyle.SP_MediaPause if on else QtWidgets.QStyle.SP_MediaPlay))
        if on:
            if self.slider.value() >= self.n - 1:
                self.slider.setValue(0)
            self._restart_clock()
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self._show_full()

    def _tick(self):
        el = time.perf_counter() - self._t0
        target = self._f0 + int(el * self.fps * self._rate())
        if target >= self.n - 1:
            self.slider.setValue(self.n - 1)
            self.play_btn.setChecked(False)
            return
        if target == self.slider.value():
            return
        self.slider.blockSignals(True)
        self.slider.setValue(target)
        self.slider.blockSignals(False)
        # real-time speeds: exact full frames (sequential decoding keeps up);
        # fast speeds: preview frames, which cost nothing
        if self.fps * self._rate() <= 80 or self.cache is None or self.cache.get(target) is None:
            fr = self._read_full(target)
            if fr is not None:
                self._show(fr, True)
        else:
            self._show(self.cache.get(target), False)
        self._shown += 1

    def _cache_status(self):
        c = self.cache
        if c is None:
            return
        if c.error:
            self.cache_lbl.setText(f"preview failed: {c.error}")
            self._cache_timer.stop()
        elif c.progress >= 1 or not c.t.is_alive():
            self.cache_lbl.setText(f"fast scrubbing ready (preview every {c.step} frame"
                                   f"{'s' if c.step > 1 else ''})")
            self._cache_timer.stop()
        else:
            self.cache_lbl.setText(f"building fast-scrub preview… {c.progress * 100:.0f}% "
                                   "(the decoded part can already be scrubbed)")

    def hideEvent(self, ev):
        try:
            self.play_btn.setChecked(False)
        except RuntimeError:          # app closing, widgets already gone
            pass
        super().hideEvent(ev)
