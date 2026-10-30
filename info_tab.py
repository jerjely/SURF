"""Tab 0 - recording info: animal, age, condition, ... plus objective / turret / eye.

Saved in the video's session file (<video>_wavegui.json, key "info"), written into the
result workbook (summary sheet, info.* rows) and picked up by the Compare tab.
The objective, turret and eye widgets mirror the ones on the Rotate tab.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

import wave_core as wc

FIELDS = [  # key, label, kind
    ("animal", "Animal ID", "combo"),
    ("age", "Age", "combo"),
    ("condition", "Condition", "combo"),
    ("treatment", "Treatment / drug, dose", "combo"),
    ("genotype", "Genotype / line", "combo"),
    ("sex", "Sex", "sex"),
    ("date", "Recording date", "line"),
    ("experimenter", "Experimenter", "combo"),
    ("notes", "Notes", "text"),
]
COPY_KEYS = [k for k, _, _ in FIELDS if k != "notes"]


def guess_age(path):
    m = re.findall(r"(?<![A-Za-z0-9])P(\d{1,3})(?![0-9])", path)
    return f"P{m[-1]}" if m else ""


class ApplyDialog(QtWidgets.QDialog):
    """Pick fields and target videos for 'Apply to other videos'."""

    def __init__(self, videos, current, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply info to other videos")
        self.resize(520, 560)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Copy these fields:"))
        self.f_checks = {}
        grid = QtWidgets.QGridLayout()
        for i, (k, lab, _) in enumerate(FIELDS):
            if k == "notes":
                continue
            c = QtWidgets.QCheckBox(f"{lab}: {values.get(k) or '–'}")
            c.setChecked(k in ("animal", "age", "genotype", "sex", "date", "experimenter") and bool(values.get(k)))
            grid.addWidget(c, i // 2, i % 2)
            self.f_checks[k] = c
        self.c_obj = QtWidgets.QCheckBox("Objective + turret")
        self.c_eye = QtWidgets.QCheckBox("Eye")
        grid.addWidget(self.c_obj, 10, 0)
        grid.addWidget(self.c_eye, 10, 1)
        lay.addLayout(grid)
        lay.addWidget(QtWidgets.QLabel("…to these videos:"))
        self.lst = QtWidgets.QListWidget()
        for full, label in videos:
            if os.path.abspath(full) == os.path.abspath(current):
                continue
            it = QtWidgets.QListWidgetItem(label)
            it.setData(Qt.UserRole, full)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.lst.addItem(it)
        lay.addWidget(self.lst, 1)
        b_all = QtWidgets.QPushButton("Tick all")
        b_all.clicked.connect(lambda: [self.lst.item(i).setCheckState(Qt.Checked) for i in range(self.lst.count())])
        b_none = QtWidgets.QPushButton("Untick all")
        b_none.clicked.connect(lambda: [self.lst.item(i).setCheckState(Qt.Unchecked) for i in range(self.lst.count())])
        h = QtWidgets.QHBoxLayout()
        h.addWidget(b_all)
        h.addWidget(b_none)
        h.addStretch(1)
        lay.addLayout(h)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def chosen(self):
        keys = [k for k, c in self.f_checks.items() if c.isChecked()]
        vids = [self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())
                if self.lst.item(i).checkState() == Qt.Checked]
        return keys, self.c_obj.isChecked(), self.c_eye.isChecked(), vids


class InfoTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._loading = False
        from player import VideoPlayer
        self.player = VideoPlayer(main.settings)
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter()
        outer.addWidget(split)
        split.addWidget(self.player)
        form_w = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(form_w)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QtWidgets.QFrame.NoFrame)
        sc.setWidget(form_w)
        sc.setMinimumWidth(380)
        split.addWidget(sc)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 420])
        self._form_w = form_w

        self.video_lbl = QtWidgets.QLabel("Open a folder and select a video.")
        self.video_lbl.setWordWrap(True)
        col.addWidget(self.video_lbl)

        g = QtWidgets.QGroupBox("Recording")
        f = QtWidgets.QFormLayout(g)
        self.w = {}
        for k, lab, kind in FIELDS:
            if kind == "combo":
                w = QtWidgets.QComboBox()
                w.setEditable(True)
                w.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
                w.lineEdit().editingFinished.connect(self._edited)
                w.activated.connect(self._edited)
            elif kind == "sex":
                w = QtWidgets.QComboBox()
                w.addItems(["", "female", "male", "mixed / pooled", "unknown"])
                w.currentIndexChanged.connect(self._edited)
            elif kind == "line":
                w = QtWidgets.QLineEdit()
                w.editingFinished.connect(self._edited)
            else:
                w = QtWidgets.QPlainTextEdit()
                w.setFixedHeight(90)
                self._notes_timer = QtCore.QTimer(self, singleShot=True, interval=800)
                self._notes_timer.timeout.connect(self._edited)
                w.textChanged.connect(lambda: self._loading or self._notes_timer.start())
            f.addRow(lab, w)
            self.w[k] = w
        self.w["animal"].lineEdit().setPlaceholderText("e.g. M12 – same ID for every video of this animal")
        self.w["age"].lineEdit().setPlaceholderText("e.g. P7")
        self.w["condition"].lineEdit().setPlaceholderText("e.g. control, drug A")
        self.w["date"].setPlaceholderText("YYYY-MM-DD")
        col.addWidget(g)

        g2 = QtWidgets.QGroupBox("Microscope and eye (same settings as the Rotate tab)")
        f2 = QtWidgets.QFormLayout(g2)
        self.obj = QtWidgets.QComboBox()
        self.tur = QtWidgets.QComboBox()
        self.obj.currentTextChanged.connect(self._obj_changed)
        self.tur.currentTextChanged.connect(self._tur_changed)
        self.scale_lbl = QtWidgets.QLabel("")
        self.scale_lbl.setWordWrap(True)
        self.eye_r = QtWidgets.QRadioButton("Right retina")
        self.eye_l = QtWidgets.QRadioButton("Left retina")
        self.eye_r.setChecked(True)
        self.eye_r.toggled.connect(self._eye_changed)
        eh = QtWidgets.QHBoxLayout()
        eh.addWidget(self.eye_r)
        eh.addWidget(self.eye_l)
        eh.addStretch(1)
        f2.addRow("Objective", self.obj)
        f2.addRow("Turret", self.tur)
        f2.addRow("", self.scale_lbl)
        f2.addRow("Eye", eh)
        col.addWidget(g2)

        # ---- dF/F view (deltaF_F prepare / viewer notebooks) ----
        g3 = QtWidgets.QGroupBox("ΔF/F view (F0 = temporal mean, as the Ackman 2012 macro)")
        f3 = QtWidgets.QFormLayout(g3)
        self.dff_chk = QtWidgets.QCheckBox("Show ΔF/F in the player")
        self.dff_chk.toggled.connect(self._dff_changed)
        self.dff_skip = QtWidgets.QDoubleSpinBox(); self.dff_skip.setRange(0, 1e5); self.dff_skip.setValue(1.0)
        self.dff_skip.setSuffix(" s"); self.dff_skip.setToolTip("Dark / warm-up frames at the start, left out of F0")
        self.dff_end = QtWidgets.QDoubleSpinBox(); self.dff_end.setRange(-1, 1e6); self.dff_end.setValue(-1)
        self.dff_end.setSuffix(" s"); self.dff_end.setSpecialValueText("end")
        self.dff_lo = QtWidgets.QDoubleSpinBox(); self.dff_lo.setRange(-5, 20); self.dff_lo.setValue(3.0)
        self.dff_lo.setSingleStep(0.1); self.dff_lo.setPrefix("−"); self.dff_lo.setSuffix(" σ")
        self.dff_hi = QtWidgets.QDoubleSpinBox(); self.dff_hi.setRange(0, 40); self.dff_hi.setValue(7.0)
        self.dff_hi.setSingleStep(0.1); self.dff_hi.setPrefix("+"); self.dff_hi.setSuffix(" σ")
        self.dff_mask = QtWidgets.QSpinBox(); self.dff_mask.setRange(0, 95); self.dff_mask.setValue(30)
        self.dff_mask.setSuffix(" %"); self.dff_mask.setToolTip("Black out pixels whose F0 is below this percentile")
        for w in (self.dff_lo, self.dff_hi, self.dff_mask):
            w.valueChanged.connect(self._dff_changed)
        b_f0 = QtWidgets.QPushButton("Compute F0")
        b_f0.clicked.connect(self._compute_f0)
        self.dff_speed = QtWidgets.QComboBox(); self.dff_speed.setEditable(True)
        self.dff_speed.addItems(["1", "2", "4", "8"]); self.dff_speed.setCurrentText("4")
        b_ren = QtWidgets.QPushButton("Render ΔF/F video…")
        b_ren.clicked.connect(self._render_dff)
        self.dff_lbl = QtWidgets.QLabel("F0 not computed yet.")
        self.dff_lbl.setWordWrap(True)
        self.dff_lbl.setStyleSheet("color: gray")
        f3.addRow(self.dff_chk)
        f3.addRow("Skip start", self.dff_skip)
        f3.addRow("F0 until", self.dff_end)
        f3.addRow("Black at", self.dff_lo)
        f3.addRow("White at", self.dff_hi)
        f3.addRow("Background mask", self.dff_mask)
        h3 = QtWidgets.QHBoxLayout(); h3.addWidget(b_f0); h3.addWidget(QtWidgets.QLabel("speed ×"))
        h3.addWidget(self.dff_speed); h3.addWidget(b_ren)
        f3.addRow(h3)
        f3.addRow(self.dff_lbl)
        col.addWidget(g3)
        self.f0d = None

        self.copy_btn = QtWidgets.QPushButton("Copy info from another video…")
        self.copy_btn.setCheckable(True)
        self.copy_btn.toggled.connect(lambda on: self.main._pick_toggled("info", on))
        b_apply = QtWidgets.QPushButton("Apply to other videos…")
        b_apply.setToolTip("Write animal / age / … to several videos at once, e.g. all videos of one animal")
        b_apply.clicked.connect(self._apply_to_others)
        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.copy_btn)
        h.addWidget(b_apply)
        col.addLayout(h)
        hint = QtWidgets.QLabel("Saved with the video and in its result workbook; the Compare tab uses "
                                "Animal, Age, Condition and Objective. Videos of the same animal must share "
                                "the same Animal ID for the paired / per-animal statistics.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        col.addWidget(hint)
        col.addStretch(1)
        form_w.setEnabled(False)

    # ------------------------------------------------------------------ values
    def values(self):
        out = {}
        for k, _, kind in FIELDS:
            w = self.w[k]
            if kind in ("combo", "sex"):
                out[k] = w.currentText().strip()
            elif kind == "line":
                out[k] = w.text().strip()
            else:
                out[k] = w.toPlainText().strip()
        return out

    def _set(self, k, v):
        w = self.w[k]
        kind = dict((a, c) for a, _, c in FIELDS)[k]
        v = "" if v is None else str(v)
        if kind in ("combo", "sex"):
            if w.findText(v) < 0 and kind == "combo" and v:
                w.addItem(v)
            w.setCurrentText(v)
        elif kind == "line":
            w.setText(v)
        else:
            w.setPlainText(v)

    def _known_values(self):
        """Values used in the other videos of this folder (for the drop-downs)."""
        known = {k: set() for k, _, _ in FIELDS}
        for full in self.main._listed_videos():
            try:
                with open(wc.session_path(full), encoding="utf-8") as f:
                    inf = json.load(f).get("info") or {}
            except Exception:
                continue
            for k, v in inf.items():
                if k in known and v:
                    known[k].add(str(v))
        return known

    def load_video(self):
        m = self.main
        self._loading = True
        self._form_w.setEnabled(bool(m.video))
        if m.video:
            i = m.info or {}
            self.player.set_video(m.video, i.get("n_frames", 1), i.get("fps", 25.0),
                                  i.get("width", 0), i.get("height", 0))
            self._load_f0()
            self.video_lbl.setText(f"<b>{os.path.basename(m.video)}</b> · {i.get('width')}×{i.get('height')} px · "
                                   f"{i.get('fps', 0):.2f} fps · {i.get('n_frames', 0) / max(i.get('fps', 1), 1e-9) / 60:.1f} min"
                                   f"<br><span style='color:gray'>{os.path.dirname(m.video)}</span>")
            known = self._known_values()
            for k, _, kind in FIELDS:
                if kind == "combo":
                    w = self.w[k]
                    w.blockSignals(True)
                    w.clear()
                    w.addItems([""] + sorted(known[k]))
                    w.blockSignals(False)
            inf = dict(m.session.get("info") or {})
            if not inf:                      # first time: sensible guesses (saved once you edit)
                inf["age"] = guess_age(m.video)
                try:
                    inf["date"] = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(m.video)))
                except OSError:
                    pass
            for k, _, _ in FIELDS:
                self._set(k, inf.get(k, ""))
        self._loading = False
        self.sync_from_main()

    # ------------------------------------------------------------------ dF/F
    def _f0_path(self):
        return os.path.splitext(self.main.video)[0] + "_F0.npz" if self.main.video else None

    def _load_f0(self):
        import cells_core as cc
        self.f0d = None
        p = self._f0_path()
        if p and os.path.exists(p):
            try:
                self.f0d = cc.load_f0(p)
            except Exception:
                self.f0d = None
        self._dff_status()
        self._dff_changed()

    def _dff_status(self):
        d = self.f0d
        if d is None:
            self.dff_lbl.setText("F0 not computed yet – press Compute F0 (one pass through the video).")
        else:
            self.dff_lbl.setText(f"F0 ready: {d['F0'].shape[1]}×{d['F0'].shape[0]} (2×2 binned), frames "
                                 f"{d['skip']}–{d['end']}, ΔF/F σ = {d['std']:.4f} → black {-self.dff_lo.value() * d['std']:.3f}, "
                                 f"white {self.dff_hi.value() * d['std']:.3f}")

    def _dff_changed(self, *_):
        import cells_core as cc
        if self.dff_chk.isChecked() and self.f0d is not None:
            F0 = self.f0d["F0"]
            self.player.set_dff(dict(f0d=self.f0d, low=self.dff_lo.value(), high=self.dff_hi.value(),
                                     mask=self.dff_mask.value(), lut=cc.lut_imagej16(),
                                     maskarr=F0 < np.percentile(F0, self.dff_mask.value())))
            self._dff_status()
        else:
            if self.dff_chk.isChecked() and self.f0d is None:
                self.dff_lbl.setText("<span style='color:#d33'>Compute F0 first.</span>")
            self.player.set_dff(None)

    def _compute_f0(self):
        import cells_core as cc
        m = self.main
        if not m.video:
            return
        fps = (m.info or {}).get("fps", 25.0)
        skip = int(round(self.dff_skip.value() * fps))
        end = None if self.dff_end.value() < 0 else int(round(self.dff_end.value() * fps))
        path, out = m.video, self._f0_path()

        def job(progress=None, cancel=None):
            d = cc.compute_f0(path, skip, end, progress=progress, cancel=cancel)
            cc.save_f0(out, d)
            return d

        def done(d):
            self.f0d = d
            self.dff_chk.setChecked(True)
            self._dff_changed()

        m.run_task(job, on_done=done, label="Computing F0")

    def _render_dff(self):
        import cells_core as cc
        m = self.main
        if self.f0d is None:
            m.error("Compute F0 first.")
            return
        try:
            speed = float(self.dff_speed.currentText().replace("×", "").replace(",", "."))
        except ValueError:
            speed = 4.0
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ΔF/F video",
                                                       os.path.splitext(m.video)[0] + "_dFoF.mp4", "MP4 (*.mp4)")
        if not out:
            return
        args = (out, m.video, self.f0d, self.dff_lo.value(), self.dff_hi.value(), self.dff_mask.value(), speed)
        m.run_task(cc.render_dff_video, *args, on_done=lambda p: m.log(f"ΔF/F video written: {p}"),
                   label="Rendering ΔF/F video")

    def _edited(self, *_):
        m = self.main
        if self._loading or not m.video:
            return
        new = self.values()
        if new == (m.session.get("info") or {}):
            return
        m.session["info"] = new
        m.save_session()
        m._refresh_list_label(m.video)

    # ------------------------------------------------------------------ mirrors of the Rotate tab
    def sync_from_main(self):
        m = self.main
        if not hasattr(m, "obj_combo"):
            return
        for src, dst in ((m.obj_combo, self.obj), (m.tur_combo, self.tur)):
            dst.blockSignals(True)
            items = [src.itemText(i) for i in range(src.count())]
            if [dst.itemText(i) for i in range(dst.count())] != items:
                dst.clear()
                dst.addItems(items)
            dst.setCurrentText(src.currentText())
            dst.blockSignals(False)
        self.tur.setEnabled(m.tur_combo.isEnabled())
        self.scale_lbl.setText(m.scale_lbl.text())
        for w in (self.eye_r, self.eye_l):
            w.blockSignals(True)
        (self.eye_l if m._eye() == "left" else self.eye_r).setChecked(True)
        for w in (self.eye_r, self.eye_l):
            w.blockSignals(False)

    def _obj_changed(self, t):
        if self.main.obj_combo.currentText() != t:
            self.main.obj_combo.setCurrentText(t)
        self._edited()

    def _tur_changed(self, t):
        if self.main.tur_combo.currentText() != t:
            self.main.tur_combo.setCurrentText(t)

    def _eye_changed(self, *_):
        m = self.main
        (m.eye_left if self.eye_l.isChecked() else m.eye_right).setChecked(True)

    # ------------------------------------------------------------------ copy / apply
    def copy_from(self, src):
        m = self.main
        if os.path.abspath(src) == os.path.abspath(m.video or ""):
            m.log("That is the current video - pick another one.")
            return
        ses = wc.load_session(src)
        inf = ses.get("info") or {}
        if not inf:
            m.log(f"{os.path.basename(src)} has no info yet.")
            return
        self._loading = True
        for k in COPY_KEYS:
            if inf.get(k):
                self._set(k, inf[k])
        self._loading = False
        self._edited()
        m.log(f"Info copied from {os.path.basename(src)} (notes not copied)")

    def _apply_to_others(self):
        m = self.main
        if not m.video:
            return
        vids = [(f, os.path.relpath(f, m.folder or os.path.dirname(f))) for f in m._listed_videos()]
        dlg = ApplyDialog(vids, m.video, self.values(), self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        keys, do_obj, do_eye, targets = dlg.chosen()
        if not targets or not (keys or do_obj or do_eye):
            return
        vals = self.values()
        n = 0
        for full in targets:
            try:
                ses = wc.load_session(full)
                inf = ses.setdefault("info", {})
                for k in keys:
                    inf[k] = vals[k]
                if do_obj:
                    ses["scale"] = m._scale_dict()
                if do_eye:
                    ses["eye"] = m._eye()
                wc.save_session(full, ses)
                m._refresh_list_label(full)
                n += 1
            except Exception as e:
                m.log(f"{os.path.basename(full)}: {e}")
        m.log(f"Info written to {n} video(s)")
