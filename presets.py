"""Analysis presets for SURF: named sets of event-detection + wave-tracking settings
(optionally the grid square size), e.g. "40x - stage II" or "10x - stage III".

Distances are stored in µm when the video they were saved from had a scale, so a preset
gives the same *physical* distances on any objective / turret.  Presets live in the
per-computer settings file and can be exported / imported to share with the lab.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from PySide6 import QtCore, QtWidgets

import wave_core as wc

BUILTIN = "Notebook defaults (built-in)"
SKIP_WAVE = {"t_start", "t_end", "workers"}       # per-recording / per-computer, never in a preset


def builtin_preset():
    return {"name": BUILTIN, "unit": "px", "objective": "", "notes": "the original notebook values",
            "events": asdict(wc.EventParams()),
            "waves": {k: v for k, v in asdict(wc.WaveParams()).items() if k not in SKIP_WAVE},
            "cells": _cell_defaults()}


def _cell_defaults():
    try:
        import cells_core as cc
        return {k: v for k, v in asdict(cc.CellParams()).items() if k not in SKIP_WAVE}
    except Exception:
        return {}


def all_presets(settings):
    d = {BUILTIN: builtin_preset()}
    d.update(settings.get("presets") or {})
    return d


def describe(p):
    bits = []
    if p.get("objective"):
        bits.append(p["objective"])
    if p.get("notes"):
        bits.append(p["notes"])
    parts = [k for k in ("events", "waves", "cells") if p.get(k)]
    if p.get("grid_size") is not None:
        parts.append("grid size")
    bits.append("contains: " + ", ".join(parts))
    bits.append("distances in " + ("µm" if p.get("unit") == "um" else "px"))
    return " · ".join(bits)


class SaveDialog(QtWidgets.QDialog):
    def __init__(self, objective, has_scale, existing, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save analysis preset")
        self.existing = existing
        f = QtWidgets.QFormLayout(self)
        self.name = QtWidgets.QLineEdit(f"{objective} – " if objective else "")
        self.name.setPlaceholderText("e.g. 40x – stage II")
        self.obj = QtWidgets.QLineEdit(objective or "")
        self.notes = QtWidgets.QLineEdit()
        self.notes.setPlaceholderText("e.g. stage II waves, P3-P5, large slow fronts")
        f.addRow("Name", self.name)
        f.addRow("Objective", self.obj)
        f.addRow("Notes / stage", self.notes)
        self.c_ev = QtWidgets.QCheckBox("Event detection settings (Events tab)")
        self.c_wv = QtWidgets.QCheckBox("Wave tracking settings (Waves tab; not the analysed time window)")
        self.c_gr = QtWidgets.QCheckBox("Grid square size (Measure tab)")
        self.c_ce = QtWidgets.QCheckBox("Cell analysis settings (Cells tab: event rule, diameter, wave linking)")
        self.c_ce.setChecked(True)
        self.c_ev.setChecked(True)
        self.c_wv.setChecked(True)
        self.c_gr.setChecked(True)
        self.c_def = QtWidgets.QCheckBox("Default for this objective: apply automatically when the "
                                         "objective is selected or a new video is opened")
        self.c_def.setChecked(bool(objective))
        for c in (self.c_ev, self.c_wv, self.c_gr, self.c_ce, self.c_def):
            f.addRow(c)
        lab = QtWidgets.QLabel("Distances are saved in µm, so the preset keeps the same physical "
                               "sizes on other objectives." if has_scale else
                               "<span style='color:#d33'>This video has no µm scale: distances are "
                               "saved in px and will not adapt to other objectives.</span>")
        lab.setWordWrap(True)
        f.addRow(lab)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self.resize(480, 0)

    def _ok(self):
        n = self.name.text().strip()
        if not n or n == BUILTIN:
            QtWidgets.QMessageBox.information(self, "Preset", "Enter a name.")
            return
        if n in self.existing and QtWidgets.QMessageBox.question(
                self, "Preset", f"Replace the existing preset '{n}'?") != QtWidgets.QMessageBox.Yes:
            return
        self.accept()


class PresetBar(QtWidgets.QWidget):
    """Preset chooser shown on the Events and Waves tabs (all instances stay in sync)."""
    changed = QtCore.Signal()

    def __init__(self, main):
        super().__init__()
        self.main = main
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QtWidgets.QComboBox()
        self.combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.currentIndexChanged.connect(self._show_info)
        b_apply = QtWidgets.QPushButton("Apply")
        b_apply.setToolTip("Load these settings into the Events and Waves tabs (and the grid size, "
                           "if the preset has one). Then run the detection / analysis again.")
        b_apply.clicked.connect(self.apply_selected)
        b_save = QtWidgets.QPushButton("Save current as…")
        b_save.clicked.connect(self.save_current)
        b_more = QtWidgets.QToolButton()
        b_more.setText("More ▾")
        b_more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        m = QtWidgets.QMenu(b_more)
        m.addAction("Make selected the default for the current objective", self.set_default)
        m.addAction("No default for the current objective", self.clear_default)
        m.addSeparator()
        m.addAction("Delete selected preset", self.delete_selected)
        m.addSeparator()
        m.addAction("Export presets…", self.export)
        m.addAction("Import presets…", self.import_)
        b_more.setMenu(m)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.combo, 1)
        row.addWidget(b_apply)
        row.addWidget(b_save)
        row.addWidget(b_more)
        lay.addLayout(row)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray")
        lay.addWidget(self.info)
        self.refresh()

    # ------------------------------------------------------------------ list
    def refresh(self, select=None):
        cur = select or self.combo.currentData()
        obj = self._objective()
        ps = all_presets(self.main.settings)
        names = sorted((n for n in ps if n != BUILTIN),
                       key=lambda n: (ps[n].get("objective", "") != obj, n.lower()))
        self.combo.blockSignals(True)
        self.combo.clear()
        defaults = self.main.settings.get("objective_defaults") or {}
        for n in names + [BUILTIN]:
            mark = "  ★" if obj and ps[n].get("objective") == obj else ""
            dfor = [o for o, dn in defaults.items() if dn == n]
            if dfor:
                mark += f"  (default for {', '.join(dfor)})"
            self.combo.addItem(n + mark, n)
        i = self.combo.findData(cur)
        self.combo.setCurrentIndex(i if i >= 0 else 0)
        self.combo.blockSignals(False)
        self._show_info()

    def sync_video(self):
        """Pre-select this video's preset, else the objective's default, else the built-in one."""
        ses = self.main.session or {}
        ps = all_presets(self.main.settings)
        name = ses.get("preset") if ses.get("preset") in ps else None
        if not name:
            d = default_for(self.main.settings, self._objective())
            name = d["name"] if d else BUILTIN
        self.refresh(select=name)
        self.update_status()

    def update_status(self):
        st = self.main.preset_state()
        if st["name"]:
            txt = f"In use for this video: <b>{st['name']}</b>"
            if st["modified"]:
                txt += " <span style='color:#b36b00'>(modified – settings changed after applying)</span>"
            self.status.setText(txt)
        elif st["legacy"]:
            self.status.setText("Using the settings saved with this video (analysed before presets). "
                                "Apply a preset to switch.")
        elif self.main.video:
            self.status.setText("<span style='color:#d33'><b>Choose a preset and press Apply</b> before "
                                "detecting events / analysing waves. The default is pre-selected.</span>")
        else:
            self.status.setText("")

    def _objective(self):
        try:
            return self.main.obj_combo.currentText()
        except Exception:
            return ""

    def _show_info(self, *_):
        p = all_presets(self.main.settings).get(self.combo.currentData())
        obj = self._objective()
        txt = describe(p) if p else ""
        if obj:
            d = (self.main.settings.get("objective_defaults") or {}).get(obj)
            txt += f"<br>★ = saved for {obj}. Default for {obj}: " + (f"<b>{d}</b>" if d else "none")
        self.info.setText(txt)

    def _store(self, presets):
        self.main.settings["presets"] = presets
        self.main.save_app_settings()
        self.main._presets_changed()

    # ------------------------------------------------------------------ actions
    def save_current(self):
        m = self.main
        s = m.um_per_px()
        user = dict(m.settings.get("presets") or {})
        dlg = SaveDialog(self._objective(), bool(s), user, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        k = s if s else 1.0
        p = {"name": dlg.name.text().strip(), "objective": dlg.obj.text().strip(),
             "notes": dlg.notes.text().strip(), "unit": "um" if s else "px",
             "created": time.strftime("%Y-%m-%d %H:%M"),
             "source_video": getattr(m, "video", "") or ""}
        if dlg.c_ev.isChecked():
            p["events"] = asdict(m.event_form.value())
        if dlg.c_wv.isChecked():
            p["waves"] = {key: (v * k if key in wc.WAVE_LENGTH_PARAMS else v)
                          for key, v in asdict(m.wave_form.value()).items() if key not in SKIP_WAVE}
        if dlg.c_gr.isChecked():
            p["grid_size"] = m._grid_spec().size * k
        if dlg.c_ce.isChecked() and hasattr(m, "cells_tab"):
            p["cells"] = {key: v for key, v in asdict(m.cells_tab.form.value()).items() if key not in SKIP_WAVE}
        user[p["name"]] = p
        if dlg.c_def.isChecked() and p["objective"]:
            m.settings.setdefault("objective_defaults", {})[p["objective"]] = p["name"]
        self._store(user)
        self.refresh(select=p["name"])
        m.log(f"Preset '{p['name']}' saved")

    def apply_selected(self):
        p = all_presets(self.main.settings).get(self.combo.currentData())
        if p:
            self.main.apply_preset(p)

    def set_default(self):
        obj = self._objective()
        if not obj:
            return
        self.main.settings.setdefault("objective_defaults", {})[obj] = self.combo.currentData()
        self._store(dict(self.main.settings.get("presets") or {}))

    def clear_default(self):
        (self.main.settings.get("objective_defaults") or {}).pop(self._objective(), None)
        self._store(dict(self.main.settings.get("presets") or {}))

    def delete_selected(self):
        n = self.combo.currentData()
        if n == BUILTIN:
            QtWidgets.QMessageBox.information(self, "Preset", "The built-in preset cannot be deleted.")
            return
        if QtWidgets.QMessageBox.question(self, "Preset", f"Delete the preset '{n}'?") != QtWidgets.QMessageBox.Yes:
            return
        user = dict(self.main.settings.get("presets") or {})
        user.pop(n, None)
        defs = self.main.settings.get("objective_defaults") or {}
        for o in [o for o, dn in defs.items() if dn == n]:
            defs.pop(o)
        self._store(user)

    def export(self):
        user = self.main.settings.get("presets") or {}
        if not user:
            QtWidgets.QMessageBox.information(self, "Export", "No saved presets yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export presets", "SURF_presets.json",
                                                        "SURF presets (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"surf_presets": list(user.values())}, f, indent=2)

    def import_(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import presets", "", "SURF presets (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                got = json.load(f).get("surf_presets", [])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import", f"Could not read the file:\n{e}")
            return
        user = dict(self.main.settings.get("presets") or {})
        n = 0
        for p in got:
            if isinstance(p, dict) and p.get("name") and p["name"] != BUILTIN:
                user[p["name"]] = p
                n += 1
        self._store(user)
        QtWidgets.QMessageBox.information(self, "Import", f"{n} preset(s) imported.")


def default_for(settings, objective):
    n = (settings.get("objective_defaults") or {}).get(objective)
    return all_presets(settings).get(n) if n else None


def apply_preset(main, p, allow_grid=True):
    """Load preset p into the forms of the main window. Returns a short report."""
    s = main.um_per_px()
    um = p.get("unit") == "um"
    notes = []
    if um and not s:
        notes.append("This video has no µm scale yet (Rotate tab), so the distance settings were "
                     "NOT applied. Set the objective / turret and apply again.")

    def conv(v):            # preset value -> px
        return v / s if um else v

    if p.get("events"):
        main.event_form.set_value({k: v for k, v in p["events"].items()
                                   if k in main.event_form.widgets})
    if p.get("waves"):
        cur = asdict(main.wave_form.value())
        for k, v in p["waves"].items():
            if k in SKIP_WAVE or k not in cur:
                continue
            if k in wc.WAVE_LENGTH_PARAMS:
                if um and not s:
                    continue
                v = conv(v)
            cur[k] = v
        cur.pop("workers", None)
        main.wave_form.set_value(cur)
    if p.get("grid_size") is not None and not allow_grid:
        notes.append("Grid size not changed: this video is already measured (change it on the "
                     "Measure tab and measure again if you want the preset's grid).")
    elif p.get("grid_size") is not None and not (um and not s):
        g = main._grid_spec()
        new = max(1, int(round(conv(p["grid_size"]))))
        if new != g.size:
            g2 = wc.GridSpec(g.x0, g.y0, new, max(1, int(round(g.height / new))),
                             max(1, int(round(g.width / new))))
            main._set_grid_spins(g2)
            notes.append(f"Grid square size set to {new} px ({g2.rows}×{g2.cols} squares over the same "
                         "area) – run the measurement again to use it.")
    if p.get("cells") and hasattr(main, "cells_tab"):
        main.cells_tab.form.set_value({k: v for k, v in p["cells"].items() if k in main.cells_tab.form.widgets})
        notes.append("Cell settings loaded (Cells tab).")
    if p.get("events"):
        notes.append("Run 'Detect events' again.")
    if p.get("waves"):
        notes.append("Run the wave analysis again.")
    return notes
