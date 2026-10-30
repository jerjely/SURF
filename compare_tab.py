"""Tab 5 - combine analysed recordings, label them (animal / age / condition) and compare groups."""
from __future__ import annotations

import os
import traceback

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import figexport

import wave_stats as ws
import wave_plots as wpl

COLS = ["Use", "Recording", "Animal", "Age", "Condition", "Objective", "Units", "Waves", "Path"]
EDITABLE = {"Animal": "animal", "Age": "age", "Condition": "condition"}


class _Canvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tb = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(tb)
        lay.addWidget(self.canvas, 1)
        from mpl_nav import AxisNavigator, HINT
        self.nav = AxisNavigator(self.canvas, tb)       # drag / wheel on the axes
        self.canvas.setToolTip(HINT)


def _df_to_table(tw: QtWidgets.QTableWidget, df: pd.DataFrame, highlight_p=True):
    tw.clear()
    tw.setRowCount(len(df))
    tw.setColumnCount(len(df.columns))
    tw.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for r, (_, row) in enumerate(df.iterrows()):
        for c, col in enumerate(df.columns):
            v = row[col]
            if isinstance(v, (float, np.floating)):
                txt = "" if not np.isfinite(v) else (f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}")
            else:
                txt = str(v)
            it = QtWidgets.QTableWidgetItem(txt)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            if highlight_p and col in ("p", "p_holm", "rayleigh_p") and isinstance(v, float) \
                    and np.isfinite(v) and v < 0.05:
                ph = row.get("p_holm", np.nan) if col == "p" else np.nan
                if isinstance(ph, float) and np.isfinite(ph) and ph >= 0.05:
                    # significant only before the multiple-comparison correction
                    it.setBackground(QtGui.QColor(255, 245, 200))
                    it.setToolTip("p < 0.05 uncorrected, but not significant after Holm correction (p_holm)")
                else:
                    it.setBackground(Qt.yellow)
            tw.setItem(r, c, it)
    tw.resizeColumnsToContents()


class CompareTab(QtWidgets.QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.recs: list[ws.Recording] = []
        self.project_path = None
        self.result = None
        self._building = False

        # ---------------- left: recordings ----------------
        b_folder = QtWidgets.QPushButton("Add folder…")
        b_folder.setToolTip("Adds every analysed recording (…_wave_results.xlsx) in the folder and its subfolders")
        b_folder.clicked.connect(self._add_folder)
        b_files = QtWidgets.QPushButton("Add result files…")
        b_files.clicked.connect(self._add_files)
        b_rm = QtWidgets.QPushButton("Remove selected")
        b_rm.clicked.connect(self._remove_selected)
        b_open = QtWidgets.QPushButton("Open project…")
        b_open.clicked.connect(self._open_project)
        b_save = QtWidgets.QPushButton("Save project…")
        b_save.clicked.connect(lambda: self._save_project(ask=True))
        b_reload = QtWidgets.QPushButton("Reload results")
        b_reload.setToolTip("Re-read the workbooks, e.g. after re-analysing a video")
        b_reload.clicked.connect(self._reload)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._item_changed)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._table_menu)
        # bulk edit
        self.bulk_field = QtWidgets.QComboBox()
        self.bulk_field.addItems(["Condition", "Animal", "Age"])
        self.bulk_value = QtWidgets.QComboBox()
        self.bulk_value.setEditable(True)
        self.bulk_value.setMinimumWidth(140)
        self.bulk_field.currentIndexChanged.connect(self._fill_bulk_values)
        b_apply = QtWidgets.QPushButton("Apply to selected")
        b_apply.clicked.connect(self._apply_bulk)
        self.proj_lbl = QtWidgets.QLabel("No project saved yet.")
        self.proj_lbl.setStyleSheet("color: gray")
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        r1 = QtWidgets.QHBoxLayout()
        for b in (b_folder, b_files, b_rm, b_reload):
            r1.addWidget(b)
        r1.addStretch(1)
        r2 = QtWidgets.QHBoxLayout()
        for b in (b_open, b_save):
            r2.addWidget(b)
        r2.addWidget(self.proj_lbl, 1)
        r3 = QtWidgets.QHBoxLayout()
        r3.addWidget(QtWidgets.QLabel("Set"))
        r3.addWidget(self.bulk_field)
        r3.addWidget(QtWidgets.QLabel("="))
        r3.addWidget(self.bulk_value)
        r3.addWidget(b_apply)
        r3.addStretch(1)
        hint = QtWidgets.QLabel("Double-click Animal / Age / Condition to edit, or select several rows "
                                "and use 'Apply to selected'. Recordings of the same retina/animal "
                                "must share the same Animal name.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        ll.addLayout(r1)
        ll.addLayout(r2)
        ll.addWidget(self.table, 1)
        ll.addLayout(r3)
        ll.addWidget(hint)

        # ---------------- right: options + results ----------------
        self.group_by = QtWidgets.QComboBox()
        self.group_by.addItems(["Condition", "Age", "Animal"])
        self.filt_age = QtWidgets.QComboBox()
        self.filt_cond = QtWidgets.QComboBox()
        self.filt_obj = QtWidgets.QComboBox()
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["Animal (average recordings per animal)", "Recording (each video independent)"])
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["Non-parametric (Kruskal-Wallis + Dunn / Mann-Whitney; paired: Friedman / Wilcoxon)",
                              "Parametric (t-test / ANOVA)",
                              "Mixed model (recordings nested in animals)"])
        self.paired = QtWidgets.QComboBox()
        self.paired.addItems(["Auto-detect", "Paired", "Unpaired"])
        self.reference = QtWidgets.QComboBox()
        self.posthoc = QtWidgets.QComboBox()
        self.posthoc.addItems(["Dunn (after Kruskal-Wallis, pooled ranks)", "Mann-Whitney for each pair"])
        self.posthoc.setToolTip("Pairwise follow-up for independent groups with the non-parametric test. "
                                "Dunn uses the ranks of all groups together (more power with small groups).")
        self.trend_chk = QtWidgets.QCheckBox("Trend test across groups (in plot order)")
        self.trend_chk.setToolTip("Jonckheere-Terpstra (permutation) + Spearman: one test for a steady "
                                  "increase / decrease, e.g. with age. Uses the numbers in the group names "
                                  "(P7 -> 7), else the plot order. Change the order in 'Order / colour'.")
        self.group_by.currentIndexChanged.connect(self._refresh_choices)
        for w in (self.filt_age, self.filt_cond, self.filt_obj):
            w.currentIndexChanged.connect(self._refresh_reference)
        b_run = QtWidgets.QPushButton("▶  Compare groups")
        b_run.setStyleSheet("font-weight: bold; padding: 6px")
        b_run.clicked.connect(self._run)
        b_exp = QtWidgets.QPushButton("Export (xlsx + figures)…")
        b_exp.clicked.connect(self._export)
        form = QtWidgets.QFormLayout()
        form.addRow("Compare by", self.group_by)
        form.addRow("Only age", self.filt_age)
        form.addRow("Only condition", self.filt_cond)
        form.addRow("Only objective", self.filt_obj)
        form.addRow("Unit of analysis", self.unit)
        form.addRow("Test", self.method)
        form.addRow("Pairing", self.paired)
        form.addRow("Reference group", self.reference)
        form.addRow("Post-hoc", self.posthoc)
        form.addRow("", self.trend_chk)
        opts = QtWidgets.QGroupBox("Statistics")
        ol = QtWidgets.QVBoxLayout(opts)
        ol.addLayout(form)
        ol.addWidget(b_run)
        ol.addWidget(b_exp)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ol.addWidget(self.info)

        self.res_tabs = QtWidgets.QTabWidget()
        plot_page = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(plot_page)
        pl.setContentsMargins(0, 0, 0, 0)
        self.level_combo = QtWidgets.QComboBox()
        self.level_combo.addItem("Recording means", "rec")
        self.level_combo.addItem("Individual waves (violin)", "wave")
        self.level_combo.addItem("Individual cells (violin)", "cell")
        self.level_combo.setToolTip("Recording means: one value per recording / animal.\n"
                                    "Individual waves: every wave of every recording (duration, path, "
                                    "speed, distance, inter-wave interval).")
        self.level_combo.currentIndexChanged.connect(self._level_changed)
        self.metric_combo = QtWidgets.QComboBox()
        self._fill_metric_combo("rec")
        self.metric_combo.currentIndexChanged.connect(self._draw_metric)
        self.style_combo = QtWidgets.QComboBox()
        for k, lab in wpl.STYLES.items():
            self.style_combo.addItem(lab, k)
        self.style_combo.currentIndexChanged.connect(self._plot_cfg_changed)
        self.points_chk = QtWidgets.QCheckBox("points")
        self.points_chk.setChecked(True)
        self.lines_chk = QtWidgets.QCheckBox("join same animal")
        self.lines_chk.setChecked(True)
        for w in (self.points_chk, self.lines_chk):
            w.toggled.connect(self._plot_cfg_changed)
        self.brk_combo = QtWidgets.QComboBox()
        self.brk_combo.addItem("significant only", "sig")
        self.brk_combo.addItem("all p values", "all")
        self.brk_combo.addItem("none", "none")
        self.brk_combo.setToolTip("Pairwise brackets on the plot. Significance uses the Holm-corrected p "
                                  "(the p shown on the bracket).")
        self.brk_combo.currentIndexChanged.connect(self._plot_cfg_changed)
        pl.addLayout(_hrow(QtWidgets.QLabel("Data"), self.level_combo,
                           QtWidgets.QLabel("Metric"), self.metric_combo, QtWidgets.QLabel("Show"),
                           self.style_combo, self.points_chk, self.lines_chk,
                           QtWidgets.QLabel("brackets"), self.brk_combo))
        # group order + colours
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.group_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.group_list.setMaximumWidth(210)
        self.group_list.setToolTip("Drag to change the order; double-click to change the colour")
        self.group_list.model().rowsMoved.connect(lambda *_: self._order_changed())
        self.group_list.itemDoubleClicked.connect(self._pick_color)
        b_up = QtWidgets.QToolButton(text="↑")
        b_dn = QtWidgets.QToolButton(text="↓")
        b_up.clicked.connect(lambda: self._move_group(-1))
        b_dn.clicked.connect(lambda: self._move_group(1))
        b_col = QtWidgets.QPushButton("Colour…")
        b_col.clicked.connect(lambda: self.group_list.currentItem() and self._pick_color(self.group_list.currentItem()))
        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("Order / colour"))
        side.addWidget(self.group_list, 1)
        side.addLayout(_hrow(b_up, b_dn, b_col))
        b_comb = QtWidgets.QPushButton("Combine…")
        b_comb.setToolTip("Select several groups (Ctrl / Shift + click) and combine them into one, "
                          "e.g. P7 + P10 + P11 = 'stage II'")
        b_comb.clicked.connect(self._combine_groups)
        b_split = QtWidgets.QPushButton("Split")
        b_split.setToolTip("Undo a combined group")
        b_split.clicked.connect(self._split_groups)
        self.grp_chk = QtWidgets.QCheckBox("use combined groups")
        self.grp_chk.setChecked(True)
        self.grp_chk.toggled.connect(self._groupings_toggled)
        side.addLayout(_hrow(b_comb, b_split))
        side.addWidget(self.grp_chk)
        body = QtWidgets.QHBoxLayout()
        body.addLayout(side)
        self.metric_canvas = _Canvas()
        body.addWidget(self.metric_canvas, 1)
        pl.addLayout(body, 1)
        self.plot_cfg = {"order": {}, "colors": {}, "style": "box", "points": True, "lines": True,
                         "groupings": {}, "use_groupings": {}}
        self.res_tabs.addTab(plot_page, "Plot")
        self.dir_canvas = _Canvas()
        self.res_tabs.addTab(self.dir_canvas, "Directions")
        self.tests_tw = QtWidgets.QTableWidget()
        self.res_tabs.addTab(self.tests_tw, "Tests")
        self.desc_tw = QtWidgets.QTableWidget()
        self.res_tabs.addTab(self.desc_tw, "Group summary")
        self.data_tw = QtWidgets.QTableWidget()
        self.res_tabs.addTab(self.data_tw, "Data")

        right = QtWidgets.QSplitter(Qt.Vertical)
        rs = QtWidgets.QScrollArea()
        rs.setWidgetResizable(True)
        rs.setWidget(opts)
        right.addWidget(rs)
        right.addWidget(self.res_tabs)
        right.setSizes([330, 600])
        sp = QtWidgets.QSplitter()
        sp.addWidget(left)
        sp.addWidget(right)
        sp.setSizes([700, 750])
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(sp)
        self._opened = False            # last project is loaded when the tab is first shown

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._opened:
            self._opened = True
            QtCore.QTimer.singleShot(0, self._reopen_last_project)

    # ------------------------------------------------------------------ loading
    def _reopen_last_project(self):
        p = self.main.settings.get("last_compare_project")
        if p and os.path.exists(p) and not self.recs and self.main._thread is None:
            try:
                self._load_project_file(p)
            except Exception:
                pass

    def _add_paths(self, paths):
        known = {os.path.normcase(os.path.abspath(r.path)) for r in self.recs}
        new = [p for p in paths if os.path.normcase(os.path.abspath(p)) not in known]
        # one result per video: a video measured with and without rotation has two workbooks
        new, skipped = ws.split_duplicates(new)
        have = {ws.video_key(r.path): r for r in self.recs}
        for p in list(new):
            if ws.video_key(p) in have:
                old = have[ws.video_key(p)]
                if ws.current_result([p, old.path]) == p:      # the new one is the current one: replace
                    self.recs.remove(old)
                    skipped.append((old.path, p))
                else:
                    new.remove(p)
                    skipped.append((p, old.path))
        if skipped:
            self.main.log("Skipped older / other results of the same video (one per recording):\n" +
                          "\n".join(f"  {os.path.basename(a)}  (using {os.path.basename(b)})" for a, b in skipped))
        if not new:
            self.main.log("No new analysed recordings found.")
            return
        recs = []
        for p in new:
            m = ws.guess_metadata(p)
            inf = ws.read_session_info(p)          # entered on the Info tab
            r = ws.Recording(p, m["name"], inf.get("animal", m["animal"]), inf.get("age", m["age"]),
                             inf.get("condition", ""), objective=inf.get("objective", ""))
            r._from_session = bool(inf)
            recs.append(r)
        self._load_data(recs, append=True)

    def _load_data(self, recs, append):
        def job(progress=None, cancel=None):
            for i, r in enumerate(recs):
                try:
                    r.data = ws.load_recording(r.path) if os.path.exists(r.path) else {}
                except Exception as e:
                    r.data = {"error": str(e)}
                if progress:
                    progress(i + 1, len(recs), f"reading {r.name}")
            return recs

        def done(rs):
            for r in rs:                           # fall back to the info stored in the workbook
                inf = (r.data or {}).get("info") or {}
                if not getattr(r, "_from_session", False) and inf:
                    r.animal = inf.get("animal") or r.animal
                    r.age = inf.get("age") or r.age
                    r.condition = inf.get("condition") or r.condition
                if not r.objective:
                    r.objective = inf.get("objective", "")
            self.recs = (self.recs + rs) if append else rs
            self._fill_table()
            self._autosave()
        self.main.run_task(job, on_done=done, label="Reading results")

    def _add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Folder with analysed recordings",
                                                       self.main.folder or "")
        if d:
            files = ws.find_result_files(d)
            if not files:
                self.main.error(f"No …_wave_results.xlsx files found in\n{d}\n\nRun the wave analysis "
                                "(tab 4) on the videos first.")
                return
            self._add_paths(files)

    def _add_files(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Wave result workbooks", self.main.folder or "",
                                                       "Wave results (*_wave_results.xlsx)")
        if ps:
            self._add_paths(ps)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.recs[r]
        self._fill_table()
        self._autosave()

    def _one_per_video(self, recs):
        """Drop extra results of the same video (keeps the current one), logging what was dropped."""
        groups = {}
        for r in recs:
            groups.setdefault(ws.video_key(r.path), []).append(r)
        out, dropped = [], []
        for r in recs:
            g = groups[ws.video_key(r.path)]
            if len(g) == 1:
                out.append(r)
                continue
            cur = ws.current_result([x.path for x in g if os.path.exists(x.path)] or [g[0].path])
            if r.path == cur:
                out.append(r)
            else:
                dropped.append((r.path, cur))
        if dropped:
            self.main.log("Removed other results of the same video (one per recording):\n" +
                          "\n".join(f"  {os.path.basename(a)}  (using {os.path.basename(b)})" for a, b in dropped))
        return out

    def _reload(self):
        self._load_data(self._one_per_video(list(self.recs)), append=False)

    # ------------------------------------------------------------------ project
    def _open_project(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Comparison project", self.main.folder or "",
                                                     "Wave comparison (*.wavecompare.json)")
        if p:
            self._load_project_file(p)

    def _load_project_file(self, p):
        recs = ws.load_project(p)
        cfg = ws.load_project_extra(p)
        if cfg:
            self.plot_cfg.update(cfg)
            self._apply_plot_cfg_widgets()
        self.project_path = p
        self.proj_lbl.setText(p)
        self.main.settings["last_compare_project"] = p
        self._save_settings()
        self._load_data(self._one_per_video(recs), append=False)

    def _save_project(self, ask=False):
        p = self.project_path
        if ask or not p:
            start = p or os.path.join(self.main.folder or os.path.expanduser("~"), "waves.wavecompare.json")
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save comparison project", start,
                                                         "Wave comparison (*.wavecompare.json)")
            if not p:
                return
            if not p.endswith(".wavecompare.json"):
                p = os.path.splitext(p)[0] + ".wavecompare.json"
        ws.save_project(p, self.recs, self.plot_cfg)
        self.project_path = p
        self.proj_lbl.setText(p)
        self.main.settings["last_compare_project"] = p
        self._save_settings()
        self.main.log(f"Project saved: {p}")

    def _autosave(self):
        if self.project_path:
            try:
                ws.save_project(self.project_path, self.recs, self.plot_cfg)
            except Exception:
                pass

    def _save_settings(self):
        self.main.save_app_settings()

    # ------------------------------------------------------------------ table
    def _fill_table(self):
        self._building = True
        t = self.table
        t.setRowCount(len(self.recs))
        for r, rec in enumerate(self.recs):
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if rec.include else Qt.Unchecked)
            t.setItem(r, 0, use)
            d = rec.data or {}
            waves = d.get("n_wave_events", np.nan)
            vals = [rec.name, rec.animal, rec.age, rec.condition, rec.objective or "?",
                    "missing!" if not d else ("error" if "error" in d else
                                              d.get("unit", "") + ("" if d.get("angles_ok", True) else " · old angles")),
                    "" if not isinstance(waves, float) or not np.isfinite(waves) else f"{waves:.0f}",
                    rec.path]
            for c, v in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(str(v))
                if COLS[c] not in EDITABLE:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if COLS[c] == "Condition" and not rec.condition:
                    it.setBackground(Qt.yellow)
                if COLS[c] == "Units" and (not d or "error" in d):
                    it.setForeground(Qt.red)
                t.setItem(r, c, it)
        # warn about two results of the same video (e.g. projects saved before this check)
        keys = [ws.video_key(rec.path) for rec in self.recs]
        for r, k in enumerate(keys):
            if keys.count(k) > 1:
                for c in range(t.columnCount()):
                    it = t.item(r, c)
                    if it is not None:
                        it.setBackground(QtGui.QColor(255, 200, 150))
                        it.setToolTip("This video has more than one result (e.g. with and without rotation) – "
                                      "press 'Reload results' to keep only the current one, or remove / untick it yourself.")
        t.resizeColumnsToContents()
        self._building = False
        self._fill_bulk_values()
        self._refresh_choices()

    def _item_changed(self, it):
        if self._building:
            return
        r, c = it.row(), it.column()
        if r >= len(self.recs):
            return
        rec = self.recs[r]
        if c == 0:
            rec.include = it.checkState() == Qt.Checked
        elif COLS[c] in EDITABLE:
            setattr(rec, EDITABLE[COLS[c]], it.text().strip())
            self._write_back(rec)
            if COLS[c] == "Condition":
                it.setBackground(Qt.white if it.text().strip() else Qt.yellow)
        self._autosave()
        self._fill_bulk_values()
        self._refresh_choices()

    def _table_menu(self, pos):
        r = self.table.rowAt(pos.y())
        if r < 0 or r >= len(self.recs):
            return
        p = self.recs[r].path
        menu = QtWidgets.QMenu(self)
        menu.addAction("Show results in folder", lambda: self.main._reveal(p))
        menu.addAction("Open results workbook",
                       lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(p)))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _fill_bulk_values(self):
        key = EDITABLE[self.bulk_field.currentText()]
        cur = self.bulk_value.currentText()
        vals = sorted({getattr(r, key) for r in self.recs if getattr(r, key)})
        if key == "condition":
            vals = sorted(set(vals) | {"Control"})
        self.bulk_value.blockSignals(True)
        self.bulk_value.clear()
        self.bulk_value.addItems(vals)
        self.bulk_value.setCurrentText(cur)
        self.bulk_value.blockSignals(False)

    def _write_back(self, rec):
        """Keep the video's Info tab in step with labels edited here."""
        f = dict(animal=rec.animal, age=rec.age, condition=rec.condition)
        m = self.main
        sp = ws.session_for_result(rec.path)
        if m.video and sp and os.path.normcase(os.path.abspath(sp)) == \
                os.path.normcase(os.path.abspath(os.path.splitext(m.video)[0] + "_wavegui.json")):
            m.session.setdefault("info", {}).update(f)       # the open video: update it live
            m.save_session()
            m.info_tab.load_video()
        else:
            ws.write_session_info(rec.path, **f)
        if sp:
            vid = os.path.splitext(sp)[0][: -len("_wavegui")]
            for full in m._listed_videos():
                if os.path.splitext(full)[0] == vid:
                    m._refresh_list_label(full)

    def _apply_bulk(self):
        key = EDITABLE[self.bulk_field.currentText()]
        val = self.bulk_value.currentText().strip()
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            self.main.error("Select one or more rows first.")
            return
        for r in rows:
            setattr(self.recs[r], key, val)
            self._write_back(self.recs[r])
        self._fill_table()
        self._autosave()

    # ------------------------------------------------------------------ options
    def _gcol(self):
        return self.group_by.currentText().lower()

    def _refresh_choices(self, *_):
        if hasattr(self, "grp_chk"):
            self._sync_grp_chk()
        for combo, key in ((self.filt_age, "age"), (self.filt_cond, "condition"), (self.filt_obj, "objective")):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            combo.addItems(sorted({getattr(r, key) for r in self.recs if getattr(r, key)}))
            if combo.findText(cur) >= 0:
                combo.setCurrentText(cur)
            combo.blockSignals(False)
        self.filt_age.setEnabled(self._gcol() != "age")
        self.filt_cond.setEnabled(self._gcol() != "condition")
        self._refresh_reference()

    def _selected_df(self):
        df = ws.recordings_table(self.recs)
        if df.empty:
            return df
        if self.filt_age.isEnabled() and self.filt_age.currentText() not in ("", "All"):
            df = df[df.age == self.filt_age.currentText()]
        if self.filt_cond.isEnabled() and self.filt_cond.currentText() not in ("", "All"):
            df = df[df.condition == self.filt_cond.currentText()]
        if self.filt_obj.currentText() not in ("", "All"):
            df = df[df.objective == self.filt_obj.currentText()]
        return df

    # --- combined groups (e.g. stage II = P7 + P10 + P11) -------------------
    def _mapping(self, gc=None):
        gc = gc or self._gcol()
        if not self.plot_cfg.get("use_groupings", {}).get(gc, True):
            return {}
        return dict(self.plot_cfg.get("groupings", {}).get(gc, {}))

    def _grouped_df(self):
        df = self._selected_df()
        mp = self._mapping()
        if mp and not df.empty:
            gc = self._gcol()
            df = df.copy()
            df[f"{gc}_original"] = df[gc]
            df[gc] = df[gc].map(lambda v: mp.get(v, v))
        return df

    def _combine_groups(self):
        if not self.result:
            self.main.error("Run 'Compare groups' first, then select the groups to combine.")
            return
        gc = self.result["opt"].group_col
        sel = [it.data(Qt.UserRole) for it in self.group_list.selectedItems()]
        if len(sel) < 2:
            self.main.error("Select two or more groups in the list (Ctrl / Shift + click).")
            return
        mp = self.plot_cfg.setdefault("groupings", {}).setdefault(gc, {})
        members = []
        for g in sel:                         # a selected combined group brings its members along
            inner = [o for o, n in mp.items() if n == g]
            members += inner or [g]
        name, ok = QtWidgets.QInputDialog.getText(self, "Combine groups",
                                                  f"Name for {' + '.join(map(str, members))}:",
                                                  text="stage II")
        name = name.strip()
        if not ok or not name:
            return
        for o in list(mp):
            if mp[o] in sel:
                del mp[o]
        for m_ in members:
            if m_ != name:
                mp[m_] = name
        self.plot_cfg.setdefault("use_groupings", {})[gc] = True
        self.grp_chk.blockSignals(True)
        self.grp_chk.setChecked(True)
        self.grp_chk.blockSignals(False)
        order = self.plot_cfg["order"].get(gc, [])
        if order:                             # the new group takes the place of its first member
            pos = min([order.index(g) for g in sel if g in order] or [len(order)])
            order = [g for g in order if g not in sel]
            order.insert(min(pos, len(order)), name)
            self.plot_cfg["order"][gc] = order
        self._autosave()
        self._run()

    def _split_groups(self):
        if not self.result:
            return
        gc = self.result["opt"].group_col
        mp = self.plot_cfg.get("groupings", {}).get(gc, {})
        sel = [it.data(Qt.UserRole) for it in self.group_list.selectedItems()]
        hit = [o for o, n in mp.items() if n in sel]
        if not hit:
            self.main.error("Select a combined group to split.")
            return
        order = []
        for g in self.plot_cfg["order"].get(gc, []):   # members go back where the group was
            order += [o for o in hit if mp.get(o) == g] if g in sel else [g]
        for o in hit:
            del mp[o]
        self.plot_cfg["order"][gc] = order
        self._autosave()
        self._run()

    def _groupings_toggled(self, on):
        self.plot_cfg.setdefault("use_groupings", {})[self._gcol()] = on
        self._autosave()
        if self.result:
            self._run()

    def _refresh_reference(self, *_):
        df = self._grouped_df()
        g = self._gcol()
        groups = sorted({str(v) for v in df[g] if str(v).strip()}) if not df.empty else []
        cur = self.reference.currentText()
        self.reference.clear()
        self.reference.addItem("(all pairwise)")
        self.reference.addItems(groups)
        if cur not in ("", "(all pairwise)") and self.reference.findText(cur) >= 0:
            self.reference.setCurrentText(cur)
        elif "Control" in groups:
            self.reference.setCurrentText("Control")
        elif cur == "(all pairwise)":
            self.reference.setCurrentIndex(0)

    def _options(self):
        method = ["nonparametric", "parametric", "mixed"][self.method.currentIndex()]
        return ws.StatsOptions(group_col=self._gcol(),
                               unit="animal" if self.unit.currentIndex() == 0 else "recording",
                               method=method,
                               paired=["auto", "yes", "no"][self.paired.currentIndex()],
                               reference="" if self.reference.currentIndex() <= 0 else self.reference.currentText(),
                               posthoc="dunn" if self.posthoc.currentIndex() == 0 else "mannwhitney",
                               trend=self.trend_chk.isChecked())

    # ------------------------------------------------------------------ run
    def _run(self):
        df = self._grouped_df()
        if df.empty:
            self.main.error("No usable recordings. Add analysed recordings and tick 'Use'.")
            return
        opt = self._options()
        self.plot_cfg.update(posthoc=opt.posthoc, trend=opt.trend)
        order, full = [], self.plot_cfg.get("groupings", {}).get(opt.group_col, {})
        live = self._mapping(opt.group_col)
        for g in self.plot_cfg["order"].get(opt.group_col, []):
            inner = [o for o, n in full.items() if n == g]
            if inner and not live:            # combined group switched off: its members, in place
                order += [o for o in inner if o not in order]
            else:                             # a member of a combined group: the group, in place
                g2 = live.get(g, g)
                if g2 not in order:
                    order.append(g2)
        opt.groups = order
        if opt.method == "mixed":
            opt.unit = "recording"
        try:
            agg, desc, tests, dirs, info, groups = ws.run_stats(df, opt)
        except Exception:
            self.main.error(traceback.format_exc())
            return
        if len(groups) < 2:
            self.main.error(f"Need at least two {opt.group_col} groups (found: {', '.join(groups) or 'none'}). "
                            "Fill in the column in the table.")
            return
        try:
            wl = ws.waves_table(df, {r.path: r for r in self.recs}, opt.group_col)
            wtests = ws.run_wave_stats(wl, opt, groups) if len(wl) else pd.DataFrame()
        except Exception:
            wl, wtests = pd.DataFrame(), pd.DataFrame()
            self.main.log("Wave-level statistics failed: " + traceback.format_exc().splitlines()[-1])
        if len(wl):
            info += (f"\nIndividual waves: {len(wl[wl.metric == 'wave_speed'])} waves from "
                     f"{wl.recording.nunique()} recordings."
                     + ("" if opt.method == "mixed" else
                        " Waves of one recording are not independent, so wave-level p values are "
                        "optimistic – choose 'Mixed model' for a valid wave-level test."))
        self.result = dict(df=df, agg=agg, desc=desc, tests=tests, dirs=dirs, info=info,
                           groups=groups, opt=opt, waves=wl, wave_tests=wtests)
        self._fill_group_list(groups)
        objs = sorted({str(o) for o in df.get("objective", []) if str(o)})
        if len(objs) > 1 or (len(df) and (df.get("objective", "") == "").any()):
            info += ("\n⚠ These recordings come from different / unknown objectives (" +
                     ", ".join(objs or ["?"]) + "). µm values are comparable once calibrated, but "
                     "detection differs with magnification – consider 'Only objective'.")
        self.info.setText(info.replace("\n", "<br>"))
        both = tests.assign(level="recording")
        if len(wtests):
            both = pd.concat([both, wtests.assign(level="individual waves")], ignore_index=True)
        _df_to_table(self.tests_tw, both)
        _df_to_table(self.desc_tw, desc)
        show = agg.drop(columns=[c for c in ("path",) if c in agg.columns])
        _df_to_table(self.data_tw, show, highlight_p=False)
        self._draw_metric()
        self._draw_dirs()
        self.res_tabs.setCurrentIndex(0)

    # --- plot order / colours / style -------------------------------------
    def _plot_groups(self):
        R = self.result
        order = [self.group_list.item(i).data(Qt.UserRole) for i in range(self.group_list.count())]
        return [g for g in order if g in R["groups"]] + [g for g in R["groups"] if g not in order]

    def _fill_metric_combo(self, level):
        cur = self.metric_combo.currentData()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        items = [(k, lab) for k, lab, _ in ws.METRICS] if level == "rec" else \
            [(k, lab) for k, lab, _, _, _ in (ws.WAVE_METRICS if level == "wave" else ws.CELL_METRICS)]
        for k, lab in items:
            self.metric_combo.addItem(lab, k)
        # keep the same quantity when switching level (e.g. speed -> speed)
        twin = {"mean_long_track_duration_s": "wave_duration_s", "mean_long_track_path": "wave_path",
                "mean_long_track_speed": "wave_speed", "mean_long_track_net_speed": "wave_net_speed",
                "mean_inter_wave_interval_s": "wave_interval_s"}
        twin.update({v: k for k, v in twin.items()})
        i = self.metric_combo.findData(twin.get(cur, cur))
        self.metric_combo.setCurrentIndex(max(i, 0))
        self.metric_combo.blockSignals(False)

    def _level(self):
        return self.level_combo.currentData()

    def _level_changed(self, *_):
        lvl = self._level()
        self._fill_metric_combo(lvl)
        self.style_combo.setEnabled(lvl == "rec")
        self.lines_chk.setEnabled(lvl == "rec")
        self.plot_cfg["level"] = lvl
        self._autosave()
        self._draw_metric()

    def _plot_kw(self):
        return dict(colors=self.plot_cfg["colors"], style=self.plot_cfg["style"],
                    points=self.plot_cfg["points"], lines=self.plot_cfg["lines"],
                    brackets=self.plot_cfg.get("brackets", "sig"))

    def _swatch(self, color):
        pm = QtGui.QPixmap(14, 14)
        pm.fill(QtGui.QColor(color))
        return QtGui.QIcon(pm)

    def _fill_group_list(self, groups):
        gc = self.result["opt"].group_col
        order = list(groups)              # already in the user's order (see _run)
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for i, g in enumerate(order):
            c = self.plot_cfg["colors"].setdefault(str(g), wpl.default_color(len(self.plot_cfg["colors"])))
            members = [o for o, n in self._mapping(gc).items() if n == g]
            label = f"{g}  ({', '.join(map(str, members))})" if members else str(g)
            it = QtWidgets.QListWidgetItem(self._swatch(c), label)
            it.setData(Qt.UserRole, g)
            self.group_list.addItem(it)
        self.group_list.blockSignals(False)
        self.plot_cfg["order"][gc] = order

    def _order_changed(self):
        if not self.result:
            return
        self.plot_cfg["order"][self.result["opt"].group_col] = \
            [self.group_list.item(i).data(Qt.UserRole) for i in range(self.group_list.count())]
        self._autosave()
        if self.result["opt"].trend:          # the trend test depends on the order
            self._run()
            return
        self._draw_metric()
        self._draw_dirs()

    def _move_group(self, step):
        r = self.group_list.currentRow()
        n = r + step
        if r < 0 or not (0 <= n < self.group_list.count()):
            return
        it = self.group_list.takeItem(r)
        self.group_list.insertItem(n, it)
        self.group_list.setCurrentRow(n)
        self._order_changed()

    def _pick_color(self, it):
        g = str(it.data(Qt.UserRole))
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.plot_cfg["colors"].get(g, "#1f77b4")), self,
                                            f"Colour for {g}")
        if c.isValid():
            self.plot_cfg["colors"][g] = c.name()
            it.setIcon(self._swatch(c.name()))
            self._autosave()
            self._draw_metric()
            self._draw_dirs()

    def _plot_cfg_changed(self, *_):
        self.plot_cfg.update(style=self.style_combo.currentData(), points=self.points_chk.isChecked(),
                             lines=self.lines_chk.isChecked(), brackets=self.brk_combo.currentData())
        self._autosave()
        self._draw_metric()

    def _apply_plot_cfg_widgets(self):
        for w in (self.style_combo, self.points_chk, self.lines_chk, self.brk_combo):
            w.blockSignals(True)
        self.brk_combo.setCurrentIndex(max(self.brk_combo.findData(self.plot_cfg.get("brackets", "sig")), 0))
        i = self.style_combo.findData(self.plot_cfg.get("style", "box"))
        self.style_combo.setCurrentIndex(max(i, 0))
        self.points_chk.setChecked(bool(self.plot_cfg.get("points", True)))
        self.lines_chk.setChecked(bool(self.plot_cfg.get("lines", True)))
        for w in (self.style_combo, self.points_chk, self.lines_chk, self.brk_combo):
            w.blockSignals(False)
        self.posthoc.setCurrentIndex(0 if self.plot_cfg.get("posthoc", "dunn") == "dunn" else 1)
        self.level_combo.blockSignals(True)
        self.level_combo.setCurrentIndex({"wave": 1, "cell": 2}.get(self.plot_cfg.get("level"), 0))
        self.level_combo.blockSignals(False)
        self._fill_metric_combo(self._level())
        self.style_combo.setEnabled(self._level() == "rec")
        self.lines_chk.setEnabled(self._level() == "rec")
        self.trend_chk.setChecked(bool(self.plot_cfg.get("trend", False)))
        self._sync_grp_chk()

    def _sync_grp_chk(self):
        self.grp_chk.blockSignals(True)
        self.grp_chk.setChecked(self.plot_cfg.get("use_groupings", {}).get(self._gcol(), True))
        self.grp_chk.blockSignals(False)

    def _unit_of(self, df):
        u = df["unit"].dropna().unique() if "unit" in df else []
        return u[0] if len(u) == 1 else "px"

    def _draw_metric(self, *_):
        R = self.result
        if not R:
            return
        key = self.metric_combo.currentData()
        if self._level() in ("wave", "cell"):
            if R.get("waves") is None or not len(R["waves"]):
                self.metric_canvas.fig.clear()
                self.metric_canvas.fig.text(0.5, 0.5, "No per-wave data in these result files.\n"
                                            "Re-run the wave analysis to create them.", ha="center")
            else:
                wpl.wave_violin(self.metric_canvas.fig, R["waves"], R["opt"].group_col, self._plot_groups(), key,
                                ws.wave_metric_label(key, self._unit_of(R["df"])), R["wave_tests"],
                                unit_name="cells" if self._level() == "cell" else "waves", **self._plot_kw())
        else:
            wpl.group_plot(self.metric_canvas.fig, R["agg"], R["opt"].group_col, self._plot_groups(), key,
                           ws.metric_label(key, self._unit_of(R["df"])), R["tests"], R["opt"].reference,
                           **self._plot_kw())
        self.metric_canvas.canvas.draw_idle()

    def _group_angles(self):
        R = self.result
        out = {}
        for g in self._plot_groups():
            names = set(R["df"].loc[R["df"][R["opt"].group_col] == g, "path"])
            arrs = [r.data.get("angles", np.array([])) for r in self.recs if r.path in names and r.data]
            out[g] = np.concatenate(arrs) if arrs else np.array([])
        return out

    def _draw_dirs(self):
        R = self.result
        if not R:
            return
        wpl.group_polar(self.dir_canvas.fig, self._group_angles(), R["dirs"], colors=self.plot_cfg["colors"])
        self.dir_canvas.canvas.draw_idle()

    # ------------------------------------------------------------------ export
    def _export(self):
        R = self.result
        if not R:
            self.main.error("Run 'Compare groups' first.")
            return
        start = os.path.join(os.path.dirname(self.project_path) if self.project_path else (self.main.folder or ""),
                             f"comparison_by_{R['opt'].group_col}.xlsx")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export comparison", start, "Excel (*.xlsx)")
        if not p:
            return
        out_dir = os.path.splitext(p)[0] + "_figures"
        os.makedirs(out_dir, exist_ok=True)
        try:
            meta = pd.DataFrame([r.meta() for r in self.recs])
            opt = pd.DataFrame(list(vars(R["opt"]).items()), columns=["option", "value"])
            with pd.ExcelWriter(p, engine="openpyxl") as xw:
                pd.DataFrame({"info": R["info"].splitlines()}).to_excel(xw, sheet_name="info", index=False)
                R["tests"].to_excel(xw, sheet_name="tests", index=False)
                if R.get("wave_tests") is not None and len(R["wave_tests"]):
                    R["wave_tests"].to_excel(xw, sheet_name="tests_individual_waves", index=False)
                if R.get("waves") is not None and len(R["waves"]):
                    R["waves"].to_excel(xw, sheet_name="individual_waves", index=False)
                R["desc"].to_excel(xw, sheet_name="group_summary", index=False)
                R["dirs"].to_excel(xw, sheet_name="directions", index=False)
                R["agg"].to_excel(xw, sheet_name="analysed_units", index=False)
                R["df"].to_excel(xw, sheet_name="recordings", index=False)
                meta.to_excel(xw, sheet_name="labels", index=False)
                opt.to_excel(xw, sheet_name="options", index=False)
            unit = self._unit_of(R["df"])
            fig = Figure(figsize=(5, 4.5), tight_layout=True)
            FigureCanvasQTAgg(fig)
            for k, lab, _ in ws.METRICS:
                wpl.group_plot(fig, R["agg"], R["opt"].group_col, self._plot_groups(), k, ws.metric_label(k, unit),
                               R["tests"], R["opt"].reference, **self._plot_kw())
                figexport.save_publication(fig, os.path.join(out_dir, k))
            if R.get("waves") is not None and len(R["waves"]):
                for k, _, _, _, _ in ws.WAVE_METRICS + ws.CELL_METRICS:
                    if not (R["waves"].metric == k).any():
                        continue
                    wpl.wave_violin(fig, R["waves"], R["opt"].group_col, self._plot_groups(), k,
                                    ws.wave_metric_label(k, unit), R["wave_tests"],
                                    unit_name="cells" if k.startswith("cell_") else "waves", **self._plot_kw())
                    figexport.save_publication(fig, os.path.join(out_dir, f"waves_{k}"))
            fig2 = Figure(figsize=(4 * max(len(R["groups"]), 1), 4.5), tight_layout=True)
            FigureCanvasQTAgg(fig2)
            wpl.group_polar(fig2, self._group_angles(), R["dirs"], colors=self.plot_cfg["colors"])
            figexport.save_publication(fig2, os.path.join(out_dir, "directions"))
        except PermissionError as e:
            self.main.error(f"Could not write (file open in Excel?):\n{e}")
            return
        self.main.log(f"Exported {p} and figures to {out_dir}")


def _hrow(*ws_):
    l = QtWidgets.QHBoxLayout()
    for w in ws_:
        l.addWidget(w)
    l.addStretch(1)
    return l
