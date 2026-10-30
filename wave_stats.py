"""
wave_stats.py - combine analysed recordings and compare groups.

Each analysed recording is one `<name>_wave_results.xlsx` written by the Waves tab.
Recordings carry metadata (animal, age, condition). Statistics take the nesting into
account:

* unit = "recording": every recording is a data point (only valid if recordings are
  independent, e.g. one per animal)
* unit = "animal": recordings of the same animal and group are averaged first
* paired: the same animals appear in several groups (e.g. control and drug videos of the
  same retina) -> Wilcoxon / Friedman (or paired t / repeated-measures by rank)
* mixed model: value ~ group + (1 | animal) on the recording level (statsmodels)

Directions are circular: per-recording mean direction of the wave segments, Rayleigh test
within a group and Watson-Williams between groups.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------- metrics
# (key in the summary sheet without unit suffix, label, unit-dependent?)
METRICS = [
    ("wave_frequency_per_min", "Wave frequency (/min)", False),
    ("mean_inter_wave_interval_s", "Inter-wave interval (s)", False),
    ("n_wave_events", "Wave events (n)", False),
    ("mean_cell_rate_per_min", "Events per ROI (/min)", False),
    ("fraction_active_rois", "Active ROIs (fraction > 0.2/min)", False),
    ("mean_long_track_duration_s", "Wave duration (s)", False),
    ("mean_long_track_path", "Wave path length", True),
    ("mean_long_track_speed", "Wave speed (path)", True),
    ("mean_long_track_net_speed", "Wave speed (start→end)", True),
    ("direction_R", "Directionality (mean resultant length R)", False),
    # single cells (Cells tab, mostly 40x); empty for recordings without a cell analysis
    ("cells.fraction_active", "Cells: active fraction", False),
    ("cells.mean_rate_per_min", "Cells: event rate (/min)", False),
    ("cells.mean_participation", "Cells: wave participation", False),
    ("cells.fraction_events_in_waves", "Cells: events inside waves (fraction)", False),
    ("cells.mean_fraction_recruited", "Cells: fraction recruited per wave", False),
    ("cells.mean_amplitude_pct", "Cells: event amplitude (% F0)", False),
    ("cells.mean_decay_tau_s", "Cells: event decay τ (s)", False),
    ("cells.mean_cell_speed", "Cells: wave speed from cell timing", True),
    ("cells.copart_fraction_pairs_sig", "Cells: pairs in the same waves more than chance (fraction)", False),
    ("cells.copart_fraction_beyond_distance", "Cells: pairs together beyond distance (fraction)", False),
    ("cells.wave_similarity_jaccard", "Cells: waves recruit the same cells (Jaccard)", False),
]
CIRC = "direction_mean_deg"

# per-wave values (long tracks, the same waves the recording means are made of)
WAVE_METRICS = [  # key, label, length-dependent, sheet, column stem
    ("wave_duration_s", "Wave duration (s)", False, "tracks", "duration_s"),
    ("wave_path", "Wave path length", True, "tracks", "path_length"),
    ("wave_speed", "Wave speed (path)", True, "tracks", "speed"),
    ("wave_net_speed", "Wave speed (start→end)", True, "tracks", "net_speed"),
    ("wave_distance", "Wave travel distance (start→end)", True, "tracks", "start_end_length"),
    ("wave_interval_s", "Inter-wave interval (s)", False, "wave_events", "interval_s"),
]


# per-cell values (sheet cells_single), for the 'Individual cells' plot level
CELL_METRICS = [
    ("cell_rate", "Cell event rate (/min)", False, "cells_single", "rate_per_min"),
    ("cell_participation", "Cell wave participation", False, "cells_single", "participation"),
    ("cell_in_waves", "Cell events inside waves (fraction)", False, "cells_single", "fraction_events_in_waves"),
    ("cell_isi", "Cell mean inter-event interval (s)", False, "cells_single", "mean_isi_s"),
    ("cell_amplitude", "Cell event amplitude (% F0)", False, "cells_single", "mean_amplitude_pct"),
    ("cell_tau", "Cell event decay τ (s)", False, "cells_single", "mean_decay_tau_s"),
]


def wave_metric_label(key, unit):
    for k, lab, dep, _, _ in WAVE_METRICS + CELL_METRICS:
        if k == key:
            if not dep:
                return lab
            u = "µm" if unit == "um" else "px"
            return f"{lab} ({u}/s)" if "speed" in k else f"{lab} ({u})"
    return key


def _wave_values(xl, unit):
    """{wave metric: array} from the sheets of one result workbook."""
    out = {}
    for key, _, dep, sheet, stem in WAVE_METRICS:
        t = xl.get(sheet)
        if t is None or not len(t):
            out[key] = []
            continue
        if sheet == "tracks" and "long" in t:
            t = t[t["long"].astype(bool)]
        cands = [stem] if not dep else [f"{stem}_{unit}", f"{stem}_{unit}_s", f"{stem}_px", f"{stem}_px_s"]
        col = next((c for c in cands if c in t.columns), None)
        out[key] = (pd.to_numeric(t[col], errors="coerce").dropna().to_list() if col else [])
    return out


def metric_label(key, unit):
    for k, lab, unit_dep in METRICS:
        if k == key:
            if unit_dep:
                u = "µm" if unit == "um" else "px"
                return f"{lab} ({u}/s)" if "speed" in k else f"{lab} ({u})"
            return lab
    return key


def find_result_files(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.endswith("_wave_results.xlsx") and not f.startswith("~$"):
                out.append(os.path.join(dp, f))
    return sorted(out)


def video_key(path):
    """Same key for the rotated and the unrotated result of one video."""
    name = guess_metadata(path)["name"]
    return os.path.normcase(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(path)), name)))


def current_result(paths):
    """Of several results of the same video (e.g. measured with and without rotation) the one that
    matches the video's current setting (session 'apply_rotation'), else the newest."""
    paths = list(paths)
    if len(paths) == 1:
        return paths[0]
    sp = session_for_result(paths[0])
    want = None
    if sp:
        try:
            with open(sp, encoding="utf-8") as f:
                ses = json.load(f)
            if "apply_rotation" in ses:
                want = bool(ses.get("apply_rotation")) and bool(ses.get("rotation"))
        except Exception:
            pass
    if want is not None:
        match = [p for p in paths if ("_rotated_grid" in os.path.basename(p)) == want]
        if match:
            return max(match, key=os.path.getmtime)
    return max(paths, key=os.path.getmtime)


def split_duplicates(paths):
    """(paths to use, [(skipped, kept), ...]) – one result per video."""
    groups = {}
    for p in paths:
        groups.setdefault(video_key(p), []).append(p)
    keep, skipped = [], []
    for k, ps in groups.items():
        c = current_result(ps)
        keep.append(c)
        skipped += [(p, c) for p in ps if p != c]
    return sorted(keep), skipped


def guess_metadata(path):
    """Name, age (P<n> anywhere in the path) and animal (= the recording's folder)."""
    name = os.path.basename(path).replace("_wave_results.xlsx", "")
    for suf in ("_rotated_grid", "_grid"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    results_dir = os.path.dirname(path)                 # <video>_grid/
    video_dir = os.path.dirname(results_dir)            # folder with the videos
    age = ""
    m = re.findall(r"(?<![A-Za-z0-9])P(\d{1,3})(?![0-9])", path)
    if m:
        age = f"P{m[-1]}"
    return dict(name=name, animal=os.path.basename(video_dir) or video_dir, age=age, condition="")


INFO_FIELDS = ("animal", "age", "condition", "treatment", "genotype", "sex", "date", "experimenter", "notes")


def session_for_result(path):
    """The <video>_wavegui.json that belongs to a …_wave_results.xlsx (None if not found)."""
    name = guess_metadata(path)["name"]
    video_dir = os.path.dirname(os.path.dirname(path))
    p = os.path.join(video_dir, name + "_wavegui.json")
    return p if os.path.exists(p) else None


def read_session_info(path):
    """{animal, age, condition, ..., objective} from the video's session file, or {}."""
    sp = session_for_result(path)
    if not sp:
        return {}
    try:
        with open(sp, encoding="utf-8") as f:
            ses = json.load(f)
    except Exception:
        return {}
    out = {k: v for k, v in (ses.get("info") or {}).items() if v not in (None, "")}
    obj = (ses.get("scale") or {}).get("objective")
    if obj:
        out["objective"] = obj
    return out


def write_session_info(path, **fields):
    """Write Animal / Age / Condition edited in the Compare tab back to the video's session."""
    sp = session_for_result(path)
    if not sp:
        return False
    try:
        with open(sp, encoding="utf-8") as f:
            ses = json.load(f)
        ses.setdefault("info", {}).update(fields)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(ses, f, indent=2, default=float)
        return True
    except Exception:
        return False


def load_recording(path) -> dict:
    """Metrics of one recording (from its summary + travelDetails sheets)."""
    xl = pd.read_excel(path, sheet_name=None)
    summ = xl.get("summary")
    d = {}
    if summ is not None:
        for k, v in zip(summ["metric"], summ["value"]):
            if str(k).startswith("info."):
                d[str(k)] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                continue
            try:
                d[str(k)] = float(v)
            except (TypeError, ValueError):
                d[str(k)] = v
    unit = "um" if any(k.endswith("_um") or "_um_" in k for k in d) else "px"
    out = {"unit": unit,
           "info": {k[5:]: str(v) for k, v in d.items() if k.startswith("info.") and str(v) not in ("", "nan")}}
    for key, _, unit_dep in METRICS:
        if key == "fraction_active_rois":
            n = d.get("n_rois")
            a = d.get("active_cells_gt_0p2_per_min")
            out[key] = (a / n) if isinstance(n, float) and n and isinstance(a, float) else np.nan
        elif key == "direction_R":
            continue
        elif unit_dep:
            cand = [f"{key}_{unit}", f"{key}_{unit}_s"]
            v = next((d[c] for c in cand if c in d), np.nan)
            out[key] = v if isinstance(v, float) else np.nan
        else:
            v = d.get(key, np.nan)
            out[key] = v if isinstance(v, float) else np.nan
    out["waves"] = _wave_values(xl, unit)
    cs = xl.get("cells_single")
    if cs is not None and len(cs):
        if "analysed" in cs:
            cs = cs[cs["analysed"].astype(bool)]
        out["waves"].update({k: pd.to_numeric(cs[col], errors="coerce").dropna().to_list() if col in cs else []
                             for k, _, _, _, col in CELL_METRICS})
    seg = xl.get("travelDetails")
    ang = seg["angle"].dropna().to_numpy(float) if seg is not None and "angle" in seg else np.array([])
    conv = str(d.get("angle_convention", ""))
    out["angles_ok"] = conv.startswith("anatomical")
    out["eye"] = "left" if "left" in str(d.get("eye", "")) else ("right" if d.get("eye") else "")
    if not out["angles_ok"]:
        ang = np.array([])            # old convention (relative to N->T, y down): not comparable
    out["angles"] = ang
    if len(ang):
        out[CIRC], out["direction_R"] = circ_mean_R(ang)
    else:
        out[CIRC], out["direction_R"] = np.nan, np.nan
    out["scale_um_per_px"] = d.get("scale_um_per_px", np.nan)
    return out


# ----------------------------------------------------------------------------- project
@dataclass
class Recording:
    path: str
    name: str = ""
    animal: str = ""
    age: str = ""
    condition: str = ""
    include: bool = True
    objective: str = ""
    data: dict = field(default_factory=dict, repr=False)

    def meta(self):
        d = asdict(self)
        d.pop("data")
        return d


def _natural_key(v):
    """P7 < P10 < P12 (numbers compared as numbers)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(v))]


def save_project(path, recs, extra=None):
    base = os.path.dirname(os.path.abspath(path))
    items = []
    for r in recs:
        m = r.meta()
        try:
            m["path_rel"] = os.path.relpath(r.path, base)
        except ValueError:          # different drive on Windows
            m["path_rel"] = None
        items.append(m)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "recordings": items, "plot": extra or {}}, f, indent=2)


def load_project_extra(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("plot", {}) or {}
    except Exception:
        return {}


def load_project(path):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    recs = []
    for m in d.get("recordings", []):
        p = m.get("path", "")
        if not os.path.exists(p) and m.get("path_rel"):
            alt = os.path.normpath(os.path.join(base, m["path_rel"]))
            if os.path.exists(alt):
                p = alt          # project folder was moved / synced elsewhere
        recs.append(Recording(p, m.get("name", ""), m.get("animal", ""), m.get("age", ""),
                              m.get("condition", ""), bool(m.get("include", True)),
                              m.get("objective", "")))
    return recs


def recordings_table(recs) -> pd.DataFrame:
    rows = []
    for r in recs:
        if not r.include or not r.data:
            continue
        row = dict(recording=r.name, animal=r.animal, age=r.age, condition=r.condition,
                   objective=r.objective,
                   unit=r.data.get("unit"), path=r.path, angles_ok=bool(r.data.get("angles_ok", False)))
        for k, _, _ in METRICS:
            row[k] = r.data.get(k, np.nan)
        row[CIRC] = r.data.get(CIRC, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- circular
def circ_mean_R(deg):
    a = np.deg2rad(np.asarray(deg, float))
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan
    C, S = np.cos(a).mean(), np.sin(a).mean()
    return float(np.rad2deg(np.arctan2(S, C)) % 360), float(np.hypot(C, S))


def rayleigh(deg):
    """Rayleigh test of uniformity (Zar's approximation). Returns (R, z, p)."""
    a = np.deg2rad(np.asarray(deg, float))
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return np.nan, np.nan, np.nan
    Rn = np.hypot(np.cos(a).sum(), np.sin(a).sum())
    R = Rn / n
    z = n * R ** 2
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n ** 2 - Rn ** 2)) - (1 + 2 * n))
    return float(R), float(z), float(min(max(p, 0.0), 1.0))


def _a1inv(x):
    if x < 0.53:
        return 2 * x + x ** 3 + 5 * x ** 5 / 6
    if x < 0.85:
        return -0.4 + 1.39 * x + 0.43 / (1 - x)
    return 1 / (x ** 3 - 4 * x ** 2 + 3 * x)


def watson_williams(groups_deg):
    """Watson-Williams test for equal mean directions in k groups. Returns (F, df1, df2, p, note)."""
    gs = [np.deg2rad(np.asarray(g, float)[np.isfinite(g)]) for g in groups_deg]
    gs = [g for g in gs if len(g) >= 2]
    k = len(gs)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan, "needs ≥ 2 groups with ≥ 2 values"
    N = sum(len(g) for g in gs)
    Ri = [np.hypot(np.cos(g).sum(), np.sin(g).sum()) for g in gs]
    allg = np.concatenate(gs)
    R = np.hypot(np.cos(allg).sum(), np.sin(allg).sum())
    rbar = sum(Ri) / N
    kappa = _a1inv(rbar)
    K = 1 + 3 / (8 * kappa) if kappa > 0 else np.nan
    F = K * (N - k) * (sum(Ri) - R) / ((k - 1) * (N - sum(Ri)))
    p = float(stats.f.sf(F, k - 1, N - k))
    note = "" if rbar > 0.45 else "low concentration (R̄ < 0.45): test is approximate"
    return float(F), k - 1, N - k, p, note


# ----------------------------------------------------------------------------- stats
def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * p[ok][i])
        adj[i] = min(run, 1.0)
    out[ok] = adj
    return out


def aggregate(df, group_col, unit):
    """unit 'animal': mean of recordings per (animal, group)."""
    cols = [k for k, _, _ in METRICS]
    if unit == "recording":
        out = df.copy()
        out["n_recordings"] = 1
        return out
    num = df.groupby(["animal", group_col], dropna=False)[cols].mean().reset_index()
    num["n_recordings"] = df.groupby(["animal", group_col], dropna=False).size().values
    # circular mean of the recording mean directions
    circ = df.groupby(["animal", group_col], dropna=False)[CIRC].apply(
        lambda s: circ_mean_R(s.dropna())[0] if s.notna().any() else np.nan).reset_index(drop=True)
    num[CIRC] = circ.values
    return num


def is_paired(agg, group_col, groups):
    """True if at least 2 animals have values in ≥ 2 of the groups."""
    if "animal" not in agg:
        return False
    per = agg[agg[group_col].isin(groups)].groupby("animal")[group_col].nunique()
    return int((per >= 2).sum()) >= 2


@dataclass
class StatsOptions:
    group_col: str = "condition"
    unit: str = "animal"              # 'recording' | 'animal'
    method: str = "auto"              # 'auto' | 'nonparametric' | 'parametric' | 'mixed'
    paired: str = "auto"              # 'auto' | 'yes' | 'no'
    reference: str = ""               # '' = all pairs
    groups: list = field(default_factory=list)   # order; empty = all
    posthoc: str = "dunn"             # non-parametric, independent groups: 'dunn' | 'mannwhitney'
    trend: bool = False               # trend test across the groups in their (plot) order


def describe(agg, group_col, groups, metric):
    rows = []
    for g in groups:
        v = agg.loc[agg[group_col] == g, metric].dropna().to_numpy(float)
        rows.append(dict(metric=metric, group=g, n=len(v),
                         mean=v.mean() if len(v) else np.nan,
                         sem=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                         sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                         median=np.median(v) if len(v) else np.nan,
                         q1=np.percentile(v, 25) if len(v) else np.nan,
                         q3=np.percentile(v, 75) if len(v) else np.nan))
    return rows


def _pairs(groups, reference):
    if reference and reference in groups:
        return [(reference, g) for g in groups if g != reference]
    return list(combinations(groups, 2))


def dunn_pairs(vals, groups, pairs):
    """Dunn's test (pooled ranks of all groups, tie-corrected); two-sided normal p."""
    use = [g for g in groups if len(vals[g])]
    pooled = np.concatenate([vals[g] for g in use]) if use else np.array([])
    N = len(pooled)
    out = {}
    if N < 3:
        return out
    ranks = stats.rankdata(pooled)
    _, t = np.unique(pooled, return_counts=True)
    factor = N * (N + 1) / 12.0 - (t ** 3 - t).sum() / (12.0 * (N - 1))
    mean_rank, n, i = {}, {}, 0
    for g in use:
        k = len(vals[g])
        mean_rank[g], n[g] = ranks[i:i + k].mean(), k
        i += k
    for a, b in pairs:
        if a in n and b in n and factor > 0:
            z = (mean_rank[b] - mean_rank[a]) / np.sqrt(factor * (1.0 / n[a] + 1.0 / n[b]))
            out[(a, b)] = (float(z), float(2 * stats.norm.sf(abs(z))))
    return out


def _levels(groups):
    """Numeric level of each group for the trend test: the number in the name (P7 -> 7) when
    every group has a distinct one, else the position in the given order (1, 2, 3 ...)."""
    nums = []
    for g in groups:
        m = re.search(r"-?\d+(?:\.\d+)?", str(g))
        nums.append(float(m.group()) if m else None)
    if all(v is not None for v in nums) and len(set(nums)) == len(nums) and \
            all(a < b for a, b in zip(nums, nums[1:])):
        return dict(zip(groups, nums)), "group numbers"
    return {g: float(i + 1) for i, g in enumerate(groups)}, "plot order"


def trend_rows(agg, metric, gc, groups, paired=False, n_perm=5000):
    """Jonckheere-Terpstra (permutation p) + Spearman for a monotonic trend over ordered groups."""
    lv, how = _levels(groups)
    d = agg[agg[gc].isin(groups)][[gc, metric]].dropna()
    v = d[metric].to_numpy(float)
    L = d[gc].map(lv).to_numpy(float)
    rows = []
    desc = ", ".join(f"{g}={lv[g]:g}" for g in groups)
    note = f"levels from {how}: {desc}"
    if paired:
        note += "; assumes independent groups (same animals appear in several groups)"
    if len(v) < 4 or len(np.unique(L)) < 2:
        return [dict(metric=metric, comparison="trend", test="Jonckheere-Terpstra", statistic=np.nan,
                     p=np.nan, n=str(len(v)), note="too few values")]
    G = (v[None, :] > v[:, None]).astype(float) + 0.5 * (v[None, :] == v[:, None])
    np.fill_diagonal(G, 0)

    def J(lab):
        return float((G * (lab[:, None] < lab[None, :])).sum())

    n_i = np.array([np.sum(L == x) for x in np.unique(L)])
    N = len(v)
    E = (N ** 2 - (n_i ** 2).sum()) / 4.0
    j_obs = J(L)
    if N <= 120:
        rng = np.random.default_rng(12345)
        dev = abs(j_obs - E)
        hits = sum(abs(J(rng.permutation(L)) - E) >= dev - 1e-9 for _ in range(n_perm))
        p = (hits + 1) / (n_perm + 1)
        test = "Jonckheere-Terpstra (permutation)"
    else:
        var = (N ** 2 * (2 * N + 3) - (n_i ** 2 * (2 * n_i + 3)).sum()) / 72.0
        p = float(2 * stats.norm.sf(abs(j_obs - E) / np.sqrt(var)))
        test = "Jonckheere-Terpstra (normal approx.)"
    direction = "increasing" if j_obs > E else "decreasing"
    rows.append(dict(metric=metric, comparison="trend", test=test, statistic=j_obs, p=p,
                     n=str(N), note=f"{direction} along the order; {note}"))
    rho, ps = stats.spearmanr(L, v)
    rows.append(dict(metric=metric, comparison="trend", test="Spearman (value vs level)", statistic=float(rho),
                     p=float(ps), n=str(N), note=f"rho = {rho:.2f}; {note}"))
    return rows


def compare_metric(df_rec, agg, metric, opt: StatsOptions, groups):
    """Returns list of test rows (omnibus + pairwise)."""
    gc = opt.group_col
    rows = []
    paired = (opt.paired == "yes") or (opt.paired == "auto" and opt.unit == "animal"
                                       and is_paired(agg, gc, groups))
    method = opt.method
    if method == "auto":
        method = "nonparametric"
    k = len(groups)
    vals = {g: agg.loc[agg[gc] == g, metric].dropna().to_numpy(float) for g in groups}

    def row(comp, test, stat, p, n, note=""):
        return dict(metric=metric, comparison=comp, test=test, statistic=stat, p=p, n=n, note=note)

    if method == "mixed":
        return _mixed(df_rec, metric, gc, groups, opt.reference or groups[0])

    if paired:
        wide = agg[agg[gc].isin(groups)].pivot_table(index="animal", columns=gc, values=metric, aggfunc="mean")
        wide = wide[[g for g in groups if g in wide.columns]]
        if k >= 3:
            comp = wide.dropna()
            if len(comp) >= 2 and comp.shape[1] >= 3:
                if method == "parametric":
                    # repeated-measures ANOVA via statsmodels AnovaRM
                    try:
                        from statsmodels.stats.anova import AnovaRM
                        long = comp.reset_index().melt(id_vars="animal", var_name="group", value_name="y")
                        r = AnovaRM(long, "y", "animal", within=["group"]).fit()
                        t = r.anova_table.iloc[0]
                        rows.append(row("all groups", "repeated-measures ANOVA", t["F Value"], t["Pr > F"],
                                        f"{len(comp)} animals"))
                    except Exception as e:
                        rows.append(row("all groups", "repeated-measures ANOVA", np.nan, np.nan, "", str(e)))
                else:
                    s, p = stats.friedmanchisquare(*[comp[c].values for c in comp.columns])
                    rows.append(row("all groups", "Friedman", s, p, f"{len(comp)} animals (complete)"))
            else:
                rows.append(row("all groups", "Friedman", np.nan, np.nan, f"{len(comp)} complete animals",
                                "needs ≥ 2 animals measured in every group"))
        pw = []
        for a, b in _pairs(groups, opt.reference):
            if a not in wide or b not in wide:
                continue
            pr = wide[[a, b]].dropna()
            if len(pr) < 2:
                pw.append(row(f"{b} vs {a}", "paired", np.nan, np.nan, f"{len(pr)} animals", "too few pairs"))
                continue
            if method == "parametric":
                s, p = stats.ttest_rel(pr[b], pr[a])
                name = "paired t-test"
            else:
                try:
                    s, p = stats.wilcoxon(pr[b], pr[a])
                except ValueError as e:
                    s, p = np.nan, np.nan
                name = "Wilcoxon signed-rank"
            note = ""
            if name.startswith("Wil") and len(pr) < 6:
                note = f"only {len(pr)} pairs: Wilcoxon cannot reach p < 0.05; consider the paired t-test or more animals"
            pw.append(row(f"{b} vs {a}", name, s, p, f"{len(pr)} animals", note))
    else:
        if k >= 3:
            arrs = [vals[g] for g in groups if len(vals[g]) >= 2]
            if len(arrs) >= 2:
                if method == "parametric":
                    s, p = stats.f_oneway(*arrs)
                    rows.append(row("all groups", "one-way ANOVA", s, p, ", ".join(str(len(a)) for a in arrs)))
                else:
                    s, p = stats.kruskal(*arrs)
                    rows.append(row("all groups", "Kruskal-Wallis", s, p, ", ".join(str(len(a)) for a in arrs)))
        pw = []
        dunn = None
        if method != "parametric" and opt.posthoc == "dunn" and k >= 3:
            dunn = dunn_pairs(vals, [g for g in groups if len(vals[g]) >= 2], _pairs(groups, opt.reference))
        for a, b in _pairs(groups, opt.reference):
            x, y = vals[a], vals[b]
            if len(x) < 2 or len(y) < 2:
                pw.append(row(f"{b} vs {a}", "", np.nan, np.nan, f"{len(x)}, {len(y)}", "needs ≥ 2 per group"))
                continue
            if method == "parametric":
                s, p = stats.ttest_ind(y, x, equal_var=False)
                name = "Welch t-test"
            elif dunn is not None:
                s, p = dunn.get((a, b), (np.nan, np.nan))
                name = "Dunn (z, pooled ranks)"
            else:
                s, p = stats.mannwhitneyu(y, x, alternative="two-sided")
                name = "Mann-Whitney U"
            pw.append(row(f"{b} vs {a}", name, s, p, f"{len(x)}, {len(y)}"))
    if pw:
        adj = holm([r["p"] for r in pw]) if len(pw) > 1 else [pw[0]["p"]]
        for r, pa in zip(pw, adj):
            r["p_holm"] = pa
        rows += pw
    if opt.trend and k >= 3:
        rows += trend_rows(agg, metric, gc, groups, paired=paired)
    return rows


def _mixed(df_rec, metric, gc, groups, ref):
    import warnings
    rows = []
    d = df_rec[df_rec[gc].isin(groups)][[metric, gc, "animal"]].dropna()
    d = d.rename(columns={metric: "y", gc: "g"})
    n_an = d["animal"].nunique()
    if n_an < 2 or d["g"].nunique() < 2:
        return [dict(metric=metric, comparison="all groups", test="mixed model", statistic=np.nan,
                     p=np.nan, n=f"{len(d)} rec / {n_an} animals", note="needs ≥ 2 animals and ≥ 2 groups")]
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"y ~ C(g, Treatment(reference={ref!r}))", d, groups=d["animal"]).fit(reml=True)
        pw = []
        for name in fit.params.index:
            if name.startswith("C(g"):
                lvl = name.split("[T.")[-1].rstrip("]")
                pw.append(dict(metric=metric, comparison=f"{lvl} vs {ref}",
                               test="mixed model (animal random intercept)",
                               statistic=float(fit.params[name]), p=float(fit.pvalues[name]),
                               n=f"{len(d)} rec / {n_an} animals",
                               note="statistic = estimated difference" + ("" if fit.converged else "; did not converge")))
        adj = holm([r["p"] for r in pw]) if len(pw) > 1 else [r["p"] for r in pw]
        for r, pa in zip(pw, adj):
            r["p_holm"] = pa
        rows += pw
    except Exception as e:
        rows.append(dict(metric=metric, comparison="all groups", test="mixed model", statistic=np.nan,
                         p=np.nan, n=f"{len(d)} rec / {n_an} animals", note=f"failed: {e}"))
    return rows


def compare_directions(agg, group_col, groups, reference):
    """Rayleigh per group + Watson-Williams between groups on (recording/animal) mean directions."""
    desc, tests = [], []
    per = {}
    for g in groups:
        a = agg.loc[agg[group_col] == g, CIRC].dropna().to_numpy(float)
        per[g] = a
        m, R = circ_mean_R(a) if len(a) else (np.nan, np.nan)
        Rr, z, p = rayleigh(a)
        desc.append(dict(group=g, n=len(a), circular_mean_deg=m, R=R, rayleigh_z=z, rayleigh_p=p))
    if len(groups) >= 2:
        F, d1, d2, p, note = watson_williams([per[g] for g in groups])
        tests.append(dict(metric=CIRC, comparison="all groups", test="Watson-Williams",
                          statistic=F, p=p, n=f"df {d1}, {d2}", note=note))
        if len(groups) > 2:
            pw = []
            for a, b in _pairs(groups, reference):
                F, d1, d2, p, note = watson_williams([per[a], per[b]])
                pw.append(dict(metric=CIRC, comparison=f"{b} vs {a}", test="Watson-Williams",
                               statistic=F, p=p, n=f"df {d1}, {d2}", note=note))
            adj = holm([r["p"] for r in pw])
            for r, pa in zip(pw, adj):
                r["p_holm"] = pa
            tests += pw
    return desc, tests


def waves_table(df_rec, recs_by_path, group_col):
    """One row per wave: recording, animal, group, metric, value (from the recordings in df_rec)."""
    rows = []
    for _, r in df_rec.iterrows():
        rec = recs_by_path.get(r["path"])
        if rec is None:
            continue
        for key, vals in ((rec.data or {}).get("waves") or {}).items():
            for v in vals:
                rows.append({"recording": r["recording"], "animal": r["animal"], group_col: r[group_col],
                             "metric": key, "value": float(v)})
    return pd.DataFrame(rows, columns=["recording", "animal", group_col, "metric", "value"])


def _mixed_nested(wl, gc, groups, ref, metric):
    """Waves nested in recordings nested in animals: random intercepts for both."""
    import warnings
    d = wl[(wl.metric == metric) & wl[gc].isin(groups)][["value", gc, "animal", "recording"]].dropna()
    d = d.rename(columns={gc: "g", "value": "y"})
    n_an, n_rec = d["animal"].nunique(), d["recording"].nunique()
    n = f"{len(d)} waves / {n_rec} rec / {n_an} animals"
    if n_rec < 3 or d["g"].nunique() < 2:
        return [dict(metric=metric, comparison="all groups", test="mixed model", statistic=np.nan, p=np.nan,
                     n=n, note="needs ≥ 3 recordings and ≥ 2 groups")]
    try:
        import statsmodels.formula.api as smf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"y ~ C(g, Treatment(reference={ref!r}))", d, groups=d["animal"],
                              vc_formula={"rec": "0 + C(recording)"}).fit(reml=True)
        pw = []
        for name in fit.params.index:
            if name.startswith("C(g"):
                lvl = name.split("[T.")[-1].rstrip("]")
                pw.append(dict(metric=metric, comparison=f"{lvl} vs {ref}",
                               test="mixed model (animal + recording random intercepts)",
                               statistic=float(fit.params[name]), p=float(fit.pvalues[name]), n=n,
                               note="statistic = estimated difference" + ("" if fit.converged else "; did not converge")))
        adj = holm([r["p"] for r in pw]) if len(pw) > 1 else [r["p"] for r in pw]
        for r, pa in zip(pw, adj):
            r["p_holm"] = pa
        return pw
    except Exception as e:
        return [dict(metric=metric, comparison="all groups", test="mixed model", statistic=np.nan, p=np.nan,
                     n=n, note=f"failed: {e}")]


def run_wave_stats(wl, opt: StatsOptions, groups):
    """Group comparisons with every wave as one value. Waves of one recording are not independent:
    the 'Mixed model' choice accounts for that (random intercepts for recording and animal)."""
    import copy
    import warnings
    tests = []
    gc = opt.group_col
    o = copy.copy(opt)
    o.unit, o.paired = "recording", "no"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key, _, _, _, _ in WAVE_METRICS + CELL_METRICS:
            sub = wl[wl.metric == key]
            if sub.empty:
                continue
            if opt.method == "mixed":
                tests += _mixed_nested(wl, gc, groups, opt.reference or groups[0], key)
                continue
            agg = sub.rename(columns={"value": key})
            tests += compare_metric(sub, agg, key, o, groups)
    return pd.DataFrame(tests)


def run_stats(df_rec: pd.DataFrame, opt: StatsOptions):
    """Returns (agg table, descriptive table, tests table, direction table, info text)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _run_stats(df_rec, opt)


def _run_stats(df_rec: pd.DataFrame, opt: StatsOptions):
    gc = opt.group_col
    df = df_rec[df_rec[gc].astype(str).str.strip() != ""].copy()
    present = sorted({g for g in df[gc].dropna().unique()}, key=_natural_key)
    if opt.groups:            # user order first, then any new groups
        groups = [g for g in opt.groups if g in present] + [g for g in present if g not in opt.groups]
    else:
        groups = present
        if opt.reference and opt.reference in groups:
            groups = [opt.reference] + [g for g in groups if g != opt.reference]
    agg = aggregate(df, gc, opt.unit)
    desc, tests = [], []
    for k, _, _ in METRICS:
        if k not in agg or agg[k].notna().sum() < 2:
            continue                       # e.g. cell metrics when no recording has a cell analysis
        desc += describe(agg, gc, groups, k)
        if len(groups) >= 2:
            tests += compare_metric(df, agg, k, opt, groups)
    ddesc, dtests = compare_directions(agg, gc, groups, opt.reference)
    tests += dtests
    paired = (opt.paired == "yes") or (opt.paired == "auto" and opt.unit == "animal" and is_paired(agg, gc, groups))
    units = df["unit"].dropna().unique() if "unit" in df else []
    n_old = int((~df["angles_ok"].astype(bool)).sum()) if "angles_ok" in df else 0
    info = [f"{len(df)} recordings, {df['animal'].nunique()} animals, groups: {', '.join(map(str, groups))}",
            ("design: mixed model on recordings, animal = random effect (handles repeated and paired animals)"
             if opt.method == "mixed" else
             f"unit of analysis: {opt.unit}; design: {'paired (same animals across groups)' if paired else 'independent groups'}"),
            f"method: {opt.method}"
            + (f"; post-hoc: {'Dunn' if opt.posthoc == 'dunn' else 'Mann-Whitney'}"
               if opt.method == "nonparametric" and not paired else "")
            + ("; trend test over the groups in plot order" if opt.trend else "")]
    if len(units) > 1:
        info.append("WARNING: some recordings are in µm and some in px - length/speed metrics are not comparable.")
    if opt.method == "mixed" and df["animal"].nunique() < 10:
        info.append("Mixed-model p-values (Wald) are optimistic with fewer than ~10 animals; "
                    "report them together with the per-animal test.")
    if n_old:
        info.append(f"{n_old} recording(s) were analysed before the anatomical angle convention "
                    "(0° T, 90° D, 180° N, 270° V) and are left out of the direction statistics - "
                    "re-run Waves on them.")
    dropped = len(df_rec) - len(df)
    if dropped:
        info.append(f"{dropped} recording(s) without a {gc} label were left out.")
    if opt.unit == "recording" and df.groupby("animal").size().max() > 1 and opt.method != "mixed":
        info.append("Note: several recordings come from the same animal; treating them as independent "
                    "inflates n. Consider unit = animal or the mixed model.")
    return agg, pd.DataFrame(desc), pd.DataFrame(tests), pd.DataFrame(ddesc), "\n".join(info), groups
