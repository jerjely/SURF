"""Matplotlib figures for the wave GUI (same conventions as the notebook)."""
from __future__ import annotations

import numpy as np
import matplotlib.patches as patches

import wave_core as wc


def _u(s):
    """(factor, unit) for um-per-px s."""
    return (s, "µm") if s else (1.0, "px")


def _bg(ax, image, meas, s=None):
    k, unit = _u(s)
    if image is not None:
        h, w = image.shape[:2]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, extent=(0, w * k, h * k, 0))
    pad = 50
    ax.set_xlim((meas.x.min() - pad) * k, (meas.x.max() + pad) * k)
    ax.set_ylim((meas.y.max() + pad) * k, (meas.y.min() - pad) * k)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")


def _orientation(ax, ori: wc.Orientation, s=None):
    """Nasal-temporal axis + derived dorsal-ventral axis."""
    k, _ = _u(s)
    if ori.temporal is None or ori.nasal is None:
        return
    T, N = np.asarray(ori.temporal) * k, np.asarray(ori.nasal) * k
    ax.plot([T[0], N[0]], [T[1], N[1]], color="purple")
    ax.text(*T, "T", color="purple", fontweight="bold")
    ax.text(*N, "N", color="purple", fontweight="bold")
    D, V = ori.dv_points()
    D, V = np.asarray(D) * k, np.asarray(V) * k
    ax.plot([D[0], V[0]], [D[1], V[1]], color="orange")
    ax.text(*D, "D", color="darkorange", fontweight="bold")
    ax.text(*V, "V", color="darkorange", fontweight="bold")


def _anat_polar(ax, temporal_left=True):
    """Anatomical polar axes with dorsal up. temporal_left=True is the view of a right retina
    from the GCL (T left, N right); False is a left retina (T right, N left)."""
    if temporal_left:
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
    else:
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["T", "", "D", "", "N", "", "V", ""], fontweight="bold")


def polar_hist(fig, angles_deg, title="", bins=20, ylim=20, temporal_left=True):
    """Notebook polar histogram: percentages, zero at West, clockwise."""
    fig.clear()
    ax = fig.add_subplot(111, polar=True)
    a = np.radians(np.asarray(angles_deg, float))
    a = a[np.isfinite(a)]
    if len(a):
        n, edges = np.histogram(a, bins=bins, range=(0, 2 * np.pi))
        pct = n / n.sum() * 100
        ax.bar(edges[:-1], pct, width=2 * np.pi / bins, bottom=0.0, align="edge",
               color="blue", alpha=0.75, edgecolor="black")
        ax.set_ylim(0, max(ylim, float(pct.max()) * 1.1))
        m = wc.circ_mean_deg(np.degrees(a))
        title = f"{title}\n n={len(a)}, circular mean {m:.0f}°"
    _anat_polar(ax, temporal_left)
    ax.set_title(title, fontsize=9)
    return ax


def raster(fig, peaks, fps, t_max=None):
    fig.clear()
    ax = fig.add_subplot(111)
    ev = [p / fps for p in peaks]
    ax.eventplot(ev, lineoffsets=np.arange(len(ev)), linelengths=0.9, linewidths=1, colors="k")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ROI")
    if t_max:
        ax.set_xlim(0, t_max)
    ax.set_ylim(-1, len(ev))
    ax.set_title("Event raster")
    return ax


def rate_hist(fig, rates, vmax=3.0):
    fig.clear()
    ax = fig.add_subplot(111)
    r = rates[rates > 0.2]
    ax.hist(r, bins=np.arange(0, max(vmax, r.max() if len(r) else vmax) + 0.1, 0.1),
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Events / min")
    ax.set_ylabel("ROIs")
    ax.set_title(f"Cell rate (ROIs > 0.2/min: {len(r)})")
    return ax


MAP_CMAPS = ["viridis", "magma", "inferno", "plasma", "hot", "turbo", "jet", "cividis", "Greys_r", "coolwarm"]


def map_limits(rates, lo_pct=2.0, hi_pct=98.0):
    """Automatic colour range: percentiles of the non-zero rates."""
    r = np.asarray(rates, float)
    r = r[np.isfinite(r) & (r > 0)]
    if not len(r):
        return 0.0, 1.0
    lo, hi = np.percentile(r, (lo_pct, hi_pct))
    return float(lo), float(max(hi, lo + 1e-3))


def activity_map(fig, meas, rates, image=None, vmax=3.0, s=None, vmin=0.2, cmap="viridis", alpha=1.0,
                 show_bg=True, hide_below=False):
    """Rate per grid square. Squares are drawn at their real size when the grid is known."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    fig.clear()
    ax = fig.add_subplot(111)
    k, _ = _u(s)
    _bg(ax, image if show_bg else None, meas, s)
    if not show_bg:
        ax.set_facecolor("black")
    rates = np.asarray(rates, float)
    keep = np.ones(len(rates), bool) if not hide_below else rates >= vmin
    size = (meas.grid or {}).get("size") if isinstance(meas.grid, dict) else None
    if size:
        h = size * k
        rects = [Rectangle((x * k - h / 2, y * k - h / 2), h, h) for x, y in zip(meas.x[keep], meas.y[keep])]
        sc = PatchCollection(rects, cmap=cmap, alpha=alpha, linewidths=0)
        sc.set_array(rates[keep])
        sc.set_clim(vmin, vmax)
        ax.add_collection(sc)
    else:
        sc = ax.scatter(meas.x[keep] * k, meas.y[keep] * k, s=40, marker="s", c=rates[keep], cmap=cmap,
                        vmin=vmin, vmax=vmax, alpha=alpha)
    fig.colorbar(sc, ax=ax, label="events / min", shrink=0.8,
                 extend="both" if (rates.min(initial=0) < vmin and rates.max(initial=0) > vmax) else
                 "max" if rates.max(initial=0) > vmax else "min" if rates.min(initial=0) < vmin else "neither")
    ax.set_title("Cell activity")
    return ax


def trace(fig, meas, k, peaks, sigma, s=None, t_mark=None, preview=False):
    fig.clear()
    ax = fig.add_subplot(111)
    t = np.arange(meas.n_frames) / meas.fps
    raw = meas.mean[:, k]
    sm = wc.smooth_traces(raw[:, None], sigma)[:, 0]
    ax.plot(t, raw, color="0.7", lw=0.6, label="raw")
    ax.plot(t, sm, color="C0", lw=1.0, label=f"smoothed (σ={sigma:g})")
    pk = peaks[k] if peaks is not None else []
    if len(pk):
        ax.plot(pk / meas.fps, sm[pk], "v", color="magenta" if preview else "red", ms=6,
                label=f"{len(pk)} events" + (" (live preview)" if preview else ""))
    elif preview:
        ax.plot([], [], "v", color="magenta", label="0 events (live preview)")
    f, unit = _u(s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean grey value")
    ax.set_title(f"ROI {meas.names[k]}  (x={meas.x[k] * f:.1f}, y={meas.y[k] * f:.1f} {unit})")
    ax.legend(loc="upper right", fontsize=8)
    ax._tmark = ax.axvline(t_mark if t_mark is not None else np.nan, color="orange", lw=1.2)
    return ax


def tracks_map(fig, meas, res: wc.WaveResults, image=None, min_steps=0, s=None):
    fig.clear()
    ax = fig.add_subplot(111)
    k, _ = _u(s)
    _bg(ax, image, meas, s)
    for t in res.tracks:
        if t.n < min_steps:
            continue
        pts = np.vstack([t.prev[:1], t.cur]) * k
        ax.plot(pts[:, 0], pts[:, 1], color="orange", lw=1.2)
        ax.scatter(*(t.start * k), s=40, color="green", zorder=3)
        ax.scatter(*(t.end * k), s=40, color="red", zorder=3)
    _orientation(ax, res.orientation, s)
    ax.set_title(f"Tracks (n={len(res.tracks)}; green start, red end)")
    return ax


def smoothed_paths(fig, meas, res: wc.WaveResults, image=None, s=None):
    fig.clear()
    ax = fig.add_subplot(111)
    k, _ = _u(s)
    _bg(ax, image, meas, s)
    for _, sm in res.smoothed_paths:
        sm = sm * k
        ax.plot(sm[:, 0], sm[:, 1], color="grey")
        ax.scatter(*sm[0], color="green", s=25, zorder=3)
        ax.scatter(*sm[-1], color="red", s=25, zorder=3)
    _orientation(ax, res.orientation, s)
    ax.set_title(f"Smoothed paths (path > threshold, n={len(res.smoothed_paths)})")
    return ax


def arrow_map(fig, meas, res: wc.WaveResults, image=None, arrow_len=10, s=None):
    fig.clear()
    ax = fig.add_subplot(111)
    k, _ = _u(s)
    _bg(ax, image, meas, s)
    gv = res.grid_vectors
    if len(gv):
        width = gv.attrs.get("bin_width", 1.0) * k
        L = arrow_len * k
        for _, r in gv.iterrows():
            X, Y = r.X * k, r.Y * k
            ax.add_patch(patches.Rectangle((X - width / 2, Y - width / 2), width, width,
                                           edgecolor="orange", alpha=0.5, facecolor="none"))
            if np.isfinite(r.angle_img_deg):
                a = np.deg2rad(r.angle_img_deg)
                ax.arrow(X, Y, L * np.cos(a), L * np.sin(a),
                         head_width=5 * k, head_length=10 * k, fc="r", ec="r")
    ev = res.wave_events
    if len(ev):
        ax.scatter(ev.start_x * k, ev.start_y * k, s=120, color="C0", zorder=4, label="wave start")
        ax.legend(loc="upper right", fontsize=8)
    _orientation(ax, res.orientation, s)
    ax.set_title("Local mean propagation direction")
    return ax


def log_isi(fig, res: wc.WaveResults, cutoff):
    fig.clear()
    ax = fig.add_subplot(111)
    tt = res.track_table
    if len(tt) > 1:
        isi = np.diff(np.sort(tt.start_s.values))
        isi = isi[isi > 0]
        if len(isi):
            ax.hist(np.log(isi), bins=np.arange(np.log(isi).min(), np.log(isi).max() + 0.1, 0.1),
                    color="0.4")
        ax.axvline(np.log(cutoff), color="r", ls="--", label=f"wave cutoff {cutoff:g} s")
        ax.legend(fontsize=8)
    ax.set_xlabel("log(inter-track start interval, s)")
    ax.set_ylabel("count")
    ax.set_title("Track start intervals")
    return ax


# ----------------------------------------------------------------------------- comparison
def _p_text(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return f"p={p:.3g} {stars}"


STYLES = {
    "box": "Box plot (median, IQR)",
    "median_iqr": "Median ± IQR",
    "mean_sem": "Mean ± SEM",
    "mean_sd": "Mean ± SD",
    "bar_sem": "Bar: mean ± SEM",
}


def default_color(i):
    import matplotlib
    return matplotlib.colors.to_hex(f"C{i % 10}")


def _significance(ax, tests, key, pos, ymax, brackets, label):
    """Omnibus / trend lines for the title and pairwise brackets (Holm p). Returns the title."""
    t = tests[tests.metric == key] if len(tests) else tests
    title = label
    if len(t):
        omni = t[t.comparison == "all groups"]
        if len(omni):
            title += f"\n{omni.iloc[0].test}: {_p_text(omni.iloc[0].p)}"
        tr = t[(t.comparison == "trend") & t.test.astype(str).str.startswith("Jonckheere")]
        if len(tr):
            r0 = tr.iloc[0]
            way = "↑" if str(r0.get("note", "")).startswith("increasing") else "↓"
            title += f"\ntrend {way} (Jonckheere-Terpstra): {_p_text(r0.p)}"
        if np.isfinite(ymax):
            lo, hi = ax.get_ylim()
            step = (hi - lo) * 0.08
            y = ymax + step
            n_sig = n_pair = 0
            for _, r in t[~t.comparison.isin(["all groups", "trend"])].iterrows():
                try:
                    b, a = [s.strip() for s in r.comparison.split(" vs ")]
                except ValueError:
                    continue
                if a not in pos or b not in pos:
                    continue
                p = r.get("p_holm", np.nan)
                p = r.p if not np.isfinite(p) else p
                n_pair += 1
                sig = np.isfinite(p) and p < 0.05
                n_sig += sig
                if brackets == "none" or (brackets == "sig" and not sig):
                    continue
                x1, x2 = pos[a], pos[b]
                ax.plot([x1, x1, x2, x2], [y - step * 0.25, y, y, y - step * 0.25], color="k", lw=0.8)
                ax.text((x1 + x2) / 2, y, _p_text(p), ha="center", va="bottom", fontsize=8)
                y += step
            ax.set_ylim(top=max(hi, y + step * 0.5))
            if brackets == "sig" and n_pair and not n_sig:
                title += "\nno significant pairwise differences (Holm)"
    return title


def group_plot(fig, agg, group_col, groups, key, label, tests, reference="", colors=None,
               style="box", points=True, lines=True, brackets="sig"):
    """Groups in the given order; summary per `style`; one dot per analysed unit;
    same-animal points joined; p values vs reference (Holm)."""
    colors = colors or {}
    fig.clear()
    ax = fig.add_subplot(111)
    pos = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    ymax = -np.inf
    data = []
    for g in groups:
        sub = agg[agg[group_col] == g]
        v = sub[key].to_numpy(float)
        ok = np.isfinite(v)
        data.append(v[ok])
        if ok.any():
            ymax = max(ymax, v[ok].max())
    col = [colors.get(g) or default_color(i) for i, g in enumerate(groups)]
    if style == "box":
        bp = ax.boxplot([d if len(d) else [np.nan] for d in data], positions=range(len(groups)), widths=0.5,
                        showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=1.5),
                        whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
        for patch, c in zip(bp["boxes"], col):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(c)
    else:
        for i, d in enumerate(data):
            if not len(d):
                continue
            if style == "median_iqr":
                c_, lo, hi = np.median(d), np.percentile(d, 25), np.percentile(d, 75)
            else:
                c_ = d.mean()
                sp = d.std(ddof=1) if len(d) > 1 else 0.0
                if style in ("mean_sem", "bar_sem"):
                    sp = sp / np.sqrt(len(d)) if len(d) > 1 else 0.0
                lo, hi = c_ - sp, c_ + sp
            if style == "bar_sem":
                ax.bar(i, c_, width=0.6, color=col[i], alpha=0.45, edgecolor=col[i], zorder=0)
            else:
                ax.plot([i - 0.25, i + 0.25], [c_, c_], color=col[i], lw=2.5, zorder=4)
            ax.errorbar(i, c_, yerr=[[c_ - lo], [hi - c_]], color="k" if style == "bar_sem" else col[i],
                        capsize=6, lw=1.5, zorder=4)
            ymax = max(ymax, hi)
    # lines for animals present in several groups
    if lines and "animal" in agg.columns:
        for an, sub in agg.groupby("animal"):
            sub = sub[sub[group_col].isin(groups)].dropna(subset=[key])
            if sub[group_col].nunique() >= 2:
                xs = [pos[g] for g in sub[group_col]]
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], sub[key].to_numpy(float)[o], color="0.75", lw=0.8, zorder=1)
    for i, g in enumerate(groups):
        d = data[i]
        if points:
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(d)), d, s=28, color=col[i], zorder=3,
                       edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={len(d)})" for g, d in zip(groups, data)])
    ax.set_ylabel(label)
    title = _significance(ax, tests, key, pos, ymax, brackets, label)
    ax.set_title(title + f"\n({STYLES.get(style, style)})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def wave_violin(fig, wl, group_col, groups, key, label, tests, colors=None, points=True,
                brackets="sig", unit_name="waves", **_):
    """Every individual wave as one value: violin per group with median and IQR, the waves as
    dots, and the median of each recording as a diamond (shows the recording-to-recording spread)."""
    colors = colors or {}
    fig.clear()
    ax = fig.add_subplot(111)
    sub = wl[wl.metric == key] if len(wl) else wl
    pos = {g: i for i, g in enumerate(groups)}
    col = [colors.get(g) or default_color(i) for i, g in enumerate(groups)]
    rng = np.random.default_rng(0)
    data, nrec = [], []
    for g in groups:
        d = sub[sub[group_col] == g]
        data.append(d["value"].to_numpy(float))
        nrec.append(d["recording"].nunique())
    ymax = max([d.max() for d in data if len(d)] or [np.nan])
    have = [i for i, d in enumerate(data) if len(d) >= 2]
    if have:
        vp = ax.violinplot([data[i] for i in have], positions=have, widths=0.8, showextrema=False)
        for body, i in zip(vp["bodies"], have):
            body.set_facecolor(col[i])
            body.set_edgecolor(col[i])
            body.set_alpha(0.35)
    for i, d in enumerate(data):
        if not len(d):
            continue
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color="k", lw=4, solid_capstyle="butt", zorder=4)
        ax.scatter([i], [med], s=30, color="white", edgecolor="k", zorder=5)
        if points:
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=7, color=col[i], alpha=0.5,
                       linewidth=0, zorder=2)
        g = groups[i]
        meds = sub[sub[group_col] == g].groupby("recording")["value"].median().to_numpy(float)
        ax.scatter(i + 0.3 + rng.uniform(-0.03, 0.03, len(meds)), meds, marker="D", s=26, facecolor="white",
                   edgecolor=col[i], linewidth=1.3, zorder=6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n({len(d)} {unit_name},\n{r} rec)" for g, d, r in zip(groups, data, nrec)])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(label)
    title = _significance(ax, tests, key, pos, ymax if np.isfinite(ymax) else -np.inf, brackets, label)
    ax.set_title(title + f"\n(each {unit_name[:-1]}; ● median, bar IQR, ◇ median of each recording)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def group_polar(fig, angles_by_group, dirs_df, bins=20, colors=None, temporal_left=True):
    """One polar histogram per group (pooled segment angles, descriptive) with the group mean
    direction of the recordings as an arrow (length = R)."""
    fig.clear()
    groups = list(angles_by_group)
    n = max(len(groups), 1)
    for i, g in enumerate(groups):
        ax = fig.add_subplot(1, n, i + 1, polar=True)
        a = np.radians(np.asarray(angles_by_group[g], float))
        a = a[np.isfinite(a)]
        top = 20
        if len(a):
            cnt, edges = np.histogram(a, bins=bins, range=(0, 2 * np.pi))
            pct = cnt / cnt.sum() * 100
            ax.bar(edges[:-1], pct, width=2 * np.pi / bins, align="edge",
                   color=(colors or {}).get(g) or default_color(i),
                   alpha=0.7, edgecolor="black")
            top = max(20, pct.max() * 1.1)
        ax.set_ylim(0, top)
        row = dirs_df[dirs_df.group == g] if len(dirs_df) else dirs_df
        title = str(g)
        if len(row) and int(row.iloc[0].n or 0) > 0:
            r = row.iloc[0]
            sig = np.isfinite(r.rayleigh_p) and r.rayleigh_p < 0.05
            # the mean arrow only when there is a significant preferred direction
            if sig and np.isfinite(r.circular_mean_deg):
                ax.annotate("", xy=(np.radians(r.circular_mean_deg), top * r.R), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
                title += f"\nmean {r.circular_mean_deg:.0f}°, R={r.R:.2f}"
            else:
                title += f"\nno preferred direction (R={r.R:.2f})"
            title += f"\nRayleigh {_p_text(r.rayleigh_p)} (n={int(r.n)})"
        else:
            title += "\nno direction data"
        _anat_polar(ax, temporal_left)
        ax.set_title(title, fontsize=9)
    return fig
