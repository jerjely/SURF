"""Publication-ready figure export.

* Text stays text: SVG keeps real <text> (svg.fonttype 'none'); EPS/PDF embed TrueType fonts
  (fonttype 42) instead of outlines / Type 3 glyphs, so Illustrator can edit the words.
* Same text size in every exported figure: the figure is saved at a fixed physical size with every
  text element set to the same point size (ticks / labels / titles), whatever the size of the plot
  on screen. Placed at 100 % in a layout program, all panels then share the same font size.
"""
import matplotlib
from matplotlib.text import Text

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
SIZES = dict(tick=6.0, label=7.0, title=7.0, legend=6.0, other=6.0)   # points
PANEL_W = 3.3          # inches: one panel; wider figures (several axes in a row) get more


def setup_rc():
    matplotlib.rcParams.update({
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
    })


def _role(t, fig):
    for ax in fig.axes:
        if t is ax.title or t is getattr(ax, "_left_title", None) or t is getattr(ax, "_right_title", None):
            return "title"
        if t is ax.xaxis.label or t is ax.yaxis.label:
            return "label"
        if t in ax.get_xticklabels() + ax.get_yticklabels():
            return "tick"
        leg = ax.get_legend()
        if leg is not None and (t in leg.get_texts() or t is leg.get_title()):
            return "legend"
    if fig._suptitle is not None and t is fig._suptitle:
        return "title"
    return "other"


def save_publication(fig, base, exts=("png", "svg", "eps", "pdf"), dpi=300, sizes=None):
    """Save `fig` as base.<ext> at a fixed physical size and uniform font sizes, then restore it."""
    setup_rc()
    sizes = dict(SIZES, **(sizes or {}))
    w0, h0 = fig.get_size_inches()
    aspect = w0 / max(h0, 1e-6)
    ncols = max(1, len({round(ax.get_position().x0, 2) for ax in fig.axes if ax.get_visible()
                        and ax.get_label() != "<colorbar>"}))
    w = PANEL_W * min(ncols, 3)
    h = max(1.6, min(w / aspect, 7.0))
    texts = [t for t in fig.findobj(Text) if t.get_text().strip()]
    old = [(t, t.get_fontsize(), t.get_fontfamily()) for t in texts]
    # tick labels are re-created at every draw: set their size on the axis instead
    old_ticks = []
    for ax in fig.axes:
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            tl = axis.get_ticklabels(which="both")
            old_ticks.append((ax, name, tl[0].get_fontsize() if tl else matplotlib.rcParams["xtick.labelsize"]))
    # markers and lines shrink with the figure so they keep their proportions
    from matplotlib.lines import Line2D
    from matplotlib.collections import Collection
    f = min(1.0, max(0.35, w / max(w0, 1e-6)))
    lines = [(l, l.get_markersize(), l.get_linewidth()) for l in fig.findobj(Line2D)]
    colls = []
    for c in fig.findobj(Collection):
        try:
            colls.append((c, c.get_sizes().copy() if hasattr(c, "get_sizes") else None, c.get_linewidths().copy()))
        except Exception:
            pass
    try:
        if f < 1.0:
            for l, ms, lw in lines:
                l.set_markersize(ms * f)
                l.set_linewidth(max(0.4, lw * f))
            for c, sz, lw in colls:
                if sz is not None and len(sz):
                    c.set_sizes(sz * f * f)
                if len(lw):
                    c.set_linewidths(lw * f)
        for ax, name, _ in old_ticks:
            ax.tick_params(axis=name, which="both", labelsize=sizes["tick"])
        fig.set_size_inches(w, h)
        for t in texts:
            t.set_fontsize(sizes[_role(t, fig)])
            t.set_fontfamily("sans-serif")
        try:
            fig.tight_layout()
        except Exception:
            pass
        for ext in exts:
            fig.savefig(f"{base}.{ext}", dpi=dpi)
    finally:
        for t, s, f in old:
            t.set_fontsize(s)
            t.set_fontfamily(f)
        for l, ms, lw in lines:
            l.set_markersize(ms)
            l.set_linewidth(lw)
        for c, sz, lw in colls:
            if sz is not None and len(sz):
                c.set_sizes(sz)
            if len(lw):
                c.set_linewidths(lw)
        for ax, name, sz in old_ticks:
            try:
                ax.tick_params(axis=name, which="both", labelsize=sz)
            except Exception:
                pass
        fig.set_size_inches(w0, h0)
        try:
            fig.tight_layout()
        except Exception:
            pass
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
