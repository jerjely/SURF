"""Direct mouse navigation for matplotlib figures (all SURF plots).

Over an AXIS (the tick labels below / left of a plot):
    drag            pan that axis only
    wheel           zoom that axis only (around the cursor)
    right-drag      stretch / squeeze that axis
Inside the PLOT:
    drag            pan
    wheel           zoom time (x);  Shift+wheel = y;  Ctrl+wheel = both  (images: both)
    right-drag      zoom x (left/right) and y (up/down) independently
    double-click    reset the view
Clicks without movement still reach the plot (e.g. picking a square on the activity map).
The matplotlib toolbar modes (pan / zoom) take over while they are switched on.
"""
from __future__ import annotations

import math

from PySide6 import QtCore

AXIS_MARGIN_X = 55      # px left of the axes that count as the y axis
AXIS_MARGIN_Y = 40      # px below the axes that count as the x axis
DRAG_THRESHOLD = 4


class AxisNavigator:
    def __init__(self, canvas, toolbar=None):
        self.canvas = canvas
        self.toolbar = toolbar
        self.drag = None
        self.last_moved = False
        self._cursor = None
        canvas.mpl_connect("scroll_event", self._scroll)
        canvas.mpl_connect("button_press_event", self._press)
        canvas.mpl_connect("button_release_event", self._release)
        canvas.mpl_connect("motion_notify_event", self._motion)

    # ------------------------------------------------------------------ helpers
    def _busy(self):
        return bool(self.toolbar is not None and getattr(self.toolbar, "mode", ""))

    def _hit(self, ev):
        """(axes, zone) for the mouse position; zone in {'x', 'y', 'in'}."""
        best = None
        for ax in self.canvas.figure.axes:
            if not ax.get_visible() or ax.name != "rectilinear" or not ax.get_navigate():
                continue
            b = ax.bbox
            if b.x0 <= ev.x <= b.x1 and b.y0 <= ev.y <= b.y1:
                return ax, "in"
            if b.x0 <= ev.x <= b.x1 and b.y0 - AXIS_MARGIN_Y <= ev.y < b.y0:
                best = best or (ax, "x")
            elif b.x0 - AXIS_MARGIN_X <= ev.x < b.x0 and b.y0 <= ev.y <= b.y1:
                best = best or (ax, "y")
        return best or (None, None)

    @staticmethod
    def _to_data(ax, x, y):
        return ax.transData.inverted().transform((x, y))

    @staticmethod
    def _zoom_lim(lim, centre, f):
        a, b = lim
        return centre - (centre - a) * f, centre + (b - centre) * f

    @staticmethod
    def _remember(ax):
        """Original view of this plot (restored by double-click)."""
        if not hasattr(ax, "_surf_home"):
            ax._surf_home = (ax.get_xlim(), ax.get_ylim())

    def _set_cursor(self, zone):
        shape = {"x": QtCore.Qt.SizeHorCursor, "y": QtCore.Qt.SizeVerCursor}.get(zone)
        if shape == self._cursor:
            return
        self._cursor = shape
        if shape is None:
            self.canvas.unsetCursor()
        else:
            self.canvas.setCursor(shape)

    # ------------------------------------------------------------------ wheel
    def _scroll(self, ev):
        if self._busy():
            return
        ax, zone = self._hit(ev)
        if ax is None:
            return
        self._remember(ax)
        f = 0.85 ** (ev.step if ev.step else (1 if ev.button == "up" else -1))
        cx, cy = self._to_data(ax, ev.x, ev.y)
        key = (ev.key or "").lower()
        if zone == "x":
            do_x, do_y = True, False
        elif zone == "y":
            do_x, do_y = False, True
        elif ax.images or "control" in key or "ctrl" in key:
            do_x, do_y = True, True
        elif "shift" in key:
            do_x, do_y = False, True
        else:
            do_x, do_y = True, False
        if do_x:
            ax.set_xlim(*self._zoom_lim(ax.get_xlim(), cx, f))
        if do_y:
            ax.set_ylim(*self._zoom_lim(ax.get_ylim(), cy, f))
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ drag
    def _press(self, ev):
        if self._busy() or ev.button not in (1, 3):
            return
        ax, zone = self._hit(ev)
        if ax is None:
            return
        if ev.dblclick and ev.button == 1:
            self._reset(ax)
            return
        self._remember(ax)
        self.drag = dict(ax=ax, zone=zone, button=ev.button, x=ev.x, y=ev.y,
                         xlim=ax.get_xlim(), ylim=ax.get_ylim(), moved=False,
                         inv=ax.transData.inverted(), c=self._to_data(ax, ev.x, ev.y))

    def _release(self, ev):
        self.last_moved = bool(self.drag and self.drag["moved"])   # was this a drag, not a click?
        self.drag = None

    def _motion(self, ev):
        d = self.drag
        if d is None:
            if not self._busy():
                self._set_cursor(self._hit(ev)[1])
            return
        dx, dy = ev.x - d["x"], ev.y - d["y"]
        if not d["moved"] and abs(dx) + abs(dy) < DRAG_THRESHOLD:
            return
        d["moved"] = True
        ax, zone = d["ax"], d["zone"]
        x0, x1 = d["xlim"]
        y0, y1 = d["ylim"]
        if d["button"] == 1:                          # pan
            p0 = d["inv"].transform((d["x"], d["y"]))
            p1 = d["inv"].transform((ev.x, ev.y))
            ddx, ddy = p1[0] - p0[0], p1[1] - p0[1]
            if zone in ("x", "in"):
                ax.set_xlim(x0 - ddx, x1 - ddx)
            if zone in ("y", "in"):
                ax.set_ylim(y0 - ddy, y1 - ddy)
        else:                                         # right-drag: stretch / squeeze
            cx, cy = d["c"]
            if zone in ("x", "in"):
                ax.set_xlim(*self._zoom_lim((x0, x1), cx, math.exp(-dx / 150.0)))
            if zone in ("y", "in"):
                ax.set_ylim(*self._zoom_lim((y0, y1), cy, math.exp(-dy / 150.0)))
        self.canvas.draw_idle()

    def _reset(self, ax):
        home = getattr(ax, "_surf_home", None)
        if home is not None:
            ax.set_xlim(*home[0])
            ax.set_ylim(*home[1])
            self.canvas.draw_idle()
            return
        ax.relim()
        ax.autoscale_view()
        self.canvas.draw_idle()


HINT = ("Drag / wheel on an axis = pan / zoom that axis · right-drag = stretch · "
        "Shift+wheel = y · double-click = reset")
