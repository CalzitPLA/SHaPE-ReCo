#!/usr/bin/env python3
"""
Live statistics of the run: the process sampled every second, and every step as it arrives, on one time axis.

Small multiples (one quantity per panel, never two y-scales in one), all against the seconds since Start:
  CPU        cores busy (this process and its children, e.g. freecadcmd)
  memory     GB resident (this process + children) and GB still available on the machine
  steps      each step's wall time, at the moment it finished
  residual   voxels missing (aqua) / in excess (red), the colours of the 3D view's residual layers
  operations accepted operations so far

A pie (fixed size, beside the readout) splits the voxels of the selected step: explained by the model, missing
(leftover input the model does not cover) and in excess (model outside the input). The layout is fixed - margins, tick
label widths and legends do not move while the run grows - so the panels keep their size on screen.

A slider picks a step: a cursor marks it in every panel, the line under the slider gives its numbers, and the window
shows that step (`stepSelected(index)`). Moving the mouse over the charts reads the values at that time. The slider
follows the newest step until it is moved by hand; "live" takes it back.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"      # palette slots 1, 2, 3, 8


def _fixed_width(value, _pos=None) -> str:
    """Tick labels of one width (figure space: U+2007), so the plot area does not move when values grow."""
    text = f"{value:,.0f}" if abs(value) >= 10 else f"{value:.1f}"
    return text.rjust(6, "\u2007")


class StatsPanel(QtWidgets.QWidget):
    stepSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, sample_ms: int = 1000):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(6, 7), facecolor=SURFACE)
        self.fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.07, hspace=0.55)   # fixed: no reflow
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumHeight(360)
        self.pie_fig = Figure(figsize=(2.3, 2.7), facecolor=SURFACE)
        self.pie_canvas = FigureCanvasQTAgg(self.pie_fig)
        self.pie_canvas.setFixedSize(230, 270)                          # fixed: the pie keeps its size on screen
        self.pie_ax = self.pie_fig.add_axes((0.1, 0.27, 0.8, 0.7))
        names = [("CPU", "cores"), ("memory", "GB"), ("step time", "s"), ("residual", "voxels"), ("operations", "")]
        self.axes = []
        for i, (title, unit) in enumerate(names):
            ax = self.fig.add_subplot(len(names), 1, i + 1, sharex=self.axes[0] if self.axes else None)
            ax.set_facecolor(SURFACE)
            ax.set_title(f"{title}" + (f" ({unit})" if unit else ""), loc="left", fontsize=9, color=INK)
            ax.tick_params(labelsize=8, colors=INK2, length=0)
            ax.yaxis.set_major_formatter(_fixed_width)                  # same label width at 0.5 and at 16000
            ax.grid(True, axis="y", color=GRID, linewidth=0.8)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if i < len(names) - 1:
                ax.tick_params(labelbottom=False)
            self.axes.append(ax)
        self.axes[-1].set_xlabel("seconds since Start", fontsize=8, color=INK2)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        self.live = QtWidgets.QToolButton()
        self.live.setText("live")
        self.live.setCheckable(True)
        self.live.setChecked(True)
        self.live.setToolTip("the slider follows the newest step")
        self.live.toggled.connect(lambda on: on and self._follow_newest())
        self.readout = QtWidgets.QLabel("No run yet.")
        self.readout.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.readout.setWordWrap(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("step"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.live)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(row)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.pie_canvas)
        top.addWidget(self.readout, 1)
        layout.addLayout(top)
        layout.addWidget(self.canvas, 1)

        self._proc = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(sample_ms)
        self._timer.timeout.connect(self._sample)
        self._dirty = False
        self._cursor = None
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.reset()

    # ---------------------------------------------------------------------------------------- data in
    def reset(self):
        """A new run: clear the series and start sampling the process."""
        self.t0 = time.time()
        self.samples = dict(t=[], cores=[], rss=[], avail=[])
        self.steps: List[dict] = []
        self.input_voxels: Optional[int] = None
        self._cpu_prev = None
        self.slider.blockSignals(True)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.blockSignals(False)
        self.readout.setText("Running ..." if self._timer.isActive() else "No run yet.")
        self._dirty = True
        self._redraw()

    def start_sampling(self):
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            self._proc = None
        self._timer.start()

    def stop_sampling(self):
        self._timer.stop()

    def _cpu_times(self) -> Optional[float]:
        if self._proc is None:
            t = os.times()
            return t.user + t.system + t.children_user + t.children_system
        total = sum(self._proc.cpu_times()[:2])
        for child in self._proc.children(recursive=True):
            try:
                total += sum(child.cpu_times()[:2])
            except Exception:                                   # the child ended meanwhile
                pass
        return total

    def _sample(self):
        now = time.time() - self.t0
        cpu = self._cpu_times()
        if self._cpu_prev is not None and now > self._cpu_prev[0]:
            self.samples["t"].append(now)
            self.samples["cores"].append(max(0.0, (cpu - self._cpu_prev[1]) / (now - self._cpu_prev[0])))
            rss, avail = float("nan"), float("nan")
            if self._proc is not None:
                import psutil
                rss = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
                avail = psutil.virtual_memory().available
            self.samples["rss"].append(rss / 2 ** 30)
            self.samples["avail"].append(avail / 2 ** 30)
            self._dirty = True
        self._cpu_prev = (now, cpu)
        if self.isVisible():
            self._redraw()

    def add_step(self, state, wall: Optional[float], cpu: Optional[float]):
        """One finished step (a ViewState of any mode; the cascade's carry the residual and the program)."""
        step = state.step
        program = getattr(state, "program", None)
        n_in = (getattr(state, "info", None) or {}).get("input_voxels")
        if n_in:
            self.input_voxels = int(n_in)
        self.steps.append(dict(
            index=state.index, t=time.time() - self.t0, stage=step.stage, message=step.message, wall=wall, cpu=cpu,
            pos=getattr(state, "residual_pos", None), neg=getattr(state, "residual_neg", None),
            ops=len(program) if program is not None else None))
        self.slider.blockSignals(True)
        self.slider.setEnabled(True)
        self.slider.setRange(0, len(self.steps) - 1)
        if self.live.isChecked():
            self.slider.setValue(len(self.steps) - 1)
        self.slider.blockSignals(False)
        self._show_readout(self.slider.value())
        self._dirty = True

    # ---------------------------------------------------------------------------------------- selection
    def _follow_newest(self):
        if self.steps:
            self.slider.setValue(len(self.steps) - 1)

    def _on_slider(self, value: int):
        if not self.steps:
            return
        if value != len(self.steps) - 1 and self.live.isChecked():
            self.live.blockSignals(True)
            self.live.setChecked(False)                     # moved by hand: stop following
            self.live.blockSignals(False)
        self._show_readout(value)
        self._dirty = True
        self._redraw()
        self.stepSelected.emit(int(self.steps[value]["index"]))

    def select_step(self, index: int):
        """The window shows step `index` (timeline click etc.): move the cursor there without echoing it back."""
        for i, s in enumerate(self.steps):
            if s["index"] == index:
                self.slider.blockSignals(True)
                self.slider.setValue(i)
                self.slider.blockSignals(False)
                self._show_readout(i)
                self._dirty = True
                self._redraw()
                return

    def _show_readout(self, i: int):
        s = self.steps[i]
        parts = [f"<b>{s['index'] + 1} {s['stage']}</b> at {s['t']:.1f} s"]
        if s["wall"] is not None:
            parts.append(f"took {s['wall']:.2f} s" + (f", CPU {s['cpu']:.1f} s" if s["cpu"] is not None else ""))
        if s["pos"] is not None:
            parts.append(f"residual +{s['pos']} / -{s['neg']} voxels")
        if s["ops"] is not None:
            parts.append(f"{s['ops']} operation(s)")
        if self.input_voxels and s["pos"] is not None:
            parts.append(f"<b>{s['pos']} voxels left over</b> ({s['pos'] / self.input_voxels:.1%} of the input)")
        self.readout.setText(" &middot; ".join(parts) + f"<br><span style='color:{INK2}'>{s['message'][:160]}</span>")

    def _on_hover(self, event):
        if event.inaxes is None or event.xdata is None or not self.samples["t"]:
            return
        import bisect
        t = self.samples["t"]
        k = min(max(bisect.bisect_left(t, event.xdata), 0), len(t) - 1)
        text = f"{t[k]:.0f} s: {self.samples['cores'][k]:.1f} cores, {self.samples['rss'][k]:.2f} GB used, " \
               f"{self.samples['avail'][k]:.1f} GB available"
        steps_before = [s for s in self.steps if s["t"] <= event.xdata]
        if steps_before:
            s = steps_before[-1]
            text += f" &middot; step {s['index'] + 1} {s['stage']}"
        self.readout.setText(text)

    # ---------------------------------------------------------------------------------------- drawing
    def _draw_pie(self):
        """The selected step's voxels: explained (in the input and in the model), missing (the leftover), excess."""
        ax = self.pie_ax
        ax.clear()
        ax.set_facecolor(SURFACE)
        s = self.steps[self.slider.value()] if self.steps else None
        if s is None or s["pos"] is None or not self.input_voxels:
            ax.set_axis_off()
            ax.set_title("voxels: no step yet", fontsize=8, color=INK2)
            self.pie_canvas.draw_idle()
            return
        missing, excess = int(s["pos"]), int(s["neg"] or 0)
        explained = max(self.input_voxels - missing, 0)
        parts = [(explained, "explained", BLUE), (missing, "missing", AQUA), (excess, "excess", RED)]
        shown = [(v, n, c) for v, n, c in parts if v > 0] or [(1, "empty", GRID)]
        total = sum(v for v, _, _ in shown)
        ax.pie([v for v, _, _ in shown], colors=[c for _, _, c in shown], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=2))         # a ring: 2 px surface gaps
        ax.set_aspect("equal")
        ax.text(0, 0.08, f"{missing:,}", ha="center", va="center", fontsize=11, color=INK, weight="bold")
        ax.text(0, -0.18, "left over", ha="center", va="center", fontsize=7, color=INK2)
        legend = [f"{n} {v:,} ({v / total:.1%})" for v, n, _ in shown]
        ax.legend(legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False,
                  labelcolor=INK2, ncol=1, handlelength=0.8)                    # under the ring, inside the canvas
        self.pie_canvas.draw_idle()

    def _redraw(self):
        if not self._dirty:
            return
        self._dirty = False
        cpu_ax, mem_ax, step_ax, res_ax, ops_ax = self.axes
        for ax in self.axes:
            for artist in list(ax.lines) + list(ax.collections):
                artist.remove()
        S = self.samples
        if S["t"]:
            cpu_ax.plot(S["t"], S["cores"], color=BLUE, linewidth=2)
            mem_ax.plot(S["t"], S["rss"], color=BLUE, linewidth=2)
            mem_ax.plot(S["t"], S["avail"], color=ORANGE, linewidth=2)
            cpu_ax.set_ylim(bottom=0)
            mem_ax.set_ylim(bottom=0)
        st = [s for s in self.steps if s["wall"] is not None]
        if st:
            step_ax.vlines([s["t"] for s in st], 0, [s["wall"] for s in st], color=BLUE, linewidth=2)
            step_ax.scatter([s["t"] for s in st], [s["wall"] for s in st], s=14, color=BLUE, zorder=3)
            step_ax.set_ylim(bottom=0)
        rs = [s for s in self.steps if s["pos"] is not None]
        if rs:
            tt = [s["t"] for s in rs]
            res_ax.step(tt, [s["pos"] for s in rs], where="post", color=AQUA, linewidth=2)
            res_ax.step(tt, [s["neg"] for s in rs], where="post", color=RED, linewidth=2)
            res_ax.set_ylim(bottom=0)
        os_ = [s for s in self.steps if s["ops"] is not None]
        if os_:
            ops_ax.step([s["t"] for s in os_], [s["ops"] for s in os_], where="post", color=BLUE, linewidth=2)
            ops_ax.set_ylim(bottom=0)
        if self.steps:
            t = self.steps[self.slider.value()]["t"]
            for ax in self.axes:
                ax.axvline(t, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        right = max([S["t"][-1] if S["t"] else 0.0] + [s["t"] for s in self.steps] + [1.0])
        cpu_ax.set_xlim(0, right * 1.02)
        self._legends()
        self.canvas.draw_idle()
        self._draw_pie()

    def _legends(self):
        """Legends from the start, in the title row (they never cover data and never move the layout)."""
        from matplotlib.lines import Line2D
        for ax, entries in ((self.axes[1], (("used by the run", BLUE), ("available", ORANGE))),
                            (self.axes[3], (("missing (left over)", AQUA), ("excess", RED)))):
            if ax.get_legend() is None:
                ax.legend([Line2D([], [], color=c, linewidth=2) for _, c in entries], [n for n, _ in entries],
                          loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, frameon=False, labelcolor=INK2,
                          ncol=2, borderaxespad=0.1)
