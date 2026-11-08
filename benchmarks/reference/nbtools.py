"""Timing, table and circuit-drawing helpers shared by the notebooks.

These are teaching aids, not part of lazyqc. Jupyter runs a notebook with its
own folder as the working directory, so the notebooks import this module
directly. ``draw_circuit`` is also used by ``scripts/render_cone_figure.py``.
"""

from __future__ import annotations

import math
import time
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

TEAL = "#007D84"
FADED = "#D5DBDB"
CUT = "#C0392B"
INK = "#222222"


def timed(fn, repeats=5):
    """Call ``fn`` ``repeats`` times; return its last value and the median wall time in ms."""
    value, stamps = None, []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        stamps.append((time.perf_counter_ns() - start) / 1e6)
    return value, median(stamps)


def compare(runs, rounds=15):
    """Time several callables round-robin; return the median ms of each, keyed like ``runs``.

    Interleaving spreads slow drifts of the machine (thermal throttling, other
    processes) evenly over the contenders instead of penalising whichever runs last.
    """
    stamps = {name: [] for name in runs}
    for _ in range(rounds):
        for name, fn in runs.items():
            start = time.perf_counter_ns()
            fn()
            stamps[name].append((time.perf_counter_ns() - start) / 1e6)
    return {name: median(values) for name, values in stamps.items()}


def label(observable):
    """Readable Pauli label: ``{0: "X", 1: "X"}`` or a lazyqc ``Pauli`` becomes ``X0 X1``."""
    factors = observable.items() if isinstance(observable, dict) else observable.factors
    return " ".join(f"{pauli}{qubit}" for qubit, pauli in sorted(factors)) or "I"


def _cell(value, digits):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.1e}"
        decimals = max(0, digits - 1 - math.floor(math.log10(abs(value))))
        return f"{value:.{decimals}f}"
    return str(value)


def table(rows, headers, digits=3):
    """Display rows as a Markdown table, showing floats with ``digits`` significant digits."""
    from IPython.display import Markdown, display

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v, digits) for v in row) + " |" for row in rows]
    display(Markdown("\n".join(lines)))


def cone_indices(operations, observable):
    """Return indices into ``operations`` retained by the backward causal cone."""
    from lazyqc.passes import causal_cone

    positions = {id(gate): i for i, gate in enumerate(operations)}
    _, cone = causal_cone(operations, observable)
    return {positions[id(gate)] for gate in cone}


def _gate_label(gate):
    from lazyqc.instruments import Projector

    if isinstance(gate, Projector):
        return f"P{gate.outcome}"
    if gate.name in ("RX", "RY", "RZ") and isinstance(gate.angle, float):
        return f"{gate.name}\n{gate.angle:.2f}"
    return gate.name


def _columns(operations, n_qubits):
    """Pack gates into layers: each gate takes the first column free on every wire it spans."""
    free = [0] * n_qubits
    columns = []
    for gate in operations:
        span = range(min(gate.qubits), max(gate.qubits) + 1)
        column = max(free[q] for q in span)
        for q in span:
            free[q] = column + 1
        columns.append(column)
    return columns


def draw_circuit(program, ax=None, *, keep=None, cuts=None, fragments=None, title=None,
                 legend=False):
    """Draw a program's resolved operations on a wire grid, qubit 0 at the top.

    Args:
        program: A bound :class:`Program` (or any object with ``qubits``,
            ``n_qubits`` and ``resolved_operations()``).
        ax: Optional axes to draw into; a new figure is created otherwise.
        keep: Optional set of operation indices to emphasize; the others fade.
        cuts: Optional set of operation indices (CX gates) marked as cuts.
        fragments: Optional mapping wire -> fragment id, drawn as row bands.
        title: Optional axes title.
        legend: Add a legend explaining kept, dropped and cut gates.
    """
    operations = list(program.resolved_operations())
    n_qubits = program.n_qubits
    keep = set(range(len(operations))) if keep is None else set(keep)
    cuts = set(cuts or ())
    fragments = dict(fragments or {})
    columns = _columns(operations, n_qubits)
    depth = max(columns, default=-1) + 1

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.0, 0.7 * depth + 1.5), 0.7 * n_qubits + 1.2))
    # Gate tuples carry integer wire positions; labels come from qubit names.
    y_of = {i: n_qubits - 1 - i for i in range(n_qubits)}
    for i, q in enumerate(program.qubits):
        y = y_of[i]
        if i in fragments or q in fragments:
            frag = fragments.get(i, fragments.get(q))
            ax.axhspan(y - 0.45, y + 0.45, color=f"C{frag % 10}", alpha=0.12, zorder=0)
        ax.plot([-0.6, depth - 0.4], [y, y], color="#999999", linewidth=1, zorder=1)
        ax.text(-0.85, y, str(q), ha="right", va="center", fontsize=10, zorder=4)
    for index, (x, gate) in enumerate(zip(columns, operations)):
        wires = [y_of[q] for q in gate.qubits]
        emphasized = index in keep
        color = TEAL if emphasized else FADED
        edge = INK if emphasized else "#AAAAAA"
        alpha = 1.0 if emphasized else 0.45
        if gate.name == "CX":
            control, target = wires
            ax.plot([x, x], [control, target], color=edge, linewidth=2, alpha=alpha, zorder=2)
            for y, marker in ((control, "o"), (target, "$\\oplus$")):
                ax.plot(x, y, marker=marker, ms=11 if marker == "o" else 13,
                        mfc=edge, mec=edge, alpha=alpha, zorder=4)
        else:
            y = wires[0]
            box = FancyBboxPatch((x - 0.38, y - 0.32), 0.76, 0.64,
                                 boxstyle="round,pad=0.02", fc="white", ec=edge,
                                 linewidth=1.6, zorder=3)
            ax.add_patch(box)
            ax.text(x, y, _gate_label(gate), ha="center", va="center", fontsize=8,
                    color=color if emphasized else "#888888", alpha=alpha, zorder=4)
        if index in cuts:
            halo = FancyBboxPatch((x - 0.46, min(wires) - 0.42),
                                  0.92, max(wires) - min(wires) + 0.84,
                                  boxstyle="round,pad=0.02", fc="none", ec=CUT,
                                  linewidth=2, linestyle="--", zorder=5)
            ax.add_patch(halo)
            if not legend:  # the legend already names the dashed boxes
                ax.text(x, max(wires) + 0.5, "cut", ha="center", va="bottom",
                        fontsize=8, color=CUT, zorder=6)
    ax.set_xlim(-1.3, depth - 0.2)
    ax.set_ylim(-0.7, n_qubits - 0.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12)
    if legend:
        handles = [Patch(fc="white", ec=INK, label="kept gate")]
        if len(keep) < len(operations):
            handles.append(Patch(fc="white", ec="#AAAAAA", label="dropped gate"))
        if cuts:
            handles.append(Line2D([], [], color=CUT, linestyle="--", label="cut CX"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                  ncol=len(handles), frameon=False, fontsize=9)
    if fig is not None:
        fig.tight_layout()
    return fig, ax
