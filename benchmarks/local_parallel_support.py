"""Independent NumPy oracle and Linux accounting for the local worker study."""

import os
import threading
from pathlib import Path
from time import perf_counter_ns


def numpy_state(width, operations):
    """State-vector gate application; independent of Qulacs and planner lowering."""
    import numpy as np

    if not 1 <= width <= 16:
        raise ValueError("Reference width must be 1..16")
    state = np.zeros(1 << width, dtype=complex)
    state[0] = 1
    matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
        "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    }
    indices = np.arange(1 << width)
    permutations = {}
    for gate in operations:
        if gate.name == "CX":
            c, t = gate.qubits
            if (c, t) not in permutations:
                permutations[c, t] = indices ^ (((indices >> c) & 1) << t)
            state = state[permutations[c, t]]
        else:
            if gate.name.startswith("R"):
                angle = gate.angle
                matrix = (
                    np.cos(angle / 2) * np.eye(2) - 1j * np.sin(angle / 2) * matrices[gate.name[1]]
                )
            else:
                matrix = matrices[gate.name]
            view = state.reshape(-1, 2, 1 << gate.qubits[0])
            a, b = view[:, 0, :].copy(), view[:, 1, :].copy()
            view[:, 0, :] = matrix[0, 0] * a + matrix[0, 1] * b
            view[:, 1, :] = matrix[1, 0] * a + matrix[1, 1] * b
    return state


def numpy_expectation(state, factors):
    import numpy as np

    indices = np.arange(len(state))
    target = indices.copy()
    phase = np.ones(len(state), dtype=complex)
    for q, axis in factors:
        bit = (indices >> q) & 1
        if axis in "XY":
            target ^= 1 << q
        if axis == "Y":
            phase *= 1j * (1 - 2 * bit)
        elif axis == "Z":
            phase *= 1 - 2 * bit
    value = np.sum(np.conj(state[target]) * phase * state)
    if abs(value.imag) > 1e-10:
        raise AssertionError("Nonreal reference expectation")
    return float(value.real)


def process_snapshot(pids):
    """Instantaneous RSS and user+system CPU; excludes child accounting in stat."""
    ticks = os.sysconf("SC_CLK_TCK")
    pages = os.sysconf("SC_PAGE_SIZE")
    rows = {}
    for pid in pids:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            rows[str(pid)] = {
                "rss_bytes": int(stat[21]) * pages,
                "cpu_ns": round((int(stat[11]) + int(stat[12])) * 1e9 / ticks),
            }
        except FileNotFoundError:
            continue
    return rows


class ResourceSampler:
    """Sample a specified parent/pool set in a separate, untimed diagnostic run."""

    def __init__(self, pids, interval_s=0.002):
        self.pids = tuple(pids)
        self.interval_s = interval_s
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        self.samples.append(
            {"timestamp_ns": perf_counter_ns(), "processes": process_snapshot(self.pids)}
        )

    def _run(self):
        while not self.stop.wait(self.interval_s):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()

    def report(self):
        totals = [sum(p["rss_bytes"] for p in row["processes"].values()) for row in self.samples]
        first, last = self.samples[0]["processes"], self.samples[-1]["processes"]
        return {
            "interval_target_ns": round(self.interval_s * 1e9),
            "sample_count": len(self.samples),
            "pids": self.pids,
            "baseline_sum_rss_bytes": totals[0],
            "sampled_peak_sum_rss_bytes": max(totals),
            "cpu_ns_by_pid": {
                pid: last[pid]["cpu_ns"] - first[pid]["cpu_ns"] for pid in first if pid in last
            },
            "samples": self.samples,
            "scope": (
                "parent plus this pool only; summed RSS double-counts shared pages; "
                "sampled, not guaranteed peak; CPU tick resolution; profiling overhead included"
            ),
        }
