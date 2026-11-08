"""Direct dense baselines and Jacobian batches for the planner-versus-direct study.

Every converter starts from the same LazyQC `Program` and `Pauli` objects, so the
planner and each direct simulator evaluate identical circuits. Imports of the
optional simulators happen inside the functions that need them.
"""

import importlib.metadata
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter_ns

THREAD_KEYS = (
    "QULACS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PACKAGES = (
    "lazyqc",
    "qulacs",
    "numpy",
    "scipy",
    "qiskit",
    "qiskit-aer-gpu",
    "qiskit-aer",
    "pennylane",
    "pennylane-lightning",
    "pennylane-lightning-gpu",
    "custatevec-cu12",
    "quimb",
)


def pin_threads(count):
    """Fix native thread pools before any simulator initializes."""
    for key in THREAD_KEYS:
        os.environ[key] = str(count)


def environment():
    """Record host, GPU, thread and package versions for provenance."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    gpu = None
    try:
        query = "name,memory.total,driver_version,clocks.max.sm,power.limit"
        gpu = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    cpu = platform.processor()
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo").read().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "gpu": gpu,
        "python": platform.python_version(),
        "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
        "packages": versions,
    }


def atomic_write(path, text):
    """Replace `path` through a temporary file, so a reader never sees a partial file.

    Results are copied off cloud machines while runs are still writing them.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(handle, "w") as stream:
        stream.write(text)
    os.replace(temporary, path)


def timed(function):
    """Return `(result, elapsed_ns)` for one call."""
    start = perf_counter_ns()
    result = function()
    return result, perf_counter_ns() - start


def summarize(samples):
    """Median of nanosecond samples, in milliseconds."""
    return median(samples) / 1e6


# -- Jacobian batches -------------------------------------------------------------


def trainable_indices(program):
    """Occurrence indices of trainable rotations: every RY in the generated families."""
    return tuple(i for i, g in enumerate(program.operations) if g.name == "RY")


def shifted_programs(program, indices):
    """Base program followed by +pi/2 and -pi/2 shifts of each trainable occurrence."""
    from lazyqc import Gate, Program

    base = list(program.operations)
    programs = [program]
    for i in indices:
        gate = base[i]
        for sign in (1, -1):
            operations = list(base)
            operations[i] = Gate(gate.name, gate.qubits, gate.angle + sign * math.pi / 2)
            programs.append(Program(program.n_qubits, tuple(operations)))
    return programs


def with_angles(program, indices, angles):
    """Return `program` with the trainable occurrences set to `angles`."""
    from lazyqc import Gate, Program

    operations = list(program.operations)
    for i, angle in zip(indices, angles, strict=True):
        operations[i] = Gate(operations[i].name, operations[i].qubits, float(angle))
    return Program(program.n_qubits, tuple(operations))


def jacobian_requests(programs, observables):
    """One request per (program, observable), program-major."""
    return [p.expectation(o) for p in programs for o in observables]


def assemble_jacobian(values, parameters, outputs):
    """Split program-major values into forward values and the shift-rule Jacobian."""
    import numpy as np

    table = np.asarray(values, dtype=float).reshape(1 + 2 * parameters, outputs)
    forward = table[0]
    jacobian = 0.5 * (table[1::2] - table[2::2]).T
    return forward, jacobian


# -- Direct dense simulators -------------------------------------------------------


def qulacs_direct(program, observables):
    """Prebuilt Qulacs circuit and observables; the callable evolves one fresh state."""
    from qulacs import Observable, QuantumCircuit, QuantumState

    n = program.n_qubits
    circuit = QuantumCircuit(n)
    for gate in program.resolved_operations():
        if gate.name == "CX":
            circuit.add_CNOT_gate(*gate.qubits)
        elif gate.name in ("RX", "RY", "RZ"):
            # Qulacs rotates by exp(+i theta P / 2); LazyQC and Qiskit use the opposite sign.
            getattr(circuit, f"add_{gate.name}_gate")(gate.qubits[0], -gate.angle)
        else:
            getattr(circuit, f"add_{gate.name}_gate")(gate.qubits[0])
    operators = []
    for pauli in observables:
        operator = Observable(n)
        operator.add_operator(1.0, " ".join(f"{p} {q}" for q, p in pauli.factors))
        operators.append(operator)

    def run():
        state = QuantumState(n)
        state.set_zero_state()
        circuit.update_quantum_state(state)
        values = [float(o.get_expectation_value(state)) for o in operators]
        del state
        return values

    return run


def aer_circuit(program, observables):
    """Qiskit circuit that saves every requested Pauli expectation."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    n = program.n_qubits
    circuit = QuantumCircuit(n)
    for gate in program.resolved_operations():
        if gate.name == "CX":
            circuit.cx(*gate.qubits)
        elif gate.name in ("RX", "RY", "RZ"):
            getattr(circuit, gate.name.lower())(gate.angle, gate.qubits[0])
        else:
            getattr(circuit, gate.name.lower())(gate.qubits[0])
    for k, pauli in enumerate(observables):
        operator = SparsePauliOp.from_sparse_list(
            [("".join(p for _, p in pauli.factors), [q for q, _ in pauli.factors], 1)], n
        )
        circuit.save_expectation_value(operator, range(n), label=f"o{k}")
    return circuit


def aer_simulator(device="GPU"):
    """Double-precision Aer statevector simulator with default gate fusion."""
    from qiskit_aer import AerSimulator

    return AerSimulator(method="statevector", device=device, precision="double")


def aer_direct(program, observables, simulator):
    """Prebuilt Aer circuit; the callable runs it and reads every saved expectation."""
    circuit = aer_circuit(program, observables)
    labels = [f"o{k}" for k in range(len(observables))]

    def run():
        data = simulator.run(circuit).result().data(0)
        return [float(complex(data[label]).real) for label in labels]

    return run


def aer_batch(programs, observables, simulator):
    """Prebuilt Aer circuits for a list of programs, run as one job."""
    circuits = [aer_circuit(p, observables) for p in programs]
    labels = [f"o{k}" for k in range(len(observables))]

    def run():
        result = simulator.run(circuits).result()
        return [
            float(complex(result.data(i)[label]).real)
            for i in range(len(circuits))
            for label in labels
        ]

    return run


def pennylane_tape(program, observables, trainable=None):
    """PennyLane tape for the program; `trainable` lists trainable operation indices."""
    import pennylane as qml

    operations, parameter_index, trainable_parameters = [], 0, []
    trainable = set(trainable or ())
    for i, gate in enumerate(program.resolved_operations()):
        if gate.name == "CX":
            operations.append(qml.CNOT(wires=list(gate.qubits)))
            continue
        if gate.name in ("RX", "RY", "RZ"):
            operations.append(getattr(qml, gate.name)(gate.angle, wires=gate.qubits[0]))
            if i in trainable:
                trainable_parameters.append(parameter_index)
            parameter_index += 1
        else:
            operations.append(
                {"H": qml.Hadamard, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[gate.name](
                    wires=gate.qubits[0]
                )
            )
    measurements = []
    for pauli in observables:
        factors = [
            {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[p](wires=q)
            for q, p in pauli.factors
        ]
        measurements.append(qml.expval(factors[0] if len(factors) == 1 else qml.prod(*factors)))
    return qml.tape.QuantumScript(operations, measurements, trainable_params=trainable_parameters)


def lightning_device(name, wires):
    """PennyLane Lightning device with an adjoint execution configuration."""
    import pennylane as qml
    from pennylane.devices import ExecutionConfig

    device = qml.device(name, wires=wires)
    config = device.setup_execution_config(ExecutionConfig(gradient_method="adjoint"))
    return device, config


def quimb_direct(program, observables, optimize="auto-hq"):
    """Direct tensor-network evaluation with quimb's own light-cone simplification.

    quimb caches simplified networks on a circuit object, so each call builds a
    fresh circuit from the prebuilt gate list; nothing carries over between runs.
    """
    import quimb.tensor as qtn

    from lazyqc.backends.tensor import contract_groups, pauli_groups

    gates = [
        (g.name, g.angle, *g.qubits) if g.name in ("RX", "RY", "RZ") else (g.name, *g.qubits)
        for g in program.resolved_operations()
    ]
    groups = pauli_groups(
        [(None, pauli) for pauli in observables], {q: q for q in range(program.n_qubits)}
    )

    def run():
        circuit = qtn.Circuit(program.n_qubits)
        for gate in gates:
            circuit.apply_gate(*gate)
        values = contract_groups(circuit, groups, len(observables), optimize)
        return [complex(v).real for v in values]

    return run


def lightning_direct(program, observables, device):
    """Prebuilt tape; the callable executes it on the given Lightning device."""
    tape = pennylane_tape(program, observables)

    def run():
        return [float(v) for v in device.execute(tape)]

    return run
