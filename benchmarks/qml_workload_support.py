"""Deterministic QML-inspired request batches for sharing experiments."""

import math
import random

import numpy as np


def _operations(family, n, depth, seed, data_id=0, theta=0.0):
    """Build one concrete circuit; data_id and theta change values, not topology."""
    from lazyqc import Gate

    rng = random.Random(seed + 1009 * data_id)
    operations = []
    if family == "hardware_efficient_local":
        for layer in range(depth):
            for q in range(n):
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (Gate("RX", (q,), feature), Gate("RY", (q,), theta + 0.13 * (q + layer)))
                )
            start = layer % 2
            operations.extend(Gate("CX", (q, q + 1)) for q in range(start, n - 1, 2))
            operations.extend(Gate("CX", (q, q + 1)) for q in range(1 - start, n - 1, 2))
    elif family == "qcnn_like_local":
        for layer in range(depth):
            stride = 1 << min(layer, 2)
            for block in range(0, n - stride, 2 * stride):
                left, right = block, block + stride
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (
                        Gate("RY", (left,), theta + feature),
                        Gate("RX", (right,), feature),
                        Gate("CX", (left, right)),
                        Gate("RY", (right,), theta - feature / 2),
                        Gate("CX", (right, left)),
                    )
                )
    elif family == "data_reuploading":
        for layer in range(depth):
            for q in range(n):
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (
                        Gate("RX", (q,), feature + 0.07 * data_id),
                        Gate("RY", (q,), theta + 0.11 * (q + layer)),
                    )
                )
            edges = (
                ((q, q + 1) for q in range(n - 1))
                if layer % 2 == 0
                else ((q, q + 1) for q in range(n - 2, -1, -1))
            )
            operations.extend(Gate("CX", edge) for edge in edges)
    elif family == "tree_mera_like":
        # Scales cycle 1, 2, 4, ...: isometries join block heads at distance
        # `stride`, nearest-neighbour disentanglers act on block boundaries.
        scales = max(1, math.ceil(math.log2(n)))
        for layer in range(depth):
            stride = 1 << (layer % scales)
            for boundary in range(2 * stride - 1, n - 1, 2 * stride):
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (
                        Gate("RX", (boundary,), feature),
                        Gate("CX", (boundary, boundary + 1)),
                    )
                )
            for left in range(0, n - stride, 2 * stride):
                right = left + stride
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (
                        Gate("RY", (left,), theta + 0.13 * (left + layer)),
                        Gate("RX", (right,), feature),
                        Gate("CX", (left, right)),
                        Gate("RY", (right,), theta - feature / 2),
                    )
                )
    elif family == "strongly_entangling":
        # PennyLane StronglyEntanglingLayers wiring: Rot on every wire, then a
        # CX ring with range (layer mod (n - 1)) + 1. Predeclared saturation control.
        for layer in range(depth):
            for q in range(n):
                feature = rng.uniform(-math.pi, math.pi)
                operations.extend(
                    (
                        Gate("RZ", (q,), feature),
                        Gate("RY", (q,), theta + 0.11 * (q + layer)),
                        Gate("RZ", (q,), 0.5 * feature),
                    )
                )
            reach = layer % (n - 1) + 1
            operations.extend(Gate("CX", (q, (q + reach) % n)) for q in range(n))
    else:
        raise ValueError(f"Unknown QML circuit family: {family}")
    return tuple(operations)


def _observables(pattern, n, program_index=0):
    """Return ordered Pauli supports for one program in a batch."""
    if pattern == "same_state_local_outputs":
        return tuple({q: "Z"} for q in range(n))
    if pattern == "same_state_shared_cone":
        q = program_index % max(1, n // 4)
        return tuple({q: pauli} for pauli in "XYZ")
    if pattern == "sparse_local_outputs":
        positions = sorted({0, n // 3, (2 * n) // 3, n - 1})
        return tuple({q: "Z"} for q in positions)
    if pattern in ("different_examples", "parameter_shift_pair", "parameter_trajectory"):
        offset = (program_index * max(1, n // 4)) % n
        return tuple({(offset + k) % n: p} for k, p in enumerate("ZXYZ"))
    raise ValueError(f"Unknown QML request pattern: {pattern}")


def layer_ry_occurrence_groups(family, n, depth):
    """Return RY operation indices grouped by the layer-level shared offset.

    The generated circuits add one common trainable offset to every RY angle
    within a layer. RX feature-encoding angles and entanglers are fixed.
    """
    groups = []
    operation_index = 0
    for layer in range(depth):
        group = []
        if family in ("hardware_efficient_local", "data_reuploading"):
            for _q in range(n):
                # Each wire contributes RX then RY in these two generators.
                operation_index += 1
                group.append(operation_index)
                operation_index += 1
            if family == "hardware_efficient_local":
                start = layer % 2
                operation_index += len(range(start, n - 1, 2))
                operation_index += len(range(1 - start, n - 1, 2))
            else:
                operation_index += n - 1
        elif family == "qcnn_like_local":
            stride = 1 << min(layer, 2)
            for _block in range(0, n - stride, 2 * stride):
                group.append(operation_index)
                operation_index += 1  # RY(left)
                operation_index += 1  # RX(right)
                operation_index += 1  # CX(left, right)
                group.append(operation_index)
                operation_index += 1  # RY(right)
                operation_index += 1  # CX(right, left)
        else:
            raise ValueError(f"Unknown QML circuit family: {family}")
        groups.append(tuple(group))
    return tuple(groups)


def make_batch(family, n, depth, pattern, seed):
    """Materialize a deterministic tuple of Requests and its structural metadata."""
    from lazyqc import Gate, Pauli, Program

    count = {
        "same_state_local_outputs": 1,
        "same_state_shared_cone": 1,
        "sparse_local_outputs": 1,
        "different_examples": 4,
        "parameter_shift_pair": 2,
        "parameter_trajectory": 4,
    }[pattern]
    requests = []
    for i in range(count):
        data_id = i if pattern == "different_examples" else 0
        theta = 0.2 + 0.17 * i
        if pattern == "parameter_shift_pair":
            theta = 0.4
        elif pattern == "parameter_trajectory":
            theta = -0.6 + 0.4 * i
        operations = list(_operations(family, n, depth, seed, data_id=data_id, theta=theta))
        if pattern == "parameter_shift_pair":
            occurrence = next(index for index, gate in enumerate(operations) if gate.name == "RY")
            gate = operations[occurrence]
            shift = (-1 if i == 0 else 1) * math.pi / 2
            operations[occurrence] = Gate(gate.name, gate.qubits, gate.angle + shift)
        program = Program(n, operations)
        requests.extend(
            program.expectation(Pauli(tuple(sorted(observable.items()))))
            for observable in _observables(pattern, n, i)
        )
    return tuple(requests)


def trajectory_batch(family, n, depth, seed, theta, *, shifted_pair=False):
    """Build one shared-bound state or a two-program single-occurrence shift batch."""
    from lazyqc import Gate, Pauli, Program

    programs = []
    for index in range(2 if shifted_pair else 1):
        operations = list(_operations(family, n, depth, seed, theta=theta))
        if shifted_pair:
            occurrence = next(i for i, gate in enumerate(operations) if gate.name == "RY")
            gate = operations[occurrence]
            sign = -1 if index == 0 else 1
            operations[occurrence] = Gate(
                gate.name, gate.qubits, gate.angle + sign * math.pi / 2
            )
        programs.append(Program(n, operations))

    request_programs = programs if shifted_pair else programs * 1
    observables_per_program = min(6, n)
    requests = []
    for program in request_programs:
        for q in range(observables_per_program):
            requests.append(program.expectation(Pauli(((q, "Z"),))))
    return tuple(requests)


def workload_cases(manifest, smoke=False):
    """Yield frozen cases; smoke mode selects small cases without changing generators."""
    patterns = tuple(manifest["request_patterns"])
    families = manifest["families"]
    seed = manifest["seed"]
    cases = []
    for family, settings in families.items():
        for n in settings["qubits"]:
            for depth in settings["depths"]:
                for offset in manifest["circuit_seed_offsets"]:
                    for pattern in patterns:
                        case_id = f"{family}_{n}q_d{depth}_{pattern}_s{offset}"
                        cases.append(
                            {
                                "case_id": case_id,
                                "family": family,
                                "qubits": n,
                                "depth": depth,
                                "pattern": pattern,
                                "seed": seed + offset,
                                "seed_offset": offset,
                            }
                        )
    if not smoke:
        return cases
    return [
        next(case for case in cases if case["family"] == family and case["seed_offset"] == 0)
        for family in families
    ]


def numpy_oracle_expectations(requests):
    """Independent vectorized NumPy statevector reference; memory scales as 2**n."""
    states = {}
    for request in requests:
        program = request.program
        if program not in states:
            n = program.n_qubits
            state = np.zeros(1 << n, dtype=np.complex128)
            state[0] = 1.0
            for operation in program.resolved_operations():
                name, wires, angle = operation.name, operation.qubits, operation.angle
                if name == "CX":
                    control, target = wires
                    indices = np.arange(1 << n)
                    mask = ((indices >> control) & 1).astype(bool) & ~(
                        ((indices >> target) & 1).astype(bool)
                    )
                    left = indices[mask]
                    right = left ^ (1 << target)
                    state[left], state[right] = state[right].copy(), state[left].copy()
                    continue

                q = wires[0]
                step = 1 << q
                blocks = state.reshape((-1, 2 * step))
                zero = blocks[:, :step].copy()
                one = blocks[:, step:].copy()
                cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
                if name == "RX":
                    blocks[:, :step] = cosine * zero - 1j * sine * one
                    blocks[:, step:] = -1j * sine * zero + cosine * one
                elif name == "RY":
                    blocks[:, :step] = cosine * zero - sine * one
                    blocks[:, step:] = sine * zero + cosine * one
                elif name == "RZ":
                    blocks[:, :step] = np.exp(-0.5j * angle) * zero
                    blocks[:, step:] = np.exp(0.5j * angle) * one
                else:
                    raise ValueError(f"Unsupported gate in QML oracle: {name}")
            states[program] = state

    values = []
    for request in requests:
        state = states[request.program]
        transformed = state.copy()
        n = request.program.n_qubits
        for q, pauli in request.observable.factors:
            step = 1 << q
            blocks = transformed.reshape((-1, 2 * step))
            zero = blocks[:, :step].copy()
            one = blocks[:, step:].copy()
            if pauli == "X":
                blocks[:, :step] = one
                blocks[:, step:] = zero
            elif pauli == "Y":
                blocks[:, :step] = -1j * one
                blocks[:, step:] = 1j * zero
            elif pauli == "Z":
                blocks[:, step:] = -one
            elif pauli != "I":
                raise ValueError(f"Unsupported Pauli in QML oracle: {pauli}")
        value = np.vdot(state, transformed)
        if abs(value.imag) > 1e-11:
            raise ArithmeticError(f"Hermitian expectation has imaginary part {value.imag}")
        values.append(float(value.real))
    return tuple(values)
