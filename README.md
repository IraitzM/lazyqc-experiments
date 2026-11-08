# LazyQC experiments

Benchmarks, raw results and paper sources for *LazyQC: Avoiding Simulation Work in Batched Quantum Expectation Workloads*. The library itself lives in its own repository, [LazyQC](https://github.com/IraitzM/LazyQC). This one only depends on it.

## Layout

- `benchmarks/`: runners, frozen manifests, reference data (`benchmarks/reference/`) and the script lockfiles of the two GPU runners
- `scripts/`: figure rendering, the archival bundle and the remote GPU launcher
- `cloud/`: the SkyPilot task for the GPU confirmation runs

## Setup

The experiments use the released package, `lazyqc` 0.5.0 from PyPI, and nothing else from the library repository:

```bash
git clone <this repository> lazyqc-experiments
cd lazyqc-experiments
uv sync --locked
```

`pyproject.toml` pins `lazyqc[zx,cutting,tensor,mqt]==0.5.0` and `uv.lock` fixes the rest. The two GPU runners declare the same release in their inline script metadata and keep their own lockfiles, so run them with `uv run --locked --script`.

The NumPy reference, the frozen circuit fixtures and the drawing helpers the runners need are in `benchmarks/reference/`. Source fingerprints in new result files hash the installed modules of the package and name them `lazyqc/<module>.py`.

## Running

```bash
task --list            # direct-summary, remote-gpu
task laptop-gpu        # rerun the laptop GPU benchmarks (locked script environments)
task local-studies     # rerun the CPU-only studies (QML matrix, held-out families, probes, workers, sensitivity)
task -d slides --list  # talks
```

The GPU confirmation runs install the same release on the cluster, so they launch from this folder alone.
