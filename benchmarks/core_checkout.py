"""Locate the installed ``lazyqc`` package and the reference data of these experiments.

Everything here goes through the package as ``pip install lazyqc`` delivers it. The
fingerprints that go into result files hash the installed modules of the package, and
the NumPy reference and the frozen fixtures live in ``benchmarks/reference/``, so no
checkout of the library is needed next to this repository.
"""

import importlib.metadata
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "benchmarks" / "reference"
FIXTURES = REFERENCE / "fixtures"


def package_dir():
    """Directory of the installed ``lazyqc`` package."""
    spec = importlib.util.find_spec("lazyqc")
    if spec is None or spec.origin is None:
        raise RuntimeError("lazyqc is not installed in this environment")
    return Path(spec.origin).parent


def package_version():
    return importlib.metadata.version("lazyqc")


def package_files():
    """Python modules of the installed package, sorted."""
    return sorted(package_dir().rglob("*.py"))


def label(path):
    """Name a file as ``lazyqc/<module>.py`` for the package or relative to this repository."""
    path = Path(path).resolve()
    package = package_dir()
    if path.is_relative_to(package):
        return f"lazyqc/{path.relative_to(package)}"
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    raise ValueError(f"{path} is neither in {ROOT} nor in the lazyqc package")


def resolve(name):
    """Inverse of ``label``: the file a fingerprint entry names."""
    if name.startswith("lazyqc/"):
        return package_dir() / name.removeprefix("lazyqc/")
    return ROOT / name
