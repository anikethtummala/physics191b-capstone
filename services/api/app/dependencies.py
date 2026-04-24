from __future__ import annotations

from importlib.util import find_spec


def has_stim() -> bool:
    return find_spec("stim") is not None


def has_pymatching() -> bool:
    return find_spec("pymatching") is not None


def missing_quantum_dependencies() -> list[str]:
    missing: list[str] = []
    if not has_stim():
        missing.append("stim")
    if not has_pymatching():
        missing.append("pymatching")
    return missing
