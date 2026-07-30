"""Role-separated loaders for public candidates and official references."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from uuid import uuid4


def _load(path: str | Path, role: str) -> ModuleType:
    source = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"orbitq_boundary_{role}_{uuid4().hex}", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {role} module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_candidate_solution(path: str | Path) -> ModuleType:
    """Load only the public ``run_solution(config)`` contract."""

    module = _load(path, "candidate")
    if not callable(getattr(module, "run_solution", None)):
        raise RuntimeError("candidate solution lacks run_solution(config)")
    return module


def load_official_solution(path: str | Path) -> ModuleType:
    """Load the richer trusted reference contract used for calibration."""

    module = _load(path, "official")
    missing = [
        name
        for name in ("run_solution", "build_circuit", "K")
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "official solution lacks " + "/".join(missing) + " interface"
        )
    return module
