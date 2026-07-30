"""Framework-native expert-reference interface and local environment gate."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable, Mapping, Protocol

from ...models import JsonRecord
from .loaders import load_official_solution


class ExpertSolution(Protocol):
    K: object

    @staticmethod
    def build_circuit(config: Mapping[str, object]) -> object: ...


@dataclass(frozen=True)
class ExpertEnvironmentAssessment(JsonRecord):
    ready: bool
    blockers: tuple[str, ...]
    official_solution_path: str


def generate_reference_values(
    solution: ExpertSolution,
    config: Mapping[str, object],
    supports: Mapping[str, tuple[tuple[int, ...], ...]],
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, tuple[float, ...]]:
    """Evaluate Z strings through the official framework circuit interface."""

    circuit = solution.build_circuit(config)
    backend = solution.K
    values: dict[str, tuple[float, ...]] = {}
    for category, category_supports in supports.items():
        category_values = []
        total = len(category_supports)
        for index, support in enumerate(category_supports, start=1):
            # Avoid materializing the full state vector.  This keeps the
            # framework-native expectation contraction viable beyond G5.
            raw = circuit.expectation_ps(z=list(support), reuse=False)
            converted = backend.numpy(raw)
            category_values.append(float(complex(converted).real))
            if progress is not None:
                progress(category, index, total)
        values[category] = tuple(category_values)
    return values


def assess_expert_environment(
    repository_root: str | Path,
) -> ExpertEnvironmentAssessment:
    root = Path(repository_root)
    solution = root / "tasks/challenge-08/solution/solution_8.py"
    blockers: list[str] = []
    if not solution.exists():
        blockers.append("official-challenge08-solution-missing")
    for module_name, blocker in (
        ("tensorcircuit", "tensorcircuit-ng-not-importable"),
        ("jax", "jax-not-importable"),
    ):
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            blockers.append(blocker)
    return ExpertEnvironmentAssessment(
        ready=not blockers,
        blockers=tuple(blockers),
        official_solution_path=str(solution),
    )
