"""Configuration-independent records and statistics for matched task pairs."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt
from typing import Mapping

from .models import JsonRecord, JsonValue, content_digest
from .statistics import wilson_interval


FAILURE_TAXONOMY = frozenset(
    {
        "semantic_parameterization_failure",
        "topology_or_indexing_failure",
        "sampling_semantics_failure",
        "representation_selection_failure",
        "framework_discoverability_failure",
        "generic_implementation_failure",
        "scalability_failure",
        "resource_failure",
        "static_policy_failure",
        "infrastructure_invalid",
        "passed",
    }
)


@dataclass(frozen=True)
class MatchedPairSpec(JsonRecord):
    capability_id: str
    pair_id: str
    base_candidate_id: str
    probe_candidate_id: str
    sole_semantic_delta: str
    controlled_fields: tuple[str, ...]
    deterministically_derived_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "capability_id",
            "pair_id",
            "base_candidate_id",
            "probe_candidate_id",
            "sole_semantic_delta",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self, "controlled_fields", tuple(self.controlled_fields)
        )
        object.__setattr__(
            self,
            "deterministically_derived_fields",
            tuple(self.deterministically_derived_fields),
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class MatchedPairEvidence(JsonRecord):
    pair_id: str
    witness_digest: str
    evaluator_digest: str
    base_config_digest: str
    probe_config_digest: str
    expert_runtime_ratio: float | None
    expert_peak_memory_ratio: float | None
    verifier_parity: bool
    agent_fingerprint: str | None
    base_outcomes: tuple[dict[str, JsonValue], ...]
    probe_outcomes: tuple[dict[str, JsonValue], ...]
    paired_effect: dict[str, JsonValue] | None
    failure_taxonomy: tuple[str, ...]
    readiness_status: str

    def __post_init__(self) -> None:
        unknown = set(self.failure_taxonomy) - FAILURE_TAXONOMY
        if unknown:
            raise ValueError(
                f"unknown failure taxonomy values: {sorted(unknown)}"
            )
        object.__setattr__(self, "base_outcomes", tuple(self.base_outcomes))
        object.__setattr__(self, "probe_outcomes", tuple(self.probe_outcomes))
        object.__setattr__(
            self, "failure_taxonomy", tuple(self.failure_taxonomy)
        )


def _validate_counts(passed: int, total: int, prefix: str) -> None:
    if total <= 0:
        raise ValueError(f"{prefix}_total must be positive")
    if not 0 <= passed <= total:
        raise ValueError(
            f"{prefix}_passed must be between zero and {prefix}_total"
        )


def newcombe_difference_interval(
    base_passed: int,
    base_total: int,
    probe_passed: int,
    probe_total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Newcombe score interval for two independent binomial proportions."""

    _validate_counts(base_passed, base_total, "base")
    _validate_counts(probe_passed, probe_total, "probe")
    base_rate = base_passed / base_total
    probe_rate = probe_passed / probe_total
    base_low, base_high = wilson_interval(
        base_passed, base_total, confidence_level
    )
    probe_low, probe_high = wilson_interval(
        probe_passed, probe_total, confidence_level
    )
    difference = base_rate - probe_rate
    lower = difference - sqrt(
        (base_rate - base_low) ** 2
        + (probe_high - probe_rate) ** 2
    )
    upper = difference + sqrt(
        (base_high - base_rate) ** 2
        + (probe_rate - probe_low) ** 2
    )
    return max(-1.0, lower), min(1.0, upper)


def fisher_exact_two_sided(
    base_passed: int,
    base_total: int,
    probe_passed: int,
    probe_total: int,
) -> float:
    """Return the probability-ordering two-sided Fisher exact p-value."""

    _validate_counts(base_passed, base_total, "base")
    _validate_counts(probe_passed, probe_total, "probe")
    total_passed = base_passed + probe_passed
    grand_total = base_total + probe_total
    minimum = max(0, base_total - (grand_total - total_passed))
    maximum = min(base_total, total_passed)
    denominator = comb(grand_total, base_total)

    def probability(base_successes: int) -> float:
        return (
            comb(total_passed, base_successes)
            * comb(
                grand_total - total_passed,
                base_total - base_successes,
            )
            / denominator
        )

    observed = probability(base_passed)
    tolerance = 1e-12
    return min(
        1.0,
        sum(
            probability(value)
            for value in range(minimum, maximum + 1)
            if probability(value) <= observed + tolerance
        ),
    )


def summarize_matched_effect(
    base_passed: int,
    base_total: int,
    probe_passed: int,
    probe_total: int,
) -> dict[str, JsonValue]:
    """Summarize the frozen independent-attempt matched comparison."""

    interval = newcombe_difference_interval(
        base_passed,
        base_total,
        probe_passed,
        probe_total,
    )
    base_rate = base_passed / base_total
    probe_rate = probe_passed / probe_total
    return {
        "attempt_design": "independent_attempts",
        "base_passed": base_passed,
        "base_total": base_total,
        "base_pass_rate": base_rate,
        "probe_passed": probe_passed,
        "probe_total": probe_total,
        "probe_pass_rate": probe_rate,
        "effect_definition": "base_pass_rate - probe_pass_rate",
        "effect_estimate": base_rate - probe_rate,
        "effect_interval_method": (
            "Newcombe 95% CI based on Wilson score intervals"
        ),
        "effect_interval": list(interval),
        "hypothesis_test": "two-sided Fisher exact test",
        "p_value": fisher_exact_two_sided(
            base_passed,
            base_total,
            probe_passed,
            probe_total,
        ),
    }
