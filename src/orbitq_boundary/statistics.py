"""Pass-rate estimation, uncertainty, and candidate classification."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist, median
from typing import Iterable, Literal

from .models import (
    AgentRunRecord,
    CandidateTaskSpec,
    CandidateVerdict,
    ClassificationThresholds,
    ExpertValidationRecord,
    OutcomeSummary,
    content_digest,
)


def wilson_interval(
    n_passed: int,
    n_runs: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute a two-sided Wilson score interval for a binomial proportion."""

    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if not 0 <= n_passed <= n_runs:
        raise ValueError("n_passed must be between zero and n_runs")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    point = n_passed / n_runs
    denominator = 1.0 + z * z / n_runs
    center = (point + z * z / (2.0 * n_runs)) / denominator
    radius = (
        z
        * sqrt(
            point * (1.0 - point) / n_runs
            + z * z / (4.0 * n_runs * n_runs)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def validate_thresholds(thresholds: ClassificationThresholds) -> None:
    """Reject ambiguous or numerically invalid threshold configurations."""

    if not 0.0 <= thresholds.hard_max < thresholds.easy_min <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= hard_max < easy_min <= 1"
        )
    if (
        thresholds.screening_checkpoint,
        thresholds.formal_checkpoint,
    ) != (5, 16):
        raise ValueError(
            "this MVP permits only the predeclared checkpoints n=5 and n=16"
        )
    if not 0.0 < thresholds.confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def _require_unique_non_empty(
    values: list[str], label: str
) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{label} must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


def classify_candidate(
    candidate: CandidateTaskSpec,
    expert_record: ExpertValidationRecord,
    runs: Iterable[AgentRunRecord],
    thresholds: ClassificationThresholds,
    outcome_kind: Literal[
        "correctness", "resource_gate", "eligible"
    ] = "eligible",
) -> CandidateVerdict:
    """Classify a validated candidate using same-configuration isolated runs."""

    validate_thresholds(thresholds)
    if expert_record.candidate_id != candidate.candidate_id:
        raise ValueError("expert validation belongs to another candidate")
    if not expert_record.passed:
        raise ValueError("expert validation did not pass every required gate")

    ordered_runs = sorted(tuple(runs), key=lambda record: record.run_id)
    if not ordered_runs:
        raise ValueError("at least one Agent run is required")
    if any(
        record.candidate_id != candidate.candidate_id
        for record in ordered_runs
    ):
        raise ValueError("Agent run belongs to another candidate")

    fingerprints = {
        record.agent_config_fingerprint for record in ordered_runs
    }
    if len(fingerprints) != 1:
        raise ValueError("mixed AgentConfig fingerprints cannot be aggregated")
    fingerprint = next(iter(fingerprints))
    if not fingerprint.strip():
        raise ValueError("agent_config_fingerprint must be non-empty")

    _require_unique_non_empty(
        [record.run_id for record in ordered_runs], "run_id"
    )
    _require_unique_non_empty(
        [record.isolation_id for record in ordered_runs], "isolation_id"
    )

    if outcome_kind == "correctness":
        outcomes = [record.correctness_passed for record in ordered_runs]
    elif outcome_kind == "resource_gate":
        outcomes = [record.resource_gate_passed for record in ordered_runs]
    elif outcome_kind == "eligible":
        outcomes = [record.passed for record in ordered_runs]
    else:
        raise ValueError(f"unsupported outcome_kind: {outcome_kind}")
    if any(outcome is None for outcome in outcomes):
        raise ValueError(f"missing {outcome_kind} outcome evidence")

    n_runs = len(ordered_runs)
    n_passed = sum(bool(outcome) for outcome in outcomes)
    pass_rate = n_passed / n_runs
    low, high = wilson_interval(
        n_passed, n_runs, thresholds.confidence_level
    )

    checkpoints = {
        thresholds.screening_checkpoint,
        thresholds.formal_checkpoint,
    }
    if n_runs not in checkpoints:
        raise ValueError(
            "classification is allowed only at a predeclared checkpoint: "
            f"{sorted(checkpoints)}"
        )

    if n_runs == thresholds.screening_checkpoint:
        checkpoint_kind = "screening"
        endpoint_classification = "unresolved"
        sampling_decision = "continue_to_formal_checkpoint"
        if pass_rate >= thresholds.easy_min:
            screening_classification = "easy_candidate"
        elif pass_rate <= thresholds.hard_max:
            screening_classification = "hard_candidate"
        else:
            screening_classification = "transition_candidate"
        reason = (
            "screening label only; formal endpoint claims require the "
            f"predeclared n={thresholds.formal_checkpoint} checkpoint"
        )
    else:
        checkpoint_kind = "formal"
        screening_classification = None
        if low >= thresholds.easy_min:
            endpoint_classification = "easy"
            sampling_decision = "stop"
            reason = "Wilson lower bound reached the formal easy threshold"
        elif high <= thresholds.hard_max:
            endpoint_classification = "hard"
            sampling_decision = "stop"
            reason = "Wilson upper bound reached the formal hard threshold"
        else:
            endpoint_classification = "unresolved"
            sampling_decision = "await_approved_sampling_policy"
            reason = (
                "formal Wilson endpoint not established; additional looks "
                "require an approved maximum budget and adjusted error "
                "control or an anytime-valid confidence sequence"
            )

    verdict_payload = {
        "candidate_id": candidate.candidate_id,
        "agent_config_fingerprint": fingerprint,
        "run_ids": [record.run_id for record in ordered_runs],
        "thresholds": thresholds.to_dict(),
        "outcome_kind": outcome_kind,
    }
    return CandidateVerdict(
        verdict_id=f"verdict-{content_digest(verdict_payload)[:16]}",
        candidate_id=candidate.candidate_id,
        agent_config_fingerprint=fingerprint,
        n_runs=n_runs,
        n_passed=n_passed,
        pass_rate=pass_rate,
        confidence_level=thresholds.confidence_level,
        interval_low=low,
        interval_high=high,
        screening_classification=screening_classification,
        endpoint_classification=endpoint_classification,
        checkpoint_kind=checkpoint_kind,
        sampling_decision=sampling_decision,
        reason=reason,
        outcome_kind=outcome_kind,
    )


def summarize_run_outcomes(
    runs: Iterable[AgentRunRecord],
) -> OutcomeSummary:
    """Summarize separate binary outcomes and the runtime distribution."""

    ordered = sorted(tuple(runs), key=lambda record: record.run_id)
    if not ordered:
        raise ValueError("at least one Agent run is required")
    candidate_ids = {record.candidate_id for record in ordered}
    fingerprints = {record.agent_config_fingerprint for record in ordered}
    if len(candidate_ids) != 1:
        raise ValueError("mixed candidates cannot be summarized")
    if len(fingerprints) != 1:
        raise ValueError("mixed AgentConfig fingerprints cannot be summarized")

    def optional_rate(values: list[bool | None]) -> float | None:
        if any(value is None for value in values):
            return None
        return sum(bool(value) for value in values) / len(values)

    runtimes = sorted(
        record.runtime_sec for record in ordered if record.runtime_sec >= 0.0
    )
    return OutcomeSummary(
        candidate_id=next(iter(candidate_ids)),
        agent_config_fingerprint=next(iter(fingerprints)),
        n_runs=len(ordered),
        correctness_pass_rate=(
            sum(record.correctness_passed for record in ordered) / len(ordered)
        ),
        resource_gate_pass_rate=optional_rate(
            [record.resource_gate_passed for record in ordered]
        ),
        eligible_pass_rate=optional_rate(
            [record.passed for record in ordered]
        ),
        runtime_count=len(runtimes),
        runtime_min_sec=runtimes[0] if runtimes else None,
        runtime_median_sec=median(runtimes) if runtimes else None,
        runtime_max_sec=runtimes[-1] if runtimes else None,
    )
