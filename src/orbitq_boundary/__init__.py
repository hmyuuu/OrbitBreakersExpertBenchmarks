"""Generic, task-independent ORBIT-Q capability-boundary primitives."""

from .comparison import assert_controlled_mutation, require_expert_validation
from .manifest import build_manifest, render_manifest, write_manifest
from .models import (
    AgentConfig,
    AgentRunRecord,
    BoundaryEvidence,
    BoundarySearchState,
    CandidateTaskSpec,
    CandidateVerdict,
    ClassificationThresholds,
    DifficultyParameter,
    EvidenceValue,
    ExpertValidationRecord,
    FormalScaleObservation,
    NonMonotonicObservation,
    OutcomeSummary,
    TaskFamilySpec,
)
from .search import (
    SearchDecision,
    record_non_monotonic_observation,
    select_next_candidate,
)
from .specs import (
    family_spec_from_mapping,
    generate_candidate,
    load_family_spec,
    validate_parameter_vector,
)
from .statistics import classify_candidate, summarize_run_outcomes, wilson_interval


def __getattr__(name):
    if name == "run_synthetic_demo":
        from .synthetic import run_synthetic_demo

        return run_synthetic_demo
    raise AttributeError(name)

__all__ = [
    "AgentConfig",
    "AgentRunRecord",
    "BoundaryEvidence",
    "BoundarySearchState",
    "CandidateTaskSpec",
    "CandidateVerdict",
    "ClassificationThresholds",
    "DifficultyParameter",
    "EvidenceValue",
    "ExpertValidationRecord",
    "FormalScaleObservation",
    "NonMonotonicObservation",
    "OutcomeSummary",
    "TaskFamilySpec",
    "SearchDecision",
    "assert_controlled_mutation",
    "build_manifest",
    "classify_candidate",
    "family_spec_from_mapping",
    "generate_candidate",
    "load_family_spec",
    "record_non_monotonic_observation",
    "render_manifest",
    "require_expert_validation",
    "run_synthetic_demo",
    "select_next_candidate",
    "summarize_run_outcomes",
    "validate_parameter_vector",
    "wilson_interval",
    "write_manifest",
]
