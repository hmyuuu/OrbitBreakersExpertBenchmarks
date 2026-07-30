"""Core records and deterministic serialization for boundary experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list[Any] | dict[str, Any]


def _jsonable(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return a stable UTF-8 JSON representation suitable for hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_digest(value: JsonValue) -> str:
    """Return a SHA-256 digest of the canonical JSON representation."""

    return sha256(canonical_json_bytes(value)).hexdigest()


class JsonRecord:
    """Mixin that converts dataclass records to JSON-compatible mappings."""

    def to_dict(self) -> dict[str, JsonValue]:
        value = _jsonable(asdict(self))
        if not isinstance(value, dict):
            raise TypeError("record serialization must produce a mapping")
        return value


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, JsonValue]:
    copied = _jsonable(value)
    if not isinstance(copied, dict):
        raise TypeError("expected a JSON-compatible mapping")
    return copied


@dataclass(frozen=True)
class EvidenceValue(JsonRecord):
    status: Literal["recovered", "partial", "unknown"]
    value: JsonValue
    source: str
    limitation: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"recovered", "partial", "unknown"}:
            raise ValueError(f"invalid evidence status: {self.status}")
        if self.status == "unknown" and self.value is not None:
            raise ValueError("unknown evidence must not carry a value")
        if self.status != "unknown" and self.value is None:
            raise ValueError("recovered or partial evidence requires a value")
        if not self.source.strip():
            raise ValueError("evidence source must be non-empty")
        object.__setattr__(self, "value", _jsonable(self.value))


@dataclass(frozen=True)
class AgentConfig(JsonRecord):
    config_id: str
    model: EvidenceValue
    harness: EvidenceValue
    tools: EvidenceValue
    prompt: EvidenceValue
    docs: EvidenceValue
    budget: EvidenceValue
    framework: EvidenceValue
    framework_version: EvidenceValue

    def __post_init__(self) -> None:
        if not self.config_id.strip():
            raise ValueError("config_id must be non-empty")

    @property
    def fingerprint(self) -> str:
        return content_digest(self.to_dict())


@dataclass(frozen=True)
class DifficultyParameter(JsonRecord):
    name: str
    kind: Literal["capability", "scale", "support"]
    values: tuple[JsonScalar, ...]
    harder_direction: Literal["ascending", "descending"]
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class TaskFamilySpec(JsonRecord):
    family_id: str
    title: str
    scientific_contract: dict[str, JsonValue]
    generator_id: str
    verifier_contract: dict[str, JsonValue]
    parameters: tuple[DifficultyParameter, ...]
    resource_budget: dict[str, JsonValue]
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scientific_contract", _copy_mapping(self.scientific_contract)
        )
        object.__setattr__(
            self, "verifier_contract", _copy_mapping(self.verifier_contract)
        )
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self, "resource_budget", _copy_mapping(self.resource_budget)
        )
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))

    @property
    def scientific_contract_digest(self) -> str:
        return content_digest(self.scientific_contract)


@dataclass(frozen=True)
class CandidateTaskSpec(JsonRecord):
    candidate_id: str
    family_id: str
    parameters: dict[str, JsonScalar]
    scientific_contract_digest: str
    parent_candidate_id: str | None
    changed_parameter: str | None
    generation_seed: int
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(frozen=True)
class ExpertValidationRecord(JsonRecord):
    candidate_id: str
    validation_id: str
    seed: int
    functional_passed: bool
    static_policy_passed: bool
    semantic_audit_passed: bool
    resource_budget_passed: bool
    runtime_sec: float
    evidence_paths: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_paths", tuple(self.evidence_paths))

    @property
    def passed(self) -> bool:
        return all(
            (
                self.functional_passed,
                self.static_policy_passed,
                self.semantic_audit_passed,
                self.resource_budget_passed,
            )
        )


@dataclass(frozen=True)
class AgentRunRecord(JsonRecord):
    candidate_id: str
    run_id: str
    agent_config_fingerprint: str
    isolation_id: str
    seed: int
    correctness_passed: bool
    resource_gate_passed: bool | None
    failure_category: str | None
    runtime_sec: float
    artifact_paths: tuple[str, ...]
    notes: str
    static_policy_passed: bool | None = None
    semantic_audit_passed: bool | None = None
    eligible_passed: bool | None = None
    candidate_digest: str | None = None
    provenance: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_paths", tuple(self.artifact_paths))
        object.__setattr__(self, "provenance", _copy_mapping(self.provenance))

    @property
    def passed(self) -> bool | None:
        """Return gate eligibility without turning runtime into a score."""

        if self.eligible_passed is not None:
            return self.eligible_passed
        if self.resource_gate_passed is None:
            return None
        return self.correctness_passed and self.resource_gate_passed


@dataclass(frozen=True)
class ClassificationThresholds(JsonRecord):
    easy_min: float = 0.8
    hard_max: float = 0.2
    screening_checkpoint: int = 5
    formal_checkpoint: int = 16
    confidence_level: float = 0.95


@dataclass(frozen=True)
class CandidateVerdict(JsonRecord):
    verdict_id: str
    candidate_id: str
    agent_config_fingerprint: str
    n_runs: int
    n_passed: int
    pass_rate: float
    confidence_level: float
    interval_low: float
    interval_high: float
    screening_classification: (
        Literal[
            "easy_candidate",
            "hard_candidate",
            "transition_candidate",
        ]
        | None
    )
    endpoint_classification: Literal["easy", "hard", "unresolved"]
    checkpoint_kind: Literal["screening", "formal"]
    sampling_decision: Literal[
        "continue_to_formal_checkpoint",
        "stop",
        "await_approved_sampling_policy",
    ]
    reason: str
    outcome_kind: Literal["correctness", "resource_gate", "eligible"] = "eligible"


@dataclass(frozen=True)
class OutcomeSummary(JsonRecord):
    candidate_id: str
    agent_config_fingerprint: str
    n_runs: int
    correctness_pass_rate: float
    resource_gate_pass_rate: float | None
    eligible_pass_rate: float | None
    runtime_count: int
    runtime_min_sec: float | None
    runtime_median_sec: float | None
    runtime_max_sec: float | None


@dataclass(frozen=True)
class NonMonotonicObservation(JsonRecord):
    easier_candidate_id: str
    harder_candidate_id: str
    parameter_name: str
    easier_pass_rate: float
    harder_pass_rate: float
    notes: str


@dataclass(frozen=True)
class FormalScaleObservation(JsonRecord):
    candidate_id: str
    parameter_name: str
    parameter_value: JsonScalar
    endpoint_classification: Literal["easy", "hard", "unresolved"]
    pass_rate: float


@dataclass(frozen=True)
class BoundarySearchState(JsonRecord):
    family_id: str
    agent_config_fingerprint: str
    axis_order: tuple[str, ...]
    axis_index: int
    current_candidate_id: str
    visited_parameter_vectors: tuple[str, ...]
    tested_candidate_ids: tuple[str, ...]
    non_monotonic_observations: tuple[NonMonotonicObservation, ...]
    status: str
    formal_scale_observations: tuple[FormalScaleObservation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_order", tuple(self.axis_order))
        object.__setattr__(
            self, "visited_parameter_vectors", tuple(self.visited_parameter_vectors)
        )
        object.__setattr__(
            self, "tested_candidate_ids", tuple(self.tested_candidate_ids)
        )
        object.__setattr__(
            self,
            "non_monotonic_observations",
            tuple(self.non_monotonic_observations),
        )
        object.__setattr__(
            self,
            "formal_scale_observations",
            tuple(self.formal_scale_observations),
        )


@dataclass(frozen=True)
class BoundaryEvidence(JsonRecord):
    family_id: str
    agent_config_fingerprint: str
    screening_easy_candidate_ids: tuple[str, ...]
    screening_hard_candidate_ids: tuple[str, ...]
    screening_transition_candidate_ids: tuple[str, ...]
    formal_easy_candidate_ids: tuple[str, ...]
    formal_hard_candidate_ids: tuple[str, ...]
    formal_unresolved_candidate_ids: tuple[str, ...]
    matched_control_candidate_id: str | None
    screening_verdict_ids: tuple[str, ...]
    formal_verdict_ids: tuple[str, ...]
    expert_validation_ids: tuple[str, ...]
    manifest_digest: str | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        tuple_fields = (
            "screening_easy_candidate_ids",
            "screening_hard_candidate_ids",
            "screening_transition_candidate_ids",
            "formal_easy_candidate_ids",
            "formal_hard_candidate_ids",
            "formal_unresolved_candidate_ids",
            "screening_verdict_ids",
            "formal_verdict_ids",
            "expert_validation_ids",
            "limitations",
        )
        for field_name in tuple_fields:
            object.__setattr__(
                self,
                field_name,
                tuple(getattr(self, field_name)),
            )
