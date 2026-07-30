"""Controlled candidate comparisons and task-validation gates."""

from __future__ import annotations

from collections.abc import Iterable

from .models import (
    CandidateTaskSpec,
    ExpertValidationRecord,
    TaskFamilySpec,
    canonical_json_bytes,
)
from .specs import validate_parameter_vector


def changed_parameters(
    parent: CandidateTaskSpec, child: CandidateTaskSpec
) -> tuple[str, ...]:
    """Return parameter names whose canonical JSON values differ."""

    keys = set(parent.parameters) | set(child.parameters)
    return tuple(
        sorted(
            key
            for key in keys
            if key not in parent.parameters
            or key not in child.parameters
            or canonical_json_bytes(parent.parameters[key])
            != canonical_json_bytes(child.parameters[key])
        )
    )


def assert_controlled_mutation(
    family: TaskFamilySpec,
    parent: CandidateTaskSpec,
    child: CandidateTaskSpec,
) -> str:
    """Validate a matched comparison and return its sole changed parameter."""

    if parent.family_id != family.family_id or child.family_id != family.family_id:
        raise ValueError("candidate belongs to a different family")
    expected_contract = family.scientific_contract_digest
    if (
        parent.scientific_contract_digest != expected_contract
        or child.scientific_contract_digest != expected_contract
    ):
        raise ValueError("scientific contract changed between candidates")
    if child.parent_candidate_id != parent.candidate_id:
        raise ValueError("child candidate does not reference the expected parent")
    validate_parameter_vector(family, parent.parameters)
    validate_parameter_vector(family, child.parameters)
    differences = changed_parameters(parent, child)
    if len(differences) != 1:
        raise ValueError(
            "controlled mutation must change exactly one parameter; "
            f"changed={list(differences)}"
        )
    actual = differences[0]
    if child.changed_parameter != actual:
        raise ValueError(
            "changed_parameter does not match actual difference: "
            f"declared={child.changed_parameter!r}, actual={actual!r}"
        )
    declared = {parameter.name for parameter in family.parameters}
    if actual not in declared:
        raise ValueError(f"changed parameter is not declared by family: {actual}")
    return actual


def require_expert_validation(
    candidate: CandidateTaskSpec,
    records: Iterable[ExpertValidationRecord],
) -> ExpertValidationRecord:
    """Return deterministic passing expert evidence or reject the candidate."""

    passing = sorted(
        (
            record
            for record in records
            if record.candidate_id == candidate.candidate_id and record.passed
        ),
        key=lambda record: record.validation_id,
    )
    if not passing:
        raise ValueError(
            f"candidate is not expert-validated: {candidate.candidate_id}"
        )
    return passing[0]
