"""Loading, validating, and instantiating task-family specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import (
    CandidateTaskSpec,
    DifficultyParameter,
    JsonScalar,
    TaskFamilySpec,
    canonical_json_bytes,
    content_digest,
)

REQUIRED_FAMILY_KEYS = {
    "family_id",
    "title",
    "scientific_contract",
    "generator_id",
    "verifier_contract",
    "parameters",
    "resource_budget",
    "metadata",
}

REQUIRED_PARAMETER_KEYS = {
    "name",
    "kind",
    "values",
    "harder_direction",
    "description",
}

PARAMETER_KINDS = {"capability", "scale", "support"}
HARDER_DIRECTIONS = {"ascending", "descending"}


def _require_exact_keys(
    data: Mapping[str, Any], required: set[str], label: str
) -> None:
    actual = set(data)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(f"invalid {label} keys: missing={missing}, extra={extra}")


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _parameter_from_mapping(data: Any) -> DifficultyParameter:
    mapping = _require_mapping(data, "parameter")
    _require_exact_keys(mapping, REQUIRED_PARAMETER_KEYS, "parameter")
    name = _require_non_empty_string(mapping["name"], "parameter name")
    kind = mapping["kind"]
    if kind not in PARAMETER_KINDS:
        raise ValueError(f"invalid parameter kind for {name}: {kind}")
    values = mapping["values"]
    if not isinstance(values, list) or not values:
        raise ValueError(f"parameter values must be a non-empty list: {name}")
    if not all(
        item is None or isinstance(item, (str, int, float, bool))
        for item in values
    ):
        raise ValueError(f"parameter values must be JSON scalars: {name}")
    encoded = [canonical_json_bytes(item) for item in values]
    if len(set(encoded)) != len(encoded):
        raise ValueError(f"parameter contains duplicate values: {name}")
    direction = mapping["harder_direction"]
    if direction not in HARDER_DIRECTIONS:
        raise ValueError(
            f"invalid harder_direction for {name}: {direction}"
        )
    description = _require_non_empty_string(
        mapping["description"], f"parameter description for {name}"
    )
    return DifficultyParameter(
        name=name,
        kind=kind,
        values=tuple(values),
        harder_direction=direction,
        description=description,
    )


def family_spec_from_mapping(data: Mapping[str, Any]) -> TaskFamilySpec:
    """Validate a JSON-like mapping and build a task-family specification."""

    if not isinstance(data, Mapping):
        raise ValueError("family specification must be a JSON object")
    mapping = dict(data)
    _require_exact_keys(mapping, REQUIRED_FAMILY_KEYS, "family")
    raw_parameters = mapping["parameters"]
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("parameters must be a non-empty list")
    parameters = tuple(
        _parameter_from_mapping(parameter) for parameter in raw_parameters
    )
    names = [parameter.name for parameter in parameters]
    if len(set(names)) != len(names):
        raise ValueError("duplicate parameter name in family specification")
    return TaskFamilySpec(
        family_id=_require_non_empty_string(mapping["family_id"], "family_id"),
        title=_require_non_empty_string(mapping["title"], "title"),
        scientific_contract=_require_mapping(
            mapping["scientific_contract"], "scientific_contract"
        ),
        generator_id=_require_non_empty_string(
            mapping["generator_id"], "generator_id"
        ),
        verifier_contract=_require_mapping(
            mapping["verifier_contract"], "verifier_contract"
        ),
        parameters=parameters,
        resource_budget=_require_mapping(
            mapping["resource_budget"], "resource_budget"
        ),
        metadata=_require_mapping(mapping["metadata"], "metadata"),
    )


def load_family_spec(path: str | Path) -> TaskFamilySpec:
    """Load a UTF-8 JSON family specification from disk."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("family specification root must be a JSON object")
    return family_spec_from_mapping(data)


def validate_parameter_vector(
    family: TaskFamilySpec, parameters: Mapping[str, JsonScalar]
) -> None:
    """Reject vectors that do not exactly match the declared parameter space."""

    expected = {parameter.name for parameter in family.parameters}
    actual = set(parameters)
    if actual != expected:
        raise ValueError(
            "parameter keys do not match family specification: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for parameter in family.parameters:
        value = parameters[parameter.name]
        encoded_value = canonical_json_bytes(value)
        encoded_domain = {
            canonical_json_bytes(domain_value)
            for domain_value in parameter.values
        }
        if encoded_value not in encoded_domain:
            raise ValueError(
                f"value outside declared domain for {parameter.name}: {value!r}"
            )


def generate_candidate(
    family: TaskFamilySpec,
    parameters: Mapping[str, JsonScalar],
    generation_seed: int,
    parent: CandidateTaskSpec | None = None,
    changed_parameter: str | None = None,
) -> CandidateTaskSpec:
    """Generate deterministic candidate metadata from a parameter vector."""

    validate_parameter_vector(family, parameters)
    if parent is None and changed_parameter is not None:
        raise ValueError("root candidate cannot declare changed_parameter")
    if parent is not None and changed_parameter is None:
        raise ValueError("child candidate requires changed_parameter")
    if parent is not None:
        if parent.family_id != family.family_id:
            raise ValueError("parent candidate belongs to a different family")
        if (
            parent.scientific_contract_digest
            != family.scientific_contract_digest
        ):
            raise ValueError("parent candidate has a different contract")
    normalized_parameters = {
        parameter.name: parameters[parameter.name]
        for parameter in family.parameters
    }
    candidate_payload = {
        "family_id": family.family_id,
        "scientific_contract_digest": family.scientific_contract_digest,
        "parameters": normalized_parameters,
        "generation_seed": generation_seed,
    }
    candidate_id = f"candidate-{content_digest(candidate_payload)[:16]}"
    return CandidateTaskSpec(
        candidate_id=candidate_id,
        family_id=family.family_id,
        parameters=normalized_parameters,
        scientific_contract_digest=family.scientific_contract_digest,
        parent_candidate_id=None if parent is None else parent.candidate_id,
        changed_parameter=changed_parameter,
        generation_seed=generation_seed,
        metadata={
            "generator_id": family.generator_id,
            "synthetic": bool(family.metadata.get("synthetic", False)),
        },
    )
