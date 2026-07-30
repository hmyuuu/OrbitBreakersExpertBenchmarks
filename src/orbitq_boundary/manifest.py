"""Deterministic reproducibility manifests for boundary experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .models import (
    AgentConfig,
    AgentRunRecord,
    BoundaryEvidence,
    BoundarySearchState,
    CandidateTaskSpec,
    CandidateVerdict,
    ClassificationThresholds,
    ExpertValidationRecord,
    JsonValue,
    TaskFamilySpec,
    canonical_json_bytes,
    content_digest,
)


def _normalized_json_mapping(
    value: Mapping[str, JsonValue], label: str
) -> dict[str, JsonValue]:
    normalized = json.loads(canonical_json_bytes(dict(value)))
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _require_unique(values: list[str], label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} in manifest inputs")


def build_manifest(
    *,
    created_at: str,
    synthetic: bool,
    seed: int,
    agent_config: AgentConfig,
    family: TaskFamilySpec,
    thresholds: ClassificationThresholds,
    candidates: Iterable[CandidateTaskSpec],
    expert_validations: Iterable[ExpertValidationRecord],
    agent_runs: Iterable[AgentRunRecord],
    screening_verdicts: Iterable[CandidateVerdict],
    formal_verdicts: Iterable[CandidateVerdict],
    search_state: BoundarySearchState,
    boundary_evidence: BoundaryEvidence,
    environment_evidence: Mapping[str, JsonValue],
    limitations: Iterable[str],
) -> dict[str, JsonValue]:
    """Build a complete canonical manifest and attach its content digest."""

    if not created_at.strip():
        raise ValueError("created_at must be an explicit non-empty value")
    fingerprint = agent_config.fingerprint
    if search_state.family_id != family.family_id:
        raise ValueError("search state family does not match manifest family")
    if boundary_evidence.family_id != family.family_id:
        raise ValueError("boundary evidence family does not match manifest family")
    if search_state.agent_config_fingerprint != fingerprint:
        raise ValueError("search state AgentConfig does not match manifest")
    if boundary_evidence.agent_config_fingerprint != fingerprint:
        raise ValueError("boundary evidence AgentConfig does not match manifest")

    ordered_candidates = sorted(
        tuple(candidates), key=lambda item: item.candidate_id
    )
    ordered_expert = sorted(
        tuple(expert_validations), key=lambda item: item.validation_id
    )
    ordered_runs = sorted(tuple(agent_runs), key=lambda item: item.run_id)
    ordered_screening_verdicts = sorted(
        tuple(screening_verdicts), key=lambda item: item.verdict_id
    )
    ordered_formal_verdicts = sorted(
        tuple(formal_verdicts), key=lambda item: item.verdict_id
    )

    _require_unique(
        [item.candidate_id for item in ordered_candidates], "candidate_id"
    )
    _require_unique(
        [item.validation_id for item in ordered_expert], "validation_id"
    )
    _require_unique([item.run_id for item in ordered_runs], "run_id")
    all_verdict_ids = [
        item.verdict_id
        for item in (
            *ordered_screening_verdicts,
            *ordered_formal_verdicts,
        )
    ]
    _require_unique(all_verdict_ids, "verdict_id")

    for candidate in ordered_candidates:
        if candidate.family_id != family.family_id:
            raise ValueError("candidate belongs to another family")
        if (
            candidate.scientific_contract_digest
            != family.scientific_contract_digest
        ):
            raise ValueError("candidate scientific contract does not match family")
    for record in ordered_runs:
        if record.agent_config_fingerprint != fingerprint:
            raise ValueError("Agent run belongs to another AgentConfig")
    for verdict in (
        *ordered_screening_verdicts,
        *ordered_formal_verdicts,
    ):
        if verdict.agent_config_fingerprint != fingerprint:
            raise ValueError("verdict belongs to another AgentConfig")
    if any(
        verdict.checkpoint_kind != "screening"
        for verdict in ordered_screening_verdicts
    ):
        raise ValueError("screening_verdicts must contain screening looks")
    if any(
        verdict.checkpoint_kind != "formal"
        for verdict in ordered_formal_verdicts
    ):
        raise ValueError("formal_verdicts must contain formal looks")

    payload: dict[str, JsonValue] = {
        "schema_version": "orbitq-boundary-manifest-v2",
        "created_at": created_at,
        "synthetic": synthetic,
        "seed": seed,
        "agent_config": agent_config.to_dict(),
        "agent_config_fingerprint": fingerprint,
        "family_spec": family.to_dict(),
        "scientific_contract_digest": family.scientific_contract_digest,
        "thresholds": thresholds.to_dict(),
        "candidates": [item.to_dict() for item in ordered_candidates],
        "expert_validations": [item.to_dict() for item in ordered_expert],
        "agent_runs": [item.to_dict() for item in ordered_runs],
        "screening_verdicts": [
            item.to_dict() for item in ordered_screening_verdicts
        ],
        "formal_verdicts": [
            item.to_dict() for item in ordered_formal_verdicts
        ],
        "search_state": search_state.to_dict(),
        "boundary_evidence": boundary_evidence.to_dict(),
        "environment_evidence": _normalized_json_mapping(
            environment_evidence, "environment_evidence"
        ),
        "limitations": list(limitations),
    }
    payload["manifest_digest"] = content_digest(payload)
    return payload


def render_manifest(manifest: Mapping[str, JsonValue]) -> bytes:
    """Render a human-readable manifest deterministically."""

    return (
        json.dumps(
            dict(manifest),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def write_manifest(
    path: str | Path, manifest: Mapping[str, JsonValue]
) -> str:
    """Atomically write a rendered manifest and return its digest."""

    digest = manifest.get("manifest_digest")
    if not isinstance(digest, str) or not digest:
        raise ValueError("manifest_digest is required before writing")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(render_manifest(manifest))
    temporary.replace(destination)
    return digest
