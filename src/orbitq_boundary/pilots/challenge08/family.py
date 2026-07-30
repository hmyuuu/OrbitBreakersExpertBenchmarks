"""Deterministic, offline Challenge 08 pilot-family construction."""

from __future__ import annotations

from typing import Mapping

from ...models import (
    CandidateTaskSpec,
    DifficultyParameter,
    TaskFamilySpec,
    content_digest,
)

FAMILY_ID = "challenge-08-grid-scale-v1"
GENERATOR_ID = "orbitq-challenge08-family-v1"
HIDDEN_GENERATOR_ID = "orbitq-challenge08-hidden-z-v1"
GENERATION_SEED = 20260729
INITIAL_GRIDS = (4, 5, 6, 7)
CONDITIONAL_GRID = 3
ALL_GRIDS = (3, 4, 5, 6, 7)
N_SAMPLES = 8192

ANGLE_CONFIG = {
    "ry_offset": 0.19,
    "ry_row_sin_scale": 0.07,
    "ry_row_sin_frequency": 0.83,
    "ry_col_cos_scale": 0.05,
    "ry_col_cos_frequency": 0.61,
    "ry_diag_sin_scale": 0.03,
    "ry_diag_sin_frequency": 0.29,
    "rzz_offset": 0.31,
    "rzz_edge_sin_scale": 0.09,
    "rzz_edge_sin_frequency": 0.47,
    "rzz_site_cos_scale": 0.06,
    "rzz_site_cos_frequency": 0.38,
    "rxx_offset": 0.27,
    "rxx_edge_cos_scale": 0.08,
    "rxx_edge_cos_frequency": 0.41,
    "rxx_site_sin_scale": 0.07,
    "rxx_site_sin_frequency": 0.33,
    "rx_offset": 0.17,
    "rx_row_cos_scale": 0.06,
    "rx_row_cos_frequency": 0.52,
    "rx_col_sin_scale": 0.04,
    "rx_col_sin_frequency": 0.44,
    "rx_diag_cos_scale": 0.02,
    "rx_diag_cos_frequency": 0.25,
}
TOLERANCES = {
    "single_z_tolerance": 0.03,
    "hidden_z_string_max_tolerance": 0.05,
    "hidden_z_string_mean_tolerance": 0.015,
}


def build_family_spec() -> TaskFamilySpec:
    return TaskFamilySpec(
        family_id=FAMILY_ID,
        title="Challenge 08 scale-consistent 2D tensor-network sampling",
        scientific_contract={
            "initial_state": "|0>^(grid_side^2)",
            "layers": ["position-dependent Ry", "horizontal RZZ", "vertical RXX", "position-dependent Rx"],
            "output": "binary framework-native samples",
            "sample_shape": ["n_samples", "grid_side^2"],
            "n_samples": N_SAMPLES,
            "angle_config": ANGLE_CONFIG,
            "tolerances": TOLERANCES,
            "forbidden": [
                "full statevector",
                "dense probability vector",
                "raw replacement simulator",
            ],
        },
        generator_id=GENERATOR_ID,
        verifier_contract={
            "official_path": "tasks/challenge-08/tests/evaluate_8.py",
            "classification_outcome": "correctness",
            "resource_reported_separately": True,
        },
        parameters=(
            DifficultyParameter(
                name="grid_side",
                kind="scale",
                values=ALL_GRIDS,
                harder_direction="ascending",
                description="only primary main-sequence scale axis",
            ),
            DifficultyParameter(
                name="api_support",
                kind="support",
                values=("baseline", "guided"),
                harder_direction="descending",
                description="matched-control condition only; never pooled with scale",
            ),
        ),
        resource_budget={
            "n_samples": N_SAMPLES,
            "hard_timeout_sec": None,
            "hard_timeout_status": "awaiting-approved-A_current-budget",
        },
        metadata={
            "initial_grids": list(INITIAL_GRIDS),
            "conditional_grid": CONDITIONAL_GRID,
            "status": "family-approved-execution-not-approved",
        },
    )


def candidate_for_grid(
    grid_side: int,
    *,
    api_support: str = "baseline",
) -> CandidateTaskSpec:
    if grid_side not in ALL_GRIDS:
        raise ValueError(f"unsupported grid_side: {grid_side}")
    if api_support not in {"baseline", "guided"}:
        raise ValueError(f"unsupported api_support: {api_support}")
    family = build_family_spec()
    prefix = "G" if api_support == "baseline" else "MC-G"
    candidate_id = f"challenge08-{prefix}{grid_side}"
    return CandidateTaskSpec(
        candidate_id=candidate_id,
        family_id=family.family_id,
        parameters={
            "grid_side": grid_side,
            "n_qubits": grid_side * grid_side,
            "n_samples": N_SAMPLES,
            "api_support": api_support,
        },
        scientific_contract_digest=family.scientific_contract_digest,
        parent_candidate_id=(
            None if api_support == "baseline" else f"challenge08-G{grid_side}"
        ),
        changed_parameter=None if api_support == "baseline" else "api_support",
        generation_seed=GENERATION_SEED,
        metadata={
            "generator_id": GENERATOR_ID,
            "main_scale_sequence": api_support == "baseline",
            "support_control": api_support == "guided",
        },
    )


def _rounded_anchor(u: float, grid_side: int) -> int:
    return min(grid_side - 1, max(0, int(u * (grid_side - 1) + 0.5)))


def generate_hidden_supports(
    grid_side: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Generate four deterministic observable classes without 7x7 truncation."""

    if grid_side not in ALL_GRIDS:
        raise ValueError(f"unsupported grid_side: {grid_side}")

    def q(row: int, col: int) -> int:
        return row * grid_side + col

    lo, mid, hi = (
        _rounded_anchor(0.0, grid_side),
        _rounded_anchor(0.5, grid_side),
        _rounded_anchor(1.0, grid_side),
    )
    raw = {
        "single_site": [
            (q(r, c),)
            for r, c in (
                (lo, lo), (lo, hi), (hi, lo), (hi, hi),
                (lo, mid), (hi, mid), (mid, lo), (mid, hi), (mid, mid),
            )
        ],
        "nearest_neighbor": [
            (q(lo, lo), q(lo, lo + 1)),
            (q(mid, max(0, mid - 1)), q(mid, min(grid_side - 1, mid))),
            (q(hi, hi - 1), q(hi, hi)),
            (q(lo, lo), q(lo + 1, lo)),
            (q(max(0, mid - 1), mid), q(min(grid_side - 1, mid), mid)),
            (q(hi - 1, hi), q(hi, hi)),
        ],
        "patch_2x2": [
            (q(r, c), q(r, c + 1), q(r + 1, c), q(r + 1, c + 1))
            for r, c in (
                (0, 0),
                (min(grid_side - 2, max(0, mid - 1)),) * 2,
                (grid_side - 2, grid_side - 2),
            )
        ],
        "long_range": [
            tuple(q(index, index) for index in range(grid_side)),
            tuple(q(index, grid_side - 1 - index) for index in range(grid_side)),
            tuple(q(mid, index) for index in range(grid_side)),
            tuple(q(index, mid) for index in range(grid_side)),
        ],
    }
    seen: set[tuple[int, ...]] = set()
    result: dict[str, tuple[tuple[int, ...], ...]] = {}
    for category, supports in raw.items():
        kept = []
        for support in supports:
            normalized = tuple(dict.fromkeys(support))
            if normalized not in seen:
                seen.add(normalized)
                kept.append(normalized)
        result[category] = tuple(kept)
    return result


def _config(grid_side: int) -> dict[str, object]:
    return {
        "grid_side": grid_side,
        "n_qubits": grid_side * grid_side,
        "n_samples": N_SAMPLES,
        **ANGLE_CONFIG,
        **TOLERANCES,
    }


def public_candidate_artifact(
    candidate: CandidateTaskSpec,
) -> dict[str, object]:
    grid_side = int(candidate.parameters["grid_side"])
    support = candidate.parameters["api_support"]
    return {
        "schema_version": "orbitq-challenge08-public-v1",
        "family_id": FAMILY_ID,
        "candidate": candidate.to_dict(),
        "candidate_digest": content_digest(candidate.to_dict()),
        "config": _config(grid_side),
        "api_support": {
            "condition": support,
            "guidance": (
                []
                if support == "baseline"
                else [
                    "tc.set_contractor('omeco-4-4')",
                    "Circuit.perfect_sampling",
                    "backend vmap",
                    "inspect installed framework source and examples",
                ]
            ),
        },
        "observable_contract": {
            "categories": [
                "single_site",
                "nearest_neighbor",
                "patch_2x2",
                "long_range",
            ],
            "values": "evaluator-only",
        },
    }


def sealed_candidate_artifact(
    candidate: CandidateTaskSpec,
) -> dict[str, object]:
    grid_side = int(candidate.parameters["grid_side"])
    supports = generate_hidden_supports(grid_side)
    candidate_digest = content_digest(candidate.to_dict())
    supports_digest = content_digest(supports)
    hidden_generator = {
        "id": HIDDEN_GENERATOR_ID,
        "seed": GENERATION_SEED,
    }
    sealed_instance_digest = content_digest(
        {
            "candidate_digest": candidate_digest,
            "hidden_generator": hidden_generator,
            "hidden_supports_digest": supports_digest,
        }
    )
    return {
        "schema_version": "orbitq-challenge08-sealed-v1",
        "family_id": FAMILY_ID,
        "candidate_id": candidate.candidate_id,
        "candidate_digest": candidate_digest,
        "hidden_generator": hidden_generator,
        "hidden_supports": supports,
        "hidden_supports_digest": supports_digest,
        "sealed_instance_digest": sealed_instance_digest,
        "expert_reference": {
            "status": "unavailable",
            "values": None,
            "method": "official TensorCircuit-NG framework-native path required",
            "limitation": "not generated until the local expert environment passes",
            "binding_digest": None,
        },
    }


def matched_control_for_screening(
    screening_labels: Mapping[str, str],
) -> CandidateTaskSpec | None:
    for target in ("transition_candidate", "hard_candidate"):
        for grid_side in INITIAL_GRIDS:
            if screening_labels.get(f"G{grid_side}") == target:
                return candidate_for_grid(grid_side, api_support="guided")
    return None


def build_dry_run_manifest() -> dict[str, object]:
    family = build_family_spec()
    candidates = [candidate_for_grid(grid) for grid in INITIAL_GRIDS]
    payload = {
        "schema_version": "orbitq-challenge08-dry-run-v1",
        "status": "challenge08_layer2_blocked",
        "family_spec": family.to_dict(),
        "family_digest": content_digest(family.to_dict()),
        "initial_candidates": [candidate.to_dict() for candidate in candidates],
        "conditional_candidate": candidate_for_grid(CONDITIONAL_GRID).to_dict(),
        "matched_control": {
            "status": "not-selected-without-screening-results",
            "selection_rule": "first transition_candidate else first hard_candidate",
        },
        "classification_outcome": "correctness",
        "agent_runs": [],
        "screening_verdicts": [],
        "formal_verdicts": [],
        "pilot_ready": False,
        "blockers": [
            "expert-references-not-generated",
            "official-scorer-migration-not-passed",
            "A_current-manifest-not-approved",
            "run-budget-not-approved",
            "hard-timeout-not-approved",
        ],
    }
    payload["manifest_digest"] = content_digest(payload)
    return payload
