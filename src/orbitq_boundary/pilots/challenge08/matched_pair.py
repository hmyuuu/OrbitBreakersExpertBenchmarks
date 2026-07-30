"""Fixed-scale BASE/PROBE pair for Challenge 08 boundary framework v2."""

from __future__ import annotations

from math import cos, sin
from typing import Mapping

from ...matched import MatchedPairSpec
from .family import ANGLE_CONFIG, N_SAMPLES, TOLERANCES

CAPABILITY_ID = "spatially_indexed_2d_circuit_semantics"
PAIR_ID = "challenge08-spatial-modulation-v2"
BASE_ID = "challenge08-v2-base"
PROBE_ID = "challenge08-v2-probe"
GRID_SIDE = 7
SEED = 2033
SOLE_SEMANTIC_DELTA = "spatial_modulation_mode"

SPATIAL_SCALE_FIELDS = (
    "ry_row_sin_scale",
    "ry_col_cos_scale",
    "ry_diag_sin_scale",
    "rzz_edge_sin_scale",
    "rzz_site_cos_scale",
    "rxx_edge_cos_scale",
    "rxx_site_sin_scale",
    "rx_row_cos_scale",
    "rx_col_sin_scale",
    "rx_diag_cos_scale",
)

_PAIR_METADATA_FIELDS = frozenset({"candidate_id", SOLE_SEMANTIC_DELTA})


def _build_config(candidate_id: str, modulation_mode: str) -> dict[str, object]:
    config: dict[str, object] = {
        "schema_version": "orbitq-challenge08-matched-config-v2",
        "candidate_id": candidate_id,
        "grid_side": GRID_SIDE,
        "n_qubits": GRID_SIDE * GRID_SIDE,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        SOLE_SEMANTIC_DELTA: modulation_mode,
        "initial_state": "|0>^49",
        "horizontal_gate": "RZZ",
        "vertical_gate": "RXX",
        "site_indexing": "row_major",
        "sampling_method": "TensorCircuit.Circuit.perfect_sampling",
        "backend": "jax",
        **ANGLE_CONFIG,
        **TOLERANCES,
    }
    if modulation_mode == "uniform":
        for field in SPATIAL_SCALE_FIELDS:
            config[field] = 0.0
    elif modulation_mode != "spatially_indexed":
        raise ValueError(f"unsupported modulation mode: {modulation_mode}")
    return config


def build_base_config() -> dict[str, object]:
    return _build_config(BASE_ID, "uniform")


def build_probe_config() -> dict[str, object]:
    return _build_config(PROBE_ID, "spatially_indexed")


def build_pair_spec() -> MatchedPairSpec:
    base = build_base_config()
    controlled = tuple(
        sorted(
            set(base)
            - _PAIR_METADATA_FIELDS
            - set(SPATIAL_SCALE_FIELDS)
        )
    )
    return MatchedPairSpec(
        capability_id=CAPABILITY_ID,
        pair_id=PAIR_ID,
        base_candidate_id=BASE_ID,
        probe_candidate_id=PROBE_ID,
        sole_semantic_delta=SOLE_SEMANTIC_DELTA,
        controlled_fields=controlled,
        deterministically_derived_fields=SPATIAL_SCALE_FIELDS,
    )


def compare_pair_artifacts(
    base: Mapping[str, object],
    probe: Mapping[str, object],
) -> dict[str, object]:
    """Verify that raw differences encode only the declared semantic delta."""

    unexpected: list[str] = []
    all_fields = set(base) | set(probe)
    allowed = _PAIR_METADATA_FIELDS | set(SPATIAL_SCALE_FIELDS)
    for field in sorted(all_fields - allowed):
        if base.get(field) != probe.get(field):
            unexpected.append(f"unexpected_difference:{field}")

    expected_metadata = (
        base.get("candidate_id") == BASE_ID
        and probe.get("candidate_id") == PROBE_ID
        and base.get(SOLE_SEMANTIC_DELTA) == "uniform"
        and probe.get(SOLE_SEMANTIC_DELTA) == "spatially_indexed"
    )
    if not expected_metadata:
        unexpected.append("unexpected_difference:pair_identity")

    derived = sorted(
        field
        for field in SPATIAL_SCALE_FIELDS
        if base.get(field) != probe.get(field)
    )
    if derived != sorted(SPATIAL_SCALE_FIELDS):
        unexpected.append("unexpected_difference:spatial_scale_derivation")
    if any(float(base.get(field, 1.0)) != 0.0 for field in SPATIAL_SCALE_FIELDS):
        unexpected.append("unexpected_difference:base_not_uniform")

    return {
        "passed": not unexpected,
        "sole_semantic_delta": SOLE_SEMANTIC_DELTA,
        "unexpected_differences": unexpected,
        "derived_differences": derived,
    }


def _rounded_unique(values: list[float]) -> int:
    return len({round(value, 12) for value in values})


def summarize_angle_nondegeneracy(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Count distinct site/edge angles using the frozen witness equations."""

    side = int(config["grid_side"])
    f = lambda name: float(config[name])
    ry: list[float] = []
    rzz: list[float] = []
    rxx: list[float] = []
    rx: list[float] = []
    for row in range(side):
        for col in range(side):
            ry.append(
                f("ry_offset")
                + f("ry_row_sin_scale")
                * sin(f("ry_row_sin_frequency") * (row + 1))
                + f("ry_col_cos_scale")
                * cos(f("ry_col_cos_frequency") * (col + 1))
                + f("ry_diag_sin_scale")
                * sin(f("ry_diag_sin_frequency") * (row + col + 2))
            )
            rx.append(
                f("rx_offset")
                + f("rx_row_cos_scale")
                * cos(f("rx_row_cos_frequency") * (row + 1))
                - f("rx_col_sin_scale")
                * sin(f("rx_col_sin_frequency") * (col + 1))
                + f("rx_diag_cos_scale")
                * cos(f("rx_diag_cos_frequency") * (row + col + 2))
            )
    edge_index = 0
    for row in range(side):
        for col in range(side - 1):
            rzz.append(
                f("rzz_offset")
                + f("rzz_edge_sin_scale")
                * sin(f("rzz_edge_sin_frequency") * (edge_index + 1))
                + f("rzz_site_cos_scale")
                * cos(f("rzz_site_cos_frequency") * (2 * row + col + 1))
            )
            edge_index += 1
    edge_index = 0
    for row in range(side - 1):
        for col in range(side):
            rxx.append(
                f("rxx_offset")
                + f("rxx_edge_cos_scale")
                * cos(f("rxx_edge_cos_frequency") * (edge_index + 1))
                + f("rxx_site_sin_scale")
                * sin(f("rxx_site_sin_frequency") * (row + 2 * col + 1))
            )
            edge_index += 1
    counts = {
        "ry": _rounded_unique(ry),
        "rzz": _rounded_unique(rzz),
        "rxx": _rounded_unique(rxx),
        "rx": _rounded_unique(rx),
    }
    return {
        "unique_value_counts": counts,
        "all_gate_families_spatially_uniform": all(
            value == 1 for value in counts.values()
        ),
        "all_gate_families_spatially_nondegenerate": all(
            value > 1 for value in counts.values()
        ),
    }


def build_negative_control_applicability() -> dict[str, list[str]]:
    both = ["base", "probe"]
    return {
        "missing_horizontal_bonds": both.copy(),
        "missing_vertical_bonds": both.copy(),
        "swapped_rzz_rxx_orientations": both.copy(),
        "incorrect_row_major_indexing": both.copy(),
        "fabricated_samples": both.copy(),
        "ignore_row_modulation": ["probe"],
        "ignore_column_modulation": ["probe"],
        "ignore_diagonal_modulation": ["probe"],
        "ignore_edge_modulation": ["probe"],
    }


def build_candidate_selection_matrix() -> list[dict[str, object]]:
    common = {
        "witness_support": "existing private Challenge 08 witness",
        "evaluator_feasibility": "high",
        "novelty": "matched semantic boundary",
    }
    return [
        {
            **common,
            "axis": SOLE_SEMANTIC_DELTA,
            "capability": CAPABILITY_ID,
            "sole_delta": "uniform versus spatially indexed angles",
            "expected_runtime_memory_ratio": "near 1 under common rendering",
            "semantic_overlap": "maximal",
            "known_confounds": [],
            "decision": "primary",
        },
        {
            **common,
            "axis": "api_guidance",
            "capability": "framework_discoverability",
            "sole_delta": "public API guidance",
            "expected_runtime_memory_ratio": "1",
            "semantic_overlap": "same scientific task",
            "known_confounds": ["prompt support changes"],
            "decision": "separate_control",
        },
        {
            **common,
            "axis": "grid_side",
            "capability": "scalability",
            "sole_delta": "grid size",
            "expected_runtime_memory_ratio": "not controlled",
            "semantic_overlap": "same circuit family",
            "known_confounds": ["runtime", "memory", "contraction complexity"],
            "decision": "scale_stress_control",
        },
    ]
