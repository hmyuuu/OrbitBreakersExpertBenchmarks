"""Compatibility and negative-control gates for the matched Challenge 08 pair."""

from __future__ import annotations

import ast
from difflib import unified_diff
from hashlib import sha256
from typing import Mapping

from ...models import content_digest
from .container_worker import (
    BOUNDED_VMAP_SAMPLING,
    render_compatible_expert_source,
)
from .matched_pair import SPATIAL_SCALE_FIELDS

PINNED_CONTRACTOR_MODE = "omeco"
PINNED_CHUNK_SIZE = 256

_EXPECTED_REJECTION_GATE = {
    "missing_horizontal_bonds": "functional",
    "missing_vertical_bonds": "functional",
    "swapped_rzz_rxx_orientations": "functional",
    "incorrect_row_major_indexing": "functional",
    "fabricated_samples": "functional",
    "ignore_row_modulation": "static_policy",
    "ignore_column_modulation": "static_policy",
    "ignore_diagonal_modulation": "static_policy",
    "ignore_edge_modulation": "static_policy",
}


def build_compatibility_transform(
    contractor_mode: str,
    chunk_size: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "orbitq-challenge08-compatibility-transform-v2",
        "contractor_alias_replacement": {
            "from": 'tc.set_contractor("omeco-4-4")',
            "mode": contractor_mode,
            "scientific_operation": "unchanged tensor-network contraction",
        },
        "sampling_chunking": {
            "method": "deterministic contiguous chunks",
            "chunk_size": chunk_size,
            "n_samples": 8192,
            "sampling_method": "perfect_sampling",
            "seed": 2033,
        },
        "validated_in_real_environment": (
            contractor_mode == PINNED_CONTRACTOR_MODE
            and chunk_size == PINNED_CHUNK_SIZE
        ),
    }
    payload["digest"] = content_digest(payload)
    return payload


def render_common_witness(source: str) -> str:
    """Apply only the approved, common compatibility transformation."""

    return render_compatible_expert_source(
        source,
        contractor_mode=PINNED_CONTRACTOR_MODE,
    )


def compatibility_binding(
    source: str,
    *,
    original_bytes: bytes | None = None,
) -> dict[str, object]:
    rendered = render_common_witness(source)
    transform = build_compatibility_transform(
        PINNED_CONTRACTOR_MODE,
        PINNED_CHUNK_SIZE,
    )
    original_digest = sha256(
        source.encode("utf-8") if original_bytes is None else original_bytes
    ).hexdigest()
    rendered_digest = sha256(rendered.encode("utf-8")).hexdigest()
    diff = "".join(
        unified_diff(
            source.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile="original_witness.py",
            tofile="rendered_witness.py",
        )
    )
    return {
        "schema_version": "orbitq-challenge08-compatibility-binding-v2",
        "validated_transform": transform["validated_in_real_environment"],
        "original_witness_digest": original_digest,
        "compatibility_transform_digest": transform["digest"],
        "rendered_witness_digest": rendered_digest,
        "base_rendered_witness_digest": rendered_digest,
        "probe_rendered_witness_digest": rendered_digest,
        "same_rendered_witness_for_base_and_probe": True,
        "transform": transform,
        "rendered_diff": diff,
    }


def _replace_exact(
    source: str,
    old: str,
    new: str,
    expected_count: int,
    control_id: str,
) -> tuple[str, int]:
    count = source.count(old)
    if count != expected_count:
        raise ValueError(
            f"{control_id} source drift: expected {expected_count} matches, "
            f"found {count}"
        )
    return source.replace(old, new), count


def _config_override(fields: tuple[str, ...]) -> str:
    entries = ", ".join(f'"{field}": 0.0' for field in fields)
    return (
        "def build_circuit(config):\n"
        "    config = dict(config)\n"
        f"    config.update({{{entries}}})\n"
        "    n_side"
    )


def render_mutant_source(
    control_id: str,
    rendered_source: str,
) -> dict[str, object]:
    """Apply one predeclared semantic mutation and fail on source drift."""

    match_count = 0
    expected_count = 1
    if control_id == "missing_horizontal_bonds":
        source, match_count = _replace_exact(
            rendered_source,
            "circuit.rzz(left, right, theta=rzz_angle(row, col, edge_index, config))",
            "pass  # mutant: omitted horizontal RZZ bond",
            1,
            control_id,
        )
    elif control_id == "missing_vertical_bonds":
        source, match_count = _replace_exact(
            rendered_source,
            "circuit.rxx(left, right, theta=rxx_angle(row, col, edge_index, config))",
            "pass  # mutant: omitted vertical RXX bond",
            1,
            control_id,
        )
    elif control_id == "swapped_rzz_rxx_orientations":
        first = (
            "circuit.rzz(left, right, "
            "theta=rzz_angle(row, col, edge_index, config))"
        )
        second = (
            "circuit.rxx(left, right, "
            "theta=rxx_angle(row, col, edge_index, config))"
        )
        if rendered_source.count(first) != 1 or rendered_source.count(second) != 1:
            raise ValueError(
                f"{control_id} source drift: expected one RZZ and one RXX call"
            )
        placeholder = "__ORBITQ_SWAP_EDGE_GATE__"
        source = rendered_source.replace(first, placeholder)
        source = source.replace(
            second,
            "circuit.rzz(left, right, "
            "theta=rxx_angle(row, col, edge_index, config))",
        )
        source = source.replace(
            placeholder,
            "circuit.rxx(left, right, "
            "theta=rzz_angle(row, col, edge_index, config))",
        )
        match_count = expected_count = 2
    elif control_id == "incorrect_row_major_indexing":
        source = rendered_source
        replacements = (
            (
                "def build_circuit(config):",
                "def _wrong_index(row, col, n_side):\n"
                "    return (row * n_side + col + 1) % (n_side * n_side)\n\n\n"
                "def build_circuit(config):",
                1,
            ),
            (
                "qubit = row * n_side + col",
                "qubit = _wrong_index(row, col, n_side)",
                2,
            ),
            (
                "left = row * n_side + col",
                "left = _wrong_index(row, col, n_side)",
                2,
            ),
            (
                "right = left + 1",
                "right = _wrong_index(row, col + 1, n_side)",
                1,
            ),
            (
                "right = (row + 1) * n_side + col",
                "right = _wrong_index(row + 1, col, n_side)",
                1,
            ),
        )
        match_count = 0
        for old, new, count in replacements:
            source, matched = _replace_exact(
                source,
                old,
                new,
                count,
                control_id,
            )
            match_count += matched
        expected_count = sum(item[2] for item in replacements)
    elif control_id == "fabricated_samples":
        replacement = (
            "    return {\"samples\": np.zeros("
            "(config[\"n_samples\"], config[\"n_qubits\"]), dtype=np.int32)}"
        )
        source, match_count = _replace_exact(
            rendered_source,
            BOUNDED_VMAP_SAMPLING,
            replacement,
            1,
            control_id,
        )
    else:
        modulation_fields = {
            "ignore_row_modulation": (
                "ry_row_sin_scale",
                "rx_row_cos_scale",
            ),
            "ignore_column_modulation": (
                "ry_col_cos_scale",
                "rx_col_sin_scale",
            ),
            "ignore_diagonal_modulation": (
                "ry_diag_sin_scale",
                "rx_diag_cos_scale",
            ),
            "ignore_edge_modulation": (
                "rzz_edge_sin_scale",
                "rzz_site_cos_scale",
                "rxx_edge_cos_scale",
                "rxx_site_sin_scale",
            ),
        }
        if control_id not in modulation_fields:
            raise ValueError(f"unknown negative control: {control_id}")
        source, match_count = _replace_exact(
            rendered_source,
            "def build_circuit(config):\n    n_side",
            _config_override(modulation_fields[control_id]),
            1,
            control_id,
        )
    return {
        "control_id": control_id,
        "source": source,
        "match_count": match_count,
        "expected_match_count": expected_count,
        "transform_count": 1,
        "source_digest": sha256(source.encode("utf-8")).hexdigest(),
    }


def check_configuration_integrity_policy(source: str) -> dict[str, object]:
    """Reject candidate code that overwrites public spatial parameters."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "passed": False,
            "score": 0.0,
            "overridden_public_fields": [],
            "syntax_error": str(exc),
        }
    overridden: set[str] = set()
    protected = set(SPATIAL_SCALE_FIELDS)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            for key in node.args[0].keys:
                if isinstance(key, ast.Constant) and key.value in protected:
                    overridden.add(str(key.value))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            for target in targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "config"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value in protected
                ):
                    overridden.add(str(target.slice.value))
    fields = sorted(overridden)
    return {
        "passed": not fields,
        "score": 1.0 if not fields else 0.0,
        "overridden_public_fields": fields,
        "reason": (
            None
            if not fields
            else "public spatial modulation fields must not be overwritten"
        ),
    }


def evaluate_control_result(
    control_id: str,
    candidate: str,
    result: Mapping[str, object],
    applicability: Mapping[str, list[str]],
) -> dict[str, object]:
    if control_id not in _EXPECTED_REJECTION_GATE:
        raise ValueError(f"unknown negative control: {control_id}")
    if candidate not in applicability.get(control_id, []):
        return {
            "control_id": control_id,
            "candidate": candidate,
            "status": "not_applicable",
            "control_validated": True,
            "expected_rejection_gate": None,
        }
    expected_gate = _EXPECTED_REJECTION_GATE[control_id]
    validated = (
        result.get("launched") is True
        and result.get("normal_schema") is True
        and result.get("candidate_passed") is False
        and result.get("rejection_gate") == expected_gate
    )
    return {
        "control_id": control_id,
        "candidate": candidate,
        "status": "rejected_as_intended" if validated else "invalid_control_result",
        "control_validated": validated,
        "expected_rejection_gate": expected_gate,
        "actual_rejection_gate": result.get("rejection_gate"),
    }
