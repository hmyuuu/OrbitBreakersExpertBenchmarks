from copy import deepcopy
from pathlib import Path

from orbitq_boundary.pilots.challenge08.matched_artifacts import (
    build_public_package,
    freeze_directory,
    scan_public_leakage,
    verify_frozen_directory,
)
from orbitq_boundary.pilots.challenge08.matched_pair import (
    BASE_ID,
    PROBE_ID,
    SPATIAL_SCALE_FIELDS,
    build_base_config,
    build_candidate_selection_matrix,
    build_negative_control_applicability,
    build_pair_spec,
    build_probe_config,
    compare_pair_artifacts,
    summarize_angle_nondegeneracy,
)


def test_fixed_base_probe_contract():
    base = build_base_config()
    probe = build_probe_config()

    assert base["candidate_id"] == BASE_ID
    assert probe["candidate_id"] == PROBE_ID
    assert base["grid_side"] == probe["grid_side"] == 7
    assert base["n_qubits"] == probe["n_qubits"] == 49
    assert base["n_samples"] == probe["n_samples"] == 8192
    assert base["seed"] == probe["seed"] == 2033
    assert all(base[name] == 0.0 for name in SPATIAL_SCALE_FIELDS)
    assert any(probe[name] != 0.0 for name in SPATIAL_SCALE_FIELDS)


def test_pair_has_exactly_one_declared_semantic_delta():
    base = build_base_config()
    probe = build_probe_config()

    assert compare_pair_artifacts(base, probe) == {
        "passed": True,
        "sole_semantic_delta": "spatial_modulation_mode",
        "unexpected_differences": [],
        "derived_differences": sorted(SPATIAL_SCALE_FIELDS),
    }
    spec = build_pair_spec()
    assert spec.sole_semantic_delta == "spatial_modulation_mode"
    assert set(spec.deterministically_derived_fields) == set(
        SPATIAL_SCALE_FIELDS
    )


def test_pair_comparison_fails_closed_on_an_unapproved_difference():
    base = build_base_config()
    probe = deepcopy(build_probe_config())
    probe["n_samples"] = 4096

    result = compare_pair_artifacts(base, probe)
    assert result["passed"] is False
    assert result["unexpected_differences"] == [
        "unexpected_difference:n_samples"
    ]


def test_base_is_uniform_and_probe_is_spatially_nondegenerate():
    base = summarize_angle_nondegeneracy(build_base_config())
    probe = summarize_angle_nondegeneracy(build_probe_config())

    assert base["all_gate_families_spatially_uniform"] is True
    assert probe["all_gate_families_spatially_nondegenerate"] is True
    assert base["unique_value_counts"] == {
        "ry": 1,
        "rzz": 1,
        "rxx": 1,
        "rx": 1,
    }
    assert all(value > 1 for value in probe["unique_value_counts"].values())


def test_negative_control_applicability_is_explicit():
    matrix = build_negative_control_applicability()
    both = {
        "missing_horizontal_bonds",
        "missing_vertical_bonds",
        "swapped_rzz_rxx_orientations",
        "incorrect_row_major_indexing",
        "fabricated_samples",
    }
    probe_only = {
        "ignore_row_modulation",
        "ignore_column_modulation",
        "ignore_diagonal_modulation",
        "ignore_edge_modulation",
    }

    assert all(matrix[name] == ["base", "probe"] for name in both)
    assert all(matrix[name] == ["probe"] for name in probe_only)


def test_candidate_selection_matrix_keeps_other_axes_separate():
    matrix = build_candidate_selection_matrix()
    assert [item["decision"] for item in matrix] == [
        "primary",
        "separate_control",
        "scale_stress_control",
    ]
    assert matrix[0]["capability"] == (
        "spatially_indexed_2d_circuit_semantics"
    )
    assert matrix[1]["axis"] == "api_guidance"
    assert matrix[2]["axis"] == "grid_side"


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_public_package_regeneration_is_byte_deterministic(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    build_public_package(left, BASE_ID, build_base_config())
    build_public_package(right, BASE_ID, build_base_config())

    assert _tree_bytes(left) == _tree_bytes(right)
    left_manifest = freeze_directory(left)
    right_manifest = freeze_directory(right)
    assert left_manifest == right_manifest
    assert verify_frozen_directory(left, left_manifest)["passed"] is True


def test_base_probe_public_differences_are_predeclared(tmp_path):
    base_root = tmp_path / "base"
    probe_root = tmp_path / "probe"
    build_public_package(base_root, BASE_ID, build_base_config())
    build_public_package(probe_root, PROBE_ID, build_probe_config())

    base_files = _tree_bytes(base_root)
    probe_files = _tree_bytes(probe_root)
    assert set(base_files) == set(probe_files)
    assert {
        name for name in base_files if base_files[name] != probe_files[name]
    } == {"instruction.md", "public_config.json"}


def test_public_task_uses_current_harbor_network_mode_symmetrically(tmp_path):
    base_root = tmp_path / "base"
    probe_root = tmp_path / "probe"
    build_public_package(base_root, BASE_ID, build_base_config())
    build_public_package(probe_root, PROBE_ID, build_probe_config())

    base_toml = (base_root / "task.toml").read_text(encoding="utf-8")
    probe_toml = (probe_root / "task.toml").read_text(encoding="utf-8")
    assert 'network_mode = "no-network"' in base_toml
    assert 'artifacts = ["/root/submission.py"]' in base_toml
    assert "evaluation_summary.json" not in base_toml
    assert base_toml == probe_toml


def test_manifest_rejects_missing_modified_and_unexpected_files(tmp_path):
    root = tmp_path / "public"
    build_public_package(root, BASE_ID, build_base_config())
    manifest = freeze_directory(root)

    (root / "instruction.md").write_text("changed", encoding="utf-8")
    assert "modified:instruction.md" in verify_frozen_directory(
        root, manifest
    )["reasons"]
    build_public_package(root, BASE_ID, build_base_config())
    (root / "extra.txt").write_text("extra", encoding="utf-8")
    assert "unexpected:extra.txt" in verify_frozen_directory(
        root, manifest
    )["reasons"]
    (root / "extra.txt").unlink()
    (root / "instruction.md").unlink()
    assert "missing:instruction.md" in verify_frozen_directory(
        root, manifest
    )["reasons"]


def test_public_leakage_scanner_rejects_predeclared_private_markers(tmp_path):
    root = tmp_path / "public"
    build_public_package(root, BASE_ID, build_base_config())
    assert scan_public_leakage(root)["passed"] is True

    for index, marker in enumerate(
        (
            "expert_reference",
            "hidden_supports",
            "solution_8.py",
            "negative_control",
        )
    ):
        leak = root / f"leak_{index}.txt"
        leak.write_text(marker, encoding="utf-8")
        report = scan_public_leakage(root)
        assert report["passed"] is False
        assert any(marker in reason for reason in report["reasons"])
        leak.unlink()
