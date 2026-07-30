from __future__ import annotations

import pytest


def test_matched_pair_has_no_ordering_requirement() -> None:
    from orbitq_boundary.matched import MatchedPairSpec

    pair = MatchedPairSpec(
        capability_id="spatially_indexed_2d_circuit_semantics",
        pair_id="challenge08-spatial-v2",
        base_candidate_id="challenge08-spatial-base",
        probe_candidate_id="challenge08-spatial-probe",
        sole_semantic_delta="spatial_modulation_mode",
        controlled_fields=("grid_side", "n_samples"),
        deterministically_derived_fields=("ry_row_sin_scale",),
    )
    serialized = pair.to_dict()
    assert serialized["sole_semantic_delta"] == "spatial_modulation_mode"
    assert "harder_direction" not in serialized
    assert pair.digest


def test_predeclared_independent_attempt_statistics() -> None:
    from orbitq_boundary.matched import summarize_matched_effect

    effect = summarize_matched_effect(5, 5, 2, 5)
    assert effect["attempt_design"] == "independent_attempts"
    assert effect["effect_estimate"] == pytest.approx(0.6)
    assert effect["effect_interval_method"] == (
        "Newcombe 95% CI based on Wilson score intervals"
    )
    assert effect["effect_interval"] == pytest.approx(
        [0.02978989079184169, 0.8823792257673521]
    )
    assert effect["hypothesis_test"] == "two-sided Fisher exact test"
    assert effect["p_value"] == pytest.approx(1.0 / 6.0)


def test_matched_statistics_reject_invalid_counts() -> None:
    from orbitq_boundary.matched import summarize_matched_effect

    with pytest.raises(ValueError, match="base_passed"):
        summarize_matched_effect(6, 5, 2, 5)
    with pytest.raises(ValueError, match="probe_total"):
        summarize_matched_effect(1, 5, 0, 0)


def test_failure_taxonomy_is_closed() -> None:
    from orbitq_boundary.matched import MatchedPairEvidence

    evidence = MatchedPairEvidence(
        pair_id="challenge08-spatial-v2",
        witness_digest="a" * 64,
        evaluator_digest="b" * 64,
        base_config_digest="c" * 64,
        probe_config_digest="d" * 64,
        expert_runtime_ratio=1.0,
        expert_peak_memory_ratio=None,
        verifier_parity=True,
        agent_fingerprint=None,
        base_outcomes=(),
        probe_outcomes=(),
        paired_effect=None,
        failure_taxonomy=("passed", "resource_failure"),
        readiness_status="layer2_ready",
    )
    assert evidence.to_dict()["failure_taxonomy"] == [
        "passed",
        "resource_failure",
    ]
    with pytest.raises(ValueError, match="failure taxonomy"):
        MatchedPairEvidence(
            pair_id="challenge08-spatial-v2",
            witness_digest="a" * 64,
            evaluator_digest="b" * 64,
            base_config_digest="c" * 64,
            probe_config_digest="d" * 64,
            expert_runtime_ratio=1.0,
            expert_peak_memory_ratio=None,
            verifier_parity=True,
            agent_fingerprint=None,
            base_outcomes=(),
            probe_outcomes=(),
            paired_effect=None,
            failure_taxonomy=("made_up_failure",),
            readiness_status="blocked",
        )
