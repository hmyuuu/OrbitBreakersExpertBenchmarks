"""Deterministic infrastructure-only demonstration of the boundary framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from .comparison import assert_controlled_mutation, require_expert_validation
from .manifest import build_manifest, write_manifest
from .models import (
    AgentConfig,
    AgentRunRecord,
    BoundaryEvidence,
    BoundarySearchState,
    ClassificationThresholds,
    EvidenceValue,
    ExpertValidationRecord,
    canonical_json_bytes,
    content_digest,
)
from .specs import generate_candidate, load_family_spec
from .statistics import classify_candidate

SYNTHETIC_LIMITATION = "infrastructure validation only"
SYNTHETIC_CREATED_AT = "2026-07-29T00:00:00Z"


def _recovered(value):
    return EvidenceValue(
        status="recovered",
        value=value,
        source="orbitq_boundary.synthetic",
        limitation=None,
    )


def _synthetic_agent_config() -> AgentConfig:
    return AgentConfig(
        config_id="A_synth",
        model=_recovered("deterministic-mock-agent"),
        harness=_recovered("in-process-mock"),
        tools=_recovered([]),
        prompt=_recovered("none"),
        docs=_recovered("none"),
        budget=_recovered(
            {
                "checkpoints": [5, 16],
                "max_runs": 64,
                "real_agent_budget": False,
            }
        ),
        framework=_recovered("none"),
        framework_version=_recovered("not-applicable"),
    )


def _expert_record(candidate, seed: int) -> ExpertValidationRecord:
    return ExpertValidationRecord(
        candidate_id=candidate.candidate_id,
        validation_id=f"expert-{candidate.candidate_id}",
        seed=seed,
        functional_passed=True,
        static_policy_passed=True,
        semantic_audit_passed=True,
        resource_budget_passed=True,
        runtime_sec=0.0,
        evidence_paths=(
            f"synthetic://expert/{candidate.candidate_id}",
        ),
        notes="deterministic mock expert passed every validation gate",
    )


def _agent_runs(
    candidate,
    fingerprint: str,
    outcomes: tuple[bool, ...],
    seed: int,
) -> tuple[AgentRunRecord, ...]:
    records = []
    for run_index, passed in enumerate(outcomes):
        identity = {
            "candidate_id": candidate.candidate_id,
            "run_index": run_index,
            "seed": seed,
        }
        run_digest = content_digest(identity)[:16]
        records.append(
            AgentRunRecord(
                candidate_id=candidate.candidate_id,
                run_id=f"run-{run_digest}",
                agent_config_fingerprint=fingerprint,
                isolation_id=f"isolation-{run_digest}",
                seed=seed + run_index,
                correctness_passed=passed,
                resource_gate_passed=True,
                failure_category=None if passed else "mock-capability-failure",
                runtime_sec=0.0,
                artifact_paths=(
                    f"synthetic://runs/{candidate.candidate_id}/{run_index}",
                ),
                notes="deterministic mock Agent outcome",
            )
        )
    return tuple(records)


def run_synthetic_demo(
    spec_path: str | Path,
    output_path: str | Path,
    seed: int,
) -> dict:
    """Run and save the deterministic infrastructure demonstration."""

    family = load_family_spec(spec_path)
    agent_config = _synthetic_agent_config()
    thresholds = ClassificationThresholds()

    easy = generate_candidate(
        family,
        {"reasoning_depth": 1, "api_hint": "absent"},
        seed,
    )
    transition = generate_candidate(
        family,
        {"reasoning_depth": 2, "api_hint": "absent"},
        seed,
        parent=easy,
        changed_parameter="reasoning_depth",
    )
    hard = generate_candidate(
        family,
        {"reasoning_depth": 3, "api_hint": "absent"},
        seed,
        parent=transition,
        changed_parameter="reasoning_depth",
    )
    control = generate_candidate(
        family,
        {"reasoning_depth": 3, "api_hint": "present"},
        seed,
        parent=hard,
        changed_parameter="api_hint",
    )
    candidates = (easy, transition, hard, control)
    for parent, child in zip(candidates, candidates[1:]):
        assert_controlled_mutation(family, parent, child)

    expert_records = tuple(
        _expert_record(candidate, seed + index)
        for index, candidate in enumerate(candidates)
    )
    transition_outcomes = (
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    outcome_profiles = (
        (True,) * 16,
        transition_outcomes,
        (False,) * 16,
        (True,) * 16,
    )
    run_groups = tuple(
        _agent_runs(
            candidate,
            agent_config.fingerprint,
            outcomes,
            seed + 100 * index,
        )
        for index, (candidate, outcomes) in enumerate(
            zip(candidates, outcome_profiles)
        )
    )
    runs = tuple(record for group in run_groups for record in group)
    screening_verdicts = tuple(
        classify_candidate(
            candidate,
            require_expert_validation(candidate, expert_records),
            group[: thresholds.screening_checkpoint],
            thresholds,
        )
        for candidate, group in zip(candidates, run_groups)
    )
    formal_verdicts = tuple(
        classify_candidate(
            candidate,
            require_expert_validation(candidate, expert_records),
            group,
            thresholds,
        )
        for candidate, group in zip(candidates, run_groups)
    )

    state = BoundarySearchState(
        family_id=family.family_id,
        agent_config_fingerprint=agent_config.fingerprint,
        axis_order=tuple(parameter.name for parameter in family.parameters),
        axis_index=1,
        current_candidate_id=control.candidate_id,
        visited_parameter_vectors=tuple(
            canonical_json_bytes(candidate.parameters).decode("utf-8")
            for candidate in candidates
        ),
        tested_candidate_ids=tuple(
            candidate.candidate_id for candidate in candidates
        ),
        non_monotonic_observations=(),
        status="synthetic-complete",
    )
    evidence = BoundaryEvidence(
        family_id=family.family_id,
        agent_config_fingerprint=agent_config.fingerprint,
        screening_easy_candidate_ids=(
            easy.candidate_id,
            control.candidate_id,
        ),
        screening_hard_candidate_ids=(hard.candidate_id,),
        screening_transition_candidate_ids=(transition.candidate_id,),
        formal_easy_candidate_ids=(
            easy.candidate_id,
            control.candidate_id,
        ),
        formal_hard_candidate_ids=(hard.candidate_id,),
        formal_unresolved_candidate_ids=(transition.candidate_id,),
        matched_control_candidate_id=control.candidate_id,
        screening_verdict_ids=tuple(
            verdict.verdict_id for verdict in screening_verdicts
        ),
        formal_verdict_ids=tuple(
            verdict.verdict_id for verdict in formal_verdicts
        ),
        expert_validation_ids=tuple(
            record.validation_id for record in expert_records
        ),
        manifest_digest=None,
        limitations=(SYNTHETIC_LIMITATION,),
    )
    manifest = build_manifest(
        created_at=SYNTHETIC_CREATED_AT,
        synthetic=True,
        seed=seed,
        agent_config=agent_config,
        family=family,
        thresholds=thresholds,
        candidates=candidates,
        expert_validations=expert_records,
        agent_runs=runs,
        screening_verdicts=screening_verdicts,
        formal_verdicts=formal_verdicts,
        search_state=state,
        boundary_evidence=evidence,
        environment_evidence={
            "execution": "in-process deterministic mock",
            "quantum_framework": "not applicable",
            "scientific_result": False,
        },
        limitations=(SYNTHETIC_LIMITATION,),
    )
    write_manifest(output_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic ORBIT-Q boundary MVP demo."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("examples/problem1/synthetic_family.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/problem1/synthetic_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = run_synthetic_demo(args.spec, args.output, args.seed)
    screening_labels = sorted(
        {
            verdict["screening_classification"]
            for verdict in manifest["screening_verdicts"]
        }
    )
    endpoint_labels = sorted(
        {
            verdict["endpoint_classification"]
            for verdict in manifest["formal_verdicts"]
        }
    )
    print(
        "合成演示完成："
        "checkpoints=5,16; "
        f"screening={','.join(screening_labels)}; "
        f"endpoints={','.join(endpoint_labels)}; "
        "matched_control=yes; "
        f"limitation={SYNTHETIC_LIMITATION}; "
        f"manifest_digest={manifest['manifest_digest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
