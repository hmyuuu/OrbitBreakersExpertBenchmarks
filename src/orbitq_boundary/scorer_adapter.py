"""Fail-closed adapter for the repository's official verifier reward payload."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

from .models import AgentRunRecord, JsonRecord, JsonValue, canonical_json_bytes

SCORER_ADAPTER_VERSION = "orbitq-official-reward-adapter-v1"
_COMPONENT_KEYS = (
    "functional_score",
    "static_policy_score",
    "llm_audit_score",
)


@dataclass(frozen=True)
class ScorerBinding(JsonRecord):
    candidate_id: str
    candidate_digest: str
    agent_config_fingerprint: str
    image_digest: str
    hard_timeout_sec: float | None
    hard_timeout_source: str

    def __post_init__(self) -> None:
        for label, value in (
            ("candidate_id", self.candidate_id),
            ("candidate_digest", self.candidate_digest),
            ("agent_config_fingerprint", self.agent_config_fingerprint),
            ("image_digest", self.image_digest),
            ("hard_timeout_source", self.hard_timeout_source),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be resolved and non-empty")
        if self.hard_timeout_sec is not None and self.hard_timeout_sec <= 0:
            raise ValueError("hard_timeout_sec must be resolved and positive")


@dataclass(frozen=True)
class MigrationInspection(JsonRecord):
    template_matches_all_tasks: bool
    verifier_smoke_passed: bool
    template_runtime_is_reporting_only: bool
    docs_formula_consistent: bool
    blockers: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def structural_migration_passed(self) -> bool:
        """Whether the tracked scorer migration is complete, excluding smoke."""

        return (
            self.template_matches_all_tasks
            and self.template_runtime_is_reporting_only
            and self.docs_formula_consistent
        )


def _score(raw: Mapping[str, object], key: str) -> float:
    if key not in raw:
        raise ValueError(f"missing official scorer field: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ambiguous official scorer field: {key}")
    result = float(value)
    if result not in {0.0, 1.0}:
        raise ValueError(f"{key} must be a binary component score")
    return result


def _assert_optional_binding(
    raw: Mapping[str, object], key: str, expected: str, label: str
) -> None:
    declared = raw.get(key)
    if declared is not None and declared != expected:
        raise ValueError(f"{label} mismatch")


def _assert_optional_numeric_binding(
    raw: Mapping[str, object],
    key: str,
    expected: float | None,
    label: str,
) -> None:
    declared = raw.get(key)
    if declared is None:
        return
    if (
        isinstance(declared, bool)
        or not isinstance(declared, (int, float))
        or expected is None
        or float(declared) != expected
    ):
        raise ValueError(f"{label} mismatch")


def adapt_official_reward(
    raw: Mapping[str, object],
    *,
    binding: ScorerBinding,
    run_id: str,
    isolation_id: str,
    seed: int,
    artifact_paths: tuple[str, ...] = (),
) -> AgentRunRecord:
    """Translate component fields without trusting top-level reward/runtime_score."""

    _assert_optional_binding(
        raw, "candidate_digest", binding.candidate_digest, "candidate digest"
    )
    _assert_optional_binding(
        raw,
        "agent_config_fingerprint",
        binding.agent_config_fingerprint,
        "AgentConfig fingerprint",
    )
    _assert_optional_binding(
        raw, "image_digest", binding.image_digest, "image digest"
    )
    _assert_optional_numeric_binding(
        raw,
        "hard_timeout_sec",
        binding.hard_timeout_sec,
        "hard timeout",
    )
    _assert_optional_binding(
        raw,
        "hard_timeout_source",
        binding.hard_timeout_source,
        "hard timeout source",
    )
    components = {key: _score(raw, key) for key in _COMPONENT_KEYS}
    functional = components["functional_score"] == 1.0
    static = components["static_policy_score"] == 1.0
    semantic = components["llm_audit_score"] == 1.0
    correctness_reward = (
        components["functional_score"]
        * components["static_policy_score"]
        * components["llm_audit_score"]
    )
    correctness = correctness_reward == 1.0

    runtime_raw = raw.get("runtime_sec")
    runtime_missing = (
        isinstance(runtime_raw, bool)
        or not isinstance(runtime_raw, (int, float))
        or float(runtime_raw) < 0.0
    )
    runtime_sec = -1.0 if runtime_missing else float(runtime_raw)
    if binding.hard_timeout_sec is None:
        resource = None
        eligible = None
    else:
        resource = (
            False
            if runtime_missing
            else runtime_sec <= binding.hard_timeout_sec
        )
        eligible = correctness and resource

    if not functional:
        failure = "functional-failure"
    elif not static:
        failure = "static-policy-failure"
    elif not semantic:
        failure = "semantic-audit-failure"
    elif runtime_missing and binding.hard_timeout_sec is not None:
        failure = "missing-runtime-evidence"
    elif resource is False:
        failure = "resource-timeout"
    else:
        failure = None

    preserved = json.loads(canonical_json_bytes(dict(raw)))
    return AgentRunRecord(
        candidate_id=binding.candidate_id,
        run_id=run_id,
        agent_config_fingerprint=binding.agent_config_fingerprint,
        isolation_id=isolation_id,
        seed=seed,
        correctness_passed=correctness,
        resource_gate_passed=resource,
        failure_category=failure,
        runtime_sec=runtime_sec,
        artifact_paths=artifact_paths,
        notes="adapted from official component fields; top-level reward ignored",
        static_policy_passed=static,
        semantic_audit_passed=semantic,
        eligible_passed=eligible,
        candidate_digest=binding.candidate_digest,
        provenance={
            "scorer_adapter_version": SCORER_ADAPTER_VERSION,
            "image_digest": binding.image_digest,
            "hard_timeout_sec": binding.hard_timeout_sec,
            "hard_timeout_source": binding.hard_timeout_source,
            "correctness_reward": correctness_reward,
            "raw_official_reward": preserved,
        },
    )


def load_official_reward(
    path: str | Path,
    *,
    binding: ScorerBinding,
    run_id: str,
    isolation_id: str,
    seed: int,
) -> AgentRunRecord:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("official reward payload must be a JSON object")
    return adapt_official_reward(
        raw,
        binding=binding,
        run_id=run_id,
        isolation_id=isolation_id,
        seed=seed,
        artifact_paths=(str(source),),
    )


def inspect_official_migration(
    repository_root: str | Path,
    *,
    verifier_smoke_passed: bool,
) -> MigrationInspection:
    """Inspect non-execution migration gates against tracked source files."""

    root = Path(repository_root)
    template = root / "templates/challenge/tests/score_submission.py"
    template_bytes = template.read_bytes()
    task_paths = [
        root / f"tasks/challenge-{index:02d}/tests/score_submission.py"
        for index in range(1, 13)
    ]
    tasks_consistent = all(
        path.exists() and path.read_bytes() == template_bytes for path in task_paths
    )
    source = template_bytes.decode("utf-8", errors="replace")
    helper = re.search(
        r"def _correctness_reward\(.*?\n(?=def |\Z)",
        source,
        flags=re.S,
    )
    runtime_reporting_only = bool(
        helper
        and "runtime_score" not in helper.group(0)
        and 'rewards["reward"] = _correctness_reward(rewards)' in source
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    docs_consistent = (
        "functional_score * static_policy_score * llm_audit_score" in readme
        and "functional_score * static_policy_score * llm_audit_score" in agents
        and runtime_reporting_only
    )
    blockers: list[str] = []
    if not tasks_consistent:
        blockers.append("template-task-scorer-mismatch")
    if not verifier_smoke_passed:
        blockers.append("verifier-smoke-not-passed")
    if not runtime_reporting_only:
        blockers.append("template-runtime-reward-coupling")
    if not docs_consistent:
        blockers.append("reward-formula-inconsistent")
    return MigrationInspection(
        template_matches_all_tasks=tasks_consistent,
        verifier_smoke_passed=verifier_smoke_passed,
        template_runtime_is_reporting_only=runtime_reporting_only,
        docs_formula_consistent=docs_consistent,
        blockers=tuple(blockers),
    )
