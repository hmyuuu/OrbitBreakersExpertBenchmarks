"""Container-side isolated execution for one matched-pair witness or mutant."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from .container_worker import evaluate_expert_samples
from .expert import generate_reference_values
from .loaders import load_candidate_solution, load_official_solution
from .matched_validation import check_configuration_integrity_policy


def _load_static_policy(path: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitq_challenge08_matched_static_policy",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load static policy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_matched_worker(
    repository_root: Path,
    source_path: Path,
    config_path: Path,
    sealed_path: Path,
    output_path: Path,
    *,
    run_id: str,
    allow_reference_generation: bool = False,
) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    source_bytes = source_path.read_bytes()
    solution = (
        load_official_solution(source_path)
        if allow_reference_generation
        else load_candidate_solution(source_path)
    )
    supports = sealed["hidden_supports"]
    references = sealed.get("expert_reference_values")
    reference_runtime = 0.0
    if references is None:
        if not allow_reference_generation:
            raise RuntimeError("sealed expert reference values are required")
        started = time.perf_counter()
        references = generate_reference_values(solution, config, supports)
        reference_runtime = time.perf_counter() - started

    started = time.perf_counter()
    result = solution.run_solution(config)
    samples = np.asarray(result["samples"])
    # The conversion above forces JAX completion before the sampling timer stops.
    sampling_runtime = time.perf_counter() - started

    evaluator_started = time.perf_counter()
    evaluator = evaluate_expert_samples(
        samples,
        supports,
        references,
        config,
    )
    evaluator_runtime = time.perf_counter() - evaluator_started
    static_policy_path = (
        repository_root / "tasks/challenge-08/tests/static_policy.py"
    )
    static = _load_static_policy(static_policy_path).check_source(
        source_path,
        "tensorcircuit",
    )
    config_integrity = check_configuration_integrity_policy(
        source_path.read_text(encoding="utf-8")
    )
    static["configuration_integrity_policy"] = config_integrity
    static["static_policy_score"] = min(
        float(static["static_policy_score"]),
        float(config_integrity["score"]),
    )
    payload = {
        "schema_version": "orbitq-challenge08-matched-worker-v2",
        "run_id": run_id,
        "candidate_id": config["candidate_id"],
        "config_digest": sha256(config_path.read_bytes()).hexdigest(),
        "sealed_digest": sha256(sealed_path.read_bytes()).hexdigest(),
        "rendered_source_digest": sha256(source_bytes).hexdigest(),
        "static_policy_digest": sha256(
            static_policy_path.read_bytes()
        ).hexdigest(),
        "sampling_runtime_sec": sampling_runtime,
        "evaluator_runtime_sec": evaluator_runtime,
        "reference_runtime_sec": reference_runtime,
        "references": references if allow_reference_generation else None,
        "evaluator": evaluator,
        "static_policy": static,
        "functional_score": 1.0 if evaluator["passed"] else 0.0,
        "static_policy_score": float(static["static_policy_score"]),
        "semantic_audit_status": "unavailable",
        "correctness_reward": None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-reference-generation", action="store_true")
    args = parser.parse_args()
    payload = run_matched_worker(
        args.repository_root,
        args.source,
        args.config,
        args.sealed,
        args.output,
        run_id=args.run_id,
        allow_reference_generation=args.allow_reference_generation,
    )
    print(
        json.dumps(
            {
                "run_id": payload["run_id"],
                "functional_score": payload["functional_score"],
                "static_policy_score": payload["static_policy_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
