"""Container-side real expert generation and family evaluator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from .expert import generate_reference_values, load_official_solution

OBSOLETE_CONTRACTOR = 'tc.set_contractor("omeco-4-4")'
SUPPORTED_CONTRACTOR = (
    "import omeco\n\n"
    'tc.set_contractor("custom", '
    "optimizer=omeco.TreeSA(ntrials=4, niters=4), preprocessing=True)"
)
GREEDY_CONTRACTOR = 'tc.set_contractor("greedy", preprocessing=True)'
MEMORY_LIMITED_GREEDY_CONTRACTOR = (
    "import opt_einsum\n\n"
    "def _memory_greedy(input_sets, output_set, size_dict, "
    "memory_limit=None, **_kws):\n"
    "    return opt_einsum.paths.greedy(\n"
    "        input_sets, frozenset(output_set), size_dict, memory_limit\n"
    "    )\n\n"
    'tc.set_contractor("custom", optimizer=_memory_greedy, '
    "memory_limit=2**26, preprocessing=True)"
)
UNBOUNDED_VMAP_SAMPLING = (
    "    sample_batch = K.jit(K.vmap(sample_one))\n"
    "    samples = sample_batch(status)\n"
    '    return {"samples": np.asarray(K.numpy(samples), dtype=np.int32)}'
)
BOUNDED_VMAP_SAMPLING = (
    "    sample_batch = K.jit(K.vmap(sample_one))\n"
    "    sampling_batch_size = 256\n"
    "    sample_chunks = []\n"
    '    for start in range(0, config["n_samples"], sampling_batch_size):\n'
    "        stop = start + sampling_batch_size\n"
    "        sample_chunks.append(K.numpy(sample_batch(status[start:stop])))\n"
    "    samples = np.concatenate(sample_chunks, axis=0)\n"
    '    return {"samples": np.asarray(samples, dtype=np.int32)}'
)


def render_compatible_expert_source(
    source: str,
    *,
    contractor_mode: str = "omeco",
) -> str:
    """Use the pinned TC-NG documented equivalent for the obsolete alias."""

    if source.count(OBSOLETE_CONTRACTOR) != 1:
        raise ValueError("canonical expert contractor alias changed unexpectedly")
    replacements = {
        "omeco": SUPPORTED_CONTRACTOR,
        "greedy": GREEDY_CONTRACTOR,
        "greedy-memory": MEMORY_LIMITED_GREEDY_CONTRACTOR,
    }
    if contractor_mode not in replacements:
        raise ValueError(f"unsupported contractor mode: {contractor_mode}")
    rendered = source.replace(
        OBSOLETE_CONTRACTOR,
        replacements[contractor_mode],
    )
    if rendered.count(UNBOUNDED_VMAP_SAMPLING) != 1:
        raise ValueError("canonical expert sampling block changed unexpectedly")
    return rendered.replace(
        UNBOUNDED_VMAP_SAMPLING,
        BOUNDED_VMAP_SAMPLING,
    )


def _flatten(
    values: Mapping[str, object],
    category_order: tuple[str, ...],
) -> list[object]:
    flattened: list[object] = []
    for category in category_order:
        category_values = values.get(category)
        if not isinstance(category_values, (list, tuple)):
            raise ValueError(f"missing category values: {category}")
        flattened.extend(category_values)
    return flattened


def evaluate_expert_samples(
    samples: np.ndarray,
    supports: Mapping[str, object],
    references: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    """Apply the canonical Challenge 08 finite-sample criteria."""

    category_order = (
        "single_site",
        "nearest_neighbor",
        "patch_2x2",
        "long_range",
    )
    ordered_supports = [
        tuple(int(index) for index in support)
        for support in _flatten(supports, category_order)
    ]
    ordered_references = np.asarray(
        _flatten(references, category_order),
        dtype=np.float64,
    )
    array = np.asarray(samples)
    expected_shape = (
        int(config["n_samples"]),
        int(config["n_qubits"]),
    )
    shape_ok = array.shape == expected_shape
    binary_ok = bool(np.all((array == 0) | (array == 1)))
    finite_ok = bool(np.all(np.isfinite(array)))
    if array.ndim == 2 and ordered_supports:
        z_samples = 1.0 - 2.0 * array.astype(np.float64)
        empirical = np.asarray(
            [
                np.mean(np.prod(z_samples[:, support], axis=1))
                for support in ordered_supports
            ],
            dtype=np.float64,
        )
    else:
        empirical = np.full(len(ordered_supports), np.nan)
    reference_shape_ok = empirical.shape == ordered_references.shape
    if reference_shape_ok:
        errors = np.abs(empirical - ordered_references)
    else:
        errors = np.asarray([np.inf])
    single_count = len(supports.get("single_site", ()))
    single_error = (
        float(np.max(errors[:single_count])) if single_count else float("inf")
    )
    hidden_max_error = float(np.max(errors)) if errors.size else float("inf")
    hidden_mean_error = float(np.mean(errors)) if errors.size else float("inf")
    criteria = {
        "sample_shape": shape_ok,
        "samples_binary": binary_ok,
        "samples_finite": finite_ok,
        "reference_shape": reference_shape_ok,
        "single_z_tolerance": (
            single_error <= float(config["single_z_tolerance"])
        ),
        "hidden_max_tolerance": (
            hidden_max_error
            <= float(config["hidden_z_string_max_tolerance"])
        ),
        "hidden_mean_tolerance": (
            hidden_mean_error
            <= float(config["hidden_z_string_mean_tolerance"])
        ),
    }
    passed = all(criteria.values())
    if not shape_ok or not binary_ok or not finite_ok:
        rejection_component = "functional-schema"
    elif not reference_shape_ok:
        rejection_component = "sealed-reference-binding"
    elif not passed:
        rejection_component = "functional-tolerance"
    else:
        rejection_component = None
    return {
        "passed": passed,
        "rejection_component": rejection_component,
        "criteria": criteria,
        "sample_shape": list(array.shape),
        "hidden_observable_count": len(ordered_supports),
        "single_z_error": single_error,
        "hidden_max_error": hidden_max_error,
        "hidden_mean_error": hidden_mean_error,
        "empirical_values": empirical.tolist(),
    }


def _load_static_policy(path: Path):
    spec = importlib.util.spec_from_file_location(
        "orbitq_challenge08_static_policy",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load static policy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker(
    repository_root: Path,
    public_path: Path,
    sealed_path: Path,
    output_path: Path,
    *,
    run_id: str,
    contractor_mode: str = "omeco",
) -> dict[str, object]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(".progress.jsonl")
    progress_path.write_text("", encoding="utf-8")

    def progress(event: str, **details: object) -> None:
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"event": event, **details},
                    sort_keys=True,
                )
                + "\n"
            )

    progress("worker-start", run_id=run_id)
    canonical_source_path = (
        repository_root / "tasks/challenge-08/solution/solution_8.py"
    )
    canonical_source = canonical_source_path.read_text(encoding="utf-8")
    compatible_source = render_compatible_expert_source(
        canonical_source,
        contractor_mode=contractor_mode,
    )
    compatible_path = output_path.parent / f"solution_8_{run_id}.py"
    compatible_path.write_text(compatible_source, encoding="utf-8")
    progress("compatible-source-written", path=str(compatible_path))
    solution = load_official_solution(compatible_path)
    config = public["config"]
    supports = sealed["hidden_supports"]

    reference_started = time.perf_counter()
    progress("reference-start")
    references = generate_reference_values(
        solution,
        config,
        supports,
        progress=lambda category, completed, total: progress(
            "reference-support-complete",
            category=category,
            completed=completed,
            total=total,
        ),
    )
    reference_runtime = time.perf_counter() - reference_started
    progress("reference-complete", runtime_sec=reference_runtime)
    sample_started = time.perf_counter()
    progress("sampling-start")
    results = solution.run_solution(config)
    sample_runtime = time.perf_counter() - sample_started
    progress("sampling-complete", runtime_sec=sample_runtime)
    evaluator = evaluate_expert_samples(
        np.asarray(results["samples"]),
        supports,
        references,
        config,
    )
    static_module = _load_static_policy(
        repository_root / "tasks/challenge-08/tests/static_policy.py"
    )
    static = static_module.check_source(compatible_path, "tensorcircuit")
    progress(
        "validation-complete",
        evaluator_passed=evaluator["passed"],
        static_policy_score=static["static_policy_score"],
    )
    payload = {
        "schema_version": "orbitq-challenge08-expert-run-v1",
        "run_id": run_id,
        "candidate_id": sealed["candidate_id"],
        "candidate_digest": sealed["candidate_digest"],
        "sealed_instance_digest": sealed["sealed_instance_digest"],
        "scientific_contract_digest": public["candidate"][
            "scientific_contract_digest"
        ],
        "generation_seed": sealed["hidden_generator"]["seed"],
        "sampling_seed": 2033,
        "expert_source_digest": sha256(
            compatible_source.encode("utf-8")
        ).hexdigest(),
        "canonical_expert_source_digest": sha256(
            canonical_source.encode("utf-8")
        ).hexdigest(),
        "contractor_compatibility": {
            "mode": contractor_mode,
            "obsolete": OBSOLETE_CONTRACTOR,
            "replacement": (
                SUPPORTED_CONTRACTOR
                if contractor_mode == "omeco"
                else (
                    GREEDY_CONTRACTOR
                    if contractor_mode == "greedy"
                    else MEMORY_LIMITED_GREEDY_CONTRACTOR
                )
            ),
        },
        "sampling_compatibility": {
            "method": "bounded-vmap-perfect-sampling",
            "batch_size": 256,
            "n_samples_unchanged": True,
            "sampling_seed_unchanged": True,
        },
        "family_evaluator_source_digest": sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "official_evaluator_source_digest": sha256(
            (
                repository_root
                / "tasks/challenge-08/tests/evaluate_8.py"
            ).read_bytes()
        ).hexdigest(),
        "static_policy_source_digest": sha256(
            (
                repository_root
                / "tasks/challenge-08/tests/static_policy.py"
            ).read_bytes()
        ).hexdigest(),
        "references": references,
        "reference_runtime_sec": reference_runtime,
        "sample_runtime_sec": sample_runtime,
        "evaluator": evaluator,
        "static_policy": static,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    progress("worker-complete", output=str(output_path))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--contractor-mode",
        choices=("omeco", "greedy", "greedy-memory"),
        default="omeco",
    )
    args = parser.parse_args()
    payload = run_worker(
        args.repository_root,
        args.public,
        args.sealed,
        args.output,
        run_id=args.run_id,
        contractor_mode=args.contractor_mode,
    )
    print(
        json.dumps(
            {
                "candidate_id": payload["candidate_id"],
                "evaluator_passed": payload["evaluator"]["passed"],
                "static_policy_score": payload["static_policy"][
                    "static_policy_score"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if payload["evaluator"]["passed"]
        and payload["static_policy"]["static_policy_score"] == 1.0
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
