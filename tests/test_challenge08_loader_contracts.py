from pathlib import Path

import pytest

from orbitq_boundary.pilots.challenge08.loaders import (
    load_candidate_solution,
    load_official_solution,
)
from orbitq_boundary.pilots.challenge08.matched_worker import run_matched_worker


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_candidate_and_official_loaders_accept_distinct_real_contracts(tmp_path):
    candidate = _write(
        tmp_path / "candidate.py",
        "def run_solution(config):\n    return {'samples': config['samples']}\n",
    )
    official = _write(
        tmp_path / "official.py",
        "K = object()\n"
        "def build_circuit(config):\n    return ('circuit', config)\n"
        "def run_solution(config):\n    return {'samples': config['samples']}\n",
    )

    assert load_candidate_solution(candidate).run_solution({"samples": [1]}) == {
        "samples": [1]
    }
    loaded = load_official_solution(official)
    assert loaded.build_circuit({"samples": []})[0] == "circuit"


def test_candidate_path_cannot_be_silently_used_as_official_path(tmp_path):
    candidate = _write(
        tmp_path / "candidate.py",
        "def run_solution(config):\n    return {'samples': []}\n",
    )

    with pytest.raises(RuntimeError, match="official solution lacks"):
        load_official_solution(candidate)


def test_matched_worker_uses_public_contract_for_candidate_evaluation(
    tmp_path, monkeypatch
):
    candidate = _write(
        tmp_path / "candidate.py",
        "def run_solution(config):\n    return {'samples': [[0, 1]]}\n",
    )
    config = _write(
        tmp_path / "config.json",
        '{"candidate_id": "probe"}\n',
    )
    sealed = _write(
        tmp_path / "sealed.json",
        '{"hidden_supports": {}, "expert_reference_values": {}}\n',
    )
    static_policy = tmp_path / "tasks/challenge-08/tests/static_policy.py"
    static_policy.parent.mkdir(parents=True)
    _write(
        static_policy,
        "def check_source(path, framework):\n"
        "    return {'static_policy_score': 1.0}\n",
    )
    monkeypatch.setattr(
        "orbitq_boundary.pilots.challenge08.matched_worker.evaluate_expert_samples",
        lambda *args: {"passed": True},
    )

    payload = run_matched_worker(
        tmp_path,
        candidate,
        config,
        sealed,
        tmp_path / "result.json",
        run_id="candidate-contract",
    )

    assert payload["functional_score"] == 1.0
    assert payload["static_policy_score"] == 1.0
