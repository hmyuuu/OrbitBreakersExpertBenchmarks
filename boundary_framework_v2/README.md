# Challenge 08 Boundary Framework v2

## Overview

Boundary Framework v2 packages a matched BASE/PROBE benchmark for spatially
indexed two-dimensional TensorCircuit-NG circuit semantics. BASE uses uniform
gate parameters; PROBE changes only the declared spatial-modulation semantic.
This supports ORBIT-Q #79 by testing whether an implementation respects
topology, row-major indexing, gate orientation, sampling, and configuration
fields—not merely whether one numerical fixture happens to match.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/orbitq_boundary/` | Matched-pair models, loaders, package builder, and evaluator helpers |
| `boundary_framework_v2/public/` | Agent-visible BASE and PROBE tasks |
| `boundary_framework_v2/controls/` | Predeclared incorrect implementations |
| `tests/test_*boundary*` | Matched-pair and contract regression tests |

## Contracts

- Candidate/public: `run_solution(config)` returning the documented
  NumPy-compatible result mapping.
- Official/reference: `run_solution(config)`, `build_circuit(config)`, and the
  configured backend `K`.
- The candidate loader never requires or supplies official-only symbols. The
  official loader fails closed when they are missing.
- Public packages contain instructions, configuration, smoke tests, and the
  submission interface. Reference values and private/sealed evaluator data are
  not included.

## Requirements

Python 3.11+ is required. Full numerical calibration uses the repository's
TensorCircuit-NG/JAX Docker environment; the contract and package tests run on
CPU without Docker. The public workload fixes `grid_side=7`,
`n_samples=8192`, and seed `2033`. Full sampling can take several minutes and
requires substantially more memory than the unit tests.

## Quick start

From a clean clone:

```bash
python -m pip install -e . pytest
python -m pytest tests/test_boundary_matched_pair_v2.py \
  tests/test_challenge08_matched_pair_v2.py \
  tests/test_challenge08_loader_contracts.py -q
```

Expected result: all contract, matched-pair, public-package, leakage, and
loader-separation tests pass.

Public smoke tests can be run independently:

```bash
python boundary_framework_v2/public/base/tests/public_smoke.py \
  boundary_framework_v2/public/submission_template.py
python boundary_framework_v2/public/probe/tests/public_smoke.py \
  boundary_framework_v2/public/submission_template.py
```

Expected result: each command exits zero after confirming that the submission
parses and declares the public `run_solution` entry point. Runtime output
semantics are checked by the private evaluator, not by this public smoke test.

## Reproducing the reported result

The shortest maintained check is the three-file pytest command above. It uses
the fixed BASE/PROBE configuration and seed, checks the sole semantic delta,
builds frozen public packages, scans them for private leakage, and exercises
candidate and official loaders separately. Test failure is `FAIL`; a test
timeout is `TIMEOUT`; inability to import or start the environment is
`INFRASTRUCTURE_INVALID`. Structured full-run evaluator records additionally
retain functional, static-policy, resource, and failure-stage fields.

## Adding a new boundary task

1. Define the public `run_solution(config)` contract.
2. Add a role-separated official implementation.
3. Declare positive and negative controls before measurement.
4. Add a structured evaluator without exposing expected values.
5. Add a public smoke test.
6. Run calibration, leakage, loader-separation, and reproducibility checks.

## Validation status and limitations

The matched BASE/PROBE specification, public-package construction, leakage
scan, predeclared controls, and loader separation are covered by repository
tests. This PR does not include Agent attempts, private/sealed data, or claim
an Agent private-evaluator pass. Full Docker calibration is environment- and
resource-dependent and is not implied by the unit-test result. Evaluator-only
helpers under `src/orbitq_boundary/` must remain outside the Agent-mounted
public package because they deterministically construct hidden supports.
