# Task 08 Autoresearch Campaign

Destination: `research/task-08/LOG.md`

Task: `task-08`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and setup

Selected task: `task-08` (49-qubit 7x7 mixed-axis grid sampling).

Base commit: `5af98f27b9404c513df8eee0f4568b1512edee19`.

Branch: `codex/orbitbreakers/task-08/extreme-native`.

Worktree:
`/Users/qqy/.codex/visualizations/2026/07/28/019fa982-7244-7e20-99f5-f609bdd0cf27/task08-extreme`.

The branch/worktree was created before the Task 08 survey files because the
user explicitly requested a new branch for the campaign. No candidate source
was edited before the survey and dataset gates were completed.

## Immutable expert bootstrap: canonical failure

Date: 2026-07-29

Reference SHA-256:
`0b0df74257e8f55d717ca29bb36e2edbb803206e9b2966fa423fefca9f15c311`.

Evaluator SHA-256:
`bffda6b012b07fd1cb8d5a1ec8a763bba4f1b967e5adc0d2b704e6ae99de9c41`.

Docker image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`).

Allocation: six CPUs, 7 GiB memory; timeout 300 seconds.

Result:

```text
workload: canonical n_samples=8192
terminal_status: RESOURCE_EXHAUSTED
valid: false
runtime_sec: none
failure_site: K.numpy(samples)
requested_buffer_bytes: 17998348288
```

Decision: `crash`. This is valid bootstrap evidence but not a runtime
baseline. Repeated canonical failures cannot support a speedup claim.

## Immutable expert scale probe

Date: 2026-07-29

Configuration difference: evaluator-supported `--n-samples 2048`; all circuit
parameters, seed, output semantics, thresholds, image and allocation
unchanged.

```text
terminal_status: SUCCESS
valid: true
runtime_sec: 48.113225
sample_shape: (2048, 49)
max_single_site_error: 0.0171194055
max_hidden_error: 0.0412643455
mean_hidden_error: 0.0101740381
```

Decision: `baseline screening probe`. Repeated paired runs are still required
before any performance claim.

## Independent exact-observable oracle

Script: `research/task-08/validate_exact_observables.py`.

Method: build the immutable expert circuit, then independently evaluate all 44
public `Z` strings with TensorCircuit `expectation_ps` and
`enable_lightcone=True`; do not call `perfect_sampling`.

Result:

```text
terminal_status: SUCCESS
passed: true
observable_count: 44
max_abs_error: 1.1976988650852505e-06
mean_abs_error: 2.878662166252735e-07
threshold: 2e-06
elapsed_sec: 11.031930867
```

The first oracle execution passed. Its immutable report is
`research/task-08/profiles/exact-observable-oracle.json`
(`sha256:871fe485ba5b168f09f8cb13a2885d76426f152f148503c4ab77961d7a567de7`).

## Experiment `e01`

Branch and fresh hypothesis worktree: pending.

### Hypothesis

The exact native rank-2 SVD split for all `RZZ` and `RXX` gates reduces
contraction width and peak mapped intermediates enough to make the canonical
8192-shot workload pass, while lowering the eligible 2048-shot runtime and
preserving all exact observables within `2e-6`.

### Pre-run frozen environment

Public dataset version: `orbitq-workloads-v20260729.4`.

Private evaluation used: `no`.

Reference/evaluator/image hashes: as recorded above.

Pair order: odd `reference -> candidate`, even
`candidate -> reference`; six pairs; 300-second cap.

### Result

Pending. Append results; never overwrite prior failure evidence.

## Append-only corrections

Append corrections below this heading. Never rewrite a result after it has
informed another experiment.
