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

## Experiment `e01` diagnostic result

Candidate commit: `794172f`.

Candidate SHA-256:
`1ed6ec65bb0402078307a4ade8e0582381dd659089b82c496c1b3a98a4946eb2`.

Diff SHA-256:
`65ceec453bcc082d4e05cbc1ce3a9aa694c5bfd51d40f400a333a0b24221818d`.

Sanitized record:
`profiles/e01-rank2-split-diagnostic.json`.

The exploratory diagnostic was run before the hypothesis commit, so it is
explicitly ineligible for a formal performance claim. It is retained because
it decisively falsifies the idea and prevents an expensive 2048-shot run.

```text
exact_observable_reference_elapsed_sec: 11.031931
exact_observable_candidate_elapsed_sec: 75.651599
candidate_max_abs_error: 8.5601921e-06
predeclared_max_abs_error: 2e-06
256_shot_reference_runtime_sec: 24.601082
256_shot_candidate_runtime_sec: 30.307426
candidate/reference_runtime_ratio: 1.231955
sample_arrays_equal: true
```

Decision: `discard`. TensorCircuit's generic SVD split doubles the entangler
node count, worsens OMECo path/contraction cost, and misses the strict exact
observable threshold. Do not benchmark it at 2048 or 8192 shots unchanged.

Next pivot: `e02`, bounded contiguous shot chunks on the original dense-gate
network.

## Experiment `e02`: 512-shot chunks

Branch: `codex/orbitbreakers/task-08/e02-shot-chunks`.

Candidate commit: `673b3b3`.

Candidate SHA-256:
`60f23741c1fcbc13ae67884b3c6f94ad939548056820bbace2b73da7c9a27e03`.

Diff SHA-256:
`90314a596f75ebc26d11bd5fdab9f3a2d18e71741fd632b327931be36c785f81`.

Hypothesis: execute the original
`K.jit(K.vmap(circuit.perfect_sampling))` on contiguous 512-row slices of the
already generated status matrix and concatenate the host arrays. This bounds
the mapped-axis intermediate memory without changing the circuit, each
conditional trajectory, random numbers, sample order, or framework path.

The candidate was committed before both screens. Sanitized record:
`profiles/e02-chunk512-screen.json`.

```text
2048_shot_reference_initial_runtime_sec: 48.113225
2048_shot_candidate_runtime_sec: 31.478739
2048_shot_screen_speedup: 1.528443
2048_shot_candidate_valid: true
observable_metrics_identical_to_reference: true

8192_shot_reference_status: RESOURCE_EXHAUSTED
8192_shot_candidate_runtime_sec: 50.843383
8192_shot_candidate_valid: true
sample_shape: (8192, 49)
max_single_z_error: 0.0046486279
max_hidden_error: 0.0188068181
mean_hidden_error: 0.0039865117
```

Decision: `keep provisionally`. The full result is an OOM-to-PASS feasibility
result, not a speedup. The 2048-shot number is a single non-paired screen and
cannot support a performance claim. Predeclared next pivot: test smaller and
larger chunk sizes in fresh worktrees, then run formal pairs only for the
frozen winner.
