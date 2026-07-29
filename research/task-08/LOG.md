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

## Experiment `e03`: 256-shot chunks

Branch: `codex/orbitbreakers/task-08/e03-chunk256`.

Candidate commit: `41713dd`.

Candidate SHA-256:
`7696f4d742d07da92a06cf5bdd4634f26ca5fe9251471163a90a4b6da280b45d`.

Diff SHA-256:
`848a3f800555f43d5e0489e1d27e5c6800ca02824de5a1859ee9fa8809b392bb`.

The candidate was committed before the canonical screen. Sanitized record:
`profiles/e03-chunk256-screen.json`.

```text
8192_shot_runtime_sec: 44.028239
valid: true
sample_shape: (8192, 49)
max_single_z_error: 0.0046682336
max_hidden_error: 0.0188068181
mean_hidden_error: 0.0039754143
chunk512_screen_runtime_sec: 50.843383
screen_improvement_over_chunk512: 13.4042%
```

Decision: `keep provisionally`. This one-run comparison selects the direction
for another isolated chunk-size experiment; it is not a formal speedup claim.
Different mapped batch shapes change complex64 rounding enough to flip a
small number of threshold-adjacent draws, so sample arrays need not be
byte-identical. The fixed circuit, complete status matrix, exact conditional
algorithm and statistical output meaning are preserved, and every functional
check passes comfortably.

## Experiment `e04`: 128-shot chunks

Branch: `codex/orbitbreakers/task-08/e04-chunk128`.

Candidate commit: `d7bd87d`.

Candidate SHA-256:
`5caecfd033af167849503c3cbc6bfa918208ff0344b72c7a58bdb82c9e02485e`.

Diff SHA-256:
`104366a653faaf2341991981fe60bc32b4bb0bfe1d750dcaa4ad5b94ddb2dc46`.

The candidate was committed before the canonical screen. Sanitized record:
`profiles/e04-chunk128-screen.json`.

```text
8192_shot_runtime_sec: 55.182176
valid: true
chunk256_screen_runtime_sec: 44.028239
regression_vs_chunk256: 25.3336%
```

Decision: `discard`. The extra 32 Python dispatch/synchronization boundaries
outweigh the smaller mapped graph. The measured optimum among 128/256/512 is
256. Next pivot: retain 256-shot memory bounding but stage all chunks with the
TensorCircuit-native `K.jaxy_scan` wrapper to remove host dispatch.

## Experiment `e05`: TensorCircuit `K.jaxy_scan` over 256-shot chunks

Branch: `codex/orbitbreakers/task-08/e05-scan256`.

Candidate commit: `c07c660`.

Candidate SHA-256:
`51c55e16ce6021b4ac3ff3da0387038beb347dd0b4105d91b416953d5b1bcb87`.

Diff SHA-256:
`6aba49e302f4e0bb2f6698b95d308fdb9d503f7b21d2af9eaa7f601e7dd30c52`.

The candidate was committed before the canonical screen. Sanitized record:
`profiles/e05-scan256-screen.json`.

```text
8192_shot_runtime_sec: 67.400915
valid: true
python_chunk256_runtime_sec: 44.028239
regression_vs_python_chunk256: 53.0855%
```

Decision: `discard`. Although `K.jaxy_scan` removes 31 Python dispatches, it
places the large mapped 49-measurement conditional-contraction body inside XLA
control flow. Compilation/optimization and loop execution cost dominate the
saved dispatch. Keep the simpler Python loop with one cached `K.jit` per
256-shot block.

## Experiment `e06`: OMECo 1x1 search budget

Branch: `codex/orbitbreakers/task-08/e06-omeco1x1`.

Parent candidate: Python-loop 256-shot chunks.

Candidate commit: `74b4be2`.

Candidate SHA-256:
`861b610773d33f4f04cbf5422aa7b71c419105811c991f7bcd5ccd36220f050b`.

Diff SHA-256:
`e9c000f299a1585021c87ba0c1ebe07b02567e1c747e3d84fb994488e5bfc321`.

The candidate was committed before the canonical screen. Sanitized record:
`profiles/e06-omeco1x1-screen.json`.

```text
8192_shot_runtime_sec: 131.958540
valid: true
omeco4x4_runtime_sec: 44.028239
regression_vs_omeco4x4: 199.7132%
```

Decision: `discard`. Reducing 16 TreeSA trial/iteration combinations to one
does reduce search work, but it selects contraction paths whose repeated
execution across 32 chunks is nearly 3x slower end to end. Preserve the
expert's `omeco-4-4` budget.

## Experiment `e07`: fuse commuting final RX gates

Branch: `codex/orbitbreakers/task-08/e07-fuse-final-rx`.

Parent candidate: Python-loop 256-shot chunks with OMECo 4x4.

Candidate commit: `caa0f2b`.

Candidate SHA-256:
`3eef7ec94a6b4268cc6806d1e1a2327b255cf40393f13b5f5bfd0169d91be09b`.

Diff SHA-256:
`b08233efdfcabdb6e930dc00451400b5c76163ad6bfb9336bdec81a7ad134b6c`.

The candidate was committed before both validations. Sanitized record:
`profiles/e07-fuse-final-rx-screen.json`.

The exact transformation uses `[RX_i tensor RX_j] RXX_ij` on 21 disjoint
vertical edges. Since every `RX_i` commutes with every vertical `RXX`, this
absorbs 42 final single-qubit nodes without changing the circuit unitary;
seven row-6 RX nodes remain.

```text
exact_observables_passed: 44/44
exact_max_abs_error: 1.0180551e-06
exact_elapsed_sec: 9.617958
reference_exact_elapsed_sec: 11.031931

8192_shot_runtime_sec: 60.989109
valid: true
unfused_chunk256_runtime_sec: 44.028239
regression_vs_unfused: 38.5222%
```

Decision: `discard`. Explicit fusion helps light-cone expectation
contractions but produces worse paths/code for the 49 repeated conditional
sampling contractions. Keep the original TensorCircuit RX/RXX nodes.

## Final accepted implementation

Candidate commit on campaign branch: `9da76f1`.

Candidate SHA-256:
`7696f4d742d07da92a06cf5bdd4634f26ca5fe9251471163a90a4b6da280b45d`.

Implementation: original TensorCircuit network and OMECo 4x4 contractor;
pre-generate the unchanged seed-2033 status matrix; evaluate the cached
`K.jit(K.vmap(perfect_sampling))` on contiguous 256-shot blocks; copy each
small output block to NumPy and concatenate in original order.

### Final canonical five-pair session

Command:

```bash
python3 research/task-08/run_docker_matrix.py \
  --mode paired --n-samples 8192 --repeat 5 \
  --timeout 300 --cpus 6 --memory 7g \
  --output results/task-08-final-canonical-5-pairs
```

Raw report SHA-256:
`02319130b267855985eb60d4e86bcc5523f1edb6df62c7695a4667ef9227148f`.

Tracked sanitized record:
`profiles/final-canonical-8192-five-pairs.json`.

```text
reference: 0/5 PASS; runtime_sec unavailable in every cell
reference terminal states: RESOURCE_EXHAUSTED/exit 137 x 5
candidate: 5/5 PASS
candidate runtimes_sec:
  40.537079, 51.583744, 55.098628, 45.220425, 44.341687
candidate mean_sec: 47.356313
candidate median_sec: 45.220425
candidate sample_stdev_sec: 5.872958
candidate stderr_sec: 2.626467
```

Decision: `keep for feasibility`. This establishes OOM-to-PASS and bounded
peak-memory behavior. It does not establish a numerical speedup because there
are zero eligible canonical runtime pairs.

### Final reduced 2048-shot six-pair session

Command:

```bash
python3 research/task-08/run_docker_matrix.py \
  --mode paired --n-samples 2048 --repeat 6 \
  --timeout 300 --cpus 6 --memory 7g \
  --output results/task-08-final-reduced-2048-pairs
```

Raw report SHA-256:
`cf254f54739057b85903702df7f824d018081106339b3b27003c7e65057dc178`.

Tracked sanitized record: `profiles/final-reduced-2048-pairs.json`.

```text
valid: 12/12 PASS
reference mean_sec: 32.246056
candidate mean_sec: 29.838621
ratio_of_means_improvement: 7.4658%
mean_pairwise_speedup: 1.089218
paired_speedup_stderr: 0.078256
paired_speedup_95pct_t_ci: [0.888056, 1.290381]
candidate pair wins: 4/6
```

Decision: `not promoted as a runtime speedup`. The descriptive mean is lower,
but the frozen rule required at least 5/6 wins and a confidence-interval lower
bound above 1.0. Both conditions fail.

Full interpretation and PR wording:
[`IMPLEMENTATION_COMPARISON.md`](IMPLEMENTATION_COMPARISON.md).

## High-memory expert-feasibility follow-up

Date: 2026-07-30 local / 2026-07-29 UTC.

Purpose: test whether the canonical expert failure is algorithmic or caused by
the fixed 7-GiB allocation. Sources, evaluator, site customization, image, six
CPU limit, 8192-shot workload, and 300-second timeout were unchanged.

### 13-GiB one-pair probe

Colima was raised from 8 GiB to 14 GiB and the container limit to 13 GiB.

```text
reference: 75.260284 s, PASS
candidate: 62.744019 s, PASS
single-pair ratio: 1.199481x
```

Decision: `feasibility only`. This proves that the immutable expert code can
complete the canonical workload, but one pair cannot support a mean, interval,
or promoted speedup.

### 13-GiB formal-repeat start

The first reference cell of the subsequent five-pair matrix failed after
26.745 controller seconds:

```text
RESOURCE_EXHAUSTED
requested_buffer_bytes: 18018189312
requested_buffer_gib: 16.780746
```

Decision: `stop matrix`. A 13-GiB allocation does not make the expert
reproducible; continuing would create ineligible pairs.

### Maximum-host-memory probe

The Apple M2 host has 16 GiB physical RAM. Apple Virtualization Framework
rejected a 20-GiB VM as greater than `maximumAllowedMemorySize`. Colima was
therefore set to its accepted 16-GiB maximum; Docker reported
16,733,048,832 bytes. The container received 15 GiB plus a temporary 8-GiB
swap file.

```text
reference: TIMEOUT
controller_wall_sec: 300.166
observed_container_memory_gib: 14.39
process_state_at_timeout: CPU-active
```

The temporary swap was disabled and deleted, all test containers were
removed, and Colima was restored to the original 8-GiB configuration.

Decision: `no canonical promotion`. The expert is algorithmically runnable,
but this host cannot produce five reproducible eligible pairs. The independent
64-GiB alternating-order run below fulfills the recommended follow-up.

Tracked record:
`profiles/high-memory-feasibility-follow-up.json`.

### 64-GiB Slurm definitive five-pair session

Date: 2026-07-30 local / 2026-07-29 UTC.

The requested canonical comparison was completed on Slurm job `23014910`,
node `a01r04n07` (AMD EPYC 7742, x86_64). The partition's 3931-MiB-per-CPU
policy required reserving 17 CPUs for a 64-GiB allocation; the benchmark
controller and all evaluator children were affinity-limited to the same six
CPU IDs. Apptainer 1.3.4 ran a pre-expanded Python 3.11.15 sandbox with the
locked TensorCircuit/JAX environment. Each cell was a fresh process with a
300-second cap and no network access.

```text
pair 1 reference -> candidate: 133.779068 -> 104.469589 s, 1.280555x
pair 2 candidate -> reference: 117.351760 -> 124.285933 s, 0.944208x
pair 3 reference -> candidate: 116.306970 -> 143.334082 s, 0.811440x
pair 4 candidate -> reference: 132.744516 -> 114.131608 s, 1.163083x
pair 5 reference -> candidate: 133.194228 -> 129.717456 s, 1.026803x

valid cells: 10/10 PASS
reference mean_sec: 126.675308
candidate mean_sec: 123.187734
ratio_of_means_speedup: 1.028311
ratio_of_means_improvement: 2.7532%
mean_pairwise_speedup: 1.045218
paired_speedup_stderr: 0.081997
paired_speedup_95pct_t_ci: [0.817558, 1.272878]
candidate pair wins: 3/5
Slurm MaxRSS: 26051648 KiB (24.8448 GiB)
```

Raw report SHA-256:
`c6f281479979d34e8293a3ad0bde749fad8140cc095daeb8bb0661921102f2b8`.

Tracked sanitized record:
`profiles/final-canonical-8192-high-memory-five-pairs.json`.

Decision: `expert feasibility confirmed; runtime speedup not promoted`. The
expert and candidate both pass 5/5 with sufficient memory. The candidate mean
is descriptively 2.75% lower, but only 3/5 pair wins and a confidence interval
including 1.0 do not support a confirmed runtime claim. Absolute times are not
combined with the Apple M2 sessions.
