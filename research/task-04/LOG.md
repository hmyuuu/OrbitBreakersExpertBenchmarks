# Task 04 Autoresearch Campaign

Destination: `research/task-04/LOG.md`

Task: `task-04`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and provenance

Selected task: `task-04` (trainable asymmetric Kraus-channel calibration).

At `2026-07-29T07:56:11Z`, the public GitHub API returned no open pull
requests on `sxzgroup/ORBIT-Q`; therefore no active Task 04 improvement PR
conflicts with this campaign. The GitHub App request failed with a transient
transport error, so the public API was used as the read-only fallback.

Campaign branch:
`codex/orbitbreakers/task-04/extreme-native`.

Base commit:
`5af98f2e` (`origin/main` after fetching on 2026-07-29).

Measured image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`).

Host fingerprint:
`c627504db97dc65b8d998afb4b9cdf73cfc3eff2ce2ac589b4a7a4aa0c7fdc48`.

All timed campaign runs use 6 CPUs, 7 GiB, Docker network disabled, and a
300-second evaluator timeout. Absolute times are not compared across hosts.

## Interrupted reference attempt

Command:

```bash
./bench run 04 --solution reference --repeat 6 --engine docker \
  --cpus 6 --memory 7g --timeout 300 --no-build \
  --output results/task-04-reference-20260729
```

The outer command session stopped after two passing cells (`15.043342 s`,
`14.636584 s`) and before a complete report was written. The container was no
longer running when checked. These cells are retained as interruption evidence
and excluded from every baseline, confidence interval, and speedup claim.

Decision: `invalid` (incomplete report, not a scientific failure).

## Complete immutable reference baseline

Command:

```bash
./bench run 04 --solution reference --repeat 6 --engine docker \
  --cpus 6 --memory 7g --timeout 300 --no-build \
  --output results/task-04-reference-20260729-v2
```

Reference SHA-256:
`04e37b73e7246599ed3eb8f65e38bb7e084db7aab511d7af3b20baa0867b21ae`.

Evaluator SHA-256:
`7f7e215064695ba3d0a0f5fb4883d78f7d225b2b7a79db70cdf3a0f3228314c5`.

Container session:
`0317899deb15820dfd2accd14bb34fbb734dae82ba7cf9d2eba3ed8f3933a410`.

Immutable report:
`results/task-04-reference-20260729-v2/results.json`
(`sha256:be8696214fa5c170398cd7a57bb3df56a476e658ebca5c99f8ec7a487735ff16`).

```text
terminal_status: SUCCESS x 6
valid: 6/6
timed_out: 0
runtime_sec: 15.250249, 14.381873, 14.605714, 14.544423, 14.697870, 14.546563
mean_runtime_sec: 14.671115
median_runtime_sec: 14.576139
sample_stdev_sec: 0.301866
stderr_sec: 0.123236
min_sec: 14.381873
max_sec: 15.250249
```

All evaluator criteria passed. This opens the repeated-reference promotion
gate and validates `task-04-canonical-fixed-v1`.

Decision: `baseline`.

## Experiment `e01-dmcircuit2`

Branch: `codex/orbitbreakers/task-04/e01-dmcircuit2`

Hypothesis: replacing only `tc.DMCircuit` with native `tc.DMCircuit2` will
defer each Kraus channel as a local superoperator TN node and reduce eager
full-density materialization without changing the circuit, channel, optimizer,
or output contract.

Parent commit: pending gate commit.

Candidate file: `src/solutions/task-04/solution_4.py`

Public dataset version: `orbitq-workloads-v20260729.1`

Private evaluation used: `no`

Pair-order pattern: odd `reference -> candidate`, even reverse

Timeout: `300 seconds`

Decision: pending.

## Result for `e03-training-scan`

Candidate commit:
`777ebfc`.

Candidate SHA-256:
`df308796deebb576ead6e2c47d6ac73494206de9e73980a7a939c8a1712c81f6`.

The enhanced reduced exact-equivalence report checks the target, one
value/gradient/update, the complete three-step loss history, final
probabilities, and final fitted expectations:
`research/task-04/profiles/e03-equivalence.json`
(`sha256:5b5cb1919d294b26177e06ef51f7c70eb397e36c62bf086a6e6a3ae699e8b3bb`).
Every field passed the `2e-6` threshold; the maximum error was `1.79e-7`.
The canonical 12-qubit evaluator also passed.

To isolate this factor, the immutable accepted e02 source was the reference
and the e03 file was supplied as an explicit candidate:

```bash
./bench run 04 --candidate <e03>/src/solutions/task-04/solution_4.py \
  --compare-to optimized --repeat 6 --engine docker \
  --cpus 6 --memory 7g --timeout 300 --no-build \
  --output results/task-04-e03-vs-e02-paired
```

Immutable report:
`results/task-04-e03-vs-e02-paired/results.json`
(`sha256:dfe56ba11cbf8a5144a19bb4e118a6ab57f258c2b54ffe9492a7140c6e9f7f89`).

```text
terminal_status: SUCCESS x 12
valid: 12/12
reference (e02) mean_runtime_sec: 6.645754
reference (e02) runtime_stderr_sec: 0.033795
candidate (e03) mean_runtime_sec: 6.736220
candidate (e03) runtime_stderr_sec: 0.061509
paired_improvement_mean: -1.374862%
paired_speedup_mean: 0.986981x
paired_speedup_stderr: 0.010315x
pair_wins: 2/6
```

Decision: `discard`. A single TensorCircuit `K.jaxy_scan` is exact but does
not meet the frozen positive-speedup or 80% pair-win gates. The earlier
profile already bounded 120 host dispatches to a small part of total runtime;
the scan wrapper instead produced a small measured regression on this
compilation-dominated workload.

## Experiment `e04-expectation-reuse`

Branch: `codex/orbitbreakers/task-04/e04-expectation-reuse`

Hypothesis: on top of accepted probe VMAP, use TensorCircuit
`DMCircuit.expectation(..., reuse=True)` so the framework contracts and caches
the final density tensor once per probe before evaluating the 12 single-Z
values and parity, instead of explicitly disabling reuse for every observable.

Parent commit: `9f01883`

Candidate file: `src/solutions/task-04/solution_4.py`

Public dataset version: `orbitq-workloads-v20260729.1`

Private evaluation used: `no`

Pair-order pattern: odd accepted parent then candidate, even reverse

Timeout: `300 seconds`

Decision: pending.

## Result for `e02-probe-vmap`

Candidate commit:
`83df63e`.

Candidate SHA-256:
`6247d16ca78364925ada690b32b350eadf3ed4c87760922e7704885514f82ec1`.

Exact-equivalence report:
`research/task-04/profiles/e02-equivalence.json`
(`sha256:6a4005fbfa954b165908c46fe04ac9f800442d0aa2874c29a6f47bf94beee01b`).
Target, observables, loss, gradient, and one Adam update passed; maximum
absolute error was `1.79e-7`. The canonical evaluator passed.

Candidate profile:
`research/task-04/profiles/e02-profile.json`
(`sha256:e99f3bd67d408493859a15082496e7e9827e82b8071bf8ac2a8084f9adb23207`).

```text
metric                         reference     probe-vmap
target generation (s)          2.629306      2.323040
lowering (s)                   3.067534      0.615647
XLA compilation (s)           10.712149      2.014091
StableHLO lines                   21720          4855
steady step mean (s)           0.001257      0.008035
projected 120 steps (s)        0.150788      0.964238
```

The factor trades a slower batched steady kernel for much smaller lowering and
compilation, which is favorable for the end-to-end timed contract.

Command:

```bash
./bench run 04 --solution optimized --compare-to reference --repeat 6 \
  --engine docker --cpus 6 --memory 7g --timeout 300 --no-build \
  --output results/task-04-e02-paired
```

Immutable report:
`results/task-04-e02-paired/results.json`
(`sha256:61fa6c4f787d3802b321da01a24ea89ce343c42509136d84a807755c4a28f350`).

```text
terminal_status: SUCCESS x 12
valid: 12/12
timed_out: 0
passing_pairs: 6
reference_mean_runtime_sec: 15.029802
reference_runtime_stderr_sec: 0.215846
reference_median_runtime_sec: 14.840418
candidate_mean_runtime_sec: 6.962292
candidate_runtime_stderr_sec: 0.203499
candidate_median_runtime_sec: 6.761579
improvement_pct: 53.676757
paired_speedup_mean: 2.165207
paired_speedup_median: 2.192234
paired_speedup_stderr: 0.050131
paired_speedup_ci_low: 2.036340
paired_speedup_ci_high: 2.294074
pair_wins: 6/6
first_five_reference_mean_sec: 15.069380
first_five_candidate_mean_sec: 7.061034
first_five_paired_speedup_mean: 2.139665
```

Decision: `keep`. All frozen promotion criteria pass. Probe VMAP is the
accepted parent for the next single-factor experiment.

## Immutable reference profile

Date: 2026-07-29

Script:
`research/task-04/profile_reference.py`.

Output:
`research/task-04/profiles/reference-profile.json`
(`sha256:eecfa9259600492fe7a8b3b3e9a4fa17a55df13395a41142683f9ece16001ebf`).

```text
target_observable_table_seconds: 2.629306
lower_seconds: 3.067534
compile_seconds: 10.712149
first_compiled_step_seconds: 0.003231
steady_step_mean_seconds: 0.001257
projected_120_step_seconds: 0.150788
stablehlo_line_count: 21720
```

Interpretation: Task 04 is compilation dominated. The training Python loop can
account for only a small fraction of runtime; the first candidate should reduce
the traced Kraus/density contraction graph before testing `K.jaxy_scan`.
## Append-only correction for `e01-dmcircuit2`

The parent commit left as pending in the predeclared entry is
`f83ab5d`. The experiment changes one factor only:
`tc.DMCircuit` to `tc.DMCircuit2`; all gate, Kraus, observable, optimizer, and
output code remains byte-for-byte unchanged.

## Result for `e01-dmcircuit2`

Candidate commit:
`c8b1850`.

Candidate SHA-256:
`54f0e4144dac3c7d757100af7acbc9dc09ce726debbff3871227717d77e8c006`.

Reduced exact-equivalence report:
`research/task-04/profiles/e01-equivalence.json`
(`sha256:48b3016ae69b7ef6905c820f7467e4f265a846ef7024f13492178bf460d05c0f`).
Target, observables, loss, gradient, and one Adam update all passed a `2e-6`
threshold; the largest error was `5.96e-8`. The canonical evaluator also
passed.

Command:

```bash
./bench run 04 --solution optimized --compare-to reference --repeat 6 \
  --engine docker --cpus 6 --memory 7g --timeout 300 --no-build \
  --output results/task-04-e01-paired
```

Immutable report:
`results/task-04-e01-paired/results.json`
(`sha256:e50489e8da594310726a457c32a55765b9ed6a6f3024fd45f3396e8c3e2014a9`).

```text
terminal_status: SUCCESS x 12
valid: 12/12
timed_out: 0
passing_pairs: 6
reference_mean_runtime_sec: 14.442584
reference_runtime_stderr_sec: 0.277754
reference_median_runtime_sec: 14.345619
candidate_mean_runtime_sec: 14.185293
candidate_runtime_stderr_sec: 0.124329
candidate_median_runtime_sec: 14.188593
improvement_pct: 1.781477
paired_speedup_mean: 1.018020
paired_speedup_median: 1.025506
paired_speedup_stderr: 0.015880
paired_speedup_ci_low: 0.977198
paired_speedup_ci_high: 1.058841
pair_wins: 4/6
```

Decision: `discard`. The lower confidence bound does not exceed one and only
four of six pairs won, below the frozen 80% rule. In this image, changing the
class name alone is not an attributable improvement.

## Experiment `e02-probe-vmap`

Branch: `codex/orbitbreakers/task-04/extreme-native`

Hypothesis: constructing the four exact probe states once with
`tc.Circuit`, then evaluating their identical noisy density-matrix TN through
TensorCircuit `K.vmap`, will reduce four repeated traced circuit graphs to one
batched graph. This should reduce the measured lowering/compilation bottleneck
without changing Kraus algebra, observables, or training.

Parent commit: `c5b1c9e`

Candidate file: `src/solutions/task-04/solution_4.py`

Public dataset version: `orbitq-workloads-v20260729.1`

Private evaluation used: `no`

Pair-order pattern: odd `reference -> candidate`, even reverse

Timeout: `300 seconds`

Decision: pending.
