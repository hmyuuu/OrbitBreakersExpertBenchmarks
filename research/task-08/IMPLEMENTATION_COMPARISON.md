# Task 08 Sampling Campaign Report

## Outcome and claim

This campaign optimized only ORBIT-Q Task 08: exact conditional sampling from
a 49-qubit 7x7 mixed-axis TensorCircuit network.

The campaign-best solution changes one execution dimension: instead of mapping
`perfect_sampling` over all 8192 shots at once, it maps the same TensorCircuit
function over contiguous 256-shot blocks and concatenates the resulting NumPy
arrays. The fixed circuit, complete RNG status matrix, sample count, sample
order and evaluator are unchanged.

On the fixed six-CPU, 7-GiB Docker allocation:

- the immutable expert failed all five canonical 8192-shot attempts with XLA
  `RESOURCE_EXHAUSTED` or cgroup exit 137 and never produced an evaluator
  runtime;
- the candidate passed all five interleaved canonical attempts in
  47.356313 s mean, 45.220425 s median, and 2.626467 s standard error;
- therefore the primary established result is **OOM to reproducible PASS**,
  not a numerical speedup.

The evaluator-supported 2048-shot scale probe provides a runtime comparison:
all 12 cells passed, with reference 32.246056 s mean and candidate
29.838621 s mean (7.4658% lower; ratio of means 1.0807x). The candidate won
only four of six pairs, however, and the mean paired-speedup 95% Student-t
interval is 0.8881x–1.2904x. This misses the frozen promotion rule, so the
7.47% number is reported descriptively and is **not claimed as a confirmed
speedup**.

No matched external system reports this evaluator workload. The implementation
is called the campaign best, not global SOTA.

| Role | Artifact | SHA-256 |
| --- | --- | --- |
| Immutable expert | `references/task-08/solution_8.py` | `0b0df74257e8f55d717ca29bb36e2edbb803206e9b2966fa423fefca9f15c311` |
| Campaign best | `src/solutions/task-08/solution_8.py` | `7696f4d742d07da92a06cf5bdd4634f26ca5fe9251471163a90a4b6da280b45d` |
| Evaluator | `tasks/task-08/evaluator/evaluate_8.py` | `bffda6b012b07fd1cb8d5a1ec8a763bba4f1b967e5adc0d2b704e6ae99de9c41` |
| Docker image | `orbitbreakers-expert-benchmarks:tensorcircuit-py311` | `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833` |

## Why the expert fails

The expert creates the entire float32 status matrix with shape `(8192, 49)`
and calls:

```python
sample_batch = K.jit(K.vmap(sample_one))
samples = sample_batch(status)
```

`perfect_sampling` performs 49 sequential conditional double-layer tensor
network contractions. Mapping the whole function over 8192 trajectories makes
the shot axis part of every XLA intermediate. In the final five-pair session,
four failures reported attempted allocations between 9.31 GB and 17.97 GB;
the fifth process was killed with exit 137. The container limit is 7 GiB.

The candidate retains the same mapped function but calls its cached JIT on 32
contiguous 256-row slices:

```python
samples = [
    np.asarray(
        K.numpy(sample_batch(status[start : start + 256])),
        dtype=np.int32,
    )
    for start in range(0, config["n_samples"], 256)
]
```

The first block pays path-search, tracing and compilation cost. Later blocks
reuse the same compiled shape while keeping only one block's contraction
intermediates live. Host-side outputs are small
(`8192 * 49` integer entries), so concatenation is negligible relative to the
network contractions.

## Preserved scientific semantics

Both implementations:

- start from `|0>^49`;
- apply the same 49 position-dependent `Ry` gates, 42 horizontal `RZZ`
  gates, 42 vertical `RXX` gates and 49 final `Rx` gates in TensorCircuit;
- use complex64 and `tc.set_contractor("omeco-4-4")`;
- generate the entire float32 status matrix with NumPy `default_rng(2033)`;
- consume every status row exactly once in original order;
- call TensorCircuit `Circuit.perfect_sampling` for every shot;
- return exactly one integer NumPy array with shape `(n_samples, 49)`;
- materialize neither a dense `2^49` state nor a dense probability vector.

Changing the mapped batch shape can change complex64 contraction rounding and
flip a small number of status values that fall extremely close to conditional
probability thresholds. This is normal finite-precision sampling behavior,
not skipped work or synthetic output. Every canonical candidate cell passes
all 44 observable checks. Separately, the unmodified circuit was validated by
44 direct TensorCircuit light-cone expectation contractions with maximum
absolute error `1.1977e-6`.

## Canonical five-pair result

One long-lived container was used, with a fresh evaluator process for every
cell. Odd pairs ran reference then candidate; even pairs reversed the order.

| Pair | Order | Expert | Candidate |
| ---: | --- | ---: | ---: |
| 1 | reference -> candidate | OOM, no runtime | 40.537079 s, PASS |
| 2 | candidate -> reference | OOM/exit 137, no runtime | 51.583744 s, PASS |
| 3 | reference -> candidate | OOM, no runtime | 55.098628 s, PASS |
| 4 | candidate -> reference | OOM, no runtime | 45.220425 s, PASS |
| 5 | reference -> candidate | OOM, no runtime | 44.341687 s, PASS |

Candidate summary:

| Metric | Value |
| --- | ---: |
| Passing cells | 5/5 |
| Mean | 47.356313 s |
| Median | 45.220425 s |
| Sample standard deviation | 5.872958 s |
| Standard error | 2.626467 s |
| Minimum / maximum | 40.537079 / 55.098628 s |

The raw untracked matrix report is
`results/task-08-final-canonical-5-pairs/results.json`
(`sha256:02319130b267855985eb60d4e86bcc5523f1edb6df62c7695a4667ef9227148f`);
the tracked sanitized record is
`profiles/final-canonical-8192-five-pairs.json`.

## 2048-shot six-pair runtime comparison

| Pair | Order | Expert | Candidate | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | reference -> candidate | 29.965212 | 27.435688 | 1.0922x |
| 2 | candidate -> reference | 39.589037 | 29.119827 | 1.3595x |
| 3 | reference -> candidate | 28.130474 | 31.500896 | 0.8930x |
| 4 | candidate -> reference | 30.583924 | 30.338020 | 1.0081x |
| 5 | reference -> candidate | 35.193469 | 27.661507 | 1.2723x |
| 6 | candidate -> reference | 30.014218 | 32.975788 | 0.9102x |

| Metric | Expert | Candidate |
| --- | ---: | ---: |
| Passing cells | 6/6 | 6/6 |
| Mean | 32.246056 s | 29.838621 s |
| Median | 30.299071 s | 29.728924 s |
| Sample standard deviation | 4.300946 s | 2.185633 s |
| Standard error | 1.755854 s | 0.892281 s |

Mean pairwise speedup is 1.0892x with standard error 0.0783x and 95%
Student-t interval `[0.8881, 1.2904]`. The candidate wins 4/6 pairs. The
frozen rule required at least 5/6 wins and a lower interval bound above 1.0,
so formal runtime promotion fails closed.

The raw untracked report is
`results/task-08-final-reduced-2048-pairs/results.json`
(`sha256:cf254f54739057b85903702df7f824d018081106339b3b27003c7e65057dc178`);
the tracked sanitized record is `profiles/final-reduced-2048-pairs.json`.

## Factor ablation and rejected approaches

Task 08 does not bundle several unexplained speedups into the promoted result.
The only promoted factor is **bounded contiguous shot batching**. The
experiments below remove or vary one factor at a time around that design.
They are full-workload screening runs unless explicitly labeled as a reduced
audit, so they rank mechanisms but are not independent five-pair speedup
claims.

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Expert monolithic 8192-shot `vmap` | 0/5 PASS; 9.31–17.97 GiB attempted buffers | The dominant contribution is bounding the mapped batch, which converts OOM into 5/5 PASS. |
| Native generic rank-2 split | 30.307 s vs 24.601 s at 256 shots; exact audit 75.652 s vs 11.032 s | Doubling entangler nodes hurts OMECo despite lower bond rank. |
| 512-shot Python chunks | 50.843 s full screen | Feasible, but larger mapped intermediates lose to 256. |
| **256-shot Python chunks (promoted)** | **44.028 s full screen** | Best measured memory/dispatch balance; the promoted candidate changes no other factor. |
| 128-shot Python chunks | 55.182 s full screen | Extra dispatch/synchronization dominates. |
| `K.jaxy_scan` over 256 blocks | 67.401 s | Staging the large contraction body in XLA control flow costs more than cached-JIT dispatch. |
| OMECo 1x1 | 131.959 s | Under-search produces paths that are expensive across 32 executions. |
| Fuse 42 final RX gates into RXX | Exact audit passes and improves 11.032 -> 9.618 s, but full sampling is 60.989 s | Fewer nodes do not guarantee better conditional-sampling paths. |

The attribution is therefore unambiguous: chunking is the sole necessary
OOM-to-PASS factor, while the 256-shot size is an empirically selected
secondary tuning choice. Scan, smaller path-search budget, gate fusion, and a
rank-2 rewrite were measured separately and rejected. The reduced 2048-shot
six-pair result remains statistically inconclusive and is not used to assign a
runtime-speedup percentage to chunking.

The evidence does not justify changing measurement order: row-major removes
complete grid rows and already exposes a width-seven frontier. With the
dominant OOM fixed and no source-supported superior order, an output-order
experiment would add semantic/random-number mapping risk without a stronger
falsifiable expectation.

## Limits and PR recommendation

- Canonical runtime speedup is unmeasurable because the expert has no valid
  runtime in this allocation.
- The reduced comparison trends positive but fails the frozen statistical
  rule; do not advertise 1.08x as confirmed.
- OMECo path search produces visible process-to-process variance.
- Evidence covers one Apple M2 host, one six-CPU/7-GiB Docker allocation, one
  image and the fixed Task 08 circuit.

The PR should be titled around bounded-memory canonical sampling, for example:
“Task 08: chunk TensorCircuit perfect sampling to make 8192 shots fit 7 GiB.”
Its headline should be “reference 0/5 vs candidate 5/5 PASS,” with the reduced
7.47% mean result explicitly labeled non-significant.
