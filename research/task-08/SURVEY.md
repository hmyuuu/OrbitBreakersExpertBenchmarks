# ORBIT-Q Task 08 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-08`

Survey freeze: `2026-07-29T03:30:00Z`

Reference commit: `5af98f27b9404c513df8eee0f4568b1512edee19`

This campaign covers only Task 08. The survey and public workload are frozen
before the first candidate edit. Runtime promotion remains closed until at
least six valid matched reference/candidate pairs exist.

## Evidence and claim boundary

The immutable human expert is
`references/task-08/solution_8.py`
(`sha256:0b0df74257e8f55d717ca29bb36e2edbb803206e9b2966fa423fefca9f15c311`).
The public contract and evaluator are `tasks/task-08/problem.md`
(`sha256:be1db9100dd3b89304fb2213d4c0e1c48cf3144572aec753e77acd43ee0243fe`)
and `tasks/task-08/evaluator/evaluate_8.py`
(`sha256:bffda6b012b07fd1cb8d5a1ec8a763bba4f1b967e5adc0d2b704e6ae99de9c41`).

The exact 49-qubit circuit, sample count, RNG seed, output contract, and
evaluator timer define the comparison. No external publication reports a
matched runtime for this evaluator, so this campaign may claim only a paired
gain over the bundled expert—not global SOTA.

The canonical expert currently has no valid runtime on the campaign host:
under the fixed six-CPU, 7-GiB Docker allocation, its monolithic 8192-shot
`vmap` reaches `K.numpy(samples)` after about 23 seconds but XLA aborts while
requesting a 17,998,348,288-byte buffer. This is preserved as a
`RESOURCE_EXHAUSTED` result, not converted into a timing. The evaluator's
supported `--n-samples 2048` scale probe passes in the same image in
48.113225 seconds and is eligible for optimization screening but not a
canonical speedup claim.

## Inspected environment and framework paths

Measurements use Docker image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with six CPUs and 7 GiB memory. The tracked dependency lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`);
the campaign intentionally uses the image's latest installed TensorCircuit-NG
rather than downgrading it.

| Component | Version | Inspected path / symbol |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `BaseCircuit.perfect_sampling` and `measure_jit` in `tensorcircuit/basecircuit.py` (`sha256:2f47be7f215c73bfbc41788661b0dd86a77054503f63e7bfbb6c4d2a84004e98`); `Circuit(split=...)` in `tensorcircuit/circuit.py` (`sha256:5c4d569325369d957dc60bbeca8a581508549ff9813a7a163de61a6294864662`) |
| TensorNetwork-NG | `0.5.1` | `tn.split_node`, called by TensorCircuit's native two-qubit split path |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | `jax.jit` and `jax.vmap` through TensorCircuit backend wrappers |
| OMECo | `0.2.0` | `omeco-4-4` contractor shortcut, configured by `envs/tensorcircuit-py311/sitecustomize.py` (`sha256:02800060761f2b15abe9055aded49d2af3877ab93d7b0fae8af94b30bac30120`) |
| Quimb | `1.11.1` | OMECo path-search support |

TensorCircuit's native two-qubit decomposition is
`tensorcircuit.simplify._split_two_qubit_gate`
(`sha256:653553fbb92508a7a758b8f929f2a3f4d253614e32b2a7e31b70b1afda3f8e04`).
It performs a TensorNetwork SVD and replaces a rank-4 gate node with two
rank-3 nodes. The official TensorCircuit advanced tensor-network tutorial
documents this `split` option and its contraction benefits:
<https://tensorcircuit.readthedocs.io/en/latest/tutorials/advanced_tn.html>.
The implementation source is public in the TensorCircuit-NG repository:
<https://github.com/tensorcircuit/tensorcircuit-ng>.

## Task 08: 7x7 mixed-axis grid sampling

### Required algorithm and output

The expert constructs a 49-qubit tensor-network circuit without a dense
`2^49` state:

1. 49 site-dependent `Ry` gates;
2. 42 horizontal nearest-neighbor `RZZ` gates;
3. 42 vertical nearest-neighbor `RXX` gates;
4. 49 site-dependent `Rx` gates;
5. 8192 exact conditional samples using the fixed NumPy RNG seed 2033.

It must return one NumPy integer array `samples` of shape `(8192, 49)`.
The evaluator checks binary values and 44 public `Z`-string expectations
against single/max/mean finite-sample tolerances of 0.03/0.05/0.015.

`perfect_sampling` implements the exact conditional-contraction algorithm
described by Ferris and Vidal,
<https://arxiv.org/abs/1201.3974>. For each of 49 bits,
`measure_jit` copies ket and bra networks, fixes all already sampled bits,
contracts a conditional 2x2 density matrix, and selects the bit from the
corresponding seed entry. The expert then applies `jit(vmap(sample_one))` to
all shots. JAX documents `vmap` as a mapped-axis transform and `jit` as XLA
compilation:
<https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html> and
<https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>.

### Dominant cost and measured bottleneck

The monolithic mapped axis is the dominant memory multiplier. XLA attempts
an approximately 18-GB buffer for 8192 shots on a 7-GiB cgroup, so the
canonical reference fails before returning. At 2048 shots it passes in
48.113225 seconds. Sampling cost comprises 49 conditional double-layer
network contractions per shot; path quality and peak intermediate size
therefore affect both compile/runtime and memory.

Large-batch tensor-network contraction can reuse work across many amplitudes,
as shown by Pan et al. for correlated bitstring batches
(Phys. Rev. Lett. 128, 030501,
<https://doi.org/10.1103/PhysRevLett.128.030501>), but TensorCircuit's current
`perfect_sampling` API maps independent conditional trajectories. A custom
raw simulator or hard-coded probability model would violate the campaign's
framework-fidelity rule.

### Independent public semantic oracle

Because the canonical expert OOMs, `validate_exact_observables.py` validates
all 44 evaluator targets with a distinct TensorCircuit path:
`Circuit.expectation_ps(..., enable_lightcone=True)`. It performs exact
observable contractions and never calls the expert's sampling routine.
The oracle must match every published exact value within `2e-6` before the
public dataset is marked ready. This validates circuit construction and
observable semantics; it does not create a reference runtime.

### Candidate hypotheses frozen before editing

Every candidate must preserve the exact circuit, complex64 semantics,
seed-2033 status matrix and row order, 8192 exact conditional samples, output
shape/type, TensorCircuit central computation, and evaluator boundary.

1. **e01—native exact rank-2 gate split.** Both
   `RZZ(theta)=cos(theta/2) I⊗I - i sin(theta/2) Z⊗Z` and
   `RXX(theta)=cos(theta/2) I⊗I - i sin(theta/2) X⊗X` have exact operator
   Schmidt rank two. Constructing `tc.Circuit(...,
   split={"max_singular_values": 2, "fixed_choice": 1})` should replace 84
   dense rank-4 nodes by native rank-3 pairs, reduce contraction width and
   peak batched intermediates, and may make the canonical workload feasible.
   Validate gate tensors and all 44 exact observables because complex64 SVD
   changes rounding.
2. **e02—bounded shot chunks.** Apply the same jitted TensorCircuit
   `vmap(perfect_sampling)` to fixed contiguous slices of the pre-generated
   status matrix, then concatenate. This preserves every random number and
   sample row while bounding peak memory. Screen 256/512/1024/2048 chunks;
   compilation and repeated launch overhead may offset memory savings.
3. **e03—measurement ordering.** Call `measure_jit` in a grid-aware order and
   invert the returned columns. Row-major already limits the square-grid
   frontier to seven; column-major and snake order are plausible but require
   measured path/intermediate evidence.
4. **e04—contractor path budget.** Compare the current OMECo
   `TreeSA(ntrials=4,niters=4)` shortcut with predeclared lower/higher budgets
   or greedy contraction. Path search lies inside the timed JIT trace; a
   cheaper path search can still lose overall if all 49 conditional
   contractions become slower or wider.
5. **e05—commuting/fusion transformations.** `RXX` gates commute with the
   final `Rx` layer, and adjacent single-qubit tensors can potentially merge.
   This has lower priority because TensorCircuit simplification/path planning
   may already exploit the local structure and because changing node order can
   increase width.

The first two ideas are complementary but constitute separate falsifiable
hypotheses until each is independently profiled. `Circuit.sample` was
inspected and rejected as the primary candidate because it dispatches
individually jitted `perfect_sampling` calls in a Python loop rather than a
native batched conditional sampler.

## Frozen measurement and promotion rule

All eligible comparisons use one long-lived container with six CPUs and
7 GiB, the image ID above, fresh evaluator processes, no network, a
300-second per-cell cap, and alternating order:

- odd pairs: reference then candidate;
- even pairs: candidate then reference.

The canonical reference is expected to fail until an optimization also
provides an apples-to-apples reference path; no canonical speedup will be
reported from a missing runtime. Candidate feasibility is reported only as
“OOM to PASS.”

For performance, the evaluator-supported 2048-shot workload will be measured
in at least six matched pairs (exceeding the user's five-run request). Report
all cells, arithmetic mean/median/sample standard deviation/standard error,
each pairwise `reference/candidate` speedup, and a two-sided 95% Student-t
interval on mean pairwise speedup (`t_(0.975,5)=2.5705818366`). Promotion
requires all functional gates, lower candidate mean and median, at least
five of six wins, and a confidence-interval lower bound above 1.0.

After the best design is frozen, run the full 8192-shot candidate at least
five times. These full runs establish reproducibility and correctness, not a
speedup against the failing reference.

## Open evidence gaps

- No valid canonical reference runtime on the available 7-GiB allocation.
- No matched external implementation/hardware result for this exact evaluator.
- No framework-level batched conditional-sampling primitive was found.
- Any result is specific to this six-CPU Docker host and fixed circuit; no
  cross-hardware or asymptotic claim is implied.
