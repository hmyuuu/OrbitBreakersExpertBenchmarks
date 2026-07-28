# Task 08 Research Insights

Task: `task-08`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

The accepted Task 08 implementation is the original TensorCircuit circuit and
OMECo 4x4 contractor with contiguous 256-shot Python chunks around one cached
`K.jit(K.vmap(perfect_sampling))`
(`sha256:7696f4d742d07da92a06cf5bdd4634f26ca5fe9251471163a90a4b6da280b45d`).

In the final canonical session, the immutable expert failed 5/5 attempts with
OOM/exit 137 while the candidate passed 5/5 at 47.356 ± 2.626 seconds (mean ±
standard error). This is an OOM-to-PASS result, not a numerical speedup.

At 2048 shots the candidate mean is 7.47% lower, but it wins only 4/6 pairs
and the paired-speedup 95% interval `[0.888, 1.290]` includes 1.0. No confirmed
runtime-speedup claim is made.

## Preserved semantics

- Build the fixed 49-qubit 7x7 TensorCircuit network from `|0>^49`.
- Apply all 49 `Ry`, 42 horizontal `RZZ`, 42 vertical `RXX`, and 49 `Rx`
  gates with the evaluator's exact angle formulas.
- Use complex64 framework semantics and the OMECo contractor path.
- Generate the complete float32 status matrix with NumPy RNG seed 2033.
- Return exact conditional computational-basis samples in the same row/column
  order as the expert, with shape `(n_samples, 49)` and integer NumPy dtype.
- Materialize neither a dense `2^49` state nor a dense probability vector.

## Confirmed bottlenecks

- The expert's monolithic `jit(vmap(perfect_sampling))` over 8192 shots asks
  XLA for a 17,998,348,288-byte buffer and fails in a 7-GiB cgroup.
- The same implementation passes at 2048 shots in 48.113225 seconds, so mapped
  batch size is a primary memory multiplier.
- `perfect_sampling` performs 49 sequential conditional double-layer
  contractions per shot; every later measurement attaches more projectors.
- `RZZ` and `RXX` have exact operator Schmidt rank two, but the expert stores
  each as one dense rank-4 node.

## What worked

- Splitting only the mapped shot axis into contiguous blocks, while leaving
  TensorCircuit's circuit and `perfect_sampling` unchanged, bounds XLA peak
  memory. A 512-shot block makes the full 8192-shot evaluator pass in
  50.843 seconds within 7 GiB. At 2048 shots it passes in 31.479 seconds with
  the same printed observable metrics as the 48.113-second reference screen.
- Reducing the block to 256 shots lowers the first full screen to 44.028
  seconds (13.4% below the 512-shot screen). Batch size therefore controls
  more than peak memory: smaller XLA graphs/temporaries currently outweigh
  the extra dispatches.

## What did not work

- The unmodified canonical expert does not fit the fixed 7-GiB environment.
- Merely requesting more CPUs is irrelevant to the semantic comparison and
  does not address the 18-GB allocation.
- TensorCircuit's generic rank-2 SVD split is counterproductive here. It
  preserves all 256 diagnostic samples exactly, but takes 30.307 seconds
  versus 24.601 seconds for the dense-gate reference; the 44 exact
  light-cone contractions take 75.652 seconds versus 11.032 seconds, and
  accumulated complex64 SVD rounding reaches `8.56e-6`. Do not repeat e01
  unless the framework gains an analytic Pauli-rotation MPO and a path
  optimizer that benefits from the larger node graph.
- 128-shot Python chunks are too small: the full run is 55.182 seconds,
  25.3% slower than the 256-shot screen. Extra dispatch and synchronization
  dominate below 256 on this host.
- Wrapping 256-shot blocks in TensorCircuit `K.jaxy_scan` is also worse:
  67.401 seconds, 53.1% above the Python-loop screen. The body contains all
  49 conditional network contractions, so staging it inside XLA control flow
  costs far more than 31 cached-jit dispatches.
- OMECo `1x1` is catastrophically under-searched for this network:
  131.959 seconds versus 44.028 seconds with `4x4`. Path quality is reused
  across 32 blocks, so the expert's modest extra TreeSA search pays back many
  times during execution.
- Absorbing 42 commuting final RX nodes into 21 vertical RXX gates is
  algebraically correct and improves the 44-observable light-cone audit
  (11.032 -> 9.618 seconds), but worsens full perfect sampling
  (44.028 -> 60.989 seconds). Contraction topology/code generation for the
  actual operation matters more than raw node count.

## Open hypotheses

1. Grid-aware measurement ordering with inverse output permutation.
2. OMECo path-search budget selection based on end-to-end contraction cost.
4. An analytic, non-SVD Pauli MPO only if paired with better graph
   simplification/path evidence.
5. Commuting/fusing the final `Rx` layer with the `RXX` network if framework
   simplification does not already achieve it.

## Evidence limits

Canonical evidence has five interleaved expert failures and five candidate
passes, but no eligible speedup pairs. The reduced six-pair comparison is
descriptive because it fails the frozen confidence/win-count rule. Results
apply only to this fixed Docker image, six-CPU/7-GiB allocation, circuit and
evaluator; no cross-hardware, global-SOTA or scaling claim is established.
