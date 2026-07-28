# Task 08 Research Insights

Task: `task-08`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Provisional only: e02 with 512-shot chunks. It turns the canonical workload
from reference OOM into a 50.843-second PASS and screens at 31.479 seconds on
2048 shots versus the initial 48.113-second reference. Formal paired evidence
is still pending, so no speedup is claimed.

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

## Open hypotheses

1. Contiguous status-matrix chunks on the original dense-gate network.
2. Grid-aware measurement ordering with inverse output permutation.
3. OMECo path-search budget selection based on end-to-end contraction cost.
4. An analytic, non-SVD Pauli MPO only if paired with better graph
   simplification/path evidence.
5. Commuting/fusing the final `Rx` layer with the `RXX` network if framework
   simplification does not already achieve it.

## Evidence limits

The campaign has one canonical OOM and one passing 2048-shot probe, not a
repeated reference baseline. Results will apply only to this fixed Docker
image, six-CPU/7-GiB host allocation, circuit and evaluator.
