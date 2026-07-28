# Task 10 Research Insights

Task: `task-10`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

None. e01 was faster in one exploratory run but failed the functional gate.

## Preserved semantics

- 22 qubits, the canonical 18 selected CMZ sites, two ansatz layers.
- Ordered `RX -> RZ -> RY` rotations followed by one exact CMZ per layer.
- Canonical initial basis state and NumPy seeded float32 parameters.
- Canonical open-boundary TFIM energy density.
- Exactly 200 Adam updates and pre-update energy history.
- Required NumPy keys, shapes, finiteness, and evaluator thresholds.
- TensorCircuit-NG remains the central quantum framework.

## Confirmed bottlenecks

- Source inspection shows the expert forms a general bra/MPO/ket tensor
  network and invokes OMECo during JAX tracing.
- The CMZ is intrinsically bond 2, and two exact applications permit an MPS
  bond-dimension bound of four; the generic contraction does not expose that
  simple sequential structure to the implementation.
- A fresh immutable-reference validation took 18.117244 seconds in the
  available newer ORBIT-Q image and passed.
- Phase-split profiling shows the 18.36-second reference is dominated by
  4.37 seconds of lowering/path search and 13.14 seconds of XLA compilation;
  the compiled optimizer step takes only 0.317 ms.

## What worked

No candidate has passed. e01 reduced exploratory wall time to 10.26 seconds,
but its 200-step trajectory failed the energy-gap criterion.

## What did not work

- The first pinned-image build failed at `apt-get update` after a network
  interruption. Retry rather than changing the environment lock.
- Prior ORBIT-Q evidence found a dense-state Task 10 implementation around
  18 times slower than the expert. Do not repeat that representation unchanged.
- Exact `MPSCircuit.apply_MPO` with QR/RQ canonicalization matches the first
  five optimization steps closely but diverges into a failing basin by step
  200. Explicit unfused rotations and complex128 intermediates both worsened
  the numerical trajectory.

## Open hypotheses

1. Reuse the local contraction kernel from `MPSCircuit.apply_MPO` without
   canonicalization. The exact rank is statically bounded at four, so QR/SVD
   is unnecessary and removing it should improve both gradient stability and
   compile size.
2. Scan-only ablation on the immutable expert objective.
3. Smaller OMECo search budgets as a reference-path ablation.

## Evidence limits

No pinned-image paired result, no memory profile, no canonical scaling study,
and no matched external SOTA comparator have been established.
