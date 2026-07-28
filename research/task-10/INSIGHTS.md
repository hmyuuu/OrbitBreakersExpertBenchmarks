# Task 10 Research Insights

Task: `task-10`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

None. Candidate work is gated on the public Task 10 workload record.

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
  available newer ORBIT-Q image and passed. Phase-split profiling is pending.

## What worked

No candidate has been measured.

## What did not work

- The first pinned-image build failed at `apt-get update` after a network
  interruption. Retry rather than changing the environment lock.
- Prior ORBIT-Q evidence found a dense-state Task 10 implementation around
  18 times slower than the expert. Do not repeat that representation unchanged.

## Open hypotheses

1. Exact `MPSCircuit.apply_MPO` CMZ plus direct local MPS contractions and a
   whole-training scan.
2. Scan-only ablation on the immutable expert objective.
3. Fusing each three-rotation block into one TensorCircuit 2x2 gate.
4. Smaller OMECo search budgets as a reference-path ablation.

## Evidence limits

No pinned-image paired result, no memory profile, no canonical scaling study,
and no matched external SOTA comparator have been established.
