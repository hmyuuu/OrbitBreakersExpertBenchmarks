# Task 07 Research Insights

Task: `task-07`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e01` contracts the final adaptive circuit once and evaluates one
TensorCircuit-native sparse data Hamiltonian. Its first canonical screen
passes in 61.396553 seconds versus the expert's 135.815605 seconds (2.2121x).
This is provisional one-run evidence, not the final paired claim.

## Preserved semantics

- 8 data and 8 ancilla qubits, two adaptive layers, and all 96 trainable
  float32 parameters in the expert's layout.
- Seed 2047 parameter initialization and seed 2048 fixed trajectory uniforms.
- Eight normalized Z-basis `cond_measure` operations per layer and trajectory,
  with the selected trainable feedback branch for every bit.
- Exactly 64 fixed trajectories averaged per objective and exactly 100
  sequential Adam updates at learning rate 0.02.
- Pre-update energy history and post-update per-trajectory energy vector.
- The eight-site open-boundary TFIM Hamiltonian and complex64 TensorCircuit
  quantum computation.

## Confirmed bottlenecks

- Every trajectory evaluates seven `ZZ` and eight `X` expectations separately,
  repeating the final circuit's bra/ket contraction 15 times in both forward
  and reverse-mode work.
- Approximately 48 seconds is fixed trace/compile/path/finalization cost; the
  additional 99 canonical updates average about 0.89 seconds each.
- The generic `conditional_gate` keeps a selected dense two-qubit `RZZ` node
  after the ancilla is already a Z eigenstate.

## What worked

- One TensorCircuit state contraction plus one native sparse eight-qubit TFIM
  expectation reduces the 100-step screen from 135.816 to 61.397 seconds and
  the 50-step passing screen from 91.540 to 52.769 seconds.
- One-trajectory energy and gradient agree within `3.34e-6` and `1.01e-6`;
  the full 50/100-step physical outputs pass and remain close.

## Measurement lesson

- Maximum first-Adam parameter difference is a poor complex64 equivalence
  gate near zero gradients. The strict e01 diagnostic failed (`0.0313`) even
  though energy, gradient, post-update energy, and complete public workloads
  passed. Preserve that failure, but use physical post-update outputs as the
  predeclared one-step semantic criterion in later experiments.

## High-confidence exact identities

- Final ancillas factor from the data state because the last operation that
  can mix their computational basis is followed by `cond_measure`, and the
  remaining feedback is diagonal. This permits one full TensorCircuit state
  contraction followed by an eight-qubit native Hamiltonian expectation.
- On a measured ancilla `|b>`, feedback `RZZ(theta_b)` is exactly data
  `RZ((1-2b) theta_b)` with the ancilla unchanged.
- A `K.jaxy_scan` can emit the same pre-update values while carrying the same
  Optax state and final parameters.

## Open hypotheses

- Single native sparse-Hamiltonian evaluation (`e01`, highest priority).
- Exact feedback-node reduction (`e02`).
- Whole-training scan (`e03`).
- OMECo budget sweep after the graph is simplified (`e04`).
- More aggressive measurement-round state reuse (`e05`, higher policy and
  numerical risk).

## Evidence limits

- The canonical baseline is currently a single bootstrap run; promotion will
  collect six matched pairs against the frozen winner.
- No matched external implementation exists, so “SOTA” can mean only the
  campaign-best implementation for this repository workload.
- No scaling or cross-hardware claim is supported.
