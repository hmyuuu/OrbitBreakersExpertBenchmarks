# Task 03 Research Insights

Task: `task-03`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

None. Candidate work starts only after the survey and canonical public workload
pass the research gate.

## Preserved semantics

- Twelve logical qubits and ten alternating brickwork layers.
- Every independent `RXX`, `RZZ`, and `RX` parameter with seed 2027.
- Sixty zero-outcome post-selection events and their conditional
  probabilities.
- Exact open TFIM energy, probability-aware loss, 300 Adam updates, and all
  pre-update histories.
- Complex64 TensorCircuit-NG computation and the original NumPy contract.

## Confirmed bottlenecks

The immutable expert averages `4.153635 s`. Source inspection shows that every
loss traces ten 12-qubit circuits, 60 separate projector insertions, and a
23-term Hamiltonian MVP. Component attribution still requires profiling.

## What worked

No candidate has yet been measured.

## What did not work

No hypothesis has yet been rejected.

## Open hypotheses

1. Exact six-single-qubit product-state reduction.
2. Simultaneous commuting post-selection on the original state.
3. Whole-training `K.jaxy_scan`.
4. `K.vmap` over the six conditional maps.

## Evidence limits

The campaign covers one fixed public workload, one image, and one host resource
profile. It establishes neither cross-hardware performance nor global SOTA.
