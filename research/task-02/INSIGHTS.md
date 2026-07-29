# Task 02 Research Insights

Task: `task-02`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

None. The immutable reference baseline passes six times, but no candidate has
yet passed the paired promotion rule.

## Preserved semantics

- 12-qubit Neel input and complex64 TensorCircuit-NG evolution;
- six brickwork sublayers with all 243 independent parameters;
- three half-chain Renyi-2 checkpoints and the fixed target profile;
- the 45-term open XXZ plus staggered-field Hamiltonian;
- fixed seed 2026, Adam learning rate `0.015`, and 500 sequential updates;
- every pre-update history and the original four-key NumPy result contract.

## Confirmed bottlenecks

- The termwise Hamiltonian MVP traces 45 independent mask/flip/accumulate
  paths.
- The generic Renyi-2 implementation computes a dense 64-by-64 matrix product
  even though Hermiticity permits an elementwise Frobenius norm.
- The circuit contains 243 gate applications per loss, although commuting
  local sequences admit exact gate fusion.
- The host dispatches one compiled optimizer update 500 times.

These are source-supported hypotheses until the component profiler and
single-factor paired benchmarks attribute runtime.

## What worked

No candidate result yet.

## What did not work

No candidate result yet.

## Open hypotheses

1. whole-training `K.jaxy_scan`;
2. exact Frobenius-purity Renyi-2 evaluation;
3. TensorCircuit-native sparse XXZ action;
4. exact `RY -> RZ` and `RXX -> RYY -> RZZ` gate fusion;
5. final leave-one-factor-out checks for every retained mechanism.

## Evidence limits

Evidence currently covers only the canonical public Task 02 configuration,
one latest TensorCircuit-NG image, and one 6-CPU/7-GiB same-host profile. It
does not establish a candidate improvement, global SOTA, or scaling result.
