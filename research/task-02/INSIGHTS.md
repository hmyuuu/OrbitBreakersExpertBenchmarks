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

- One update lowers in `0.803158 s`, compiles in `1.929769 s`, and contains
  15,479 StableHLO lines. The 500 steady updates project to `1.321653 s`, so
  both cold graph staging and steady execution are material.
- The separately compiled trajectory is about 93.5% of the forward loss
  runtime (`0.000889/0.000951 s`), making exact gate-graph reduction the
  highest-priority factor.
- The host dispatches one compiled optimizer update 500 times. A whole-training
  scan can remove that boundary, although it cannot eliminate the steady
  quantum kernel itself.
- The termwise Hamiltonian MVP traces 45 independent paths, but its separately
  measured forward action is only `0.000176 s`. Sparse conversion is therefore
  a secondary compile-size hypothesis, not the primary steady bottleneck.
- One generic Renyi-2 call is `0.000132 s`; its Frobenius-purity rewrite is
  algebraically exact but expected to be a smaller factor than gate fusion.

Profile: [`profiles/reference-profile.json`](profiles/reference-profile.json).

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
