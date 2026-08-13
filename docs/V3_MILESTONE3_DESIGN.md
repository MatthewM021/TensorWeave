# V3 Milestone 3: Physical CP-Rank Export

## Scope

Milestone 3 begins with the shared merge operator's CP interaction axis. This
is distinct from the `d_model` state axis. Reducing CP rank can physically
reduce merge tensors and contraction work without reducing persistent forest
state. All records must report those two facts separately.

The initial deterministic selector uses augmented parameter energy as a stable
channel-contribution proxy. It is suitable for implementation audits and
predeclared rank controls, but it is not a claim of predictive optimality.
Scientific compactness claims require paired validation/test evidence later.

## Physical slice contract

For sorted retained CP indices `I`, the compact merge slices:

- rows `I` of `left.weight`, `right.weight`, and `scale_to_rank.weight`;
- entries `I` of `global_rank`; and
- columns `I` of `output.weight`.

Every residual, gate, normalization, state-scale, router, encoder, and readout
tensor is copied exactly. The compact operator has rank `len(I)` and carries no
rank mask, padded dense tensor, retained-index buffer, or original-rank tensor.

## Correctness boundary

The dense selected reference zeros exactly the discarded portions of those
five CP tensors. The physical model must match that reference through both
streaming and packed-parallel execution, including routes, route logits, value
logits, and final forest state within declared dtype tolerances.

Perturbing discarded CP factor rows, scale-conditioning rows, global-rank
entries, or output columns must not affect the selected reference or compact
result. This covers inactive CP scale-conditioning channels. It does not cover
future pruning of `d_model` scale-signal channels; that broader exporter must
permanently test `scale_to_state`, gates, normalization, router, and readout.

## Remaining verification gate

Before a pruned reference campaign, a separate commit-bound audit must add and
verify:

- a deterministic, non-executable compact-model byte format;
- source/config/selection/artifact checksums;
- fresh-process load and exact replay;
- midstream forest-state serialize/reload/resume under the compact model;
- real-update parity across binary carry boundaries;
- parameter, raw byte, operation, state-scalar, RSS, and wall-time records; and
- explicit wording that wall-time or memory differences are measurements, not
  matched-quality wins.
