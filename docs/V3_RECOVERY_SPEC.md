# V3 Recovery Specification

This document records the intended V3 design in implementation-oriented form. `HANDOFF.md` remains authoritative when the two overlap.

## Package boundary

Add V3 beside V2:

```text
src/tnlm_v3/
  data.py
  routing.py
  operators.py
  forest.py
  truncation.py
  export.py
  baselines.py
  training.py
  benchmark.py
  factory.py
```

Do not rename or mutate `src/tnlm_v2/` in ways that invalidate its stored checkpoints.

## Core state

For branch `b` and scale `s`, maintain an optional state or transition object `S[b, s]`. A new routed event creates a scale-zero object. When `S[b, s]` is occupied, combine it with the carry and propagate to `s+1`; otherwise store the carry and stop.

Required properties:

- causal token-by-token update;
- masks for absent/padded/global events;
- bounded branch count in each experiment;
- active slot count scaling as `O(B log L)`;
- parameters independent of maximum configured length;
- deterministic serialized state;
- a parallel reduction whose result matches streaming within declared tolerance.

A query-aware readout may combine active branch/scale summaries through a bounded tree or another contractible map. It may not attend densely over all past token states.

## Tensor/operator parameterization

V2 used CP-factorized order-three merges. V3 may retain this as the controlled baseline. The interrupted design also explored composable transition operators to obtain associative reduction. Implement both only where scientifically useful; do not multiply variants without an explicit ablation.

Every merge must expose:

- nominal rank;
- effective/exported rank;
- parameter count;
- operation-count proxy;
- active state size;
- optional spectral or gate diagnostics.

Use nonlinear gates/residuals only when they preserve causal streaming and the exported contraction is explicit.

## Scale sharing

Default: one shared merge/update operator, or a small fixed number of scale classes, plus a bounded scale embedding. No module list whose length grows with `max_length`.

Test the same checkpoint at lengths not seen in training.

## Routing

The router's input at step `t` may include:

- current event/token representation;
- causal position features;
- branch prototypes/summaries derived only from the prefix;
- branch age/occupancy;
- bounded global state.

The router emits `B` branch logits plus an optional global/null route. Complexity should be `O(B)` per event, not `O(tB)` or `O(t^2)`.

Modes:

- `oracle`: true route is used;
- `curriculum`: route-label guidance decays during training, but evaluation is autonomous;
- `latent`: no true routes enter the training computation.

Store both raw route labels for evaluation and the model-visible fields. Add a test proving evaluation labels do not alter logits or routes.

## Dynamic document-local binding task

The generator should create episodes/documents with local symbol identities and mutable bindings. The same surface symbol must be reusable with a different route or value in another document so the model cannot solve routing via a global token lookup.

Suggested event families:

- introduce/bind entity or key;
- update/transform its state;
- copy or combine bindings;
- invalidate/expire a binding;
- query a binding or derived state;
- distractor events on interleaved bindings.

The exact grammar may differ, but record the latent dependency graph and route labels for oracle mode and diagnostics.

Vary:

- context length;
- number of simultaneous live bindings;
- document-local symbol permutation;
- update/query distance;
- collision and rebinding frequency;
- training/evaluation length distribution.

## Predictive auxiliary losses

Candidate loss decomposition:

```text
L = L_query
  + lambda_route * L_route_curriculum
  + lambda_branch * L_branch_future
  + lambda_scale * L_multihorizon
  + lambda_balance * L_router_balance
  + lambda_consistency * L_route_persistence
  + lambda_rank * L_truncation
  + lambda_cost * C_exported
```

Only activate terms with an ablation and a clear interpretation. Curriculum route loss must be zero in fully latent experiments.

## Compact export

Training-time selection may be soft, but final export must slice all dependent tensors consistently. For each retained channel set, rewrite every producer, consumer, gate, normalization, residual projection, scale signal, and readout using the compact dimensions.

Mandatory tests:

1. dense selected model versus compact model output parity;
2. perturb discarded input channels and prove retained outputs do not change;
3. serialize/reload compact model in a fresh process;
4. compare bytes, parameters, active-state scalars, operation proxy, RSS, and wall time;
5. verify no mask or zero-padded dense tensor recreates the original dimension.

The prior bug was specifically that inactive scale-signal channels could affect retained gate outputs. Keep a permanent regression for this path.

## Baselines

- GRU or another strong recurrent state baseline;
- causal Transformer with a proper key/value cache for token-by-token inference;
- corrected causal TTN/complete-tree control;
- streaming forest without learned routing where appropriate;
- optional identity-initialized MERA/disentangler switch.

Parameter matching alone is insufficient. Include matched-error comparisons and report state-memory/runtime scaling.

## Results schema

Each completed run should record at least:

```text
run_key
status
code_commit
config_sha256
model
routing_mode
seed_model
seed_train_data
seed_validation_data
seed_test_data
train_length_distribution
eval_length
accuracy/loss
route_accuracy_aligned
document_consistency
parameter_count
exported_parameter_count
nominal/effective/exported ranks
active branches and slots
state scalar count
operation proxy
training time
streaming tokens/s
peak RSS
checkpoint path
invalidated_by (nullable)
```

Write the record atomically immediately after completion.
