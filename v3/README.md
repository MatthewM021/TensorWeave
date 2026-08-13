# TensorWeave V3

This directory contains the clean V3 rebuild described in the repository handoff.
V2 remains unchanged under `v2_baseline/`.

## Milestone 1: streaming tensor forest

The first milestone establishes the correctness-critical state engine before
introducing learned routing or the dynamic-binding benchmark:

- branch-wise binary-counter state with one dedicated global lane;
- one length-independent, scale-shared CP merge with analytic scale features;
- streaming and packed parallel reductions over the same chronological dyadic
  merge DAG, agreeing within declared dtype tolerances (associativity is not
  assumed);
- query-aware readout over active slots only;
- explicit padding, null-query, local-route, and global-route semantics; and
- deterministic, canonical state bytes with validated resume.

`valid_mask = false` is a total padding no-op. At a valid event, route `-1` is
a read-only/null query, routes `0..B-1` select local lanes, and route `B`
selects the global lane. A routed event updates state before its query readout.

No model constructor accepts a maximum sequence length. Runtime scale capacity
grows from state counters while parameter keys, shapes, and counts stay fixed.
The readout visits at most `(B + 1) * ceil(log2(L + 1))` slots and never sees
the token history directly.

The parallel implementation compacts each routed lane chronologically, builds
complete dyadic nodes once, and extracts exactly the nodes represented by each
prefix counter. This construction matches streaming even for a deliberately
non-associative merge and sparse/interleaved routing.

## Reproduce the milestone

From this directory, install the package and run:

```text
python -m pip install -e .[test]
python -m pytest -p no:cacheprovider
```

The CPU reference settings and declared float64 parity tolerances live in
`configs/milestone1_cpu.yaml`.

## Scope boundary

The state serializer here is not the later compact model exporter.

## Milestone 2: dynamic binding and persistent routing

The second milestone adds:

- a deterministic document-local language with bind, update, copy,
  invalidate, query, and distractor events;
- explicit separation of model-visible fields from evaluation-only routes,
  query targets, generation IDs, and dependency parents;
- a sanitized model construction object that excludes generator-only held-out
  combinations, length limits, and mixture probabilities;
- a prefix-only `O(B)` persistent router with shared branch scoring and bounded
  prototypes/global state;
- scientifically distinct oracle, curriculum, and fully latent contracts;
- deterministic curriculum guidance that becomes fully autonomous in
  evaluation;
- exact per-document permutation-aligned route recovery, document consistency,
  local/global/null load, per-document collapse, aggregate/seen/held-out query
  accuracy, oracle-gap, and label-independence audits; and
- a straight-through selected-route surrogate so latent predictive loss has a
  gradient to the router while the executed forest remains discrete.

The surrogate is an explicitly biased optimization estimator; hard autonomous
routes and metrics are always evaluated without route labels. The fixed-batch
smoke matrix is an implementation/optimization check, not scientific evidence.
Its three strict configs live under `configs/milestone2/`, and the runner is
`scripts/run_milestone2_smoke.py`.

The runner requires a clean checkout whose `HEAD` exactly matches
`--code-commit`, and it requires `--output` outside that checkout so source
integrity can be rechecked after training. It writes atomic progress after each
condition and records failures as well as success. Once a run passes, copy the
strict JSON record into `v3_recovery/` in a separate evidence commit.

Scientific baselines, structural truncation, compact model export, paired
multi-seed campaigns, and campaign closure remain later milestones.
