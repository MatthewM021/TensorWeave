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

Structural CP-rank export is verified in Milestone 3. Recurrent,
cached-Transformer, and corrected causal-tree control source is implemented in
Milestone 4, but trained scientific baseline comparisons, paired multi-seed
campaigns, and campaign closure remain pending.

## Milestone 3: compact CP-rank export (verified implementation audit)

The first export layer now selects a deterministic subset of the shared merge
operator's CP interaction channels, builds a dense zeroed reference, and
physically slices the five tensors carrying that CP axis. The compact model has
fewer parameters, fewer raw tensor bytes, and a lower declared merge-operation
proxy. It contains no dense rank mask or original-rank buffer.

This operation does **not** reduce `d_model`, the number of forest paths, or the
number of scalars in an occupied persistent-state slot. The export manifest
records that unchanged state interface explicitly. Whole-state channel pruning
would require coordinated slicing through embeddings, router, residual and gate
paths, normalization, scale signals, and readout; it is not implied by this
CP-rank result.

The compact model now has a deterministic, non-executable byte format with a
bounded canonical header, little-endian tensor payload, whole-section and
per-tensor checksums, strict configuration reconstruction, and exact manifest,
selection, mode, gradient-flag, and model-fingerprint validation. A separate
worker reloads it in a fresh process and hashes all predictions, routes, and
persistent router/forest state under both execution engines.

Parameter-energy selection is a deterministic structural proxy, not evidence
that a chosen compact rank preserves scientific quality. The strict audit
configuration is `configs/milestone3/export_audit.yaml`; the commit-bound
runner is `scripts/run_milestone3_export_audit.py`, with isolated runtime/RSS
measurements performed by `scripts/measure_milestone3_runtime.py`.

The commit-bound audit passed from clean published commit
`7f1c9ead80e93dddb03c51bf608ed24d7e41d719`. Its strict record and native
compact artifact live in `../v3_recovery/`. The audit covers real evaluation
lengths 15, 16, 31, 32, 63, and 64; local and global carry depths 4, 5, and 6;
canonical midstream state resume; fresh-process replay; and 18 isolated
runtime/RSS workers. Dense-selected versus compact parity and streaming versus
parallel parity passed under the predeclared tolerances. The fresh worker also
matched the parent hashes separately for each execution engine.

This verifies the export implementation, not the predictive merit of rank 4.
The selector is a data-independent parameter-energy proxy, no quality threshold
was applied, persistent-state width is unchanged, and runtime/RSS observations
remain descriptive measurements rather than matched-quality or scientific wins.

## Milestone 4: causal reference controls (source contracts)

The first reference-control checkpoint adds a stacked GRU with bounded
persistent state and a causal Transformer with an explicit key/value cache for
every attention layer. Both accept only `BindingModelInputs`, keep generator and
evaluation metadata outside model configuration, support one-event stepping and
chunked continuation, and use no learned maximum-length parameter table.

Their strict configurations live under `configs/milestone4/`. Contract tests
cover full/step/chunk parity in float32 and float64, prefix causality, padding
and empty-input no-ops, finite forward/backward behavior, cache gradients,
counter overflow, canonical state validation, independent returned state, real
event cache growth, and parameter-count independence from evaluation length.

The second control checkpoint adds a fixed, one-lane causal complete-tree TTN.
It stores a canonical binary frontier, appends real events with chronological
binary carries, and rebuilds one complete-tree root using the same scale-shared
merge at every depth. It has analytic real-event positions, no route input,
learned maximum-length table, or per-level parameter list, and `O(d_model log
L)` persistent state. Tests cover an independent non-associative tree oracle,
carry boundaries, mixed padding, exact step/chunk continuation, tensor-only
model/state roundtrip, and complete logical-state and byte accounting.

The shared baseline campaign adapter applies one visible-input-only query
objective to the GRU, cached Transformer, and causal tree. It reports overall,
seen, and held-out query results plus exact model/state storage, rejects
malformed native state and forged tree-work diagnostics, and makes an
empty-query batch a true optimizer no-op. Routed oracle, curriculum, and latent
conditions continue to use the explicit routing-aware training path.

The strict campaign planner freezes three distinct stages: pilot, validation-
only screen, and promotion-bound confirmatory. It requires the routed oracle,
curriculum, and latent sources; a compact child exported from each curriculum
checkpoint; GRU; cached Transformer; and causal tree. Run IDs bind the complete
model, task, data, seed, training, source-tree, and config identity. A plan is
revalidated against its original config before resume, preventing a missing
model or seed pair from being accepted as a smaller complete campaign. No test
field exists in pilot or screen configurations.

Campaign execution is protected by three additional source contracts. The
execution adapter binds a constructed model to its exact model/pair run,
enforces the declared deterministic/thread policy, generates paired streams,
validates the full AdamW state and cursor, and only derives a compact child at
the curriculum parent's declared final cursor. The checkpoint codec is
non-pickle, allocation-bounded, canonical, checksum-bound, and restores the
exact model/optimizer/RNG continuation transactionally. The manifest stores
immutable content-addressed attempt transitions with atomic generation checks,
crash reconciliation, and checksum-bound external artifacts. The isolated
pilot worker derives its trusted checkpoint contract from the resolved plan,
while the parent independently verifies the clean tree, paired streams, every
checkpoint and compact artifact, and the manifest history. A fresh commit-bound
seven-lineage pilot passed from clean published commit
`8e5a7001fe543aabfa683fb1f8d7be393e3a0f30`. Its preserved non-claiming bundle
contains six trained-source results, one derived compact result, 18 canonical
checkpoints, one compact artifact, and the complete generation-32 immutable
manifest history. It used one development pair, 12 optimizer steps per source,
validation lengths 16/32/64, and no test or scaling stream. This verifies
execution, provenance, checkpoint creation and validation, paired streams, and
compact-parent lineage only.

The validation-only three-pair screen source is now implemented and frozen in
`configs/milestone4/validation_screen_v1.yaml`. Its plan contains 24 results:
18 independently trained sources across the six source conditions and six
rank-2/rank-4 compact children derived from their exact curriculum parents.
Each source receives 512 optimizer steps over mixed lengths 10/12/16/18, then
64 validation episodes at each of lengths 16/32/64/128/256. The runner now
validates routed recovery, consistency, and load/collapse summaries, aggregates
the three paired results, applies the predeclared source and compact-quality
gates, and writes a checksum-bound promotion record with a decision of
`complete_promote` or `complete_do_not_promote`.

The current local V3 suite collects 858 tests.  Its default run passes 854 with
four skips; the two opt-in Phase-II discovery regressions pass separately.
The remaining skips are the CUDA-only check and the deliberately unopened
40-environment Phase-II campaign regression.  This is local source
verification, not screen or scientific campaign evidence.
The screen remains non-claiming: it exposes no test or scaling stream, and its
schema-mandatory runtime declaration does not produce runtime or RSS evidence.
No screen run or promotion decision has yet been produced. Pilot-scale trained
checkpoints and commit-bound pilot evidence are published, but there is no
promotion-bound confirmatory campaign, matched-quality result, MERA result,
scientific result, or speed/RSS win. The functional cache path deliberately
performs strict public-state validation and immutable growth; comparative timing
must first use a predeclared, fair measurement path that accounts for that
overhead consistently across opponents.

## Exact algebra and trace-supervised coefficient learning

The Phase-I analysis now gives the binding task an executable exact semantic
algebra, strict legality certificate, and heldout-identifiability audit.  For
the frozen five-key/four-value task, the natural absence-aware realization has
dimension 21 while direct train support spans 20; unrestricted train behavior
therefore cannot determine the missing direction without an intervention or a
declared structural prior.

Phase II adds a trace-supervised transition-table learner, 19 causally censored
pseudoheldout folds, a shared-plus-key-local selector, and a balanced dynamic
probe suite.  In three retrospective passive seed pairs, the final models made
0 / 981 seen-query errors, learned the exact 20 generator-supported BIND,
UPDATE, and COPY entries with no realized local overrides, and answered
288 / 288 actual-cell queries and 72 / 72 focal queries correctly.  All 36
seed-by-probe-family rows and every declared path relation were exact.  The
selected penalty varied across seeds, so this supports repeatable coefficient
recovery inside the supplied addressable-register representation, not
discovery of a unique regularization strength, latent representation, or
assumption-free algebra.  A later audit found nonzero TRAIN error in 47 / 114
pilot fold-candidate fits, so their penalty comparisons are optimization-
confounded even though every final model and behavioral probe is exact.  The
records remain retrospective and non-claiming;
see `../docs/V3_PHASE2_LEARNING_THE_ALGEBRA.md` and
`../v3_recovery/PHASE2_TRACE_ALGEBRA_PILOT_SUMMARY.json`.

The subsequent frozen 40-environment execution passed every preopen fit gate
and produced exact behavior: 3,600 / 3,600 actual-cell queries and 72,000 /
72,000 rotated-control queries.  Its formal campaign result is still failed
(10 / 40), because a fixed 96-query acceptance check was valid only for
value-0 cells; the other cells generated 88 balanced queries and answered all
88 correctly.  The immutable failed record and separate behavioral audit are
in `../v3_recovery/phase2_outer_rotation_v3/`.  This outcome is not relabeled
post hoc.

The proposed Phase III follow-on removes the supplied register coordinates
from the estimator and tests exact predictive-state discovery from opaque
event/query symbols, including an actively acquired missing Hankel direction,
similarity-gauge invariance, and a fail-closed factorization certificate.  Its
preregistered design is in
`../docs/V3_PHASE3_PREDICTIVE_STATE_DISCOVERY.md`.
