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

## Portable artifact contract

Compact models are stored in a deterministic, non-executable binary format.
The fixed prefix checksum-binds a bounded canonical-JSON header and a contiguous
little-endian tensor payload. The header binds the strict model configuration,
export manifest, CP selection, model fingerprint, sorted tensor names,
dtypes, shapes, offsets, per-tensor checksums, module modes, and parameter
gradient flags. Loading rejects duplicate or noncanonical JSON, unknown fields,
unsupported devices or dtypes, malformed tensor ranges, excessive allocation,
trailing bytes, and every checksum or fingerprint mismatch. It never invokes
pickle or executes artifact-provided code.

The unkeyed digests detect corruption; they do not authenticate who produced
an artifact. Source-side manifest fields are exporter assertions that cannot be
re-derived from compact tensors alone. Commit-bound evidence must therefore
record and compare the expected source, manifest, selection, and complete
artifact fingerprints. The independent replay worker requires all four trusted
expectations rather than accepting provenance claimed only inside the artifact.

The independent replay worker accepts only bounded structured fixtures, loads
the artifact in a fresh Python process, and hashes all logits, routes, router
state, and forest state for both streaming and parallel execution.

## Commit-bound verification

Before a pruned reference campaign, a separate commit-bound audit must verify:

- source/config/selection/artifact checksums;
- midstream forest-state serialize/reload/resume under the compact model;
- real-update parity across binary carry boundaries;
- parameter, raw byte, operation, state-scalar, RSS, and wall-time records; and
- explicit wording that wall-time or memory differences are measurements, not
  matched-quality wins.

The strict source for this gate lives in
`v3/configs/milestone3/export_audit.yaml`. The commit-bound runner is
`v3/scripts/run_milestone3_export_audit.py`; it delegates every model/length
runtime and sampled-RSS observation to a separate
`v3/scripts/measure_milestone3_runtime.py` process.

That implementation gate passed on clean published commit
`7f1c9ead80e93dddb03c51bf608ed24d7e41d719` (tree
`f129139655fc6542904c96128976e2d5b6b282c0`). The reviewed evidence is
`v3_recovery/MILESTONE3_EXPORT_AUDIT.json`, accompanied by the exact evaluation
fixture, fresh-process replay record, native compact artifact, and checksum
manifest. All 24 top-level gates passed, including declared-tolerance parity,
local and global carry depths 4/5/6, canonical resume, and 18 isolated runtime
workers. The native artifact is 41,049 bytes with SHA-256
`27fdb5df3cc236bdbfd8f4a69a2344c2d26c03f2f565e8923d8eb6d12ddc9d4d`.

This is an implementation audit, not a scientific compactness campaign. The
rank-4 choice was made by the data-independent `parameter_energy_v1` structural
proxy; no predictive-quality threshold was applied. Runtime and RSS values are
descriptive, and the audit does not claim a speed, memory, matched-quality, or
persistent-state-width win.
