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

This milestone uses supplied/oracle route IDs only. The constrained causal
router, dynamic document-local binding task, training stack, scientific
baselines, structural truncation, compact model export, and campaign closure
remain later milestones. The state serializer here is not the later compact
model exporter.
