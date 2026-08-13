# V3 Milestone 2 design: dynamic binding and persistent routing

This document fixes the interfaces and evidence rules for the second recovery
milestone. `HANDOFF.md` remains authoritative.

## Evidence boundary

Every generated batch has two deliberately separate surfaces:

1. **model-visible fields** describe the current causal event (event kind,
   surface key, visible argument, composite token, and padding mask); and
2. **evaluation-only fields** contain oracle routes, query targets, generation
   identities, and dependency parents.

Autonomous curriculum evaluation and fully latent execution receive only the
first surface. The label-independence audit runs identical visible inputs with
two different evaluation-label tensors and requires bit-identical routes and
logits. Oracle routes are an ansatz upper bound, not an autonomous result.

Generator-only settings (held-out key/value pairs, length bounds, and data
mixture probabilities) are also removed from the model construction object.
The model retains only surface vocabulary cardinalities and the branch count;
the held-out answer map is never stored on the model or its checkpoint.

## Dynamic document-local language

An episode is a sequence of bind, update, copy, invalidate, query, and
distractor events. Live bindings map surface keys to mutable values and local
branch identities. The generator records the event that last determined every
queried value, plus a secondary parent for copy events.

Distractors carry one visible scope bit distinguishing global-state updates
from null/read-only events. This non-permutation-symmetric state effect is
therefore learnable from causal input rather than being a hidden random label.

Surface-key identity cannot define a global route:

- each document independently permutes key-to-branch assignments;
- invalidated keys may be rebound to a different free branch and generation;
- train and evaluation splits include an explicit held-out key/value
  combination partition; and
- metrics align symmetric branch labels independently per document.

The generator must replay its own serialized visible program to reproduce all
query targets and must never exceed the configured simultaneous-live-binding
limit.

## Causal persistent router

At event `t`, the router may use the current route feature and bounded state
computed strictly from events `< t`: shared branch prototypes, occupancy,
relative age/load, and a bounded global summary. It scores branches in one
vectorized pass. It has no branch-specific embedding, no token-history cache,
and no all-pairs search.

The route decision is made before the current feature updates its selected
prototype. Padding changes neither router state nor its event clock. Parameter
keys and shapes are independent of runtime length.

Branch scores and probabilities are permutation equivariant. Hard routes and
updated states are equivariant whenever the winning score is unique; at an
exact symmetric tie, deterministic `argmax` uses the canonical lowest local
index, so no unique permutation-equivariant hard choice is claimed.

The three modes are separate contracts:

- **oracle** requires and executes generated routes;
- **curriculum** may use deterministic teacher guidance only while training,
  according to a recorded schedule; route supervision is applied only to the
  events selected by that same guidance mask and reaches zero at its declared
  endpoint; evaluation is always autonomous; and
- **latent** rejects route labels and has zero route-supervision loss.

Global and null decisions remain distinct. A global decision updates the
bounded global lane; a null decision advances a valid-event clock without a
lane update.

## Curriculum and diagnostics

The schedule is a pure function of optimization step, with explicit start,
hold, decay, and end values. Its value and realized guided fraction are stored
with every run.

Required diagnostics are:

- exact query accuracy and oracle gap;
- separate seen- and held-out-combination query counts/accuracy;
- route recovery after an exact per-document optimal permutation;
- route consistency grouped by document, surface key, and binding generation;
- branch/global/null load, normalized entropy, active-branch count, and
  collapse indicators; and
- a scalar branch-score work proxy proportional to `N * T * B`.

Empty masks have explicit counts and finite, documented metric values; they are
never silently dropped from aggregate schemas.

## Milestone acceptance

Before the Milestone 2 commit, the repository must contain:

- deterministic generator, router, integration, loss, and diagnostic tests;
- prefix token/route causality and evaluation-label-independence tests;
- a tiny fixed-batch overfit that demonstrates the optimization path works;
- a clean smoke matrix containing oracle, curriculum-autonomous, and fully
  latent conditions with exact seeds and config hashes; and
- a machine-readable record that distinguishes test/smoke evidence from any
  later multi-seed scientific claim.

The smoke runner binds evidence to a clean Git commit/tree, committed config
bytes, paired train and held-out batch hashes, paired initial weights and
parameter shapes, and imported package origin. It atomically rewrites an
external progress record after every completed condition, preserves a failure
record on error, and rechecks the checkout/configs before declaring success.
