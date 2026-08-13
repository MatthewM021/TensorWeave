# V3 Milestone 4: Paired Reference Controls and Campaign

## Scope and sequencing

Milestone 4 begins by implementing the strong causal opponents required by the
handoff. No reference campaign starts until their state, cache, causality,
padding, and configuration contracts pass deterministic tests from a clean
published commit.

The first source checkpoint contains:

- a recurrent binding baseline with bounded hidden state; and
- a causal Transformer baseline with a real per-layer key/value cache.

A separate source checkpoint adds a corrected causal complete-tree TTN control.
MERA remains an optional identity-initialized ablation and cannot support a
positive claim without a paired multi-seed TTN comparison. Whole-`d_model`
state-channel pruning is not part of this milestone: the verified Milestone-3
export intentionally reduced only the shared CP interaction axis.

## Shared information boundary

Every opponent consumes only `BindingModelInputs`: event kind, primary and
secondary visible keys, visible argument, token identifier, and validity mask.
Model configuration contains only sanitized vocabulary/architecture sizes. It
does not contain held-out combinations, generator mixture probabilities,
sequence-length limits, oracle routes, query targets, dependency parents, or
generation identifiers.

Query targets may enter the declared predictive training loss, but evaluation
metadata and oracle routes never enter either baseline's forward API. A padded
position is a total no-op: it produces zero logits, does not advance a state or
cache, and contributes no parameter gradient even if its ignored integer fields
contain extreme or out-of-vocabulary values.

## Recurrent control

The recurrent control uses a stacked GRU transition over the visible event
encoding. Its persistent state is the hidden tensor plus per-row real-event
counts. State size is independent of context length. Full-sequence execution,
one-event stepping, chunked continuation, and resumed execution use the same
transition and must agree within the declared dtype tolerance.

## Cached causal Transformer control

The Transformer uses analytic, length-independent positions and an explicit
key/value cache for every attention layer. Each valid event appends exactly one
key and value per layer for its document; padding appends nothing. A query may
attend only to occupied current-or-past cache entries in the same document.
There is no learned maximum-length table and no constructor field that grows
parameters with evaluation length.

The cache record exposes layer, batch, head, real-event, and head-dimension
axes so scaling measurements can count actual state elements. Full-sequence
training, token stepping, and chunked resume must produce the same causal
outputs within tolerance. Campaign runtime measurements must use the cached
step path rather than repeatedly recomputing a dense prefix.

## Corrected causal complete-tree TTN control

The tree control is a fixed, single-lane causal opponent rather than a routed
forest. Its persistent state is a canonical binary frontier. Each real event is
appended by chronological binary carry, and the current complete-tree root is
reconstructed from occupied scales from low to high: an older block is always
the left child, a newer suffix is always the right child, and an absent child is
promoted unchanged. The same scale-shared CP merge is used at every depth with
its global conditioning path active. A simple root readout replaces routed
forest attention.

Positions are analytic functions of real-event counts, so the model has no
learned maximum-length table, per-depth parameter list, route input, or
constructor depth limit. State grows as `O(d_model log L)`, between the bounded
GRU state and linear Transformer cache. Full execution, token stepping, chunked
continuation, and a tensor-only model/state roundtrip follow the identical
causal transition.

This corrects the V2 control's learned maximum-length position buffer,
length-dependent level modules, whole-context-only output, and missing
persistent continuation contract. It does not revive V2 adaptive rank gates or
make a MERA claim.

## Source-checkpoint gates

Before a baseline source commit is published, tests must cover:

- strict configuration parsing and sanitized task metadata;
- deterministic construction and finite forward/backward behavior;
- full versus stepped versus chunked-resume parity;
- prefix causality and future-token independence;
- arbitrary padding values and all-padding total no-op behavior;
- recurrent-state and cache validation;
- cache growth by real events and one cache per attention layer;
- parameter-count independence from evaluated context length; and
- structural metrics for parameters and persistent state/cache elements.

Passing these tests establishes usable controls, not comparable quality.

The recurrent, cached-Transformer, and corrected causal-tree source contracts
are implemented and pass the complete local V3 suite. This establishes source
behavior only: there is no trained baseline checkpoint, commit-bound campaign
evidence, matched-quality comparison, MERA result, or scientific result. The
public cached step path prioritizes strict state validation and immutable
continuation; a comparative runtime campaign must predeclare a measurement path
that accounts for validation and cache-growth costs consistently before making
efficiency claims.

## Paired campaign boundary

The later campaign uses immutable fixtures and shared seed blocks across the
routed forest, compact exports, recurrent control, cached Transformer, and tree
controls. It separates validation-only selection from locked test evaluation,
uses at least three paired strong seeds, trains on mixed real lengths, evaluates
true longer-update sequences through 256, and reserves 512/1,024/2,048 for
scaling audits.

Matched-quality runtime or memory claims require a predeclared quality rule and
quality-qualified models. Measurements use isolated processes, batch-one
streaming, the complete router/state/cache cost, fixed thread settings, warmups,
raw timing samples, and confidence intervals. Until that campaign is complete,
baseline and compact runtime/RSS values are implementation measurements only.
