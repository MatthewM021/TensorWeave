# Phase III-T2: Learner-Selected Guarded Predictive Dynamics

Status: the official bounded T2 campaign completed and passed its frozen
acceptance gate. The result is a source/runtime-bound honest-code synthetic
rehearsal within the supplied finite grammar, legality mask, and candidate
pool; it is not a total-WFA, global-minimality, natural-language, or
assumption-free discovery result.

## Question

T1 showed that exact partial event maps can be recovered when a trusted
controller supplies a semantically designed 15-edge excitation basis. T2 asks
the next narrower question:

> Can an estimator that receives only opaque diagnostics, an explicit
> legality mask, passive edge labels, and an unordered candidate pool choose
> its own legal-domain excitation basis and thereby identify arbitrary-length
> guarded behavior?

The target remains Contract B. Illegal actions are undefined. T2 must not
fill them with zero, no-op, absence, or dead-state behavior, and it must not
serialize total event operators.

## What remains supplied

The experiment supplies:

- the complete opaque event, query, and answer vocabularies;
- the nine opaque source representatives and the full `3 x 3` categorical
  output grammar;
- the exact `9 x 10` defined/undefined source--event mask;
- 21 passive legal edge observations in each omission environment;
- the unordered 23-edge complement as the permitted membership-query pool;
- the promise that diagnostics are output channels, not event operators; and
- exact arithmetic, resource ceilings, and deterministic tie-breaking.

These are substantial structural assumptions. T2 does not discover the state
grammar, legality mask, event vocabulary, or candidate language.

The finite version space is exact. A source-assignment version is a bijection
between the currently unlabeled mask-source representatives and unused rows of
the nine-row categorical codebook. For each event, a map version is represented
by its categorical image rows on a deterministic basis of that event's legal-
source span. It survives only if its linear extension agrees with every known
constraint and maps every declared legal source back into the codebook.
Versions that differ only off the legal-source span are one equivalence class
and are never enumerated separately. Thus an event of legal-domain rank `d`
has at most `9^d` finite versions; its infinite off-domain rational extensions
remain represented only by the nullity witness.

## What the learner must choose

The trusted controller's T1 `15 + 8` ordering is erased. The T2 input
canonicalizes passive rows, both domain-mask halves, and the complete candidate
set by opaque content digests; recomputes fresh primitive-row commitments; and
drops every T1 object, input hash, response ordinal, active/sealed tag,
semantic family, cell identifier, and controller field.

At round `r`, the pure learner may receive only:

1. the immutable T2 input;
2. its own prior transcript through round `r - 1`; and
3. categorical answers already returned in those prior rounds.

It must commit the next request hash before the controller releases that
request's two diagnostic labels. Eager batches and future-response fields are
forbidden.

## Frozen acquisition rule

The policy is a causal frontier-plus-version-space rule over learner-visible
rows.

1. Reconstruct the known opaque word-to-diagnostic table from passive rows and
   prior responses.
2. Restrict the pool to unanswered requests whose source word is known.
3. Retain only requests whose source diagnostic row raises the directly
   observed-plus-inferred source rank for that event token.
4. For every eligible request, enumerate the exact categorical target rows
   across all remaining full-product source assignments and all rational
   legal-domain maps that agree with known edges **and map every declared
   legal source back into the nine-row categorical codebook**.
5. Before choosing any ambiguous request, iterate all singleton consequences
   to a deterministic fixed point. This includes a forced final source-row
   assignment and any edge whose categorical-map version set has one image.
   Every such edge is recorded as structural inference; its controller answer
   remains hidden.
6. If ambiguity remains, prefer a request whose target word is an as-yet-
   unlabeled member of the nine source representatives; within that frontier,
   maximize compatible-outcome count and then use request SHA-256. Once the
   source assignment is complete, partition the selected event's finite map
   versions by every possible two-label outcome and minimize `(worst posterior
   global-version product, worst event-version bucket, negative distinct-
   outcome count, request SHA-256)`.
7. Open exactly the committed request, update the transcript, and return to
   the singleton fixed point.
8. Stop only when the final directly observed plus structurally inferred
   constrained rank equals the legal-domain rank for every event token. The
   unqueried requests remain sealed.

The algorithm may adapt to earlier answers. Its code and scoring rule are
frozen before any answer, while each individual request is committed before
its own response. "Learner-selected" therefore means causal selection within
the supplied pool, not unconstrained program synthesis.

For the current `K=2,V=2` guarded toy, the raw exact deficit is 15 source
directions, or 75 rational image parameters. One edge response can directly
fix the five-coordinate image of one independent source direction. The exact
version-space policy obtains one further constraint without an oracle call:

- after two frontier answers, the three initially missing source-row
  assignments collapse `6 -> 2 -> 1`, fixing the last representative and its
  literal rank-gaining edge.

The primary target is consequently `21 passive + 14 queried + 1 structurally
inferred + 8 sealed = 44`. These counts are protocol expectations to be tested,
not values handed to the selector. The 15 added constraints form a minimum
relative source-rank basis, but the protocol does **not** claim that 14 is the
globally minimum number of adaptive membership calls. A truth-specific
13-query distinguishing set exists in the declared environments, but choosing
it uses knowledge of which posterior bucket is actual; it is reported only as
a posthoc teaching-set control, never as autonomous-selection evidence.

## Exact certificates

Each choice certificate binds:

- the immutable input and prior-state hashes;
- the canonical pending-pool hash;
- the known-word and observed-edge hashes;
- source availability;
- the event's before/after observed-source ranks;
- separate direct-observation, structural-inference, and final-constrained
  rank increments;
- whether the request expands the source-representative frontier;
- every learner-visible compatible categorical outcome class;
- the deterministic score and tie-break rows;
- the selected request hash;
- zero future-response fields and zero semantic/controller reads; and
- a choice commitment produced before the linked response.

The terminal learner artifact binds all choices and responses, the exact
rank-5 realization, every restricted event-map certificate, the structurally
inferred edge, the `6 -> 2 -> 1` source-assignment witness, every categorical
map-version count, the eight unopened sealed candidate requests, and the
unchanged aggregate off-domain extension nullity of 80. Each inferred-edge
certificate must be recomputable without a controller answer and must reject
any hidden-answer hash or response ordinal. Every `total_operator` remains
`null`.

Before selection, the trusted controller commits every candidate answer with
a distinct high-entropy hiding salt and binds request-SHA-ordered leaves of
`(request payload, answer, salt)` in one root. Unsalted hashes are forbidden
because each answer has only nine possible categorical values. The pure learner
receives no root, leaf hash, salt, proof, or raw controller nonce: even an
apparently opaque root would be an arbitrary controller-generated side channel.
The trusted wrapper binds a completed choice to the precommitted root only
after selection and verifies the corresponding inclusion proof when a response
opens its own answer and salt. The inferred-answer sidecar, eight independent sealed
edges, and long/path answers remain hidden until all eight omission models and
one terminal pre-open aggregate are frozen. The atomic postfit phase verifies
all 23 leaves: 14 already opened during acquisition, one inferred, and eight
independent sealed.

## Arbitrary-length guarded behavior

T2 does not claim a total weighted automaton over all strings. It constructs a
finite guarded predictive machine:

- nine categorical predictive states;
- ten opaque event tokens;
- 44 defined transitions;
- 46 explicitly undefined source--event pairs; and
- two opaque diagnostic readout channels.

After 14 selected responses plus the explicit structural inference, each
defined transition is either directly observed or exactly predicted from an
identified restricted map. The trusted postfit evaluator binds the empty-word
initial-state pair and controller-supplied opaque event-token correspondence,
then compares all nine state/readout pairs and all 90 source--event pairs with
the sealed controller graph. An exact nine-state bisimulation, closed under
all 44 legal transitions and rejecting all 46 undefined pairs, proves by
induction that every finite legal suffix has the same diagnostic behavior. The
long/path programs are implementation checks, not the proof. This is
arbitrary-length **guarded-language equivalence**. It is not global matrix
equality or behavior on illegal words.

## Experimental arms

The bounded report uses fresh T2 controller nonces while retaining T1's design
shape: two independently opaque full-support controls and eight omission
environments, four omitted cells under two nonce-keyed relabelings. Already
opened T1 environments are not reused as new evidence. Full controls need no
active responses. Omission environments run
the causal selector. Only after all eight omission models and their terminal
aggregate are frozen does one atomic postfit phase open all nine still-hidden
edge sidecars--the inferred answer plus eight independent edges--together with
the sealed long/path programs.

Postfit controller correspondences may be used to audit paired-relabel
similarity and semantic correctness. They never enter selection or fitting and
must be reported as supplied audit structure rather than learned alignment.

## Mandatory attacks and controls

Acceptance requires all of the following:

- candidate-order permutations produce the same canonical decision process;
- opaque vocabulary/output relabelings preserve counts, closure, and behavior;
- controller T1 ordering, hashes, ordinals, and active/sealed tags are absent;
- mutating the trusted answer-commitment root cannot change any pure learner
  choice, state, or model byte;
- monkeypatched semantic/controller helpers cannot be reached by pure learner
  functions;
- every choice is reproducible from only its prior transcript;
- counterfactual future answers cannot change an already committed choice;
- both alternative first-response branches are replayed from the exact visible
  version space; if their larger rank deficits exceed the frozen sealed-quota
  or call budget they return an honest `not_identified`/cap result rather than
  silently pruning those outcomes;
- removing a certified cocircuit whose loss cannot be exchanged with one of
  the eight redundant candidates yields `not_identified`, never a fallback;
- forged masks, preanswered candidates, duplicate responses, response/request
  mismatches, and out-of-order responses fail closed;
- the inferred-answer sidecar, eight unopened independent edges, and
  long/path programs remain absent from every fit-time artifact;
- all undefined pairs remain rejected rather than zero/dead encoded;
- all ten event tokens occur in the complete `9 x 10` mask,
  `passive + candidate` is exactly the 44-edge legal support, and neither query
  token enters the event alphabet;
- exact off-domain twins remain possible and the nullity stays 80;
- the finite bisimulation checks all 90 pairs, rather than a sample of paths;
- constant/default-answer, identity/no-op, event-only, source-state-only, and
  reuse-the-first-14-T1-edges shortcuts each fail a sealed or bisimulation gate
  under both relabel blocks;
  and
- nested hashes are contextual links, while the authoritative environment and
  report constructors replay every semantic invariant.

## Acceptance gate

The bounded T2 gate passes only if:

1. both full controls recover all restricted maps with zero active calls;
2. all eight omissions begin at diagnostic rank 4;
3. every request is learner-selected and causally committed;
4. all eight omissions terminate after exactly 14 responses / 28 returned
   labels plus one answer-sidecar-free structural inference, without access
   to the old T1 partition;
5. every omission has exact diagnostic rank 5 and all nine categorical
   codebook rows;
6. every event reports its direct observed rank and any provenance-distinct
   inferred increments; exactly one inferred edge occurs per omission, and
   only the final constrained rank equals each legal-domain rank;
7. all eight remaining sealed candidate edges and all sealed long/path programs are
   exact;
8. all 44 defined and 46 undefined graph entries are exact in every
   environment;
9. the guarded bisimulation certifies arbitrary-length legal behavior;
10. paired opaque realizations are gauge-equivalent under the disclosed
   controller-supplied audit alignment; and
11. every total operator is absent and total-extension nullity remains 80.

Aggregate omission-arm arithmetic must be exact: 112 oracle responses / 224
returned labels, eight structurally inferred constraints, and 64 independent
sealed edges. Including the two full controls, the complete report binds 256
passive, 112 queried, eight inferred, and 64 sealed legal edges = 440
environment-edge rows, plus 460 undefined-pair rejections and 120 long/path
programs.

All enumeration, response, exact-rank, transcript, certificate, and postfit
ceilings are preflighted before hidden-answer access. A fresh
machine-readable T2 config binds controller nonces, salted answer commitments,
source/runtime inventory, the selection rule, budgets, and failure policy.
Canonical transcript/result hashes are evidence links whose semantic authority
comes from full environment/report reconstruction. This remains an honest-code
synthetic rehearsal, not cryptographic isolation or prospective evidence about
an unknown natural system.

## Official bounded execution

The fresh official execution completed all ten scheduled arms: two full-
support controls followed by eight omissions spanning four omitted cells under
two independently opaque relabel blocks. All ten preopen records were frozen
before the terminal aggregate authorized a single atomic postfit open. The
frozen identities are:

- protocol SHA-256
  `7c5ee8bcee72e0af5ac2d8404f54b479e1b7d1b1200922ec40caf66483c04292`;
- source/runtime binding SHA-256
  `514ebb445d3eb00e456095bf3377bf4f7eb2e15a4282e0765ca307b5203e5e90`;
- terminal record SHA-256
  `31b3849c7a469ed380c68502193287e4917ee07170bbbd9c3c95d837055c1352`
  and raw-file SHA-256
  `e5e2524285c970eff1e45474d43aee43cf26615ce8ae89aa19b43dd0aa5b0819`;
  and
- opened report record SHA-256
  `9993913e5f60a73ce41fb08803c7dab24511165a4dc636e89b065d68be59c40f`
  and raw-file SHA-256
  `89b1fa8fa2d5f21fd001143d8275fcde94acdac964169828d55f566386ae5bd9`.

The primary causal learner used 112 membership responses containing 224
categorical labels across the eight omissions, made eight answer-sidecar-free
structural inferences, and left 64 candidates sealed. Every omission therefore
realized `21 passive + 14 queried + 1 inferred + 8 sealed = 44` legal edges.
Across all ten arms, all 440 defined rows and 120 sealed long/path programs were
exact, all 460 undefined pairs were rejected, and five controller-aligned
postfit similarity certificates passed. All ten restricted event-map version
spaces were singleton in every arm; no total operator was serialized and the
off-domain total-extension nullity remained 80 per environment.

The controller-supplied postfit negative control that reuses the first 14 T1
excitation edges remained nonidentified in all eight omissions: one restricted
event retained nine compatible versions in each arm. Thus the T1 ordering is
not a substitute for the T2 learner's causal selection rule.

The truth-aware postfit teaching-set control selected 104 counterfactual
queries, derived 16 answer-free singleton inferences, left 64 candidates
unopened, accounted for 208 categorical labels, and made zero new membership
calls. It closed all 80 omission-arm restricted maps, but it is explicitly
truth-specific and noncausal, and is ineligible for autonomous selection,
confirmation, a global query-minimality claim, or a total-operator claim.

The authoritative records are preserved under
`v3_recovery/phase3_t2_opaque_active_discovery/`. Full reconstruction, rather
than any nested digest considered alone, is the semantic evidence boundary.
The canonical `SUMMARY.json` has record/file SHA-256 values
`d70a52a187b8341bdfc072043fdba629803f5177385181109dd49ac191c6929b`
and `d44abe1afb0e175fb160efccabfe084390018b1c87a994ecdbacfae74a68c7a3`.
The 14-row `EVIDENCE.sha256` manifest has raw SHA-256
`1ca40ffb794134599f39ec3ce6db25b9b58f00d2430658d938df91554f3dba00`.

## Claim boundary

If the gate passes, the strongest justified statement is:

> Within a supplied finite categorical state grammar, legality mask, and
> membership-query pool, an opaque exact learner autonomously selected a
> rank-closing legal-domain excitation basis and recovered a guarded predictive
> machine that is behaviorally correct for every finite legal suffix, up to
> predictive-coordinate similarity.

It would not demonstrate autonomous vocabulary discovery, program synthesis,
total-WFA identification, off-domain semantics, tensor-factor discovery,
cryptographic isolation, or learning from natural language.
