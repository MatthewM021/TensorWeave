# Phase III: Predictive-State and Compositional-Factor Discovery

## Status and purpose

The full arbitrary-suffix operator and factorization campaign remains a
pre-execution research protocol.  It specifies the first full operator-and-
factorization experiment that removes the supplied addressable-register
coordinates used in Phase II. Bounded zero-suffix diagnostic, controller-
supplied guarded partial-operator, and learner-selected guarded partial-
operator checkpoints have now been completed and are reported explicitly
below; none of the broader total-operator, factorization, or scaling results
has yet been obtained.

The primary question is deliberately narrower than "discover a geometry of
language":

> Can an exact learner recover a minimal predictive state and its observable
> operators from opaque event symbols and query answers, close a deliberately
> missing predictive direction with an active observation, and then recover a
> role--filler factorization from the learned operators up to gauge?

There are three nested claims, and the campaign must report them separately:

1. **Predictive-state identification:** recover an exact minimal linear
   realization of the declared observable series, up to similarity.
2. **Active completion:** identify the one direction missing from passive
   support using a bounded, label-bearing intervention selected without
   semantic coordinates.
3. **Compositional-factor discovery:** recover mutually commuting role and
   filler algebras, and low operator-Schmidt ranks, from the identified
   operators rather than from supplied key/value blocks.

Success at level 1 does not imply level 2, and success at levels 1--2 does not
imply level 3. A tensor-network scaling claim requires a later experiment in
which the discovered factor ranks and the cost of identifying them remain
controlled as the task family grows.

This protocol is exact and finite. It does not claim that unrestricted
natural language has finite Hankel rank, an exact finite predictive state, or
the same role--filler decomposition.

## Questions frozen before execution

The campaign answers the following questions in order.

1. Does the sequence-only learner reproduce the known `4 -> 5` restricted-to-
   full rank change for `K=2, V=2` under the absence-aware contract?
2. Can it select one opaque membership query whose answer raises the rank,
   without being told which event denotes a key, value, bind, or query?
3. Does the same procedure reproduce the `20 -> 21` change for the full
   `K=5, V=4` task in every rotated omitted-cell environment?
4. Are independently learned minimal realizations exactly equivalent under a
   single invertible similarity map, rather than merely behaviorally accurate
   on the finite fitting table?
5. On the full 21-dimensional absence-aware realization, can an algebraic
   search recover the decomposition

   \[
   \mathbb R^{21} \cong \mathbb R\,1 \oplus
   (\mathbb R^5 \otimes \mathbb R^4)
   \]

   up to global similarity, local role/filler gauges, and the unavoidable
   factor swap when the two factor dimensions are equal?
6. Do the recovered event operators have the exact low operator-Schmidt ranks
   predicted by the binding algebra after, but not before, that factorization?

The known answers from Phase I are used only by the sealed evaluator and by
positive controls. They are not inputs to the learner.

## Observable object: a multi-output behavioral series

Let `Sigma` be a finite alphabet of opaque event tokens. A history is a word
`p in Sigma*`. Each contract defines a family of observable scalar series

\[
f_c(p) \in \mathbb Q, \qquad c \in \mathcal C,
\]

where `c` indexes an output channel. Categorical answers use exact one-hot
coding. A terminal test is a pair `(s, c)` consisting of a suffix word and an
output channel. For prefix set `P` and test set `T`, the multi-output Hankel
block is

\[
H_{P,T}[p,(s,c)] = f_c(ps).
\]

For a partial contract, this entry exists only when `ps` is in the declared
domain. Undefined entries are absent constraints. They must never be filled
with zero, an `ABSENT` label, or an illegality label, because doing so would
silently change the observable contract.

A predictive state is the row functional

\[
q(p) = \bigl(f_c(ps)\bigr)_{(s,c)\in T}.
\]

The learner's state therefore means "predictions of future observable tests."
It is not an estimate of a privileged semantic register vector. A minimal
linear realization is a tuple

\[
(\alpha,\{A_a:a\in\Sigma\},\{\omega_c:c\in\mathcal C\})
\]

satisfying

\[
f_c(a_1\cdots a_n)
= \alpha^\top A_{a_1}\cdots A_{a_n}\omega_c.
\]

All fitting, rank decisions, and equality checks in this phase use exact
rational arithmetic. Floating-point SVD thresholds are not admissible as
evidence. Modular arithmetic may accelerate elimination, but the final basis,
operators, ranks, inverses, and identities must be reconstructed over the
rationals and directly verified.

## Three contracts, three different experiments

The campaign must never pool data, ranks, or conclusions across the following
contracts.

| Contract | Domain and outputs | Frozen dimensional fact |
|---|---|---|
| **A. Promised-valid query behavior** | Only generator-promised histories; a query is scored only for a live key and returns one of `V` value classes. Invalidity and absence are unobserved. | Exact realization upper bound `1 + K(V-1)`: 3 for the toy and 16 for the full task. The minimum rank is to be measured, not assumed. |
| **B. Absence-aware diagnostic behavior** | Admissible histories plus a diagnostic query that returns `ABSENT` or one of `V` values for every key. Illegality is not made into an output. | Exact minimal dimension `1 + KV`: 5 for the toy and 21 for the full task. With one cell direction excluded, the restricted support ranks are 4 and 20 respectively. |
| **C. Strict legal-language behavior** | Every word is interpreted. An illegal action enters an observable absorbing dead state; legality/death and state diagnostics are outputs. | The full frozen `K=5, V=4` contract has certified rank 192. Its rank is not 21. The toy strict rank is recomputed exactly rather than inferred from the absence-aware result. |

The `4 -> 5` and `20 -> 21` experiments below refer only to Contract B. They
describe the rank of the zero-suffix diagnostic state table before and after
one missing predictive direction is excited. They are not claims that every
event operator has been identified. They are also not claims that the strict
series changes rank from 20 to 21, nor that the promised-valid series has
those dimensions.

Contracts A and B are masked, domain-restricted problems. The learner may use
only complete rectangular subblocks and exact equations whose words are
declared defined by the sanitized transcript. Contract C is total and includes
the dead behavior. Contract B makes absence observable while keeping illegal
words outside the series. Each report records the domain mask as visible
supervision and as part of the evidence.

For a partial contract, let `D_a` be the span of reachable predictive states
from which event `a` is legal. Observations identify at most the restricted map
`A_a|D_a`. Even when the ambient predictive dimension is `r`, a total `r x r`
extension retains at least

\[
(r-\dim D_a)r
\]

free rational parameters. Undefined actions do not supply zero rows, an
`ABSENT` result, a no-op, or a dead-state transition. A report may serialize a
legal-domain basis and its images; it must not serialize one preferred total
extension unless a separately declared total contract identifies it.

## Opaque event/query firewall

### Information available to the learner

For each environment, the learner receives only:

- opaque event tokens with equality preserved within that environment;
- episode or word boundaries;
- a per-position marker saying either `NO_OUTPUT` or `OUTPUT`;
- an opaque categorical output token where an output exists;
- the declared contract name, so partial and total tables are not confused;
- the passive observation-table entries and domain mask authorized for that
  phase; and
- in the active phase, a finite pool of opaque candidate words and a one-call
  membership interface.

The event alphabet is anonymized by a fresh bijection in each environment.
The output labels, including `ABSENT` where that output exists, are independently
permuted. Opaque identifiers are fixed-length random-looking byte strings; no
identifier contains an event kind, key number, value number, lane, arity, or
argument boundary. Repetition of a token remains observable because it is
part of the sequence behavior.

The learner is not given `K`, `V`, the live cap, canonical states, event
arguments, the omitted cell, the exact executor, transition records,
dependency parents, heldout masks, oracle routes, generation IDs, semantic
task fingerprints, or probe-family labels. It is also not given a map saying
which opaque symbols are binds, updates, copies, invalidations, or queries.

The number of distinct input and output symbols is observable. Inferring a
possible factor dimension from that cardinality is allowed and must be
reported. Reading a hidden semantic map is not.

### Controller/learner separation

The trusted controller owns semantic generation, the opaque bijections, the
contract domain, and the sealed answers. It writes a sanitized, content-bound
input and invokes the learner in a process that cannot import the exact
executor or open controller sidecars. The learner writes a model and a
certificate into a separate output directory.

Before fitting, the implementation must pass all of these firewall tests:

1. randomize every forbidden sidecar field while leaving the sanitized input
   fixed and require the learner artifact to remain byte-identical;
2. place canary values for the omitted cell and semantic event names in
   forbidden files and require that none occur in process reads, logs, or the
   artifact;
3. deny imports of the semantic executor, generator internals, Phase-I
   analysis modules, and Phase-II register learner in the learner process;
4. run the same behavior under a second input/output-token bijection and
   require equivalent behavior up to token relabeling and latent similarity;
5. invert the output permutation only in the evaluator, never in fitting or
   active-query selection; and
6. prove from an access log that the preactive learner made zero sealed
   membership calls and that the active learner made at most the declared
   number.

These checks establish an interface boundary. They do not make the controller
or operating system cryptographically trusted.

## Exact multi-output Hankel/PSR learner

The learner is deterministic once its sanitized input is fixed. It has no
random initialization, gradient descent, early stopping, or best-seed choice.

### Basis construction

1. Begin with the empty prefix and every observable zero-length output test.
2. Extend the current prefix and suffix-test frontiers by opaque tokens in
   their serialized byte order.
3. Query only entries present in the passive table. For a partial contract,
   retain the exact definedness mask and form pivots only from fully defined
   candidate subblocks.
4. Add the lexicographically first row or column whose exact residual is
   nonzero relative to the current basis.
5. Continue until one-symbol prefix and suffix closure add no rank, or until a
   frozen resource cap is reached. Hitting a cap is a failed environment, not
   a lower-bound result promoted to an exact rank.
6. Minimize the resulting representation by exact reachable/observable
   reduction and emit the pivot words, tests, masks, and elimination
   transcript.

For a total contract, let `B` be the resulting invertible `r x r` basis block
with basis prefixes `p_i` and basis tests `(s_j,c_j)`. Define

\[
H_a[i,j] = f_{c_j}(p_i a s_j).
\]

With row-state convention, one exact realization is

\[
A_a = H_a B^{-1}, \qquad
\alpha^\top = H[\epsilon,T_B]B^{-1}, \qquad
\omega_c = H[P_B,(\epsilon,c)].
\]

For masked Contracts A and B, the learner instead solves the corresponding
exact linear systems from all defined entries. For event `a`, it records a
basis of `D_a`, the images of that basis, the observed-source rank, and a basis
annihilating `D_a`. The latter supplies explicit nonzero perturbations that
leave every legal observation unchanged and certifies the total-extension
nullity above. The learner may call the restricted map identified only when
the observed legal-source span equals `D_a`. It may emit a total operator only
when a separately totalized contract gives a unique solution on the complete
reachable/observable quotient. Minimum-norm, zero, no-op, and dead-state
completions are forbidden priors.

### Exact reconstruction certificate

Every accepted model must certify:

- the exact rank and all selected pivot words/tests;
- `det(B) != 0` over the rationals;
- every table entry reconstructed exactly, including entries not used as
  pivots;
- one-symbol prefix and suffix closure;
- reachable rank and observable rank both equal to `r` after minimization;
- every transition/readout equation used to derive the model;
- exact replay of all authorized passive words;
- the complete list of undefined entries for a partial contract; and
- no NaN, infinity, tolerance, singular-value cutoff, or floating comparison.

An independent verifier reconstructs the model from the transcript rather
than trusting serialized summary booleans.

## Gauge and similarity: what representation recovery means

Latent coordinates are not identifiable. For a total realization, if `G` is
invertible, then

\[
\alpha^\top G,\quad
G^{-1}A_aG,\quad
G^{-1}\omega_c
\]

realize the same series. Raw matrix equality, coordinate correlations, and
neuron-by-neuron alignment are therefore invalid success criteria.

For a learned total minimal realization `L` and a reference realization `R`,
the evaluator solves for one rational invertible map `G` satisfying

\[
\alpha_L^\top=\alpha_R^\top G,\qquad
A_{L,a}=G^{-1}A_{R,a}G,\qquad
\omega_{L,c}=G^{-1}\omega_{R,c}
\]

for every opaque-token/output-token correspondence. Acceptance requires one
and the same `G` for the initial state, all event operators, and all readouts.
Matching each operator with a different map is forbidden.

For a partial realization, the evaluator instead requires one common `G` for
all predictive states and readouts and checks every legal edge in the form
`q_L(p) = q_R(p)G` and `q_L(pa) = q_R(pa)G`. Domain bases and their image bases
must transform by that same `G`. No equality is requested between arbitrary
off-domain matrix completions, because those completions are not behavioral
objects of the guarded contract.

`G` is first solved from a frozen alignment set and then checked on disjoint
operators, readouts, prefixes, and suffixes. For the omission experiments, a
map fitted on the passive rank-20 span cannot certify the 21st direction.
Full similarity is evaluated only after the active answer and full-rank model
are frozen. Sealed probes used to solve `G` are not reused as validation
probes.

An overcomplete realization is not called equivalent merely because it fits.
It must first reduce exactly to its reachable/observable quotient; that
minimal quotient must have the certified rank and satisfy the same single-`G`
test. Extra modes must be unreachable, unobservable, or both.

## Stage T: `K=2, V=2` rank restoration and guarded-map recovery

The toy stage is a mandatory debugging and falsification stage, not optional
warm-up evidence. It is split into two claims that must not be conflated.

### T0: zero-suffix diagnostic rank

- Use the absence-aware contract with a live cap of 2.
- Rotate the omitted pair through all four key/value cells.
- Run two independent opaque input/output bijection blocks, for eight
  environments total.
- Enumerate the complete zero-suffix diagnostic support subject to the
  one-cell exclusion; do not substitute a finite random trajectory sample.
- Precommit every passive table, opaque vocabulary, candidate-pool hash, and
  candidate order before fitting any environment.

The expected passive diagnostic rank is 4. The learner is not told that
number. It must discover rank 4, emit exact passive-consistent continuations
that disagree on an authorized active candidate, and decline to make an
identified prediction for that disagreement.

The answer-free acquisition rule selects the missing atomic vocabulary item
and its two terminal diagnostic channels. Every compatible categorical
response must raise the finite diagnostic table to rank 5. Exactly one
membership response, containing the two categorical diagnostic tokens, is
opened per environment. The learner rebuilds the finite diagnostic block from
scratch, reconstructs all nine zero-suffix states, and eliminates the exhibited
state-row disagreement. A known-row repeat control must remain rank 4.

This one response identifies the fifth diagnostic state direction. It does
**not** identify all action maps or arbitrary suffix behavior.

### T1: partial operators on legal source domains

The guarded toy has ten state-changing event tokens: four binds, two updates,
two ordered copies, and two invalidations. The two diagnostics remain output
channels rather than event operators. Across nine reachable states there are
44 legal state--event edges and 46 undefined edges. Undefined edges stay in a
learner-visible definedness mask and never receive synthetic target values.

For each of the four omissions under two nonce-keyed opaque relabelings, the
trusted controller causally censors every edge whose source or target enters
the omitted cell. This leaves exactly 21 passively labeled edges. The
controller supplies and content-binds a fixed 15-edge legal-domain excitation
basis; that basis was designed with semantic knowledge and is not claimed to
have been discovered by the learner. Its first response restores diagnostic
rank `4 -> 5` but must leave nine restricted event maps unidentified. Only
after this negative checkpoint are the other 14 responses opened. Thus an
omission environment uses 15 membership responses and 30 categorical labels,
not one answer.

The remaining eight legal edges are sealed redundancy checks. Acceptance
requires the exact partition `21 passive + 15 acquired + 8 sealed = 44`, exact
prediction of all eight sealed targets, and exact prediction of 12 separately
sealed long/path programs. Two additional full-support opaque controls receive
all 44 edge labels passively and no active labels. The ten environments form
five paired-relabel comparisons.

In the rank-5 absence-aware coordinates the legal-source ranks and unavoidable
off-domain extension nullities are:

| Event family | Tokens | Legal-source rank per token | Total-extension nullity per token |
|---|---:|---:|---:|
| bind | 4 | 3 | 10 |
| update | 2 | 4 | 5 |
| copy | 2 | 3 | 10 |
| invalidate | 2 | 4 | 5 |

The aggregate total-extension nullity is 80. Every event certificate must bind
a legal-domain basis, its image basis, and an exact annihilator basis proving
the stated ambiguity. The serialized `total_operator` is always `null`.
Success means all ten **restricted** maps are exact on their legal domains,
the legality mask and sealed compositions are exact, and one common rational
similarity relates each paired opaque realization. It is not a full WFA,
autonomous acquisition discovery, or identification of off-domain behavior.

## Stage F: full `K=5, V=4` diagnostic and operator follow-ons

The full absence-aware diagnostic stage may repeat T0 over all 20 omissions and
two opaque relabel blocks. A one-response `20 -> 21` result would identify the
missing diagnostic state direction only. It must not be reported as complete
arbitrary-suffix behavior or as identification of every `21 x 21` event
matrix.

A full guarded-operator campaign is a separate, not-yet-frozen experiment. It
must first enumerate the legal-source span `D_a` for every opaque event token,
then precommit enough active responses to drive every claimed restricted-map
nullity to zero. Each action retains `(21-dim(D_a))*21` unconstrained total
extension parameters. The campaign must keep independent legal edges and long
programs sealed after basis closure, and it must report any map that remains
underexcited as `not_identified`.

If the scientific target is instead a set of total `21 x 21` operators, the
protocol must freeze a new explicitly totalized contract and collect complete
shifted-block evidence with nullity zero. An overwrite/no-op completion would
be supplied structure; a dead-state completion is Contract C and changes the
rank. Neither may be silently attached to Contract B.

## Contract-isolation controls

The same exact learner is run separately on full-support Contracts A and C.
These are implementation and semantic-separation controls, not additional
samples for the absence-aware result.

### Promised-valid control

Run two opaque bijections at each scale. The learner must preserve undefined
entries as undefined, infer an exact minimal reachable/observable quotient if
the masked table identifies one, and never use invalidity or absence as an
output. The measured rank must not exceed the constructive upper bound of 3
for the toy and 16 for the full task. A failure to identify a unique operator
on the partial domain is reported as partial identifiability, not repaired by
borrowing Contract-B outputs.

### Strict control

Run two opaque bijections at each scale. The full task must recover the known
strict rank 192, including the observable dead behavior, under exact closure
and reconstruction. The toy strict rank is discovered and independently
verified. Neither strict model may be projected to dimension 5 or 21 and then
called exact.

The learner artifacts for A, B, and C live in disjoint directories and carry
different series fingerprints. A validator rejects any aggregate containing
more than one contract fingerprint.

## Dimension ladder and exact baselines

The main omission campaign evaluates a frozen dimension ladder in addition to
the automatically minimized learner.

| Scale | Underfit/passive diagnostic dimension | Full diagnostic dimension | Overcomplete diagnostic dimension |
|---|---:|---:|---:|
| `K=2, V=2`, absence-aware | 4 | 5 | 8 |
| `K=5, V=4`, absence-aware | 20 | 21 | 32 |

Each baseline has a predetermined interpretation:

- **Exact semantic executor:** sealed positive control for generation and
  scoring only; it is never linked into the learner.
- **Canonical-state spectral control:** uses the known absence-aware features
  and must reproduce the exact ranks; this validates table construction but
  is not sequence-only learning.
- **Prefix/suffix memorizer:** must fit every passive entry and remain
  undefined or wrong on at least one disagreement witness. It cannot count as
  active representation recovery.
- **Rank-4/rank-20 model:** must exactly fit the passive restricted table and
  fail to fit the full postactive table.
- **Rank-5/rank-21 diagnostic model:** is not identified before acquisition
  merely because the dimension was supplied. After diagnostic acquisition it
  must minimize to the exact full state rank. This alone says nothing about
  whether every guarded event map has enough legal-source excitation.
- **Rank-8/rank-32 model:** may fit, but must reduce exactly to rank 5 or 21.
  Unconstrained hidden directions are reported, never interpreted.
- **Nullspace twins:** two exact passive-consistent models must disagree on
  the selected active witness; exactly one survives the returned answer and
  complete postactive table.
- **Output-majority and class-zero controls:** output permutations and balanced
  probes must force these below exact performance.

Contract A and C use the deterministic ladder `r-1`, `r`, and the least power
of two strictly greater than `r`, where `r` is the exact rank found by the
minimal learner. This rule is frozen even when `r` differs from the known
upper bound.

## Compositional-factor discovery

This stage is **not eligible from guarded Contract-B maps alone**. Partial
event maps do not define unique elements of `End(W)`: different off-domain
extensions have identical guarded behavior but can have different commutants
and operator-Schmidt ranks. A preferred zero or minimum-norm completion would
manufacture the geometry being tested.

The algebraic search below is therefore a design for a separately frozen
totalized experiment. It may run only after every total event operator is
identified with shifted-system nullity zero, or after a new theory proves that
the proposed factor certificate is invariant across **all** total extensions
of the learned partial maps. Otherwise the mandatory output is
`factorization_not_identified_from_guarded_contract`. An eligible learner
receives the total operators as an unordered map from opaque symbols and
receives neither canonical key/value coordinates nor event-family labels.

### Recover the affine difference space

The homogeneous realization contains a one-dimensional affine lift. From
reachable predictive states, choose the lexicographically first state `q_0`
and form

\[
W = \operatorname{span}\{q(p)-q_0:p\text{ reachable}\}.
\]

Acceptance requires `dim(W)=KV`: 4 for the toy and 20 for the full task, and
requires `W` to be invariant under every induced difference operator. The
one-dimensional quotient and the invariant difference space must be recovered
algebraically; their canonical semantic coordinates are not supplied.

### Blind factor-pair search

For every nontrivial factorization `dim(W)=ab` with `2 <= a <= b`, the search
looks for a pair of event-generated unital subalgebras of `End(W)` satisfying

\[
\mathcal A \cong M_a(\mathbb Q), \qquad
\mathcal B \cong M_b(\mathbb Q), \qquad
\mathcal A' = \mathcal B, \qquad
\mathcal B' = \mathcal A.
\]

Here the prime denotes the exact commutant in `End(W)`. An arbitrary
hand-chosen tensor product on `W` is not a candidate: at least one factor
algebra must be generated by a precommitted subset or equivalence class of
recovered transition operators. Candidate generator sets are constructed
without semantic labels from exact invariant signatures:
minimal polynomials, ranks of `T-I`, idempotence, pairwise commutation, and
finite multiplication-table relations. The search closes each candidate
under rational linear combinations and products. It is capped at 4,096
candidate algebras per environment and uses a frozen lexicographic ordering.
Exhausting the cap is a failed factor-discovery run, not evidence that no
factorization exists.

A certificate must include matrix units for both algebras, their dimensions,
centers, multiplication tables, commutant bases, and exact mutual-commutant
equalities. It must also show that the generated product algebra is all of
`End(W)`. These double-centralizer conditions yield a tensor isomorphism

\[
W \cong R \otimes F

\]

up to independent changes of basis in `R` and `F`. For the full task the
candidate pairs are `2 x 10` and `4 x 5`; acceptance requires the unique
`4 x 5` certificate, oriented as `5 x 4` only by observable output
cardinality. For the toy, `2 x 2` is identifiable only up to swapping the two
factors.

The expected role algebra on the full task has dimension 25 and its commutant
has dimension 16. Those numbers are evaluator expectations, not learner
inputs.

### Operator-Schmidt tests

After a factorization is certified, reshape each induced operator on `W` as a
four-index tensor and matricize across

\[
(R_{out},R_{in})\mid(F_{out},F_{in}).
\]

Its exact matrix rank is the operator-Schmidt rank. No approximate threshold
is used. After the controller reveals semantic families for evaluation, the
following identities are required:

- role-only copy/invalidation linear parts have Schmidt rank 1;
- a nonidentity update localized to one role has Schmidt rank exactly 2 for
  `K>1`, with the form

  \[
  I_R\otimes I_F + E_{kk}\otimes(U-I_F);
  \]

- the affine injection term of a bind is a rank-1 role--filler vector;
- each key/value query slice is a rank-1 role--filler covector;
- distinct-key update actions commute exactly;
- the value update has the declared cyclic order on its identified active
  subspace;
- alternate paths denoting the same update/copy program yield exactly the same
  operator; and
- the deliberately order-sensitive copy programs remain unequal.

These ranks are meaningful only relative to the certified tensor factors.
Reshaping arbitrary learned coordinates into a `5 x 4` rectangle is forbidden.
Operator-Schmidt rank is invariant under local role/filler gauges but not
under an arbitrary unaccounted global similarity.

### Factorization negative controls

The factor search must pass all of these controls:

1. apply a dense exact unimodular global similarity to the recovered model;
   the same factorization and Schmidt-rank certificate must be recovered in
   transformed coordinates;
2. use a naive coordinate reshape before factor recovery; its ranks must not
   be substituted for the certified ranks;
3. run rank-matched generic rational weighted automata with matched alphabet
   size; the double-centralizer search must not falsely certify a `5 x 4`
   binding factorization;
4. remove the copy/invalidation family from a synthetic observation table;
   if the role algebra is then nonidentifying, the search must report that
   failure rather than use hidden labels; and
5. permute semantic event-family sidecars while holding the learned operators
   fixed; factor selection must remain byte-identical because labels are
   revealed only after selection.

The generic-automaton and family-ablation controls are generated and hashed
before the positive factor search is run.

## Behavioral probes after freezing

Exact table reconstruction is necessary but not the only check. After every
model and similarity/factorization certificate is frozen, the evaluator opens
separate opaque long-program probes containing:

- direct entry to and exit from the acquired direction;
- repeated updates through and beyond one full cycle;
- bind, invalidate, and rebind sequences;
- copy into and out of the acquired direction;
- copy-source mutation immediately after the copy;
- independent-role commutation;
- deliberate copy-order noncommutation;
- alternate paths to the same predictive state;
- dormant invalidated roles followed by long distractor runs;
- output-label-balanced terminal diagnostics; and
- lengths 2, 4, and 8 times the longest word used to select the Hankel basis,
  subject to the contract's legality domain.

Every exact probe must be correct. An aggregate average cannot compensate for
one failed family, omission, output class, or path relation.

## Frozen budgets

The implementation config must use the following caps. A cached evaluation of
one complete word counts once regardless of the number of output channels
returned. Repeated cache hits do not count again. A denied, undefined, or
malformed active request consumes one response from the applicable stage's
frozen budget and causes failure.

| Run | Environments | Maximum word length | Maximum suffix-test candidates | Maximum distinct oracle evaluations per environment | Label-bearing active responses |
|---|---:|---:|---:|---:|---:|
| Toy zero-suffix diagnostic omission (T0) | 8 | 16 | 64 | 50,000 | 1 response / 2 labels |
| Toy guarded partial operators (T1) | 8 omission + 2 full controls | 16 | 64 | 50,000 | 15 responses / 30 labels for an omission; 0 for a full control |
| Full zero-suffix diagnostic omission (F0) | 40 | 64 | 128 | 5,000,000 | 1 response / all diagnostic channels |
| Toy promised-valid, full support | 2 | 16 | 64 | 50,000 | 0 |
| Full promised-valid, full support | 2 | 64 | 128 | 5,000,000 | 0 |
| Toy strict, full support | 2 | 32 | 128 | 100,000 | 0 |
| Full strict, full support | 2 | 256 | 512 | 25,000,000 | 0 |

Additional fixed caps are:

- 4,096 active candidate words for the toy and 65,536 for the full task;
- 4,096 factor-algebra candidates per environment;
- exact reachable-prefix representatives capped at 10 for the toy strict
  machine and 822 for the full strict machine;
- exact basis dimension capped at 10 for the toy and 192 for the full strict
  control; and
- no more than one complete rebuild after the T0 response and two staged
  rebuilds for T1 (the one-response negative checkpoint and the final
  15-response fit);
- for T1, exactly 90 definedness-mask rows, 44 legal edges, 46 undefined
  edges, and the omission partition `21 + 15 + 8`; at most 30 returned
  categorical labels and 12 long/path probes;
- at most 10,000 sealed behavioral words per environment; and
- sealed behavioral word length capped at 128 for the toy and 512 for the
  full task.

The active candidate-pool scan itself may inspect every unlabeled candidate,
but it may not obtain outputs during selection. If the chosen word exceeds the
length cap or the passive basis construction exceeds any cap, the environment
fails closed.

These are computational ceilings, not sample-size claims. The passive
omission tables enumerate their declared support exactly. Scientific
active-observation counts are reported separately: T0 uses one membership
response containing two labels; T1 uses 15 membership responses containing 30
labels. Neither is abbreviated to "one answer."

## Execution order and content binding

Before any learner process starts, commit and hash:

- this protocol and a machine-readable configuration;
- the exact learner, verifier, firewall, controller, and evaluator sources;
- the input/output bijections and their derivation rule;
- every passive transcript and domain-mask digest;
- every active candidate pool and ordering rule;
- all long-program probes and negative controls in sealed form;
- the exact arithmetic/runtime manifest; and
- all resource caps and failure policies.

Execution then proceeds in six irreversible stages:

1. **Firewall qualification:** run import denial, canaries, sidecar poisoning,
   and bijection controls.
2. **Toy T0 preactive:** finish all eight rank-4 diagnostic shards and a
   terminal aggregate.
3. **Toy T0 active open:** release the eight selected diagnostic responses as
   one batch and require all eight finite rank-5 diagnostic blocks.
4. **Toy T1 guarded maps:** freeze the two full-support and eight 21-edge
   omission inputs plus the controller-supplied 15-edge bases. For each
   omission, open response one, freeze the negative checkpoint, then open the
   other 14 responses. Freeze all partial maps before opening the eight edge
   checks and 12 long/path probes.
5. **Full F0 preactive:** finish all 40 rank-20 diagnostic shards and a
   terminal aggregate.
6. **Full F0 active open:** release the 40 selected diagnostic responses as a
   batch and freeze the rank-21 diagnostic blocks. Do not run a total-operator
   or factorization audit until its distinct totalization/evidence protocol is
   frozen and eligible.

No active response may be constructed in a learner process before the matching
terminal aggregate exists. Failed environments are retained. There is no seed
replacement, budget increase, selective rerun, best-environment reporting, or
postopen change to the factor search.

Each artifact is strict canonical JSON or a separately hashed exact-matrix
payload. Reports bind parent hashes in a directed acyclic provenance graph.
The independent verifier reconstructs nested evidence objects and all hashes;
it does not accept a top-level `passed: true` field as proof.

## Acceptance gates

### Gate 0: firewall

All access-log, canary, poison, import-denial, and token/output-permutation
tests pass. Any semantic read by the learner invalidates the campaign.

### Gate 1: exact learner controls

Canonical-state controls reconstruct exactly; prefix/suffix memorization is
correctly classified as non-completing; underfit and overcomplete models have
the expected exact minimal quotients; output defaults fail the balanced
controls.

### Gate 2a: toy diagnostic-state discovery

All 8/8 T0 environments have passive diagnostic rank 4, a certified
disagreement, exactly one membership response containing two categorical
labels, postactive rank 5, and exact reconstruction of all nine zero-suffix
diagnostic states. The known-entry control remains rank 4. This gate makes no
operator claim.

### Gate 2b: toy guarded partial operators

Both full-support controls reconstruct all 44 legal edges without active
labels. All 8/8 omission environments have the exact `21 + 15 + 8` edge
partition; response one raises the diagnostic rank while leaving nine maps
unidentified; all 15 responses identify every restricted legal-domain map;
all eight sealed edges and 12 long/path programs are exact. Every total
operator is absent, the per-family extension nullities sum to 80, and five
paired-relabel similarities satisfy every state, readout, and restricted-edge
equation.

### Gate 3: full diagnostic-state discovery

All 40/40 F0 environments have passive diagnostic rank 20, a one-dimensional
state-row ambiguity, exactly one diagnostic response, postactive diagnostic
rank 21, and exact reconstruction of the declared finite diagnostic block. No
result may be replaced by an aggregate rate. Guarded operator recovery is
reported separately and may remain `not_identified`.

### Gate 4: gauge-equivalent recovery

Every minimal postactive model is related to the exact reference and to its
paired opaque-bijection model by one full-rank rational similarity map. For a
total contract it satisfies every operator and readout equation. For a partial
contract it satisfies every state, readout, legal-domain basis, and legal-edge
image equation, without comparing arbitrary off-domain completions. The map is
fitted and tested on disjoint frozen sets.

### Gate 5: contract isolation

Promised-valid runs never observe absence or illegality; strict full runs
recover rank 192 and dead behavior; no cross-contract record or rank is
pooled.

### Gate 6: compositional-factor discovery

This is a separate, stronger pass condition and is ineligible for the current
guarded Contract-B partial maps. Under a future identified totalized contract,
all full postactive environments must recover the `5 x 4`
double-centralizer certificate, exact expected operator-Schmidt ranks, and all
polynomial/path identities, while every negative control behaves as
preregistered. Until that prerequisite exists, the required verdict is
`factorization_not_identified_from_guarded_contract` even if Gate 3 passes.

## Implemented bounded checkpoints

### T0: finite zero-suffix diagnostic block

The first executable checkpoint is now implemented in
`v3/src/tnlm_v3/opaque_predictive_state.py`.  Its claim is deliberately
narrower than the full protocol above: it learns the finite **zero-suffix
multi-output absence-diagnostic block**, not arbitrary-suffix transition
operators, a general PSR, or a tensor factorization.

The controller runs all four `K=2`, `V=2` omissions under two independent
nonce-keyed opaque relabel blocks.  The learner receives only opaque event,
query, and answer tokens, fixed-length opaque words, and a content-bound
candidate pool.  It receives no `K`, `V`, event roles or arguments, semantic
states, omitted-cell ID, executor, or controller map.  Exhaustive
learner-input-only attacks leave all eight `(block, cell)` hypotheses
compatible.  This is an honest-code/API boundary, not cryptographic or
process isolation against a learner that inspects controller source or the
call stack.

Every six-row passive block has exact rational rank 4.  Three precommitted
categorical completions are compatible with the opaque vocabulary, and each
raises rank to 5.  The postactive learner is outcome-total: all 24 combinations
of eight environments and three answers rebuild rank 5 and reconstruct all
nine diagnostic rows exactly.  One response is used per environment.  Four
paired relabel comparisons satisfy exact nonsingular rational similarities
under `x_A S = x_B` and `S O_B = O_A`.

The hostile suite passes 45 / 45 tests.  The source SHA-256 is
`e63eded99f4d2d27261466df3e212253d81078ca3a46e249a4ad1726cdab0d27`;
the deterministic fixture-report SHA-256 is
`0c41157bc062264e05fc4772e4b54e8cc40e2f0aebf66fdb74c1fff407449487`.
Nested digests are contextual content links; authoritative semantic evidence
comes from reconstructing the complete environment and report.

This checkpoint demonstrates exact passive ambiguity diagnosis and active
rank restoration in the smallest opaque finite block.  It does not yet meet
the full operator, factorization, or scaling gates below.

### T1: guarded partial operators

The next bounded checkpoint is implemented in
`v3/src/tnlm_v3/opaque_partial_operators.py`.  It keeps Contract B partial and
therefore learns no invented zero, no-op, dead-state, or other total extension.
The sanitized estimator sees opaque state diagnostics, opaque event tokens,
two categorical target labels per membership response, and the complete
defined/undefined mask.  It does not receive semantic key/value coordinates,
event roles, the omitted-cell identity, or controller sidecars.  The trusted
controller supplies the semantically designed 15-edge excitation basis; the
learner neither discovers nor selects that basis.

The deterministic report contains two full-support controls and all four
omissions under two nonce-keyed relabelings.  The controls use all 44 legal
edges passively.  Every omission has the exact partition `21 passive + 15
acquired + 8 sealed = 44`.  The first response raises diagnostic rank `4 -> 5`
while nine of ten restricted maps remain unidentified; the next 14 responses
close the remaining legal-source deficits.  Across the report this is 256
passive edges, 120 membership responses containing 240 target-label fields,
64 sealed edges, and 120 separately sealed long/path programs.  Every sealed
prediction is exact.

All ten learned realizations identify the ten event maps on their legal source
domains.  They reject all `10 x 46 = 460` undefined state--event pairs.  Exact
annihilator witnesses retain total-extension nullity 80 in every environment,
and every serialized `total_operator` is `null`.  Five rational similarities
are derived from five independent controller-aligned fit states and verified
on four disjoint states, exact aligned readouts, and all 44 legal edges.  The
state, event, and readout correspondences used for this postfit audit are
supplied by the trusted controller; they are not learned from the opaque input.

The hostile suite passes 58 / 58 tests, and the two Phase-III checkpoints pass
103 / 103 together.  The T1 source SHA-256 is
`6efd6dad92e8c3c22fb787071dee599363a61a11b4d560f60d7a5d0fb20e9738`;
the test SHA-256 is
`da1b7a862d3638450773049359663a1b68a266141012b6b4cc3753dd0081152a`;
the deterministic report SHA-256 is
`26cd2a6d628abe67c3d2a16a373b2c524a1ef2b22687d8122b7777f04116f80f`.
Authoritative environment/report reconstruction, rather than standalone
nested hashes, is the semantic evidence boundary.

This demonstrates exact guarded-map recovery when a trusted controller
supplies enough legal-domain excitation.  It does not demonstrate autonomous
query design, total-WFA identification, off-domain behavior, factorization,
process isolation, or assumption-free representation discovery.

### T2: learner-selected guarded partial operators

The official bounded T2 campaign is implemented by
`v3/src/tnlm_v3/opaque_active_discovery.py` and its sealed protocol wrapper.
It retains T1's supplied nine-state categorical grammar, complete legality
mask, 21 passive omission edges, and unordered 23-edge membership-query pool,
but removes T1's controller-designed excitation ordering. The learner commits
each request before receiving its two-label response and uses only opaque
diagnostics, the legal mask, its passive input, and its causal transcript.

Two full-support controls and eight omission arms passed. Across the omissions,
the learner made 112 membership requests containing 224 returned labels,
derived eight answer-sidecar-free structural inferences, and left 64 candidates
sealed. Every omission realized `21 passive + 14 queried + 1 inferred + 8
sealed = 44` legal edges and identified all ten restricted maps. Across the ten
arms, all 440 defined rows and 120 long/path programs were exact, all 460
undefined pairs were rejected, and five exact controller-aligned postfit gauge
certificates passed. Every total operator remained absent and exact off-domain
extension nullity remained 80 per environment.

The controller-supplied postfit reuse-the-first-14-T1-edges baseline remained
nonidentified in all eight omissions. A separate truth-aware, noncausal
teaching-set control used 104 counterfactual requests, 208 labels, 16 singleton
inferences, 64 unopened candidates, and zero new membership calls. That control
is evidence about a smaller truth-specific teaching set only; it is ineligible
for autonomous selection, confirmation, or a global query-minimality claim.

The official protocol SHA-256 is
`7c5ee8bcee72e0af5ac2d8404f54b479e1b7d1b1200922ec40caf66483c04292`,
and the source/runtime binding is
`514ebb445d3eb00e456095bf3377bf4f7eb2e15a4282e0765ca307b5203e5e90`.
The terminal record/raw-file SHA-256 values are respectively
`31b3849c7a469ed380c68502193287e4917ee07170bbbd9c3c95d837055c1352`
and
`e5e2524285c970eff1e45474d43aee43cf26615ce8ae89aa19b43dd0aa5b0819`;
the opened report record/raw-file values are
`9993913e5f60a73ce41fb08803c7dab24511165a4dc636e89b065d68be59c40f`
and
`89b1fa8fa2d5f21fd001143d8275fcde94acdac964169828d55f566386ae5bd9`.
The evidence is preserved under
`v3_recovery/phase3_t2_opaque_active_discovery/` and is authoritative only
through complete environment/report reconstruction.

T2 demonstrates autonomous causal acquisition within a supplied finite
grammar, legal-domain mask, and candidate pool, followed by arbitrary-length
guarded-language equivalence. It does not demonstrate a total WFA, globally
minimal query count, autonomous vocabulary or legality discovery,
factorization, natural-language learning, or assumption-free representation
discovery.

## Required reporting

The final report gives every environment row, not only aggregates. It includes:

- passive and postactive exact diagnostic ranks;
- pivot-prefix and suffix-test lengths;
- oracle-evaluation and exact-elimination work;
- every active request hash, whether its basis was learner-selected or supplied
  by the trusted controller, returned-label count, and rank increment;
- for every partial event map, the legal-domain rank, observed-source rank,
  restricted nullity, total-extension nullity, domain/image bases, and exact
  annihilator witness;
- explicit `total_operator: null` fields for guarded contracts and an aggregate
  extension-nullity audit;
- underfit, minimal, overcomplete, memorizer, and nullspace-twin outcomes;
- finite diagnostic-table, sealed-edge, and per-probe-family exact counts;
- similarity-map determinant, exact residuals, fit-set hash, and disjoint
  test-set hash;
- candidate factor pairs and every rejected reason;
- double-centralizer dimensions and matrix-unit certificate hashes;
- per-symbol operator-Schmidt rank before semantic family labels are opened;
- posthoc event-family mapping and family-level identity checks;
- all negative-control results; and
- any cap hit or undefined operator, even if later stages pass elsewhere.

For exact arithmetic, a residual is either zero or nonzero. Condition numbers
may be reported as numerical diagnostics for future noisy work, but they do
not replace exact acceptance.

## Claim boundaries

If Gates 0--5 pass, the strongest justified statement is:

> For this finite binding system and these explicitly declared observable
> contracts, an exact opaque diagnostic learner recovered the predictive state
> coordinates up to similarity. Under the guarded absence-aware contract, one
> membership response closed the single missing diagnostic direction, while a
> larger disclosed excitation basis identified each event map only on its
> legal source domain. Off-domain total extensions remained nonidentified.

This would not show that the heldout direction was identifiable passively. It
would show that the ambiguity can be diagnosed behaviorally and removed with
a bounded intervention.

If a future eligible totalized experiment makes Gate 6 pass, the additional
justified statement is:

> Within the preregistered algebra-search language, the fully identified
> predictive operators determine a role--filler tensor factorization through
> an exact double-centralizer certificate and exhibit the predicted low
> operator-Schmidt ranks, without supplying key/value coordinates to the
> learner.

Even that result would not establish a universal "geometry of language." The
factor search, exact finite alphabet, active oracle, rational noiseless data,
and candidate-algebra language are substantive inductive biases. The result
would establish a mechanism by which a compositional geometry can be
identified in a controlled system.

## What would count as a tensor-network scaling breakthrough

The rank-21 result alone does not unlock tensor-network scaling. It gives a
small exact state space that a dense linear model can already represent. The
relevant breakthrough would be evidence that the *interfaces* needed for
prediction factor systematically:

1. predictive rank may grow with the number of roles and fillers, while
   operator-Schmidt ranks across discovered semantic cuts remain bounded or
   grow much more slowly;
2. the double-centralizer/factor search continues to identify those cuts from
   opaque behavior rather than receiving them as architecture metadata;
3. active-query and observation complexity scale with the factor dimensions
   and algebra generators, not with an exhaustive table of role--filler
   combinations;
4. approximate noisy versions remain well-conditioned and stable under
   perturbation; and
5. the factorized realization preserves long composition laws better per
   parameter and per observation than dense minimal-state, automaton, and
   memorization baselines.

Only after the exact Phase-III certificate exists should a scaling study vary
`K`, `V`, live cap, operator vocabulary, and partial observability. It must
compare dense predictive-state, TT/MPO, and hierarchical factorizations under
matched behavior and data. A useful "geometry" is then an empirically
recoverable factorization of predictive interfaces with stable cut ranks and
composition laws--not a visual arrangement of embeddings and not merely the
fact that a tensor network can encode a finite automaton.
