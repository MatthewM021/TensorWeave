# Phase II: Can the Binding Algebra Be Learned?

## Scope and provisional verdict

Phase I established that the binding task has a fixed-width exact algebra, but
that the declared train/validation distribution leaves one observable state
direction unidentified.  This note separates three questions that are easy to
conflate:

1. **Coefficient learning:** if the correct sharing law is fixed in advance,
   can its remaining numerical tables be estimated from seen-only data?
2. **Representation learning:** can a learner infer a latent state and its
   compositional operators from visible sequences and query answers, without
   being given the canonical state coordinates?
3. **Algebra discovery:** can a selection procedure choose the shared law over
   equally accurate, non-compositional extensions without consulting the
   sealed held-out combination?

For the frozen finite task, the answer to the first question is **yes**.  The
answer to the second is a testable system-identification question, with the
usual similarity gauge and complete-basis conditions.  The answer to the
third is **not from the original passive support without an inductive bias**.
It becomes scientifically testable when the bias is named and selected using
rotated pseudo-heldout cells, multiple environments, active counterexamples,
or a preregistered simplicity criterion.

The executed three-seed retrospective pilot now provides positive evidence at
that conditional boundary: a trace-supervised learner recovered the exact 20
supported transition entries inside a supplied register-transducer family and
generalized perfectly to the omitted cell.  It did not infer the register
representation or select a unique penalty, so it advances coefficient
learning without resolving representation or assumption-free algebra
discovery.

This distinction is the main result.  A hard-equivariant model that succeeds
has shown that the algebra is a useful sufficient prior.  It has not shown
that the learner discovered the law.  Evidence of discovery requires a more
flexible model class and a heldout-clean procedure that repeatedly selects the
shared algebra over pair-local alternatives.

Nothing in this note claims that unrestricted natural language has finite
Hankel rank or follows the binding task's algebra.

## The exact obstacle inherited from Phase I

The frozen task has five keys, four values, a three-binding live cap, and one
excluded combination `(key 0, value 0)`.  Its absence-aware homogeneous state
space has dimension

\[
D=1+KV=21.
\]

The complete train support spans only 20 dimensions.  If
`h = e_(0,0)` selects the missing coordinate, every train state lies in
`ker(h*)`.  Consequently, for any train-consistent operator `A` and readout
`R`,

\[
A' = A + u h^*, \qquad
R' = R + w h^*
\]

are equally train-consistent for arbitrary output vectors `u` and `w`.
Phase I counted the ambiguity even under the optimistic assumption that every
canonical train state and successor is observed:

| Model of the train system | Parameters | Constraint rank | Nullity |
|---|---:|---:|---:|
| Unrestricted canonical operators and readouts | 29,967 | 25,055 | 4,912 |
| Key-local value-block operators | 720 | 656 | 64 |
| Shared update and copy, local bind and query | 224 | 216 | 8 |
| Shared bind, update, copy, and query prototypes | 96 | 96 | 0 |
| Per-key calibrated cyclic presentation | 40 | 40 | 0 |
| Shared calibrated cyclic presentation | 8 | 8 | 0 |

The nullity-zero shared-prototype row uses the screen's five-key support.  It
is not a formula valid for every smaller task: with only two keys, the value-0
copy prototype has no train-valid source/destination pair, leaving `V` free
coefficients (four when `V=4`).  The executable analysis now accounts for this
support edge explicitly.

The nullity-zero rows are not evidence that the data selected those model
classes.  They say that, **conditional on those universal sharing laws**, the
remaining coefficients are identifiable from train support.

## What is conditionally learnable without hard-coding the cyclic table

The 96-parameter shared-prototype model is the most important positive
control.  It need not be told that updates form `C_4`, that copy is the
identity on values, or that queries decode one-hot values.  It only assumes
that all keys use the same bind, update, copy, and query tables after their
local value blocks are aligned.  All entries of those tables can then be fit
from other observed key/value combinations **if canonical pre-state,
post-state, and categorical query records are independently observed**.

That would be genuine coefficient learning from repeated structure.  The
current repository does not yet contain such independent state-transition
records: visible sequences expose events and query answers, not canonical
register values at every step.  The executable fitter therefore has two
strictly separated entry paths.  Its generic fitting function consumes
explicit transition records and never calls the exact executor; the included
smoke control generates those records with the exact executor and is labelled
`sealed_exact_executor_control`.  That circular smoke test checks optimization,
serialization, and rank bookkeeping only.  It is not empirical evidence that
the tables were learned from sequences.

Each accepted transition record is bound to the exact task fingerprint,
visible-sample digest, direct `train` split, document/event location, and
visible action fields.  It also carries canonical pre-state, post-state, and
an optional query label, plus a declared origin tag distinguishing external
records from sealed-executor controls.  The two fit entry paths reject
unmodified records carrying the other origin.  This tag is an experimental
attestation, not cryptographic proof of how a caller produced the data.  A
trusted firewall requires the locations to equal
the coverage certificate exactly and rejects any pre- or post-state containing
the sealed pair.  It then emits a sanitized batch containing no heldout-pair
identifier; only that batch reaches the coefficient estimator.  These checks
prevent an externally labelled record from silently substituting heldout or
unrelated observations while keeping sealed-cell knowledge out of estimation.
They establish provenance and split cleanliness; they do not establish that
an external state sensor is correct.  That measurement assumption must be
declared separately.

The conditional result is stronger than filling the missing cell with the
known answer, and weaker than automatic law discovery.  Its critical
assumption is the absence of a key-by-value exception term.  A single
localized residual

\[
\Delta(k,v)=\lambda\,1[k=0]1[v=0]
\]

fits every train and seen-validation trajectory while changing the sealed
answer.

The eight-parameter cyclic model is a different control.  It imposes a
calibrated transitive `C_4` action and universal bind/query covariance.  Its
success demonstrates sufficiency of the proposed algebraic presentation; it
must not be called discovery.

## Exploratory finite-sample excitation result

An exploratory, non-evidence calculation replayed generated seen-only
trajectories with the exact semantic audit and measured which columns an
independently state-supervised 96-parameter shared-prototype system *would*
excite. Each occurrence addresses one of 24 value-block columns, with four
scalar outputs per column. The replay locates columns using the known exact
state; it does not infer that state from visible sequences.

The executable analysis was run against the exact frozen SCREEN task
fingerprint (`max_length=2048`, even though every sampled document below has
length 64). Across 500 deterministic train seeds:

| Documents | Train samples with rank 96/96 | Mean rank |
|---:|---:|---:|
| 1 | 1 / 500 | 70.328 |
| 2 | 65 / 500 | 87.872 |
| 3 | 263 / 500 | 93.376 |
| 4 | 394 / 500 | 94.984 |
| 5 | 452 / 500 | 95.584 |
| 6 | 472 / 500 | 95.768 |
| 8 | 497 / 500 | 95.976 |

The default checked Phase-II record uses the first 100 of those seeds: 99 have
rank 96, the minimum rank is 92, and the mean rank is 95.96. An earlier
ad-hoc calculation used a task object with `max_length=512`; because that
field participates in deterministic seed derivation, its superficially
similar counts are not the frozen experiment and are superseded here.

These numbers show that **persistent excitation is not the practical
bottleneck for the shared canonical-state control**.  They do not establish
sequence-only representation learning: the calculation used exact semantic
states and exact successors derived by the audit.

For seed 0 at eight length-64 documents, the explicitly oracle-generated
coefficient control reconstructs 24 of 24 tables while recording that
universal key sharing was supplied and the cyclic law was not. This is the
expected consequence of a full-rank exact design, not a new empirical result.

The count is also task-specific.  A larger alphabet, rare operators, noisy
states, or natural-language supervision could make coverage and conditioning
dominant again.

## Executed trace-supervised pilot

The repository now also contains a trace-supervised coefficient learner and a
shortcut-resistant behavioral probe suite.  This is narrower than latent
representation learning.  The estimator is given an addressable register
interface, structured event kinds and arguments, the value alphabet, presence
and invalidation semantics, and an identity query gauge.  It learns 20
categorical transition entries: four BIND outputs, twelve outputs for the
three generator-supported non-identity UPDATE arguments, and four COPY
outputs.  It may also fit key-local residual entries.  It is not given
canonical pre- or post-states, the exact executor, oracle routes, dependency
metadata, or the outer heldout identifier.

A trusted controller does see oracle occupancy and causal-dependency
annotations, and knows the outer `(0, 0)` identifier, solely to censor whole
traces and score descendant queries in the 19 inner pseudoheldout folds.  The
outer probe answers do not enter fitting, validation, candidate ordering, or
early stopping.  This boundary makes the coefficient estimator trace-only;
it does not make the overall protocol metadata-free.

### Three-seed passive pilot

We froze three paired environments before comparing their outcomes.  Each
used 16 direct-`train` and 16 direct-`validation` documents of length 64, the
candidate penalties `{4, 16}`, optimizer seed 0, two restarts, and four
coordinate-descent sweeps.  All settings and probes were identical across
seeds.

| Train / validation seeds | Selected penalty | Pseudoheldout errors, penalty 4 / 16 | Final seen fit | Final local overrides | Actual `(0, 0)` queries | Focal queries |
|---|---:|---:|---:|---:|---:|---:|
| 17 / 23 | 4 | 85 / 838 vs 90 / 838 | 328 / 328 | 0 | 96 / 96 | 24 / 24 |
| 18 / 24 | 4 | 437 / 1,001 vs 478 / 1,001 | 319 / 319 | 0 | 96 / 96 | 24 / 24 |
| 19 / 25 | 16 | 147 / 992 vs 94 / 992 | 334 / 334 | 0 | 96 / 96 | 24 / 24 |

Every final model learned exactly the same shared table: BIND was the identity
on values, UPDATE was the correct cyclic shift for each of the 12 supported
source/argument combinations, and COPY was the identity on the source value.
Across the three seeds, the frozen models answered 288 / 288 actual-cell
queries and 72 / 72 focal queries correctly.  All 36 seed-by-family rows were
exact, including update-in/out, copy-in/out, copy naturality, cyclic
composition, invalidation/rebinding, independent-key commutation, deliberate
copy-order noncommutation, distractors, and long compositions.  Path-relation
accuracy was 1.0 in every seed.

The balanced rotated-cell coverage control was also exact: 5,400 / 5,400
queries and 1,440 / 1,440 focal queries across 20 cell slices.  These are
independently generated cell rotations, not literal conjugates of one base
program and not evidence of learned equivariance.  On the actual-cell suite,
the strongest shortcut control reached 88 / 96 overall and 16 / 24 focal
queries; none passed the conjunctive correctness, focal, and family gates.

The penalty itself was not stable: penalty 4 won two environments and penalty
16 won one.  The final realized solution was nevertheless stable, with zero
local overrides in all three runs.  This supports repeatable recovery of the
shared transition coefficients and behavior inside the supplied family.  It
does **not** establish discovery of a unique regularization strength or
selection of one uniquely privileged algebraic hypothesis class.

A later optimizer audit exposed an additional limitation in those immutable
pilot records: 47 of their 114 fold-candidate fits retained nonzero TRAIN
error.  The reported pseudoheldout penalty comparisons therefore mix
structural preference with optimization failure and must not be used as clean
regularization-selection evidence.  The final full-data fits were exact, so
this does not retract the coefficient-recovery or behavioral-probe result.
After the outer answers were already known, a trace/query-only initializer and
bounded pairwise search made all 114 candidate fits exact with zero local
overrides; all three retrospective replays then selected penalty 16.  That is
an optimizer diagnostic, not a replacement prospective result.

An intentionally low-budget run on seeds 17 / 23 (one restart and one sweep)
made 94 / 328 seen-training errors and then scored only 82 / 96 overall and
10 / 24 focal actual-cell queries.  It is retained as an optimization control.
The heldout result is not interpretable until the preregistered exact-seen-fit
gate passes; increasing the optimizer budget was justified by that seen-fit
failure, not by the already retrospective outer score.

All four runs are content-bound JSON records.  The three successful canonical
record SHA-256 values are:

- seeds 17 / 23: `abc45cc0d76d2d5d53f35e54ca2aaca8217661f2b397a53bd6fedd7c3034b1b0`;
- seeds 18 / 24: `36af0ed4aadcbf460a192eb2127f3ded157d3c3782021a202ce4b097143badd4`;
- seeds 19 / 25: `40412b867f147165947023274c9506853b2f4b3bb4f5c0596f640e4885e508b4`.

The run records bind the exact campaign config and the learner, probe, and
runner source hashes.  Independent reconstruction verified every record hash,
all 19 fold aggregates, the learned tables, the actual-cell results, all four
shortcut controls, and the 300-case rotated-cell summary.  Repeating the first
successful run produced the same 117,217 bytes and file SHA-256
`a5d27c361332abecfb0b3e0dcf439f091edfe6a4e2337cd2a19a0a1600410a3f`.

This remains a retrospective, non-confirmatory pilot.  Earlier phases had
already inspected the `(0, 0)` semantics, three seeds are exploratory, and
the register representation and candidate family were supplied.  The result
therefore establishes a more precise positive fact:

> Passive event traces and query labels can recover the coefficients of this
> supplied compositional register algebra and extrapolate them to an omitted
> key/value combination.  The experiment does not show that the algebra or
> its latent representation can be discovered from raw sequences.

### Active-excitation implementation control

A smaller designed corpus separately excites every supported shared table
entry while excluding `(0, 0)`.  It contains 35 visible sequences, 106 events,
and 51 query labels.  The trace-only estimator reaches 0 / 51 training errors,
uses no local override, and scores 96 / 96 actual-cell queries, 24 / 24 focal
queries, all 12 families, and every path relation exactly.  This is an active
coefficient-identification and implementation control.  It shows that the
estimator and supplied family work under sufficient excitation; it is not a
passive-data result or representation discovery.

### Observed transition-address exception power control

The selector now also has a paired, class-balanced synthetic power control.
Its positive condition places one directly observed destination-key/UPDATE
address exception in both fitting and validation traces.  Penalty 4 reaches
zero TRAIN mistakes, realizes exactly that one override, and scores 0 / 231
validation mistakes; penalty 16 attains its certified hard-shared optimum of
six TRAIN mistakes and scores 12 / 231 validation mistakes.  The matched null
condition makes both candidates exact and uses the frozen complexity
tie-break to select penalty 16 with no override.  Every fold optimum and the
semantic decomposition are reconstructed from the immutable report.

This proves power only for an **observed transition-address-local exception**.
The self-pseudoheldout folds that censor the exception are explicitly marked
nonidentifying, and the reused validation view is not an independent holdout.
It does not show that an arbitrary unseen cell-specific exception can be
predicted.  The production artifact is
`v3_recovery/PHASE2_ALGEBRA_POWER_CONTROL_V1.json`; its file SHA-256 is
`4fa8d54c636ae693fdca7931ebb3e2095488eb8cc74f8d7f49e120e4c6e230a6`
and its canonical record SHA-256 is
`831bf991a99ad51b841a916f919df94def448c8f722fd9e1f06769e873c1913f`.

### Frozen multi-environment execution protocol

Before any new outer-omission fit, V3 freezes two complete 20-cell blocks: 40
environments, 24 direct-TRAIN and 24 direct-validation documents per
environment, length 64, penalties `{4, 16}`, one restart, four coordinate
sweeps, and two bounded pairwise rounds.  The numeric seed labels are balanced
across cells but are not common-random-number matches because the omitted cell
changes the generator's task fingerprint.  Every one of the 1,520 fold
candidates must have zero TRAIN mistakes, zero local overrides, and at least
16 pseudo-dependent queries; every final model must independently replay with
zero direct-TRAIN mistakes and zero overrides.

Execution is deliberately two-phase.  All 40 preopen shards and a terminal
aggregate must be content-bound before any outer probe is constructed.  Only
if every preopen gate passes may the runner open all 40 models as one batch;
there is no failed-cell replacement or pooled-average rescue.  The frozen
protocol is `v3/configs/phase2/outer_rotation_v3.json`.  Its exact
implementation/runtime manifest is external to avoid a self-hash cycle and is
validated before generation or fitting.  At this point the protocol is ready
for execution but has no result yet.

## The leakage firewall

The scientific experiment must never reuse the Milestone-4 screen's campaign
`validation` helper.  That helper intentionally maps its validation stream to
the data generator's `eval` split, which exposes and forces the held-out pair.
It was valid for a non-claiming model screen, but it is invalid for selecting
an algebra intended to predict that pair.

For Phase II:

- fit on direct generator split `train` only;
- tune and select on direct generator split `validation`, which also excludes
  the real held-out pair;
- do not pass heldout masks, heldout pair IDs, oracle routes, generation IDs,
  dependency parents, or test seeds into fit or selection code;
- freeze hypothesis family, state dimension, rank, law weight, stopping rule,
  seeds, and checkpoint before opening any sealed probe containing the real
  held-out pair;
- poison or permute every forbidden evaluation-only metadata field (heldout
  masks, routes, dependency parents, generation metadata, and sealed seeds)
  and require fitted and selected artifacts to remain byte-identical; declared
  `train` query targets are the sole allowed supervised label;
- record explicit booleans that heldout labels were not used for fitting or
  selection.

Even choosing `D=21` from the full-support Phase-I audit is an oracle prior.
Sequence-only experiments must include preregistered `D=20`, `D=21`, and
overcomplete controls, or label `D=21` as supplied knowledge.

## Preregistered hypothesis ladder

The initial experiment should proceed in the following order.

### 0. Implementation controls

- exact 21-dimensional executor;
- explicit nullspace twins that agree on every train-valid state and disagree
  on sealed probes;
- output-label permutations that make default class-zero prediction fail.

### 1. Canonical-state system identification

Expose the canonical state and successor during fitting.  Compare:

- unrestricted operators and readouts;
- shared update/copy with local bind/query;
- bind-covariant but query-free;
- query-covariant but bind-free;
- fully shared learned prototypes;
- hard cyclic equivariant presentation.

This stage isolates coefficient optimization.  It is not representation
discovery.

### 2. Completion controls

- additive key-plus-value cell completion;
- low-rank pair-table completion with declared rank and anchor conditions;
- a shared transducer plus a flexible pair-local residual;
- minimum-norm completion only as an explicitly canonical-gauge geometric
  prior.

Additive or low-rank completion may answer the direct missing cell without
learning a closed transition algebra.  Dynamic probes must decide that.
Minimum Frobenius norm is not invariant under a nonorthogonal similarity
transform, so it cannot serve as a gauge-free discovery criterion.

### 3. Soft laws

Fit an expressive model with sampled penalties for group, bind, query, and
copy laws on seen states.  Report residuals separately on:

- sampled train trajectories;
- the complete seen semantic basis;
- the complete full basis, opened only after freezing.

A zero train-law residual does not close the missing column: the Phase-I
rank-one witnesses also have zero residual on the train span.

### 4. Hard universal laws

Parameterize the complete bind/update/query/copy/invalidate presentation so
the equations hold on all coordinates by construction.  This is the positive
sufficiency control.

### 5. Structure selection

Fit a flexible shared-plus-local-residual supermodel.  Choose residual
strength, rank, and law family only by seen-only pseudoheldout validation or a
preregistered prequential description-length criterion.  This stage is the
first one that can provide evidence of discovery.

### 6. Latent representation learning

Move beyond the now-implemented trace-supervised table learner and remove the
supplied addressable-register representation itself.  Learn a predictive
state and operators from visible event sequences and query answers.  Evaluate
behavior and operator identities up to similarity; never compare raw latent
matrices across seeds.  A similarity map may be fitted on the rank-20 seen
span only.  Extending it with the missing state is test leakage.

## Selecting structure without opening the real heldout pair

The real exclusion gives only one missing cell, so it cannot be used for
model selection.  Use the 19 observed key/value cells as inner folds:

1. retain the real `(0,0)` exclusion in every fold;
2. temporarily remove one additional observed cell from the fold's fit data;
3. fit every hypothesis from a fresh initialization;
4. evaluate balanced direct and compositional probes for that pseudoheldout
   cell;
5. rotate through all 19 observed cells and report each cell slice separately;
6. select the hypothesis by aggregate fold performance and a frozen
   complexity rule;
7. retrain the selected hypothesis on all seen-only data;
8. open the real heldout suite once.

This procedure can show that repeated structure predicts many deliberately
withheld cells.  It still does not logically rule out a unique exception at
the final cell.  It supplies empirical evidence relative to the declared
exchangeability/simplicity prior.

An active version is stronger.  One membership query that reaches the missing
state direction raises the feature span from 20 to 21.  More generally,
counterexample-guided automata learning keeps adding prefixes and suffixes
until the observation table is complete.  If active interventions are allowed,
the benchmark should measure query efficiency as well as final accuracy.

## Sealed behavioral probe families

The current evaluation fixture begins with `BIND(0,0)` followed quickly by
`QUERY(0)`, and the heldout answer is always class zero.  That is vulnerable to
echo, zero-fill, and argmax tie-breaking shortcuts.  A valid discovery suite
must balance labels and paths.

For every rotated heldout cell, include:

- direct bind-in, update-in, and copy-in entry into the missing value;
- update-out and copy-out after entering it;
- repeated cyclic updates, including the identity cycle `U^V`;
- invalidate/rebind reset;
- repeated query identity;
- alternate derivations of the same state and path-independence checks;
- distinct-key commutation and deliberate copy-order noncommutation;
- shuffled bind order, lane permutations, distractors, and long
  interleavings;
- long products at lengths beyond training.

Track causal provenance of the first entry into the heldout coordinate.
`heldout_combination_mask & QUERY` is insufficient because it misses
descendant queries after an update or copy leaves the heldout value.

All-zero or tied logits are not an identified prediction, even if deterministic
argmax happens to return the right class.  Require finite cross-entropy,
positive margin, balanced output-label and cell controls, and exact output for
exact linear controls.  The current rotated-cell suite does not certify
program-level conjugacy or equivariance.

## What counts as success

### Canonical exact controls

- exact seen fit;
- exact satisfaction of every law claimed by the model;
- 100% accuracy on every sealed probe family;
- deterministic recovery across seeds whenever the design has full rank;
- explicit disagreement from every negative-control witness.

### Neural sequence-only models

Before running, freeze numerical gates such as:

- at least 0.99 overall probe accuracy;
- no probe family below 0.95;
- at least 0.99 path-consistency;
- at least 18 of 20 paired model/data seeds passing;
- no best-seed selection.

The independent unit is the seed/fold, not thousands of correlated query
positions.  Three seeds are suitable only for an exploratory pilot.

## How to interpret each outcome

| Outcome | Defensible interpretation |
|---|---|
| Hard laws pass; flexible models fail | The algebra is useful when supplied, not discovered. |
| Shared 96-parameter model passes on independent canonical transition records | Repeated operator structure is sufficient to learn the coefficients. |
| Exact-executor oracle-state control passes | The fitter and rank bookkeeping work; no empirical learning claim. |
| Soft laws pass only on seen states | The penalty fits sampled identities but does not close the missing direction. |
| Direct bind probe passes; dynamic probes fail | Cell completion, echo, or default behavior; no learned algebra. |
| Flexible residual model repeatedly selects zero residual on pseudoheldout folds and passes sealed probes | Evidence that a named selection procedure can recover the shared algebra. |
| Behavioral probes pass but latent matrices differ | The algebra is learned up to similarity gauge. |
| Active learner closes the rank with few counterexamples | The behavior is efficiently identifiable with interventions. |

No result on a zero-support split proves assumption-free recovery.  The
strongest honest passive claim is:

> A specified selection rule, using a specified simplicity or symmetry prior,
> stably selects the intended compositional extension across rotated
> pseudoheldout environments and predicts a separately sealed combination.

## Relation to prior theory

This experiment sits at the intersection of several established theories.

- Weighted-automaton spectral recovery needs a complete finite Hankel basis:
  [Balle, Carreras, Luque, and Quattoni](https://borjaballe.github.io/papers/preprint-bclq13.pdf).
  Separately, in the canonical feature realization studied here, the observed
  span has dimension 20 while the full absence-aware span has dimension 21.
  Thus that chosen canonical basis is incomplete.  This is not a claim that
  the task's strict behavioral Hankel rank is 21; the strict legality-and-output
  series analyzed in Phase I has rank 192.
- Under arbitrary missing Hankel entries, learning becomes constrained matrix
  completion and needs structural assumptions:
  [Balle and Mohri](https://papers.neurips.cc/paper/4697-spectral-learning-of-general-weighted-automata-via-constrained-matrix-completion.pdf).
- Active membership/equivalence queries can exactly learn automata by adding
  counterexamples:
  [Angluin](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf),
  with a linear-algebraic weighted-automaton extension described by
  [Kaznatcheev and Panangaden](https://doi.org/10.1016/j.ipl.2021.106130).
- Spectral HMM and predictive-state recovery require rank, separation, and
  conditioning assumptions rather than mere recurrence:
  [Hsu, Kakade, and Zhang](https://www.sciencedirect.com/science/article/pii/S0022000012000244)
  and
  [Boots, Siddiqi, and Gordon](https://journals.sagepub.com/doi/10.1177/0278364911404092).
- Hard group-equivariant architectures reduce sample complexity by universal
  weight sharing:
  [Cohen and Welling](https://proceedings.mlr.press/v48/cohenc16.html).
  Their benefit is an imposed symmetry, exactly the role of the hard cyclic
  control here.
- Symmetries can sometimes be discovered when a candidate search family and
  adequate support are supplied.  Recent finite-group regression theory gives
  polynomial-time recovery over a subgroup lattice:
  [Soleymani, Tahmasebi, Jaillet, and Jegelka](https://proceedings.mlr.press/v336/soleymani26a.html).
  Empirical methods such as
  [LieGAN](https://proceedings.mlr.press/v202/yang23n.html)
  likewise search a declared symmetry representation.
- Unsupervised latent structure is not identifiable without inductive biases:
  [Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html).
  Interaction and transition information can provide the missing structure:
  [Caselles-Dupré, Garcia-Ortiz, and Filliat](https://arxiv.org/abs/1904.00243).
- Sparse equation discovery succeeds relative to a supplied function library:
  [Brunton, Proctor, and Kutz](https://doi.org/10.1073/pnas.1517384113).
  That is the right analogy for selecting our operator presentation: discovery
  is always relative to a hypothesis language.

The literature therefore supports a precise research claim, not a mystical
one: compositional algebra can be learned when repeated environments or
interventions identify its shared operators, and when the search space makes
the relevant law expressible.  Neither passive absence nor generic neural
capacity supplies that identification by itself.

## Immediate experimental sequence

The exact learner, nullity witnesses, excitation curves, 19 rotated
pseudoheldout folds, shared-plus-local selector, balanced behavioral probes,
and observed-exception power control are now executable and verified.  The
next sequence is:

1. commit the exact V3 protocol, implementation manifest, runtime, and power
   prerequisite before any new environment is fit;
2. execute all 40 preopen environments, form the terminal aggregate, and open
   the outer probes only if every candidate and final-fit gate passes;
3. freeze a genuinely label-held prospective environment outside this already
   inspected task family;
4. remove the supplied addressable-register coordinates and learn a minimal
   predictive state/operator realization from opaque event/query symbols;
5. test tensor geometry only after the predictive algebra is identified, by
   certifying a unique low operator-Schmidt-rank factorization up to gauge;
6. treat automatic discovery of the symmetry group or presentation as a
   separate experiment, not as a side effect of hard tying.
