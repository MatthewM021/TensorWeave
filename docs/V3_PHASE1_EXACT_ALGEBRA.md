# Phase I: Exact Algebra of the V3 Dynamic-Binding Task

## Status and scope

This note fixes the mathematical object to be studied before any further model
design. It applies to the current `binding-v1` synthetic benchmark, and in
particular to the frozen validation-screen parameters

\[
K=5,\qquad V=4,\qquad B=3,\qquad M=3,
\]

where (K) is the number of visible keys, (V) the value cardinality, (B)
the number of local route lanes, and (M) the simultaneous-live-binding cap.
The sole held-out pair is ((k_0,v_0)).

The main conclusion is deliberately narrow:

> The current bounded binding task has an exact finite compositional algebra.
> Its required realization depends strongly on what is declared observable.

This is not evidence that unrestricted natural language has finite Hankel rank,
a fixed-width state, or this particular algebra. It gives us a completely
understood control problem on which methods for discovering an algebra can be
tested.

The first executable checkpoint is now implemented in
`v3/src/tnlm_v3/exact_algebra.py`, with the reproducible analysis entry point
`v3/scripts/analyze_binding_algebra.py` and focused contract tests in
`v3/tests/test_exact_algebra.py`. The implementation includes bounded semantic
enumeration, exact rational/finite-field diagnostic ranks, homogeneous integer
operators, the symbolic segment normal form, generated-episode replay, and a
strict-rank certificate that matches a GF(2) closure lower bound to the
independent structural upper bound. It does not yet implement the guarded
segment predicate as a canonical BDD or recover the operators from sampled
Hankel blocks.

The companion
[held-out identifiability note](V3_PHASE1_HELDOUT_IDENTIFIABILITY.md) and
`v3/src/tnlm_v3/algebra_identification.py` now determine exactly why the
train-support rank is 20, construct train-equivalent systems that disagree on
the 21st direction, and certify which supplied algebraic symmetries close that
gap.

## Three different semantics must not be conflated

The same generator supports at least three legitimate mathematical questions.
They have different predictive quotients and different linear dimensions.

| Contract | What is observable | Discrete states | Exact linear result |
|---|---|---:|---:|
| **A. Promised-valid, currently scored behaviour** | Only value targets on generated queries; every queried key is promised live | Liveness may be quotiented away | Dimension at most (1+K(V-1)=16); minimum not yet established |
| **B. Absence-aware diagnostic behaviour** | Every key can be probed and returns `ABSENT` or one of (V) values | (821) | Minimal dimension exactly (1+KV=21) |
| **C. Strict legal-language behaviour** | Illegal events are observable and enter an absorbing dead state; state diagnostics expose absence/value | (821+1=822) | Verified strict-contract Hankel rank (192) |

The ranks are not competing estimates of one object. Changing the observable
contract changes the equivalence relation on histories and therefore changes
the Hankel matrix.

### A. Promised-valid, currently scored behaviour

The campaign's primary quality metric scores a (V)-class answer only at
`QUERY` events generated for live keys. It does not ask the model to:

- say that an inactive key is absent;
- reject an illegal query, copy, update, bind, or invalidation;
- predict which events the generator is allowed to emit next; or
- reproduce dependency-parent and generation annotations.

This creates an important observability loophole. An exact executor may treat
`INVALIDATE(k)` as the identity on its value memory. The stale value cannot be
queried or copied while the key is inactive, and a valid later `BIND(k,v)`
overwrites it before that key can again affect a scored answer. Invalidation is
real in the generator, but it is not identifiable from the current query-only
behaviour.

Choose (V) affinely independent simplex codes
(c_0,\ldots,c_{V-1}\in\mathbb R^{V-1}). Store one code per key and one shared
homogeneous coordinate:

\[
z=(1,c_{x_1},\ldots,c_{x_K})
   \in\mathbb R^{1+K(V-1)}.
\]

An unbound or invalidated key may retain any stale code. Bind is an affine
reset, update is an affine permutation of the simplex, copy replaces one key's
code with another's, and query has an affine (V)-class readout. These maps
become linear after the single homogeneous lift. Thus the frozen task has an
exact promised-valid realization of dimension at most

\[
1+5(4-1)=16.
\]

This is an upper bound, not yet a claim that the current scored series has
Hankel rank exactly (16). Establishing its exact minimum requires stating the
series coding and completing reachable/observable minimization under the
promised-valid domain.

If next-event prediction, explicit absence, or legality were added to the
objective, invalidation would immediately become observable and this quotient
would no longer be valid.

### B. Absence-aware diagnostic behaviour

Let

\[
X=\{\bot\}\cup\mathbb Z_V,
\]

where `ABSENT` denotes an inactive key. A semantic state is a capped partial
map

\[
x:[K]\rightharpoonup\mathbb Z_V,
\qquad |\operatorname{dom}x|\le M.
\]

The number of states is

\[
N_{K,V,M}
=\sum_{\ell=0}^{M}\binom K\ell V^\ell.
\]

For the frozen screen,

\[
N_{5,4,3}=1+5\cdot4+10\cdot16+10\cdot64=821.
\]

If a diagnostic query may return `ABSENT`, every two distinct partial maps are
distinguishable by probing a key on which they differ. Hence all (821) are
distinct deterministic predictive states under this contract.

Define coordinates

\[
x_{k,v}=\mathbf 1[x(k)=v]
\]

with every coordinate in key block (k) equal to zero when (k) is absent.
The homogeneous state

\[
z=(1,x_{1,0},\ldots,x_{K,V-1})\in\mathbb R^{1+KV}
\]

gives an exact (21)-dimensional linear realization. Absence is read as
(1-\sum_v x_{k,v}). Empty state plus the (KV) singleton key-value states
give (1+KV) independent directions, and the absence/value probes observe
those directions. Therefore (21) is both sufficient and necessary for this
diagnostic contract.

The large difference between (821) deterministic states and linear dimension
(21) is fundamental: discrete state count is not bond dimension, WFA
dimension, or Hankel rank.

### C. Strict legality with an observable sink

The strict contract retains the (821) capped partial maps and adds an
absorbing dead state (dagger). Any event that violates its precondition moves
to (dagger), and the dead condition is observable. The base observation is a
dead indicator together with, for every key, a one-hot outcome in

\[
\{\texttt{ABSENT},0,1,2,3\}.
\]

The exact finite audit used the (67) event signatures supported by the
generator:

- (KV=20) binds;
- (K(V-1)=15) non-identity updates;
- (K(K-1)=20) directed copies;
- (K=5) invalidations;
- (K=5) queries; and
- two distractor scopes.

Starting from the base output coordinates, the audit repeatedly closed the
row space under precomposition by these event transformations. Over both
`GF(2)` and the real field, the space reached rank (192) after one
full generator-extension round and was stable.

The computed basis count matches

\[
1_{\mathrm{sink}}+1_{\mathrm{empty}}
+\sum_{\substack{\varnothing\ne S\subseteq[K]\\|S|\le M}}
  \left(1+|S|(V-1)\right),
\]

which evaluates to

\[
1+1+5(4)+10(7)+10(10)=192.
\]

This formula has a clear interpretation: strict guards distinguish each active
key set (S), while values within a fixed active set contribute one constant
direction and (|S|(V-1)) value-contrast directions. The equality (192) is a
verified finite result for the observation and event coding above. A general
symbolic proof of the displayed basis formula remains a Phase-I deliverable;
it should not yet be advertised as a theorem for other configurations or
other output codings.

## Exact event algebra

The source of truth is `v3/src/tnlm_v3/data.py`. Raw mathematical key and value
IDs are zero-based; model tensors add one so that zero is reserved for an
absent field or padding.

For a state (x\in X^K), the valid task events are:

| Event | Strict precondition | Semantic action | Scored output |
|---|---|---|---|
| `BIND(k,v)` | (x_k=\bot) and fewer than (M) keys are live | (x_k\leftarrow v) | none |
| `UPDATE(k,t)` | (x_k\ne\bot) | (x_k\leftarrow x_k+t+1\pmod V) | none |
| `COPY(d,s)` | (d\ne s), both live | (x_d\leftarrow x_s) | none |
| `INVALIDATE(k)` | (x_k\ne\bot) | (x_k\leftarrow\bot) | none |
| `QUERY(k)` | (x_k\ne\bot) | identity | (x_k) |
| `DISTRACTOR(global)` | always | identity | none |
| `DISTRACTOR(null)` | always | identity | none |

The generator chooses a new value different from the current value, so it does
not emit update argument (t=V-1), which would be the identity modulo (V).
The lower-level validator can represent that no-op update, but it is not one of
the (67) generator-supported signatures used in the strict audit.

Padding is not an event in this algebra. A false `valid_mask` is a total model
no-op, emits zero logits, does not advance the forest clock, and carries ignored
evaluation sentinels.

### Homogeneous operator matrices

Use the absence-aware coordinates (z=(1,\operatorname{vec}x)) above. Each
event has an exact (0/1) homogeneous matrix:

- **Set/bind** (S_{k,v}): zero block (k), then copy the homogeneous
  coordinate into entry ((k,v)).
- **Update** (U_{k,\delta}), with (delta=t+1\pmod V): apply the cyclic
  permutation (P_\delta) to block (k). The all-zero absent block remains
  zero.
- **Copy** (C_{d\leftarrow s}): zero destination block (d), then copy
  source block (s) into it.
- **Delete/invalidate** (D_k): zero block (k).
- **Query and distractors**: the identity matrix.

For column-state convention, a sequence (a_1\cdots a_n) acts as

\[
z_n=A_{a_n}\cdots A_{a_1}z_0.
\]

Associativity is inherited from matrix multiplication rather than encouraged
by a soft loss. Some immediate laws are

\[
U_{k,\delta}U_{k,\epsilon}=U_{k,\delta+\epsilon\bmod V},
\qquad C_{d\leftarrow s}^2=C_{d\leftarrow s},
\]

and operations on disjoint key supports commute unless a copy creates a data
dependency. Set, delete, and copy are irreversible resets or assignments;
updates form cyclic reversible actions.

These matrices are natural total extensions of the memory operations. The
strict language is obtained by pairing them with the precondition tests in the
table. For example, the total extension of update leaves an absent block
absent, whereas the strict contract sends that attempted update to
(dagger).

### Symbolic segment normal form

The matrices have an even smaller symbolic representation. For each input key
variable (X_j\in X), define

\[
\sigma_\delta(\bot)=\bot,
\qquad
\sigma_\delta(v)=v+\delta\pmod V.
\]

Ignoring guards, every event segment has an output expression for each key of
one of the forms

\[
E_i=c,
\qquad c\in X,
\]

or

\[
E_i=\sigma_\delta(X_j),
\qquad j\in[K],\ \delta\in\mathbb Z_V.
\]

This follows by induction. Initially (E_i=X_i). Set and delete replace one
expression with a constant, copy duplicates an existing expression, and update
adds a modular shift. Substitution composes two segments and preserves the
same form. Normalizing shifts modulo (V) gives a canonical expression tuple
for the totalized transformation.

For strict semantics, attach a Boolean predicate (g(X_1,\ldots,X_K)) saying
that every intermediate event guard succeeds, including the live-count cap.
A strict segment is therefore represented exactly by

\[
(g;E_1,\ldots,E_K),
\]

with guard failure mapping to (dagger). A reduced truth table or canonical
BDD can make the guard component canonical. The expression tuple remains
small; the guard is where the strict contract's additional rank and
combinatorics live.

This normal form is a direct candidate for an exact segment summary and
parallel composition baseline. It should be implemented and tested before
asking a neural tensor merge to rediscover it.

## Held-out pair as one missing linear direction

Train and validation generation exclude the held-out pair ((k_0,v_0)) from
every live state, including values produced by update and copy. Evaluation and
test force a bind and later query of that pair.

For general held-out set (H), let

\[
a_k=V-|\{v:(k,v)\in H\}|.
\]

The number of train-reachable absence-aware states is

\[
N_{\mathrm{train}}
=\sum_{\substack{L\subseteq[K]\\|L|\le M}}
  \prod_{k\in L}a_k.
\]

For the frozen screen this is (708), versus (821) without the exclusion.

More revealingly, in the (21)-coordinate realization the train-state span
contains the homogeneous direction and (19) observed key-value directions,
so its exact rank is (20). Evaluation introduces the missing
((k_0,v_0)) coordinate and raises the span rank to (21).

Thus the held-out test is literally extrapolation into one unobserved linear
direction under this representation. An unconstrained learner cannot identify
arbitrary operator or readout behaviour on that direction from train states
alone. A shared compositional law can determine it—for example, the cyclic
update action relates all four value vertices—but that determination comes
from the structural assumption, not from empirical coverage.

The (20\to21) statement concerns the absence-aware state-feature span. It is
not the strict-contract Hankel rank, and it should not be mixed with the
promised-valid (16)-dimensional upper bound.

## Lanes are a gauge variable for the target task

For raw oracle routing, augment a live partial map with an injection

\[
\lambda:\operatorname{dom}x\hookrightarrow[B].
\]

The route-aware state count is

\[
N_{K,V,B,M}^{\mathrm{route}}
=\sum_{\ell=0}^{M}\binom K\ell V^\ell(B)_\ell,
\qquad
(B)_\ell=\frac{B!}{(B-\ell)!}.
\]

For the frozen task this gives (4{,}861) raw states, or (4{,}186) after the
train held-out exclusion.

These labels are not identifiable from query targets. Every document receives
its own random branch permutation, and the same surface key can occupy a
different lane in another document or after invalidation and rebinding. The
symmetric group (S_B) acts by renaming local lanes without changing target
behaviour. Route recovery therefore correctly aligns branch permutations per
document.

This is a gauge freedom, not semantic uncertainty. It is separate from the
usual weighted-automaton gauge in which minimal linear realizations are unique
only up to an invertible change of basis. A route-aware executor can store
key-value and key-lane one-hots separately, but target prediction needs no lane
coordinate at all because primary and secondary keys are already visible.

The two distractor routes do not change this conclusion. Their scope bit is
model-visible: raw argument zero denotes the dedicated global lane and raw
argument one denotes the null route. Both are identities on the target-semantic
memory.

## Consequences for the current architecture

The exact algebra exposes several mismatches in the routed forest. These are
structural observations, not yet causal claims about any particular accuracy
result.

1. **Queries should be semantic identities.** `RoutedBindingModel` classifies
   every event with a primary key as local, so `QUERY` is inserted into its
   branch before readout. Repeated queries therefore change forest counts,
   binary-carry grouping, and learned state even though the exact memory is
   unchanged.

2. **Distractors should be semantic identities.** A global distractor is
   inserted into the global forest lane, while a null distractor still advances
   the valid-event clock. Neither affects a future exact query answer.

3. **Invalidation is an exact clear, but is weakly observable in the current
   score.** The forest appends an invalidation event rather than applying a
   clearing operator. At the same time, query-only supervision permits the
   stale-value loophole described in contract A. The model is neither given an
   exact clear nor strongly required to discover one.

4. **Copy is a state-to-state operation, not merely a destination token.** The
   exact action must snapshot the source's value at copy time. The current
   event is routed only to the destination branch, and that branch update does
   not directly contract the source branch state. A later all-slot readout must
   reconstruct the historical snapshot, including cases where the source has
   since changed.

5. **The learned merge is not constrained to the task laws.** Sharing a CP
   merge across scales does not make it associative, unital, reset-preserving,
   or closed under the symbolic segment normal form. Binary-counter execution
   controls storage topology but does not by itself produce a sufficient
   semantic statistic.

6. **Routing expends capacity on a nuisance symmetry.** Random local lanes are
   useful for testing routing, but they are not required for query semantics.
   A key-indexed exact executor solves the target algebra without discovering
   a document-specific lane permutation.

7. **The held-out direction is structurally underdetermined.** Generalization
   to ((k_0,v_0)) cannot be inferred by an unrestricted tabular action on the
   train span. Parameter sharing must express an actual law linking keys and
   values if it is to fill that direction reliably.

These observations motivate an exact algebraic control and sharper diagnostic
contracts before another architecture search.

## Relation to established theory

### Myhill--Nerode, predictive states, and Hankel realization

The equivalence relation "two histories agree on every allowed future probe"
is the behavioural quotient underlying the [Myhill--Nerode
theorem](https://doi.org/10.1090/S0002-9939-1958-0135681-9). Contract A, B, and
C choose different probes, so they induce different quotients.

For real-valued string series, finite Hankel rank is equivalent to a finite
weighted-automaton realization, and the rank is its minimal linear dimension.
[Balle, Panangaden, and Precup](https://arxiv.org/abs/1501.06841) give a modern
canonical-form and approximate-minimization treatment. Predictive State
Representations similarly define state by predictions of observable future
tests rather than by a privileged hidden coordinate system; see [Singh,
James, and Rudary](https://arxiv.org/abs/1207.4167). [Jaeger's Observable
Operator Models](https://doi.org/10.1162/089976600300015411) provide a closely
related operator account for stochastic sequences.

The Phase-I calculations are a finite, deterministic instance of those ideas.
They also show why "the rank of the task" is incomplete language: a rank is
defined only after the observable series has been fixed.

### Weighted automata and tensor networks

A weighted-automaton score

\[
\alpha^\top A_{a_1}\cdots A_{a_n}\omega
\]

is a uniform Matrix Product State/Tensor Train contraction. The relationship
between weighted automata, recurrent linear models, and tensor trains is made
explicit by [Li, Precup, and
Rabusseau](https://arxiv.org/abs/2010.10029). [Adhikary et
al.](https://proceedings.mlr.press/v130/adhikary21a.html) establish broader
connections among uniform tensor-network sequence models, weighted automata,
and predictive-state/OOM-like models under their stated assumptions.

For this benchmark, the first relevant dimension is therefore (16), (21),
or (192), depending on the contract—not the number of historical tokens and
not automatically the CP rank of a nonlinear merge tensor. A more elaborate
tree geometry is justified only if it factorizes the required behavioural
operators or their cut ranks more efficiently than the exact baseline.

### Krohn--Rhodes and guarded program algebra

The primitive operations already exhibit the reversible/irreversible split
central to Krohn--Rhodes theory. Updates generate cyclic group actions on
active values; bind, delete, and copy are noninvertible reset or assignment
maps. The [Krohn--Rhodes prime decomposition
theorem](https://doi.org/10.1090/S0002-9947-1965-0188316-1)
says that finite transformation semigroups decompose through cascades of group
and reset-like components.

We have not computed the minimal Krohn--Rhodes decomposition of this task. The
theorem supplies a structural vocabulary and a later analysis target, not a
claim that the displayed primitive split is already minimal.

Strict preconditions are naturally separated from state transformers as
tests. [Kozen's Kleene Algebra with
Tests](https://doi.org/10.1145/256167.256195) is a relevant exact formalism for
composing actions, guards, choice, and iteration. It may be a better language
for the strict segment pair ((g;E)) than forcing guards into dense learned
vectors.

### Tensor-product representations

The absence-aware state can be reshaped as

\[
M=\sum_{k\text{ live}} e_k\otimes e_{x(k)},
\]

which is an exact sparse role--filler tensor-product representation. Querying a
key contracts its role; bind writes a filler, invalidate zeros a role slice,
and copy transfers a filler between roles. This is directly related to
[Smolensky's tensor-product variable-binding
framework](https://www.sciencedirect.com/science/article/pii/000437029090007M).

Here the key vocabulary is finite and fixed, so the (KV) growth is acceptable
and provides a ground-truth control. It does not establish that a fixed
role--filler table is adequate for open-ended language, unbounded entities, or
latent linguistic roles.

## Phase-I experiment plan

### 1. Freeze the three behavioural contracts

Implement A, B, and C as separate named specifications. Each must declare:

- its event alphabet and guards;
- whether absence and invalidity are observable;
- its output coding;
- whether raw lanes are included or quotiented by permutation; and
- whether held-out restrictions define the system or only a sampling split.

No Hankel rank should be reported without this declaration.

### 2. Build an independent exact enumerator

Enumerate partial maps lexicographically and, when requested, their injective
lane assignments. Produce exact tables for legality, next state, query output,
route output, held-out status, and post-event live count. Add an optional dead
state rather than silently assigning arbitrary outcomes to illegal events.

Cross-check every generated episode against the independent table, including
mandatory copy, invalidation, rebinding, global/null distractors, and padding.
Evaluation-only generation IDs and absolute dependency-parent pointers should
remain outside the finite predictive state; they are provenance annotations
whose exact replay needs length-growing indices.

### 3. Implement both exact composition representations

Provide:

- the (21\times21) homogeneous integer matrices;
- the (16)-dimensional promised-valid simplex construction;
- the symbolic segment form ((g;E_1,\ldots,E_K)); and
- exact streaming-versus-balanced-composition tests.

Matrix products and symbolic substitution must agree on every enumerated state.
This becomes the non-neural quality and length-generalization ceiling.

### 4. Certify the realization results

Use exact arithmetic, not floating-point SVD thresholds, to:

- determine the exact minimum for contract A rather than stopping at the
  (16)-dimensional upper bound;
- preserve the analytic upper/lower-bound certificate for contract B's rank
  (21);
- reproduce contract C's (192)-dimensional reachable/observable closure; and
- prove or refute the proposed active-set basis formula for general
  ((K,V,M)).

Only after the exact rank is certified should singular spectra be used to study
approximate compression.

### 5. Turn held-out structure into an identification experiment

Fit realizations on the rank-(20) train span and evaluate the missing
twenty-first direction under several assumptions:

- unrestricted tabular operators;
- shared cyclic value-update operators;
- explicit key/value factorization;
- law-regularized learned operators; and
- the hand-specified exact algebra.

The question is not merely whether held-out accuracy rises, but which algebraic
assumptions make the missing direction identifiable.

### 6. Isolate parsing, routing, and execution

Compare four controlled interfaces:

1. visible structured events into the exact executor;
2. learned event parsing into the exact executor;
3. oracle lanes into a learned executor; and
4. learned lanes into a learned executor.

Add counterfactual tests in which queries and distractors are inserted
arbitrarily, invalidated keys remain dormant for long intervals, and a copy's
source changes immediately afterward. These directly test the identity, clear,
and snapshot laws.

### 7. Investigate tensor factorization only after realization

Factor the exact operators and finite Hankel blocks into TT/MPO or hierarchical
forms, measure exact and approximate ranks across declared cuts, and compare
chronological, key-grouped, and learned trees. A tensor geometry is useful only
if it compresses the established predictive interfaces while preserving their
composition laws.

### 8. Generalize cautiously

Vary (K), (V), and (M); add typed relations, scoped entities, and partially
observable events; then repeat the quotient and rank analysis. The research
question for later language-like tasks is whether their empirically relevant
future probes admit similarly structured approximate quotients—not whether the
specific (21)-dimensional binding algebra can simply be relabelled as a
geometry of language.

## Current claims and open questions

Established analytically for the specified contracts:

- the (821)-state absence-aware count;
- the (822)-state strict machine with one observable sink;
- the exact (21)-dimensional absence-aware realization and its matching lower
  bound;
- the promised-valid realization upper bound of (16);
- the totalized symbolic segment expression form; and
- lane-permutation nonidentifiability for target behaviour.

Verified by exact finite computation for the frozen strict contract:

- Hankel/reachable-observable rank (192) under the declared output coding and
  (67)-signature alphabet; and
- the train/evaluation absence-aware feature-span change (20\to21).

Still open:

- the exact minimal rank of the promised-valid scored series;
- a general proof of the strict active-set rank formula;
- the size and minimal Krohn--Rhodes decomposition of the generated
  transformation monoid;
- whether the algebra can be recovered robustly from sampled trajectories;
- whether a tensor factorization provides any advantage over the direct exact
  executor; and
- which, if any, analogous finite or approximately finite predictive algebras
  occur in substantially richer language-like systems.
