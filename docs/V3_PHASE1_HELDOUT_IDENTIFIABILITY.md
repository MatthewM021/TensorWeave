# Phase I: Identifiability of the Held-Out Binding Coordinate

## Scope and conclusion

The validation-screen task withholds the single key/value pair

\[
(k_\star,v_\star)=(0,0)
\]

from every train and validation state. Evaluation deliberately binds and later
queries that pair. This note asks a narrower question than model accuracy:

> Is the behavior of the missing coordinate determined by the complete
> train-support transition system, and, if not, which structural assumptions
> make it determined?

The answer has three parts.

1. **Data alone do not identify the held-out behavior.** There are explicit
   21-dimensional linear operators and query readouts that agree on every
   train-valid semantic state and disagree on `(0,0)`.
2. **Generic key/value factorization is not enough.** A rank-one interaction
   can be supported only on the missing pair. Additive sharing, a suitably
   nondegenerate low-rank completion, or exact key/value equivariance can make
   the missing behavior unique.
3. **The transitive cyclic `C_4` law uniquely completes the missing successor
   table, but does not by itself identify the full transducer.** Full recovery
   additionally requires universal laws tying bind, query, copy, invalidation,
   guards, and composition to that value action.

Here, "unique" means unique **observable behavior on valid programs**. It does
not mean that learned latent coordinates or matrices are numerically unique.

The executable calculations live in
[`algebra_identification.py`](../v3/src/tnlm_v3/algebra_identification.py) and
use the exact contracts from
[`exact_algebra.py`](../v3/src/tnlm_v3/exact_algebra.py). This note extends the
[Phase-I exact-algebra note](V3_PHASE1_EXACT_ALGEBRA.md); its distinctions among
promised-valid, absence-aware, and strict legal-language behavior remain in
force. None of these finite-benchmark results establishes a finite algebra or
finite Hankel rank for unrestricted natural language.

## Frozen task and the missing direction

The screen task has

\[
K=5,\qquad V=4,\qquad M=3,
\]

where `K` is the number of surface keys, `V` the value cardinality, and `M`
the simultaneous-live-binding cap. Its canonical absence-aware homogeneous
state is

\[
\phi(s)=c+\sum_{k\text{ live}}e_{k,s(k)}
\in \mathbb{Q}^{1+KV}=\mathbb{Q}^{21}.
\]

Let

\[
h=e_{0,0}
\]

denote the held-out coordinate and let `h*` select it. Train generation
excludes `(0,0)` not only from `BIND` events but from every live result of an
update or copy. Consequently every train-state feature lies in

\[
S_{\mathrm{train}}
=\operatorname{span}\{c,e_{k,v}:(k,v)\ne(0,0)\}
=\ker h^*.
\]

The exact counts are:

| Quantity | Full contract | Train support |
|---|---:|---:|
| Semantic states | 821 | 708 |
| Homogeneous feature rank | 21 | 20 |
| Absence-aware diagnostic rank | 21 | 20 |

The 113 omitted states are exactly the states containing `key 0 = value 0`:

\[
113
=1+\binom41 4+\binom42 4^2.
\]

Although 113 discrete states disappear, they introduce only one new direction
in the canonical linear realization. This is why the train/evaluation feature
rank changes from 20 to 21.

Weighted-automaton spectral recovery requires a finite Hankel basis block whose
rank equals the rank of the full series. Separately, the chosen canonical
feature realization here loses one sampled direction: its train span has rank
20 while its full absence-aware span has rank 21. The observed canonical basis
therefore cannot determine that missing direction. This is analogous to
constrained Hankel completion under arbitrary sampling, and the localized
witnesses below prove that the behavioral extension is nonunique here. It is
not a claim that the task's behavioral Hankel rank is 21. See
[Balle, Carreras, Luque, and Quattoni](https://borjaballe.github.io/papers/preprint-bclq13.pdf)
for the complete-basis condition and
[Balle and Mohri](https://cs.nyu.edu/~mohri/pub/swa.pdf) for the Hankel-
completion framing.

This rank-one gap must not be confused with the strict-contract Hankel rank of
192 established in the main Phase-I note. They answer different questions:
20 versus 21 concerns the span of absence-aware state features under a
sampling exclusion; 192 concerns the suffix-closed strict observable series
with guards and an error sink.

## What the unrestricted train system identifies

The strongest possible data-only audit in `algebra_identification.py` assumes
that training reveals much more than the neural experiment actually reveals:

- the canonical 21-dimensional latent state is observed;
- the exact successor vector is observed for every legal train-support edge;
- every train-valid semantic state is available, rather than a finite sample;
- every generator-supported event signature is independently parameterized;
- every key-specific four-class query readout is linear.

If held-out behavior is not identifiable under this optimistic design, it
cannot become identifiable merely by hiding states and supervising only query
answers.

There are 67 generator-supported event signatures. Each unrestricted event
matrix has `21 x 21 = 441` coefficients. Five key-specific query readouts each
have `4 x 21 = 84` coefficients. Exact rational rank gives:

| Design component | Parameters | Constraint rank | Nullity |
|---|---:|---:|---:|
| 67 event operators | 29,547 | 24,675 | 4,872 |
| 5 query readouts | 420 | 380 | 40 |
| **Complete optimistic system** | **29,967** | **25,055** | **4,912** |

The operator calculation covers 16,107 legal train-support transitions. Only
66 of the 67 event signatures are observed: `BIND(0,0)` has no train edge and
its entire 441-parameter matrix is unconstrained.

The nullity 4,912 is not solely the one held-out coordinate. Guards restrict
the source subspace seen by each action, producing additional unobserved
directions. The one-dimensional feature gap is nevertheless enough to prove
behavioral non-identifiability, and the witnesses below isolate it directly.

## Explicit train-equivalent witnesses

### Unseen bind witness

Let `B_{k,v}` denote the homogeneous operator for binding key `k` to value
`v`. Consider two systems that are identical except that the second uses

\[
\widetilde B_{0,0}=B_{0,1}.
\]

No train program contains `B_{0,0}`, so the systems agree on all train data.
They disagree immediately on

\[
[B_{0,0},Q_0],
\]

which returns 0 in the exact system and 1 in the alternative.

This is the simplest witness. It applies whenever event signatures are
unrestricted or have enough pair-specific capacity to isolate the missing
bind.

### Query-readout witness

Let `R_0` be the four-class readout for key 0, and let `o_0,o_1` be output
basis vectors for labels 0 and 1. Define

\[
\widetilde R_0
=R_0+(o_1-o_0)h^*.
\]

For every train state `x`, `h*x=0`, so

\[
\widetilde R_0x=R_0x.
\]

On a state containing `h`, the alternative changes the query answer from 0 to
1. The live implementation exhaustively verifies equality on all 708 train
states before checking this disagreement.

### Operator-column witness

For any unrestricted event matrix `A_a`, every perturbation

\[
\widetilde A_a=A_a+u_a h^*
\]

agrees with `A_a` on the complete train span. This leaves 21 free coefficients
in the single unobserved input column, or 20 if the homogeneous output row is
fixed by definition.

The executable witness changes the unseen input column of key 0's
`UPDATE(argument=0)` operator. On all 708 train states the exact and
adversarial operators agree. Starting from `key 0 = value 0`, the exact update
produces value 1 while the adversarial update produces value 2.

The canonical witness report also executes the unseen-bind construction. It
records zero train-support transitions for `BIND(0,0)`, then applies the true
and aliased bind operators to empty memory and obtains values 0 and 1,
respectively. The bind, query, and update witnesses are therefore three
separately checked failures of identification, not three descriptions of one
unchecked hypothetical perturbation.

These witnesses prove more than a failure of a particular optimizer: the
observed equations admit multiple exact solutions with different evaluation
behavior.

## Which structural ties close the gap?

The executable report also compares five canonical local-block designs. This
is an intentionally optimistic experiment: the value blocks and their exact
targets are treated as observed. The counts are therefore identifiability
controls, not parameter counts for a generic neural architecture.

| Structural hypothesis | Parameters | Constraint rank | Nullity | Missing behavior fixed? |
|---|---:|---:|---:|:---:|
| Key-local bind/update/copy/query blocks | 720 | 656 | 64 | No |
| Share update and copy prototypes only | 224 | 216 | 8 | No |
| Share bind, update, copy, and query prototypes | 96 | 96 | 0 | Yes |
| Per-key calibrated cyclic bind/query orbits | 40 | 40 | 0 | Yes |
| Key-shared calibrated cyclic bind/query orbit | 8 | 8 | 0 | Yes |

The 64-dimensional local ambiguity splits exactly into 4 bind, 24 update, 32
copy, and 4 query coefficients. Sharing update and copy removes their 56
degrees but leaves the bind and query interfaces independently ambiguous:
`4 + 4 = 8`. This is a useful minimal negative result. It is not enough for
the middle of the state transition to have the right group action; the input
and output interfaces must belong to the same presented algebra.

Full prototype sharing or calibrated cyclic orbits remove the design nullity
entirely in this canonical block model. As elsewhere in this note, zero
behavioral nullity does not remove similarity gauges in a more general latent
realization, and these exact ties are supplied hypotheses rather than
conclusions learned from the unrestricted data.

## Pair completion: what sharing does and does not buy

It is tempting to say that a key embedding and a value embedding should fill
the missing cross-product cell. That conclusion depends on the exact
factorization class.

### Arbitrary pair table

A general scalar table `F(k,v)` has 20 independent entries. The 19 observed
cells have exact design rank 19, leaving nullity one. An explicit missing-cell
interaction is

\[
\Delta(k,v)=\lambda\,\mathbf 1[k=0]\mathbf 1[v=0].
\]

It vanishes on every observed pair and changes only the held-out pair. This is
itself a rank-one key/value interaction. Therefore the mere presence of a
bilinear or tensor-product factorization does not guarantee recovery; an
unused factor direction can localize on the missing cell.

### Additive key/value sharing

For the no-interaction model

\[
F(k,v)=a_k+b_v,
\]

the missing value is forced by any observed rectangle:

\[
F(0,0)=F(0,1)+F(1,0)-F(1,1).
\]

There are nine displayed parameters, the observed design has exact rank eight,
and the remaining one-dimensional parameter ambiguity is the familiar gauge

\[
a_k\mapsto a_k+t,\qquad b_v\mapsto b_v-t.
\]

The parameters are not unique, but the missing behavior is: its behavioral
nullity is zero.

### Generic low-rank completion

Write a scalar completion as

\[
M(x)=
\begin{bmatrix}
x&r^\top\\
c&A
\end{bmatrix},
\]

where `x=M_{0,0}` is missing and `A` is the fully observed `4 x 3` anchor
block. If `rank(A)=r`, the observed row and column lie in its corresponding
spaces, and `rank(M)<=r`, then

\[
x=r^\top A^\#c
\]

is unique for any consistent generalized inverse `A^#`. If `rank(A)<r`, the
rank bound need not determine `x`. The all-zero observed table plus
`lambda E_{0,0}` is already a degenerate rank-one counterexample.

For a `5 x 4` table, the generic dimensions of the rank-`r` matrix varieties
are

| Rank `r` | Variety dimension `r(K+V-r)` |
|---:|---:|
| 1 | 8 |
| 2 | 14 |
| 3 | 18 |
| 4 | 20 |

Rank four is unrestricted and necessarily leaves the missing scalar degree.
Ranks one through three can generically determine it only when the required
anchor minor is nondegenerate. Dimension counting alone is not a uniform
identifiability theorem.

### Exact equivariance

Exact symmetry is stronger than generic parameter sharing.

For a key permutation `pi` represented by `Pi_pi`, require universal operator
identities such as

\[
B_{\pi(k),v}
=\Pi_\pi B_{k,v}\Pi_\pi^{-1},
\qquad
R_{\pi(k)}\Pi_\pi=R_k,
\]

with corresponding identities for update and copy. Swapping key 0 with any
other key maps `(0,0)` to an observed `(j,0)`, so exact key equivariance
recovers the missing bind, update, readout, and all copy operators.

Exact value equivariance similarly maps an observed `(0,v)`, `v!=0`, to
`(0,0)` while transforming the output label. The live analysis verifies both
the key-swap construction and the value-shift construction as exact integer
matrix equalities.

Those equivariances are **supplied hypotheses**, not facts inferred from the
train constraints. The report marks
`assumes_universal_exact_operator_identities = true` and
`identities_are_inferred_from_train_constraints = false`. Its recovery checks
answer the conditional question "would these exact identities determine the
missing behavior?" They do not claim that the current data discover, select,
or statistically justify the identities.

This is the important boundary:

> Learned embeddings, shared layers, or low rank may still contain a
> pair-localized residual. Exact transitive equivariance forbids that residual.

## Cyclic `C_4` completion

Let `U` denote key 0's successor update, `v -> v+1 mod 4`. Because value 0 is
excluded from both source and result states, train support directly observes
only

\[
1\mapsto2,\qquad2\mapsto3.
\]

The exact completion enumeration finds:

| Hypothesis class for the successor table | Compatible completions |
|---|---:|
| Arbitrary functions on four values | 16 |
| Permutations | 2 |
| Transitive single-cycle permutations | 1 |

The two compatible permutations are

\[
(0,2,3,1)
\quad\text{and}\quad
(1,2,3,0),
\]

where a tuple lists each input's output. The first fixes 0 and cycles
`1 -> 2 -> 3 -> 1`; it is not transitive. The second is the intended `C_4`
cycle

\[
0\mapsto1\mapsto2\mapsto3\mapsto0.
\]

Thus a declared transitive cyclic law uniquely recovers the missing **update
successor table**.

### Why that is not yet the full transducer

An update group law alone says nothing about an independently parameterized
`BIND(0,0)` operator. Even if bind is tied to the cycle, a query readout may
still relabel the unseen coordinate. Even if bind and query are tied, a copy
into key 0 from a source carrying value 0 can still be altered off support.

Separate witnesses make each gap explicit:

1. **No bind covariance:** keep the cyclic updates and choose `B_{0,0}`
   arbitrarily.
2. **No query covariance:** force the missing state but apply the
   `R_0 + (o_1-o_0)h*` witness.
3. **No copy naturality:** alter `COPY(destination=0, source=j)` only when the
   source carries value 0.
4. **Laws checked only on train states:** add any `u h*` operator perturbation;
   every sampled law residual remains zero.

Accordingly, the live report deliberately records both

- `cyclic_completion_is_unique = true`, and
- `cyclic_update_law_alone_recovers_full_transducer = false`.

These statements are complementary, not contradictory.

## Sufficient universal transducer laws

Let `U_{k,delta}` add `delta` modulo `V`, let `P_delta` apply the same shift to
output labels, and let `C_{d<-s}` copy the source value into the destination.
A sufficient presentation includes the following identities on their declared
guarded or promised-valid domains:

\[
U_{k,\delta}U_{k,\epsilon}
=U_{k,\delta+\epsilon},
\qquad U_{k,0}=I,
\]

\[
U_{k,\delta}B_{k,v}=B_{k,v+\delta},
\]

\[
R_kU_{k,\delta}=P_\delta R_k,
\]

and value-natural copy, including

\[
R_dC_{d\leftarrow s}=R_s
\]

when source and destination are live, together with the corresponding state
update law. Invalidation must be an exact clear, queries and distractors must
be semantic identities, guards must be fixed, and sequence interpretation
must be an exact homomorphism under associative composition.

For the screen, choose the observed anchor value 1. The missing bind is then
forced by

\[
B_{0,0}=U_{0,3}B_{0,1},
\]

where visible update argument 2 encodes shift 3. Query covariance gives

\[
R_0B_{0,0}
=R_0U_{0,3}B_{0,1}
=P_3e_1
=e_0.
\]

Copy naturality gives, for any other key `j`,

\[
R_0C_{0\leftarrow j}B_{j,0}
=R_jB_{j,0}
=e_0.
\]

This equation is on the guarded domain where destination key 0 is already
live; equivalently, prefix the displayed composition with any legal bind of
key 0 before performing the copy.

Finally, all update-out behavior follows:

\[
R_0U_{0,\delta}B_{0,0}=e_\delta.
\]

These identities must be structural or checked on a basis that includes the
held-out direction. A soft law penalty evaluated only on sampled train states
cannot see the missing column and therefore cannot establish identification.

For a direct held-out query, any one of the following can be sufficient:

- exact key equivariance plus coverage of `(j,0)` for another key;
- calibrated transitive value equivariance plus coverage of `(0,v)` for an
  observed value;
- additive sharing; or
- a low-rank completion with a proved nondegenerate anchor condition.

For arbitrary held-out-containing programs, some complete coupling must cover
bind, update, query, copy, invalidation, and guards, along with at least one
observed representative per symmetry orbit and no pair-specific residual
channel. The presentation above is sufficient; it is not claimed to be the
only logically possible presentation.

## Behavior, realization gauge, and routing gauge

The conclusion above is about observable behavior. If

\[
A'_a=SA_aS^{-1},\qquad
b'=Sb,\qquad
R'=RS^{-1}
\]

for an invertible change of basis `S`, the realization matrices differ but all
sequence outputs agree. Minimal weighted-automaton realizations are generally
unique only up to such similarity; see the canonical-form discussion of
[Balle, Panangaden, and Precup](https://arxiv.org/abs/1501.06841). Predictive
state is fundamentally a behavioral quotient, in the spirit of the
[Myhill--Nerode theorem](https://doi.org/10.1090/S0002-9939-1958-0135681-9),
not a privileged neural coordinate system.

The random document-local lane assignment adds another gauge: a branch
permutation changes oracle route labels without changing binding answers. It
is independent of the missing-coordinate ambiguity. Exact recovery should
therefore be reported at three levels:

1. **behavioral identification:** all valid future probes agree;
2. **realization identification:** matrices agree up to similarity;
3. **raw-coordinate identification:** matrices agree in a fixed canonical
   gauge.

Only the first is necessary for correct semantic generalization. Any claim of
literal matrix recovery must additionally fix and justify a gauge.

## Exact experiment ladder

The next experiments should change one structural assumption at a time while
using the same train-support equations and held-out probes.

### 0. Exact control and witness audit

Run `analyze_heldout_identification` and preserve its canonical report. Require:

- `821/708` full/train states;
- `21/20` full/train feature and diagnostic ranks;
- total unrestricted counts `29,967/25,055/4,912` for
  parameters/constraint-rank/nullity;
- equality of the adversarial readout and update operator on all 708 train
  states;
- zero train-support uses of the held-out bind and disagreement between its
  true and aliased executions;
- disagreement on the held-out state;
- one transitive cyclic completion; and
- exact key- and value-equivariance recovery checks, explicitly recorded as
  supplied universal identities rather than identities inferred from data.

### 1. Unrestricted operator baseline

Give every visible event signature an independent operator and every key an
independent readout. This condition should fit the complete train system but
has no principled answer for the missing coordinate. Measure variation across
seeds and explicitly compare it with the constructed witnesses.

### 2. Generic key/value factorization

Use learned key and value factors without exact symmetry. Sweep factor rank
and report anchor-block rank. This tests generic matrix completion, not
algebraic recovery. Include a localized-interaction diagnostic to determine
whether spare capacity can represent `E_{0,0}`.

### 3. Additive and exact-equivariant controls

Evaluate separately:

- additive key/value sharing;
- exact key-permutation equivariance;
- exact cyclic value equivariance; and
- both symmetries together.

These are identification-positive controls. Parameter gauges may remain, but
held-out behavior should be invariant.

### 4. Cyclic-update-only condition

Force the successor operator to be the unique transitive `C_4` completion but
leave bind, readout, and copy unrestricted. This condition should recover the
value cycle while retaining the full-transducer witnesses. It is the critical
control against the false conclusion that `U^4=I` solves the task by itself.

### 5. Sampled law regularization

Penalize cyclic, bind, query, and copy equations only on train examples. Then
measure residuals both on the train span and on the complete 21-dimensional
basis. A model can have zero train residual and nonzero held-out residual; this
condition quantifies that failure mode.

### 6. Universal-law parameterization

Build bind, update, query, copy, and invalidation from a shared algebraic
presentation so the identities hold globally by construction. This is the
identification-positive compositional model. Compare its behavior with the
hand-specified exact executor, not merely with labels from sampled episodes.

### 7. Parsing and tensor factorization only afterward

Once the structured event algebra is recovered, separately test learned event
parsing and tensor factorization of the exact operators. This follows the
realization-first sequence proposed in the
[Phase-I exact-algebra plan](V3_PHASE1_EXACT_ALGEBRA.md#phase-i-experiment-plan):
first identify the predictive algebra, then ask whether a tensor geometry
compresses it without breaking its laws.

## Required counterfactual probes

Ordinary held-out accuracy should be accompanied by algebraically related
programs whose answers are known exactly:

| Probe | Program | Expected answer |
|---|---|---:|
| Direct bind | `BIND(0,0); QUERY(0)` | 0 |
| Cyclic derivation | `BIND(0,1); UPDATE(0, argument=2); QUERY(0)` | 0 |
| Cross-key copy | `BIND(0,1); BIND(1,0); COPY(0 <- 1); QUERY(0)` | 0 |
| Update out | `BIND(0,0); UPDATE(0, argument=0); QUERY(0)` | 1 |
| Clear and rebind | `BIND(0,0); INVALIDATE(0); BIND(0,2); QUERY(0)` | 2 |

The direct, cyclic, and copy programs are distinct derivations of the same
answer. A compositional model should make them agree for structural reasons.
The update-out and clear/rebind probes verify that the held-out representation
participates correctly in later composition rather than merely producing one
memorized output.

For every learned condition, report:

- train-support fit;
- held-out program accuracy;
- exact or numerical law residuals on the train span;
- the same residuals on the full semantic basis;
- constraint/design rank and nullity where available;
- behavior agreement after gauge alignment, separately from raw matrix error;
- paired seeds and paired generated streams.

## Claim boundary

The established result is an identifiability theorem and finite exact audit for
one synthetic task:

- unrestricted data admit held-out-disagreeing solutions;
- additive or exact transitive symmetry can determine the missing pair;
- the transitive `C_4` hypothesis uniquely completes the successor table;
- that cyclic completion is insufficient for the full transducer without
  universal bind, query, copy, invalidation, guard, and composition laws.

The result does not show that natural language has this algebra, that tensor
networks will discover it, or that a soft algebra loss will enforce it. Its
research value is sharper: it gives a controlled setting in which empirical
generalization can be separated exactly into data coverage, matrix completion,
symmetry, algebraic closure, and latent gauge.
