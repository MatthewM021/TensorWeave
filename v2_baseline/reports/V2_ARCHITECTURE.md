# V2 Architecture Specification

## Predictive object

Each model maps a context \(x_{1:L}\) to the omitted next-token class. The routed models construct a bounded collection of tensor-network branches rather than a dense sequence-to-sequence attention state.

## Local tensor map

Every binary coarse-graining node uses a CP-factorized order-three tensor:

\[
T_{oij}=\sum_{r=1}^{R} O_{or}L_{ri}R_{rj},
\qquad
h_o=\sum_{ij}T_{oij}a_i b_j.
\]

The implementation evaluates this as two projections, an elementwise product, and an output projection. A gated residual and layer normalization stabilize repeated contraction.

## Fixed TTN

Leaves are token plus sinusoidal-position states. Adjacent states are merged recursively. Masks pass a lone occupied child unchanged, allowing padded sequences and sparse branch leaves.

Nominal work is \(O(LRd)\) per level implementation constant, with \(O(\log L)\) depth. The code performs all leaves in parallel rather than a streaming update.

## MERA-like extension

Before each merge level, two-site linear maps act on shifted pairs

\[
(1,2),(3,4),\ldots
\]

while the isometry/merge acts on

\[
(0,1),(2,3),\ldots.
\]

The maps initialize exactly as identity, so MERA begins functionally equivalent to TTN. An orthogonality penalty discourages pathological scaling. They are called “MERA-like” because the network uses predictive feature vectors and nonlinear residual tensor maps rather than a strict quantum-state isometric circuit.

## Routed branch network

For \(B\) branches, assignments \(p_{t,b}\) mask the token leaves. Each branch contracts its own temporal TTN. The resulting branch roots are contracted by a second balanced tree in branch order.

Oracle mode uses generated branch metadata. Learned mode uses only tokens and embeddings. The learned router has vocabulary-symbol route logits and chooses among current, one-, two-, and three-token causal anchors. Its work is \(O(BL)\); it has no all-pairs attention.

## Adaptive rank

Each level can multiply channels by sigmoid gates \(g_i\). The reported effective rank is the participation ratio

\[
r_{\mathrm{eff}}=\frac{(\sum_i g_i)^2}{\sum_i g_i^2}.
\]

V2 keeps dense kernels, so this is a representational diagnostic. It becomes a computational saving only after a gate-aware compact export. The reference result did not prune.

## Baselines

- token-indexed uniform MPS transfer matrices;
- 32-state GRU;
- 32-dimensional, two-layer, four-head causal Transformer;
- fixed positional TTN/MERA.

## Structural metrics

The benchmark reports explicit **proxies**, not empirical mutual information:

- nominal/effective bond dimension;
- active branch count;
- tree depth;
- cut-capacity proxy \(\sum_e\log_2\chi_e\);
- streaming-state scalar proxy;
- Transformer KV/attention-score scalar proxies;
- measured batch throughput and complete parameter count;
- isolated process maximum RSS and warmed-forward incremental RSS for selected checkpoints.

## Known architectural limitations

1. The tree processes a complete context rather than maintaining a binary-counter streaming forest.
2. Scale-specific merge tensors impede true length extrapolation.
3. Learned routing has no entity-consistency curriculum.
4. Soft rank gates do not export smaller kernels.
5. The combined task has no intermediate branch/global predictive losses.
6. The PyTorch implementation launches many modest operations and is not fused.
7. Optimization is strongly seed-sensitive at tight rank; V2 found no replicated MERA advantage after paired controls.
8. The routed encoder uses generated branch metadata in oracle mode; only learned mode tests deployable geometry inference.
