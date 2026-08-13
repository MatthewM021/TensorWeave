# V2 Predictive-Geometry Tensor-Network Language Model

## Status

V2 is a complete, runnable benchmark rather than a weight-compression demonstration. It trains context models from scratch whose state update and readout remain in tensor-network form, and it separates three questions that V1 had conflated:

1. **Representation:** can a bounded-width tensor network solve a predictive task when the dependency geometry is known?
2. **Geometry learning:** can a cheap causal router discover that geometry from tokens alone?
3. **Disentangling:** does MERA-like cross-boundary mixing reduce the bond dimension required for the same predictive task?

The main reference CPU campaign completed **32 trained runs and 87 measured evaluation rows**, with no recorded run failures. A three-seed paired low-rank replication and a paired original-initialization control add **12 runs and 24 rows**, bringing the scientific campaign to **44 research runs**. A separate 13-run smoke campaign is included for execution validation. Raw evidence, checkpoints, exact split seeds, run histories, plots, and machine-readable summaries are included under `results/`.

## Principal result

The V2 result is differentiated rather than simply favourable or unfavourable:

- **The branch-aligned representation works when the geometry is supplied.** Oracle-routed TTN reached 100% in-distribution accuracy on interleaved state threads, permuted hierarchy, and predictive-detail retrieval. On the first two tasks it substantially exceeded the positional TNs and conventional opponents in the one-seed reference tournament.
- **The present router does not discover enough of that geometry.** Permutation-aligned route recovery was only 27.3–29.5%, and learned-route prediction remained near chance or a weak baseline.
- **The apparent low-rank MERA advantage did not survive causal controls.** The original reference rows showed 98.05% for MERA at \(\chi=4\) and 53.13% for two TTN variants, but those rows used different model seeds and test splits. Across three paired seeds, all variants remained near chance; under the original successful initialization, all three reached 97.36–98.54%. The 44.92-point gap was therefore an optimization-seed artifact, not evidence that a disentangler lowered the required rank.
- **The models do not yet maintain a stable long-context predictive state.** Oracle-routed models trained at length 32 lost much of their accuracy when the number of genuine updates increased at lengths 64 and 128.
- **The adaptive-rank mechanism did not prune.** Its participation-ratio rank remained 15.996 out of 16.
- **The current unfused implementation is not a matched-quality runtime win.** It is slower at short context, though its hierarchical contraction and memory scaling become more favourable than quadratic attention at longer nominal lengths.

The original tensor-network direction remains viable as a research programme, but V2 places its bottleneck much more sharply: **learned predictive geometry, optimization stability, and streaming multiscale state maintenance** matter more than adding MERA machinery to the present encoder.

## What V2 contains

### Four controlled causal languages

Each sample is a context whose omitted next token is the target.

| Task | Controlled structural question | Oracle geometry |
|---|---|---|
| `interleaved_threads` | Can a model retain the last value of every interleaved entity and answer a random query? | One branch per entity thread |
| `permuted_hierarchy` | Can a model restore a balanced dependency tree when leaves are serialized in random order? | Branch identity followed by fixed branch-tree contractions |
| `predictive_detail` | Can a model retain a future-relevant signal while removing a local nuisance variable? | One branch per entity; shifted `MIXA/MIXB` pairs |
| `combined_language` | Can routing, local detail extraction, state update, and global hierarchy all be composed? | Entity branches plus global tree |

The detail task serializes an update as

\[
\mathrm{ID}_i,\quad \mathrm{MIXA}_{a},\quad \mathrm{MIXB}_{b},
\qquad s=(a-b)\bmod K.
\]

For the binary reference task this is XOR. Because events have odd length, `MIXA/MIXB` pairs alternate between lying within and across first-level positional TTN cuts. The shifted MERA layer therefore has direct access to the cross-cut subset before those values are separately coarse-grained. This is a deliberately diagnostic construction, not a claim that natural language presents identical partitions.

### Tensor-network models

- **MPS:** a token-indexed transfer matrix contracted through the sequence.
- **Fixed TTN:** adjacent positional leaves reduced by low-rank three-index contractions.
- **Fixed MERA:** the same TTN with shifted two-site maps before each coarse-graining level.
- **Routed TTN:** token states assigned to bounded entity branches; every branch has its own temporal TTN, followed by a global branch tree.
- **Routed MERA:** routed TTN plus shifted disentanglers in the temporal and global trees.
- **Adaptive-rank variants:** differentiable channel gates with a rank penalty and effective-rank diagnostics.

The local merge is a CP-factorized tensor contraction,

\[
y_o=\sum_r O_{or}
\left(\sum_i L_{ri}x_i\right)
\left(\sum_j R_{rj}z_j\right),
\]

with a gated residual path for optimization stability. The tensor models never construct an attention matrix or reconstruct a dense per-token context state before readout.

### Oracle and learned routing

The oracle condition receives branch metadata generated with the task. It is an upper bound on the tensor ansatz, not a deployable language model.

The learned router receives tokens and embeddings only. It is restricted to \(O(BLd)\) local work: vocabulary-symbol route logits, a learned choice among the current token and three causal lags, and a small contextual correction. There is no all-pairs attention or unconstrained graph search. Entropy and load-balancing penalties are included.

### Conventional opponents

- one-layer GRU with 32-dimensional state;
- two-layer, four-head causal Transformer with 32-dimensional model state;
- fixed positional TTN/MERA and MPS baselines.

The reference comparison is diagnostic rather than perfectly parameter-matched, so all parameter counts are reported.

## Experimental accounting

### Main reference campaign

- PyTorch 2.10 CPU build;
- 32 trained configurations and 87 evaluation rows;
- batch 64 for measured inference;
- one deterministic training seed per configuration;
- train length 32 for the first three task families and 64 for the combined stress test;
- evaluation at lengths 32, 64, and 128 where applicable;
- all routing, tensor operations, padding work, and readout included in wall-clock measurements;
- exact training, validation, and test split seeds stored in every row and checkpoint;
- no custom CUDA, C++, Triton, or fused contraction kernels.

Because the main tournament uses one model seed per configuration, its modest differences are architectural diagnostics, not confidence intervals.

### Supplementary low-rank controls

The initial low-rank comparison was discovered to be unpaired. V2 therefore includes two corrective campaigns:

1. **Three paired seeds:** within each seed, TTN \(\chi=4,R=4\), TTN \(\chi=4,R=9\), and MERA \(\chi=4,R=4\) use the same train, validation, and test datasets and the same PyTorch initialization seed.
2. **Original successful initialization:** all three variants use model seed `20290919`, the original MERA training and validation splits, and one shared test split.

These controls are the basis for the corrected disentangler conclusion below.

## Results

### 1. Correct geometry is the dominant representational variable

In-distribution results at the training context length:

| Task | Model | Accuracy | Loss | Parameters |
|---|---:|---:|---:|---:|
| Interleaved threads | Routed TTN, oracle | **1.0000** | 0.00055 | 16,741 |
| Interleaved threads | Routed MERA, oracle | 0.9941 | 0.02663 | 26,981 |
| Interleaved threads | Transformer | 0.3926 | 1.26927 | 23,013 |
| Interleaved threads | MPS | 0.3945 | 1.28630 | 3,972 |
| Interleaved threads | Fixed TTN | 0.3477 | 1.34561 | 11,717 |
| Permuted hierarchy | Routed TTN, oracle | **1.0000** | 0.00008 | 16,675 |
| Permuted hierarchy | Fixed TTN | 0.8613 | 0.35787 | 11,651 |
| Permuted hierarchy | Transformer | 0.6602 | 0.59009 | 22,883 |
| Predictive detail | Routed TTN, oracle | **1.0000** | 0.00065 | 16,707 |
| Predictive detail | Routed MERA, oracle | 0.9961 | 0.02476 | 26,947 |
| Predictive detail | Fixed MERA | 0.6016 | 0.66068 | 18,851 |
| Predictive detail | Fixed TTN | 0.5469 | 0.68404 | 11,683 |

The interleaved result is the clearest confirmation of the geometric premise. A positional MPS or TTN must mix all live entity states through shared positional separators. The oracle-routed model gives each live thread a modest branch and solves the task almost exactly.

The hierarchy result shows that the gain is not merely the availability of recurrent memory. Supplying the dependency-aligned branch tree is worth 13.9 percentage points over the fixed TTN and roughly 34 points over the Transformer in this controlled one-seed setting.

This establishes an **ansatz upper bound**, not a learned-language result: the useful geometry is supplied rather than inferred.

### 2. The low-rank MERA result was an optimization-seed artifact

At the ordinary \(\chi=16\) budget, MERA did not improve routed TTN:

- interleaved: 99.41% MERA versus 100% TTN;
- predictive detail: 99.61% MERA versus 100% TTN;
- more parameters and lower measured throughput in both cases.

The original low-rank reference rows appeared much more dramatic:

| Unpaired reference row | Model seed | Test seed | Accuracy | Parameters |
|---|---:|---:|---:|---:|
| Routed TTN, \(\chi=4,R=4\) | 20290819 | 21290812 | 0.5313 | 1,203 |
| Routed TTN, \(\chi=4,R=9\) | 20291019 | 21291012 | 0.5313 | 1,803 |
| Routed MERA, \(\chi=4,R=4\) | 20290919 | 21290912 | **0.9805** | 1,843 |

The train and validation splits were shared, but the model initialization seeds and test splits were not. The table therefore could not establish that the disentangler caused the difference.

The three-seed paired replication gives the opposite conclusion:

| Paired variant | Accuracy mean | Sample standard deviation | Parameters |
|---|---:|---:|---:|
| TTN, \(\chi=4,R=4\) | 0.4993 | 0.0214 | 1,203 |
| TTN, \(\chi=4,R=9\) | 0.5143 | 0.0176 | 1,803 |
| MERA, \(\chi=4,R=4\) | 0.5039 | 0.0296 | 1,843 |

All three variants remained near the 50% binary chance level. MERA showed no replicated advantage.

The paired original-initialization control is equally decisive:

| Shared model/data seed condition | Accuracy | Loss |
|---|---:|---:|
| TTN, \(\chi=4,R=4\) | 0.9736 | 0.09520 |
| TTN, \(\chi=4,R=9\) | 0.9814 | 0.11892 |
| MERA, \(\chi=4,R=4\) | 0.9854 | 0.06317 |

With the successful seed, all architectures solve the task. The task therefore has a strong optimization-basin effect, and the original 44.92-point MERA gap cannot be interpreted as reduced required rank. A future disentangler claim must use paired datasets, paired model seeds, multiple seeds, and a predeclared aggregate comparison.

The raw unpaired rows remain in the bundle rather than being deleted. Their corrected interpretation and both controls are preserved in `results/reference_cpu/tables/low_rank_paired_controls.csv` and the three low-rank plots.

### 3. Learned topology is the present failure point

Permutation-aligned route accuracies at length 32 were:

| Task | Learned routed TTN route accuracy | Prediction accuracy |
|---|---:|---:|
| Interleaved threads | 0.2733 | 0.3320 |
| Permuted hierarchy | 0.2908 | 0.6465 |
| Predictive detail | 0.2954 | 0.5039 |

Random route accuracy with eight branches is 0.125. The router therefore extracts some entity structure, but not enough to recover the oracle advantage. It uses all or nearly all branches, so the failure is not simple branch collapse; assignments are semantically mixed or unstable.

This separates two hypotheses cleanly:

- the oracle tensor representation can exploit the supplied structure;
- the present fully latent routing objective does not find that structure reliably.

Increasing bond dimension or adding more disentanglers would attack the wrong bottleneck.

### 4. Context extrapolation remains poor

The oracle-routed interleaved TTN fell from 100% at length 32 to 56.64% at 64 and 26.17% at 128. The detail TTN fell from 100% to 56.64% and 50.59%. Routed MERA was somewhat better on long predictive-detail contexts—66.41% at 64 and 60.94% at 128—but still far from a solved streaming state, and this one-seed difference is not a replicated MERA result.

The hierarchy model remains at 100% for nominal lengths 32–128, but that generator contains a fixed 17 valid tokens and only adds padding. This verifies mask/padding invariance; it is **not** evidence of long-context generalization.

The failure indicates that the present complete-tree encoder learned a depth- and distribution-specific reduction. A successor needs an explicitly recurrent binary-counter forest, scale-shared operators, and mixed-length training.

### 5. Adaptive rank did not adapt

The adaptive routed TTN reached 100% at length 32, but its effective rank was **15.996 out of 16** despite the rank penalty. The current sigmoid participation-ratio gate therefore provides instrumentation rather than compression. It also fell to 70.70% at length 64.

A credible successor must use a stronger hard-concrete or loss-environment spectral-tail mechanism and export genuinely smaller contractions. Until then, adaptive-rank runtime claims would be fictitious.

### 6. The combined task is unsolved

All models scored approximately chance on `combined_language`:

| Model | Accuracy at length 64 |
|---|---:|
| Fixed TTN | 0.5430 |
| Routed TTN, oracle | 0.5293 |
| Transformer | 0.5078 |
| GRU | 0.4941 |
| Routed MERA, oracle | 0.4746 |

This is a useful negative result. Solving routing, local detail extraction, branch state update, and global composition separately does not imply that the present implementation composes them reliably. The next architecture needs intermediate predictive losses or a staged curriculum at branch and global levels.

### 7. Runtime is only a scaling hint

On interleaved length 32, the unfused oracle routed TTN achieved about 326,800 tokens/s, versus 527,400 for the Transformer and 1.19 million for the GRU. It is slower at short context.

At nominal length 128, it measured about 474,800 tokens/s versus 208,500 for the Transformer, a 2.28× throughput ratio. This is not an end-to-end victory because the routed model's long-length accuracy had collapsed and the fixed TTN and GRU remained faster. It only shows that the contraction pattern can scale more gently than quadratic attention in this implementation.

The code launches many modest PyTorch operations. No hardware-advantage claim is warranted until routing and long-context quality work and the contractions are fused.

### 8. Isolated CPU memory shows the expected attention trend

A separate batch-64 campaign launched every checkpoint/length pair in a fresh process and sampled Linux RSS during warmed inference. It measures the complete Python/PyTorch process, model, batch, and forward workspaces rather than only an analytical tensor count.

| Model | Peak RSS, L=32 | Peak RSS, L=128 | Incremental forward peak, L=128 |
|---|---:|---:|---:|
| Routed TTN, oracle | 316.71 MiB | **334.35 MiB** | **10.35 MiB** |
| Routed MERA, oracle | 319.42 MiB | 338.51 MiB | 18.87 MiB |
| Transformer | 318.45 MiB | 379.39 MiB | 30.51 MiB |
| GRU | 313.16 MiB | 319.39 MiB | 3.95 MiB |

At length 32, interpreter and framework overhead dominate and routed TTN and Transformer process peaks are nearly identical. At length 128, routed TTN used about **45.04 MiB less process peak RSS**, while its warmed-forward increment was about **2.95 times smaller** than the Transformer's. This is consistent with hierarchical linear-memory contraction versus quadratic attention workspaces.

It is not a matched-quality production memory victory: both models were poor at length 128, the GRU used less memory, and RSS depends on allocator and platform behaviour. Raw measurements are in `results/reference_cpu/memory_benchmark.csv`.

## Decision gates

| Gate | V2 outcome | Consequence |
|---|---|---|
| Oracle topology can exploit low-width structure | **Passed as an ansatz upper bound** | Continue investigating TN-native predictive state |
| Learned router approaches oracle | **Failed** | Make geometry learning the primary V3 problem |
| MERA beats TTN at ordinary rank | **Not supported** | Do not make MERA mandatory |
| Disentangler lowers required rank on the controlled cut | **Failed robust replication** | Treat disentanglers as optional ablations only; require paired multi-seed evidence |
| Rank adapts under training | **Failed** | Replace soft channel penalties with genuine truncation machinery |
| Length extrapolation is stable | **Failed** | Use scale sharing and streaming multiscale updates |
| Combined task is solved | **Failed** | Add staged and multiscale auxiliary objectives |
| Runtime is competitive at matched quality | **Failed** | Defer kernel engineering until the architecture passes |
| Isolated memory grows more gently than attention | **Passed only as a scaling hint** | Retest after long-context quality is fixed |

## Recommended V3

V3 should not be a larger V2 or a MERA-first exercise. It should make four targeted changes:

1. **Streaming binary-counter tensor forest.** Update only \(O(\log L)\) occupied scale slots per token, share operators by scale class, and train over a distribution of lengths.
2. **Router curriculum with an explicit oracle gap.** Learn event consistency and entity persistence through weak route supervision or self-supervised objectives, then progressively remove route labels. Always report oracle, weakly supervised, and fully latent routing.
3. **Predictive spectral truncation.** Replace sigmoid channel gates with loss-environment singular spectra, hard rank decisions, and an exported compact tensor state.
4. **Hierarchical auxiliary prediction.** Branch states predict pending local state/query targets; coarse states predict progressively longer horizons. This should make the combined task trainable and provide a local future-sufficiency signal.

MERA-like disentanglers should remain a controlled optional switch, not part of the default architecture. They should be promoted only after a paired, multi-seed experiment shows reduced rank, branch count, or matched-error compute relative to a parameter- and optimization-controlled TTN.

The V3 success criterion should be stringent: learned routing must recover most of the oracle advantage; accuracy must remain stable as the number of real updates grows; a compact export must actually reduce state and computation; and the complete model must beat recurrent and Transformer opponents at matched prediction error, measured memory, and wall-clock cost.

## Reproduction

From the project root:

```bash
python -m pip install -e .
python -m pytest -q
python -m tnlm_v2.benchmark \
  --config configs/reference_cpu.yaml \
  --output results/reference_cpu_reproduction
python scripts/generate_reference_artifacts.py \
  --results results/reference_cpu_reproduction
```

Replay a main checkpoint exactly:

```bash
PYTHONPATH=src python scripts/replay_checkpoint.py \
  --results results/reference_cpu \
  --run-key detail_low_rank__routed_mera_oracle__seed20290919 \
  --length 32
```

Reproduce the paired three-seed control:

```bash
PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_seed_sweep_reproduction \
  --seeds 4101 4102 4103
```

Reproduce the original-initialization control:

```bash
PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_reference_seed_control_reproduction \
  --seeds 20290919 \
  --train-data-seed 20290823 \
  --validation-data-seed 20290835 \
  --test-data-seed-32 21290912 \
  --test-data-seed-64 22290912
```

The main runner and supplementary control runner write completed runs immediately and can resume without repeating finished configurations.
