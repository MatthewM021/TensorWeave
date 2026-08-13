# Tensor-Network-Native Language Model — Work Handoff

**Handoff date:** 13 August 2026  
**Repository state supplied here:** a self-contained monorepo containing the complete verified V2 baseline under `v2_baseline/`, the original V2 ZIP, and a recovery specification for the interrupted V3 build.  
**First instruction to the next agent:** read this file completely before modifying code.

## 1. Mission

This project investigates whether tensor networks can be the **fundamental state and computational object** of a machine-learning system, especially an autoregressive language model, rather than merely a post-hoc compression format for dense Transformer weights.

The central hypothesis is:

> A causal model may be able to maintain a compact tensor-network predictive state whose size follows the number and structure of independent future-relevant dependencies, rather than the raw number of previous tokens.

The intended advantage is not simply fewer parameters. It is a different computational regime in which the context state, state update, routing, coarse-graining, readout, backward environments, and rank adaptation remain tensor-network-native. Large dense attention matrices and repeated dense context-to-context multiplication should not be reintroduced as hidden fallbacks.

Language has sequential causal order but no single fixed dependency geometry. The model therefore needs a learned, constrained dependency topology. The graph is not itself the result: it must remain the wiring of an efficiently contractible tensor state. Arbitrary sparse-graph discovery without a tensor representation or bounded contraction schedule is explicitly not useful.

## 2. Non-negotiable research constraints

1. **Do not turn this into ordinary weight compression.** Do not train a Transformer, factor its matrices, and call the result a TN-native model.
2. **The predictive state is the TN.** State update and readout must remain in tensor form. No reconstruct-dense/apply-Transformer/recompress path.
3. **Causality must be testable.** Prefix outputs and routing decisions must be independent of future tokens, future labels, padding after the prefix, and evaluation metadata.
4. **Topology learning must be cheap.** The router must not hide all-pairs attention or another operation with the cost the TN is meant to avoid.
5. **MERA is optional.** V2 found no replicated low-rank advantage from the MERA-like disentangler. Promote it only after paired, multi-seed evidence shows lower rank, fewer branches, lower matched-error cost, or better extrapolation than a controlled TTN.
6. **Keep strong opponents permanently.** At minimum retain an optimized recurrent baseline and a causal Transformer with a proper autoregressive cache when measuring streaming inference.
7. **Matched quality precedes speed claims.** Analytical FLOPs, nominal ranks, or memory at failed accuracy do not establish an advantage.
8. **Measure complete cost.** Include routing, contraction planning, canonicalization/truncation, compilation, export, and state maintenance.
9. **Use paired seeds and paired data.** The V2 false MERA result came from an unpaired initialization/data comparison. Never repeat that mistake.
10. **Commit and checkpoint frequently.** Long campaigns must be resumable and should write each completed run immediately. Do not leave hours of uncommitted or unrecorded work inside one agent turn.

## 3. What is present in this repository

The complete V2 deliverable is physically present under `v2_baseline/`. It is also embedded verbatim as `source_archives/mera_language_direction_demo_v2.zip`. No external download or prior repository is required.

Key locations:

```text
v2_baseline/src/tnlm_v2/             complete V2 implementation
v2_baseline/configs/                 V2 smoke and reference campaigns
v2_baseline/results/reference_cpu/   completed V2 evidence and checkpoints
v2_baseline/results/low_rank_seed_sweep/ paired three-seed low-rank controls
v2_baseline/results/low_rank_reference_seed_control/
v2_baseline/reports/V2_REPORT.md     complete interpretation and decision gates
v2_baseline/reports/V2_ARCHITECTURE.md exact V2 architecture
v2_baseline/scripts/validate_bundle.py 161-check V2 evidence validator
v2_baseline/BUILD_VERIFICATION.json  V2 build/test/replay verification
v2_baseline/VALIDATION.json          V2 evidence-integrity record
WORK_START_PROMPT.md                 prompt to give ChatGPT Work
v3_recovery/V3_RECOVERY_STATUS.yaml  machine-readable interrupted-build status
docs/V3_RECOVERY_SPEC.md             architecture and implementation recovery spec
docs/V3_VALIDATION_PROTOCOL.md       required tests, audits, and campaign closure
```

The supplied V2 bundle was previously verified with:

- 22 passing tests;
- 13/13 clean smoke runs;
- 161 evidence-integrity checks;
- 15 exact checkpoint replays;
- 192 original file checksums;
- successful editable installation and command-line entry point;
- strict JSON validation.

Before beginning V3, reproduce the local baseline:

```bash
python VERIFY_REPOSITORY.py
cd v2_baseline
python -m pip install -e '.[test]'
pytest -q
PYTHONPATH=src python scripts/validate_bundle.py --output /tmp/V2_VALIDATION.json
cd ..
```

The historical verification environment was Linux, Python 3.13.5, PyTorch 2.10.0+cpu, with CUDA unavailable. Exact reproduction on a different environment is not required, but any changed numerical results must be recorded rather than silently replacing the reference evidence.

## 4. Verified V2 findings

### 4.1 The representation hypothesis survived under oracle geometry

When supplied the generated dependency geometry, routed TTN reached 100% training-length accuracy on:

- interleaved state threads;
- permuted hierarchy;
- predictive-detail retrieval.

This is an ansatz upper bound, not a deployable language model. It shows that a branch-aligned TN predictive state can exploit the controlled low-width structure.

### 4.2 Learned routing was the dominant failure

At length 32, permutation-aligned route recovery was only about 27–30%. The learned router extracted some structure but did not approach the oracle result. Increasing rank or adding disentanglers would attack the wrong bottleneck unless routing is first improved.

### 4.3 The apparent low-rank MERA win was false

The original one-seed comparison appeared to show roughly 98.05% for MERA versus 53.13% for two TTN controls. It used different model seeds and test splits.

Across three paired seeds at the controlled low-rank setting, mean accuracies were:

- TTN, `chi=4`, `rank=4`: 49.93%;
- TTN, `chi=4`, `rank=9`: 51.43%;
- MERA, `chi=4`, `rank=4`: 50.39%.

Under the original successful initialization, all three reached 97.36–98.54%. The apparent MERA advantage was an optimization-basin artifact. Treat all future disentangler claims as invalid unless they survive paired model seeds, paired data splits, multiple seeds, and a predeclared aggregate statistic.

### 4.4 V2 did not extrapolate as a streaming state

The oracle-routed interleaved TTN fell from 100% at length 32 to 56.64% at 64 and 26.17% at 128. The complete-tree encoder had learned a depth/distribution-specific reduction. This motivated a recurrent binary-counter forest, shared operators, and mixed-length training.

### 4.5 The soft adaptive rank did not prune

The reported effective rank remained 15.996 out of 16. V2's gates were instrumentation, not a computational saving. V3 must produce an exported model with genuinely smaller tensors and fewer operations.

### 4.6 The combined language remained unsolved

All models were near chance on the combined task. This motivated local/branch/coarse auxiliary predictive objectives and a staged curriculum.

### 4.7 Runtime and memory were only hints

The unfused V2 implementation was slower than the Transformer at short context. At nominal length 128, routed TTN throughput scaled more gently and used about 334.35 MiB process peak RSS versus 379.39 MiB for the Transformer, with warmed-forward incremental RSS of 10.35 versus 30.51 MiB. Both models had poor long-length quality, so this is not a matched-quality win.

## 5. Recovered status of the interrupted V3 build

The previous V3 attempt was terminated by the execution environment before final validation and packaging. The following facts were recorded before termination:

- 168 reference training runs were planned;
- 127 had completed;
- zero training failures had been recorded;
- implemented components reportedly included:
  - a causal streaming tensor forest;
  - branch-wise binary-counter state updates;
  - scale-shared operators, with scale/level information retained separately;
  - a causal persistent router;
  - a dynamic document-local binding task;
  - query-aware readout;
  - corrected causal TTN and MERA controls;
  - a cached-Transformer opponent;
  - predictive truncation and compact export machinery;
  - streaming/parallel comparison machinery and auxiliary diagnostics;
- the router supported oracle, curriculum/weakly supervised, and fully latent conditions;
- the intended router used token information, position, and branch prototypes rather than hidden all-pairs attention;
- a token could select a branch or remain on a global path;
- occupied scale slots were combined into summaries for prediction;
- the first two strong seeds of the autonomous curriculum router reportedly reached roughly 98–100% dynamic-binding accuracy through length 256;
- the fully latent router and strong GRU/Transformer controls were much lower in those partial results;
- no final scientific claim was issued.

### Critical compact-export bug

An audit found that **inactive scale-signal channels could influence retained gate outputs**. The dependency was patched and a regression test was reportedly added, but 16 affected pruned runs still required regeneration.

The required invariant is stronger than simply masking inactive output channels:

> After selecting retained channels for compact export, perturbing any discarded/inactive scale-signal input must not alter any retained output, routing logit, gate, state update, or prediction.

The compact model must also use physically smaller tensors and execute fewer operations; masking a dense model is not an export.

### Work still outstanding at termination

The interrupted run still needed:

1. the third strong dynamic-binding seed;
2. additional corrected causal TTN/MERA controls;
3. the strong-opponent tournament;
4. regeneration of all 16 bug-affected pruned runs;
5. a clean 22-run smoke campaign;
6. a 2,048-token scaling run;
7. an operational routing audit;
8. an evaluation-label-independence audit;
9. complete checkpoint replay;
10. final tests, checksums, report, and archive packaging.

### Important recovery limitation

The V3 scratch directory and partial checkpoints did **not** survive into the accessible filesystem used to create this handoff. They are therefore not included. Do not claim that the 127 run files or V3 source are present. The supplied code is the verified V2 base. Reconstruct V3 from the specifications and commit it durably to the new GitHub repository. If another copy of the interrupted V3 tree is later found, audit it before merging rather than treating it as authoritative.

## 6. Required V3 architecture

The default V3 model should be a **streaming, scale-shared, routed tensor forest**, not a larger complete-tree V2 model.

### 6.1 Streaming binary-counter forest

For each branch, maintain at most one occupied state per binary scale. A new event enters scale zero. Collisions merge recursively, like carrying a binary counter. This gives `O(log L)` active scale slots and `O(log L)` worst-case merge work for an insertion, with constant amortized merge count under the ordinary binary-counter schedule.

The implementation must provide both:

- a token-by-token streaming path; and
- a parallel/tree-reduction path used for training where appropriate.

They must agree numerically under the same routing and masks.

### 6.2 Associative transition/operator state where possible

The interrupted design was moving away from a slow feature-state forest toward composable transition operators. Preserve that direction where it gives an exact or controlled associative composition law. The goal is to permit efficient parallel reduction and exact streaming equivalence without secretly materializing a dense sequence state.

Do not force associativity where it destroys predictive quality. Any approximation must be measured against the sequential update.

### 6.3 Shared operations across lengths and scales

V2's level-specific tensors damaged length extrapolation. Share merge/update tensors across all scales or across a small fixed number of scale classes. A compact level/scale embedding may condition the operator, but the parameter count must not grow with maximum context length.

### 6.4 Constrained persistent routing

Routing must preserve document-local entity/event consistency without all-pairs search. The router may use:

- current token/event embedding;
- causal position or relative age;
- summaries/prototypes of active branches;
- bounded local/global state;
- a small fixed template bank.

It must not inspect future route labels or future tokens. It should be vectorized over branches.

Run three distinct conditions:

1. **Oracle:** generated true branch assignments; ansatz upper bound only.
2. **Curriculum:** route supervision or guidance starts strong and decays according to a declared schedule; inference is autonomous.
3. **Fully latent:** no route labels, using causal consistency, persistence, entropy/load balance, and predictive objectives.

Always report the oracle gap and do not blur curriculum routing with fully latent routing.

### 6.5 Dynamic document-local binding benchmark

Rebuild the V3 dynamic-binding task as a causal language with document-local keys/entities, changing bindings/state, interleaved updates, and later queries. It must prevent trivial global token-to-branch lookup: the same symbol should be able to bind differently in different documents or episodes.

Include diagnostics for:

- exact query accuracy;
- route recovery after optimal branch permutation;
- document-local consistency;
- unseen symbol/binding combinations;
- number of simultaneous live bindings;
- length extrapolation;
- adversarial collisions and expired bindings.

### 6.6 Hierarchical predictive objectives

Use branch-local and scale-appropriate auxiliary losses so that the state is trained to retain future-relevant information, not merely reconstruct inputs. Examples include pending query/value prediction at branch level and longer-horizon targets at coarser scales. Report all auxiliary loss weights and ablate them.

### 6.7 Genuine predictive truncation and compact export

Replace V2's soft participation-ratio gate with a method that produces discrete ranks and a smaller executable model. Candidate mechanisms may include loss-environment spectral tails, hard-concrete gates, structured group sparsity followed by exact slicing, or another controlled rank-selection method.

The export must satisfy:

- retained-output parity within declared tolerance;
- no dependence on discarded inputs/channels;
- smaller serialized state;
- fewer parameters;
- fewer measured tensor operations or lower wall-clock cost;
- no dense mask retained in the exported kernels.

### 6.8 MERA-like disentanglers remain an ablation

The default should be the routed TTN/forest. Add cross-boundary disentanglers only as a controlled switch initialized to identity. Compare against a TTN with matched seed, data, parameter budget, optimizer, and training time. Promote the switch only if the multi-seed result reduces the required rank/branch count or improves matched-error cost.

## 7. Exact continuation order

Follow this sequence rather than launching another monolithic campaign.

### Milestone 0 — durable import

1. Commit this complete self-contained handoff unchanged, including `v2_baseline/` and `source_archives/`.
2. Run the V2 tests and validator.
3. Record environment information.
4. Create a `v3-recovery` branch.

Suggested first commit:

```text
import verified V2 baseline and V3 recovery handoff
```

### Milestone 1 — V3 core with tests

Create a new top-level `v3/` project containing its own `src/tnlm_v3/` package. Do not modify or overwrite `v2_baseline/`. Implement the streaming forest, masks, shared operator/merge, causal readout, and exact state serialization.

Before training, pass tests for:

- streaming versus parallel/tree-reduction parity;
- prefix causality;
- padding invariance;
- nullable/global routing;
- branch permutation equivariance or correct alignment handling;
- deterministic resume and state reload;
- no growth of parameter count with configured maximum length.

Commit this milestone before starting routing work.

### Milestone 2 — dynamic binding and router curriculum

Implement the document-local dynamic-binding generator, oracle routes, causal router, curriculum schedule, and fully latent mode. Add route accuracy and document-consistency diagnostics. Run a tiny deterministic overfit test and a clean smoke matrix.

Commit source, config, tests, and smoke results.

### Milestone 3 — truncation and export

Implement real rank selection and compact physical slicing. Add the inactive-channel regression described above. Verify the exported model independently in a fresh process. Report serialized bytes, parameter counts, state scalars, operations, and wall-clock inference.

Do not launch pruned reference runs until this audit passes.

### Milestone 4 — paired reference campaign

Use a resumable manifest. Write each run atomically with exact code commit, config hash, model seed, train/validation/test data seeds, environment, checkpoint, and metrics.

At minimum include:

- oracle, curriculum, and fully latent routed forest;
- fixed/corrected causal TTN control;
- optional causal MERA ablation;
- strong GRU;
- cached causal Transformer;
- unpruned and compact-export variants;
- at least three paired strong seeds;
- lengths through 256 for quality and 2,048 for scaling/audit;
- paired datasets and initializations for all architectural ablations.

Regenerate the 16 compact/pruned conditions under the corrected export logic. Do not reuse metrics from the buggy path.

### Milestone 5 — closure audits

Complete the audit list in `docs/V3_VALIDATION_PROTOCOL.md`, including the 22-run clean smoke campaign, operational routing, evaluation-label independence, checkpoint replay, and fresh-process compact-export checks.

### Milestone 6 — final deliverable

Produce:

- `reports/V3_REPORT.md` with negative results retained;
- `reports/V3_ARCHITECTURE.md`;
- raw run records and exact configs;
- compact tables and plots;
- replay and integrity manifests;
- final checksums;
- a complete archive or tagged GitHub release.

The report must separate verified claims from indications and failed replications.

## 8. V3 success gates

Do not call V3 complete merely because training runs finish.

1. **Routing:** curriculum routing should recover most of the oracle predictive advantage across at least three paired seeds. Fully latent routing must be reported separately.
2. **Streaming generalization:** quality must remain stable when the number of real updates grows beyond the training length; padding-only tests do not count.
3. **Compactness:** exported ranks must actually fall, with smaller tensors and measured computation.
4. **Causality:** all prefix and evaluation-label independence tests must pass.
5. **Fair baselines:** compare to a cached Transformer and recurrent opponent at matched quality, not only matched nominal dimensions.
6. **MERA:** no positive claim without paired multi-seed evidence.
7. **Runtime:** no speed claim unless routing, export, and all state-maintenance costs are included.
8. **Reproducibility:** every headline number must map to a checkpoint, exact split seeds, code commit, and replay result.

## 9. Git and campaign discipline for ChatGPT Work

The previous failure was operational: a long scratch session ended before durable packaging. Avoid recurrence.

- Commit after every completed milestone and after any bug that invalidates results.
- Store the commit hash in every run record.
- Use atomic run files and a resumable manifest.
- Never delete contradictory or failed results; supersede them with explicit status.
- Keep `main` reproducible. Use a feature branch for experimental changes.
- Tag the verified V2 import before V3 modifications, for example `v2-handoff-baseline`.
- Do not wait until the end to write the report. Maintain an evidence ledger as results arrive.
- A run affected by a correctness bug must be marked invalid and regenerated; it must not remain in aggregate plots.

## 10. Immediate command sequence

```bash
python -m pip install -e '.[test]'
pytest -q
PYTHONPATH=src python scripts/validate_bundle.py --output /tmp/V2_VALIDATION.json
git status
git add .
git commit -m "import verified V2 baseline and V3 recovery handoff"
git tag v2-handoff-baseline
git switch -c v3-recovery
```

Then create the V3 package and begin Milestone 1. Do not spend the first phase re-litigating whether tensor networks might be useful; that conceptual decision has already been narrowed into the empirical gates above.
