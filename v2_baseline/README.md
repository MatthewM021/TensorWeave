# MERA Language Direction Demo V2

A self-contained research benchmark for treating the **predictive context state** of a machine-learning model as a tensor network, rather than tensor-compressing dense Transformer weights after training.

V2 implements:

- four controlled next-token languages;
- MPS, fixed TTN, fixed MERA, routed TTN, and routed MERA models;
- oracle and fully latent learned routing;
- predictive shifted disentanglers;
- adaptive-rank diagnostics;
- GRU and causal Transformer opponents;
- context-length, quality, parameter, routing, capacity-proxy, measured CPU runtime, and isolated peak-RSS evaluations;
- resumable runs, checkpoints, exact split seeds, checkpoint replay, plots, tests, and a completed reference campaign;
- paired-seed controls for the critical low-rank MERA/TTN ablation.

## Headline findings

The main reference campaign contains **32 research runs and 87 evaluation rows**. Two supplementary causal controls add **12 runs and 24 rows**, for **44 research-campaign runs**. A separate 13-run smoke set is also included for execution validation.

- With the generated dependency geometry supplied as an oracle, routed TTN reached **100%** at the training length on interleaved threads, permuted hierarchy, and predictive detail. This confirms that the branch-aligned TN ansatz can exploit the controlled low-width structures.
- The initial one-seed low-rank table appeared to favour MERA by 44.92 percentage points, but that comparison used different initialization seeds and different test splits. The paired controls **do not replicate a disentangler advantage**: over three shared seeds, TTN \(\chi=4,R=4\), TTN \(\chi=4,R=9\), and MERA \(\chi=4,R=4\) averaged **49.93%**, **51.43%**, and **50.39%** respectively. Under the original successful initialization, all three reached **97.36–98.54%**. The raw gap was therefore an optimization-seed artifact, not evidence that MERA lowered the required rank.
- Learned route recovery remained only **27–30%** after permutation alignment, exposing geometry learning as the dominant present bottleneck.
- Adaptive rank stayed at **15.996/16** and therefore did not genuinely prune.
- Models trained at length 32 degraded sharply when the number of real updates increased at lengths 64 and 128; the combined stress task remained unsolved.
- The unfused implementation is not a matched-quality runtime win. Its scaling is nevertheless informative: at batch 64 and length 128, routed TTN used 334.35 MiB isolated process peak RSS versus 379.39 MiB for the Transformer, and 10.35 versus 30.51 MiB of warmed-forward incremental RSS. Both models had poor long-length accuracy, so this is a memory-scaling hint rather than a production victory.

See [`reports/V2_REPORT.md`](reports/V2_REPORT.md) for the complete results, controls, limitations, and V3 decision gates.

## Quick start

```bash
python -m pip install -e .
pytest -q
python -m tnlm_v2.benchmark \
  --config configs/smoke.yaml \
  --output results/my_smoke
```

Reproduce the main CPU campaign:

```bash
python -m tnlm_v2.benchmark \
  --config configs/reference_cpu.yaml \
  --output results/reference_cpu_reproduction
```

Regenerate the derived plots, compact tables, and `summary.json` without retraining:

```bash
python scripts/generate_reference_artifacts.py \
  --results results/reference_cpu
```

Reproduce the paired low-rank controls:

```bash
PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_seed_sweep_reproduction \
  --seeds 4101 4102 4103

PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_reference_seed_control_reproduction \
  --seeds 20290919 \
  --train-data-seed 20290823 \
  --validation-data-seed 20290835 \
  --test-data-seed-32 21290912 \
  --test-data-seed-64 22290912
```

Replay a stored checkpoint on its exact test split:

```bash
PYTHONPATH=src python scripts/replay_checkpoint.py \
  --results results/reference_cpu \
  --run-key detail_low_rank__routed_mera_oracle__seed20290919 \
  --length 32
```

The replay utility also accepts checkpoints in either supplementary low-rank directory.

Validate the included evidence and artifact structure without retraining:

```bash
PYTHONPATH=src python scripts/validate_bundle.py
```

Re-run the isolated CPU memory campaign:

```bash
PYTHONPATH=src python scripts/measure_checkpoint_memory.py \
  --results results/reference_cpu \
  --study interleaved_tournament \
  --models routed_ttn_oracle routed_mera_oracle fixed_ttn mps gru transformer \
  --lengths 32 64 128 \
  --output results/reference_cpu/memory_benchmark_reproduction.csv
```

The main runner is resumable. A particular study/model can be isolated:

```bash
python -m tnlm_v2.benchmark \
  --config configs/reference_cpu.yaml \
  --output results/reference_cpu_reproduction \
  --study detail_low_rank \
  --model routed_mera_oracle
```

## Project map

```text
src/tnlm_v2/data.py                    controlled causal-language generators
src/tnlm_v2/models/components.py       tensor merges, disentanglers, rank gates, router
src/tnlm_v2/models/tree_models.py      fixed and routed TTN/MERA models
src/tnlm_v2/models/mps.py              MPS baseline
src/tnlm_v2/models/baselines.py        GRU and causal Transformer
src/tnlm_v2/training.py                training, evaluation, route alignment, timing
src/tnlm_v2/benchmark.py               resumable benchmark runner
configs/                               smoke and reference campaigns
results/reference_cpu/                 main completed evidence and checkpoints
results/low_rank_seed_sweep/           three paired-seed low-rank controls
results/low_rank_reference_seed_control/ paired original-initialization control
scripts/generate_reference_artifacts.py plots, tables, and machine summary
scripts/run_low_rank_seed_sweep.py      paired low-rank replication runner
scripts/replay_checkpoint.py           deterministic checkpoint replay
scripts/verify_result_replays.py        multi-checkpoint replay verification
scripts/measure_checkpoint_memory.py   isolated CPU peak-RSS measurement
scripts/validate_bundle.py             internal evidence-integrity checks
VALIDATION.json                         completed 161-check evidence-integrity manifest
BUILD_VERIFICATION.json                 tests, install, smoke, replay, and JSON checks
reports/V2_REPORT.md                   interpretation and V3 decision gates
reports/V2_ARCHITECTURE.md             implemented architecture specification
tests/                                 correctness, gradient, and resume tests
```

## Interpretation boundary

Oracle routing is an ansatz upper bound; it is not a deployable learned-language model. The synthetic tasks intentionally expose particular dependency structures. Runtime numbers are measured on CPU with ordinary PyTorch operations and no fused tensor kernels. Most main-tournament comparisons use one seed per configuration, so modest gaps are not statistical claims. The low-rank MERA claim was explicitly subjected to paired-seed controls, and those controls reject its original causal interpretation.
