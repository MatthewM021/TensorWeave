# Completed V2 main reference campaign

This directory contains the main evidence set discussed in `../../reports/V2_REPORT.md`.

## Completion state

- 32 trained model configurations
- 87 evaluation rows
- 0 recorded failures
- CPU-only PyTorch 2.10.0 reference execution
- one deterministic model seed per configuration

`metrics.csv` is the canonical flat result table and records the exact training, validation, and test data seeds. `runs/` contains each model's full training history and evaluation payload. `checkpoints/` contains trained state dictionaries plus reproduction metadata. `executed_config.yaml` is the exact campaign definition used for the included runs.

## Derived files

- `summary.json`: machine-readable headline values, including the supplementary-control conclusion
- `replay_verification.json`: exact replay checks for three main low-rank checkpoints
- `memory_benchmark.csv`: isolated batch-64 process RSS measurements at lengths 32, 64, and 128
- `tables/in_distribution.csv`: all training-length evaluations
- `tables/headline.csv`: compact principal comparisons with the misleading unpaired low-rank rows deliberately excluded
- `tables/low_rank_paired_controls.csv`: combined paired-seed evidence from the two sibling control campaigns
- `plots/low_rank_unpaired_reference.png`: the original raw comparison, explicitly labelled non-causal
- `plots/low_rank_paired_seed_control.png` and `low_rank_paired_seed_mean.png`: corrected causal controls
- remaining plots: regenerated directly from `metrics.csv` and the isolated memory table

Regenerate the derived files without retraining:

```bash
python scripts/generate_reference_artifacts.py \
  --results results/reference_cpu
```

The generator automatically incorporates `../low_rank_seed_sweep/` and `../low_rank_reference_seed_control/` when they are present.

## Interpretation boundary

Oracle routes expose the known synthetic dependency geometry and are an ansatz upper bound, not a deployable language-model result. The main tournament is diagnostic, uses unfused PyTorch CPU contractions, and does not establish a production runtime advantage.

The original low-rank reference rows used different model seeds and test splits. They showed MERA at 98.05% and two TTN variants at 53.13%, but the paired controls in the sibling result directories reject a causal disentangler interpretation: three shared seeds put all variants near chance, while the original successful initialization makes all variants reach 97.36–98.54%. The raw rows remain here for auditability rather than being removed.

Replay a main checkpoint exactly:

```bash
PYTHONPATH=src python scripts/replay_checkpoint.py \
  --results results/reference_cpu \
  --run-key detail_low_rank__routed_mera_oracle__seed20290919 \
  --length 32
```
