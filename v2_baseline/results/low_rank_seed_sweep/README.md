# Three-seed paired low-rank control

This directory tests whether the apparent low-rank MERA advantage in the main reference campaign survives a paired design.

## Design

For each seed `4101`, `4102`, and `4103`, all three variants use:

- the same training data seed;
- the same validation data seed;
- the same length-32 and length-64 test data seeds;
- the same PyTorch model-initialization seed;
- the same optimizer and training budget.

The variants are:

- routed TTN, \(\chi=4,R=4\), 1,203 parameters;
- routed TTN, \(\chi=4,R=9\), 1,803 parameters;
- routed MERA, \(\chi=4,R=4\), 1,843 parameters.

## Result at length 32

| Variant | Mean accuracy | Sample standard deviation |
|---|---:|---:|
| TTN, \(\chi=4,R=4\) | 0.4993 | 0.0214 |
| TTN, \(\chi=4,R=9\) | 0.5143 | 0.0176 |
| MERA, \(\chi=4,R=4\) | 0.5039 | 0.0296 |

All variants remained near the 50% binary chance level. The main campaign's apparent 44.92-point MERA advantage did **not** replicate.

`metrics.csv` contains every length-32 and length-64 row; `checkpoints/` and `runs/` contain the complete trained artifacts. `manifest.json` records the paired design.

`replay_verification.json` confirms exact length-32 checkpoint replay for every included run.

Reproduce:

```bash
PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_seed_sweep_reproduction \
  --seeds 4101 4102 4103
```
