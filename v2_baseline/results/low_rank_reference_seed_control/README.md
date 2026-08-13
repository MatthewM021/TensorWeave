# Paired control using the original successful initialization

This directory asks whether the main low-rank MERA result was architectural or whether seed `20290919` placed the optimizer in a successful basin.

All three variants use:

- model seed `20290919`;
- training data seed `20290823`;
- validation data seed `20290835`;
- the same length-32 test seed `21290912` and length-64 test seed `22290912`;
- the same optimizer and training budget.

## Result at length 32

| Variant | Accuracy | Loss |
|---|---:|---:|
| TTN, \(\chi=4,R=4\) | 0.9736 | 0.09520 |
| TTN, \(\chi=4,R=9\) | 0.9814 | 0.11892 |
| MERA, \(\chi=4,R=4\) | 0.9854 | 0.06317 |

All three architectures solve the task under the original successful initialization. Together with the three-seed paired sweep, this shows that the raw main-campaign gap was an optimization-seed artifact rather than evidence that MERA lowered the required bond rank.

`replay_verification.json` confirms exact length-32 checkpoint replay for every included run.

Reproduce:

```bash
PYTHONPATH=src python scripts/run_low_rank_seed_sweep.py \
  --output results/low_rank_reference_seed_control_reproduction \
  --seeds 20290919 \
  --train-data-seed 20290823 \
  --validation-data-seed 20290835 \
  --test-data-seed-32 21290912 \
  --test-data-seed-64 22290912
```
