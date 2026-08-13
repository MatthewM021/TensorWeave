# Prompt for ChatGPT Work

Continue the tensor-network-native language-model project in the GitHub repository currently open in Work.

This is a self-contained monorepo. The complete, verified V2 project is physically present under `v2_baseline/`; the original V2 ZIP is also present under `source_archives/`. Do not assume access to any files outside this repository.

First read, in order:

1. `README_FIRST.md`
2. `HANDOFF.md`
3. `v2_baseline/reports/V2_REPORT.md`
4. `v2_baseline/reports/V2_ARCHITECTURE.md`
5. `docs/V3_RECOVERY_SPEC.md`
6. `docs/V3_VALIDATION_PROTOCOL.md`
7. `v3_recovery/V3_RECOVERY_STATUS.yaml`

Before changing code, run:

```bash
python VERIFY_REPOSITORY.py
cd v2_baseline
python -m pip install -e '.[test]'
pytest -q
PYTHONPATH=src python scripts/validate_bundle.py --output /tmp/V2_VALIDATION.json
cd ..
```

Preserve `v2_baseline/` unchanged and replayable. Implement V3 in a new top-level `v3/` project with its own package, tests, configurations, results, reports, and resumable campaign machinery. Commit the imported baseline, retain the `v2-handoff-baseline` tag, and work on the `v3-recovery` branch.

The prior V3 scratch directory and its 127 partial run files were not recovered. Do not claim they exist. Reconstruct V3 from the handoff and specifications, while preserving the recorded partial findings only as unverified historical guidance.

Implement in this order:

1. streaming scale-shared binary-counter tensor forest, with exact streaming/parallel parity and causality tests;
2. causal document-local dynamic-binding task and oracle, curriculum, and fully latent routing;
3. genuine predictive truncation and physically compact export, including the inactive-scale-signal independence regression;
4. paired multi-seed campaign with corrected causal TTN/MERA controls, a strong GRU, and a genuinely cached causal Transformer;
5. clean smoke, prefix-causality, operational-routing, evaluation-label-independence, 2,048-token scaling, checkpoint replay, integrity, and packaging audits.

Do not turn this into Transformer weight compression, arbitrary sparse-graph discovery, or a dense model hidden behind TN storage. Keep MERA optional unless paired multi-seed evidence shows a real reduction in rank, branch count, matched-error compute, or extrapolation error. Include all routing, truncation, export, state-maintenance, and compilation costs in runtime comparisons.

Commit frequently. Every experiment must be resumable and atomically recorded with commit hash, configuration hash, model seed, data seeds, checkpoint, environment, and metrics. Preserve invalidated and negative results with explicit status. Continue until a complete, verified V3 release exists in the repository; do not merely return a plan or code dump.
