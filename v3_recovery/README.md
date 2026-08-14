# V3 Recovery Directory

No V3 source tree, checkpoints, or 127 partial run records were recovered from
the interrupted execution session. Those unavailable historical artifacts are
not represented as reproducible evidence.

The new reconstruction lives under the top-level `v3/` directory. This recovery
directory now contains machine-readable status and validation records produced
from that reconstruction. `CURRENT_STATUS.yaml` is the current index;
`V3_RECOVERY_STATUS.yaml` remains the immutable historical recovery record.

Milestone 3 evidence consists of the strict commit-bound audit record,
evaluation fixture, fresh-process replay record, and native non-executable
compact artifact. `MILESTONE3_EVIDENCE.sha256` binds those four files. The audit
verifies physical CP-rank export and replay correctness only; it does not claim
that rank 4 preserves scientific quality or improves wall time, RSS, or
persistent-state width.

Milestone 4 pilot evidence consists of `MILESTONE4_PILOT_SUMMARY.json` and the
deterministic `MILESTONE4_PILOT_BUNDLE.zip`, bound by
`MILESTONE4_PILOT_EVIDENCE.sha256`. The archive preserves the generation-32
manifest, all 32 immutable transition records, seven results and subprocess
envelopes, 18 canonical checkpoints, and one compact artifact. Mutable progress
hints, PID leases, and the manifest lock are intentionally excluded. The pilot
used one paired seed block and 12 optimizer steps per source; it verifies
execution and evidence plumbing only, not scientific quality, matched-quality
compression, runtime, RSS, or generalization.

Phase II algebra-learning pilot evidence consists of three successful passive
trace records, one deliberately under-optimized control record, and
`PHASE2_TRACE_ALGEBRA_PILOT_SUMMARY.json`, all bound by
`PHASE2_TRACE_ALGEBRA_PILOT_EVIDENCE.sha256`.  The three exploratory seed pairs
learned the exact 20 supported categorical transition entries inside a
supplied addressable-register family and passed every actual-cell behavioral
probe.  This is retrospective coefficient-recovery evidence, not prospective
confirmation, latent representation discovery, assumption-free algebra
discovery, or natural-language evidence.

The paired observed-exception/null power control is preserved as
`PHASE2_ALGEBRA_POWER_CONTROL_V1.json`.  It validates the selector's ability
to realize one directly observed transition-address exception and to prefer
the shared solution on the matched null condition; it makes no unseen-
exception claim.  `PHASE2_OUTER_ROTATION_V3_IMPLEMENTATION.sha256` freezes the
complete 33-file implementation/runtime closure for the subsequent two-phase
40-environment execution.  The manifest and prerequisite record must validate
before any new environment is generated or fit.
