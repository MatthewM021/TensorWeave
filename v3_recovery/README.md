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
exception claim.  `PHASE2_ALGEBRA_POWER_CONTROL_V2.json` is the exact
source-refrozen prerequisite for V4.  `PHASE2_OUTER_ROTATION_V3_IMPLEMENTATION.sha256` freezes the
complete 33-file implementation/runtime closure for the subsequent two-phase
40-environment execution.  The manifest and prerequisite record must validate
before any new environment is generated or fit.

The completed execution is preserved under `phase2_outer_rotation_v3/`.
`terminal-preopen.json` passed all 1,520 candidate and 40 final-fit gates and
authorized one atomic batch open.  `open-campaign.json` is formally failed at
10 / 40 environments because the frozen runner required 96 actual queries at
every cell, while the valid balanced suite contains 88 queries at values
1--3.  All 3,600 realized actual queries and all 72,000 rotated-control queries
were correct.  `SUMMARY.json` records the distinction between the immutable
formal failure and the exact behavioral result; `EVIDENCE.sha256` binds all 40
preopen shards plus the terminal and opened records.

The forward-only corrective replication is preserved under
`phase2_outer_rotation_v4/`.  Its protocol was frozen only after the V3 result
was opened and changes only nonfocal balance padding; it does not reclassify
V3.  `PHASE2_OUTER_ROTATION_V4_IMPLEMENTATION.sha256` freezes the 34-file
implementation closure.  All 1,520 candidate gates and all 40 final-fit gates
passed before one atomic batch open.  V4 formally passed 40 / 40 environments:
3,840 / 3,840 actual queries, 960 / 960 focal queries, and 76,800 / 76,800
rotated-control queries were correct.  `SUMMARY.json` records the result and
its nonconfirmatory supplied-representation scope; `EVIDENCE.sha256` binds the
40 shards, terminal aggregate, and opened campaign.
Candidate-fit cleanliness remains a typed, hash-bound trusted-runner
certificate because candidate models are not serialized; the 40 final models'
direct-TRAIN fits are independently replayed.
