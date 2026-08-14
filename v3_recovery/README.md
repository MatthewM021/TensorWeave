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
