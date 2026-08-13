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
