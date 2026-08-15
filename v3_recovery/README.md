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

The later Phase-III modules are intentionally not added to either frozen
Phase-II source closure. On the current head, the historical Power V2 and V4
production loaders therefore reject execution as post-freeze source drift.
Regression tests verify the later additions, with no removed or changed
recorded file, and that rejection occurs before fitting or probing. A positive
production replay requires the frozen V4 source bundle/commit; the immutable
Phase-II artifacts are not regenerated or reclassified to follow later code.

Phase III now has three local bounded checkpoints. The first recovers the
finite zero-suffix diagnostic state block at exact rank `4 -> 5` from opaque
tokens after one two-label membership response; it makes no transition-
operator claim.  The second keeps the absence-aware contract guarded and
learns each event map only on its legal source domain.  Its two full-support
controls and eight rotated omissions reconstruct all 44 legal edges, predict
64 sealed edges and 120 long/path programs exactly, reject all 460 undefined
edges, and retain exact off-domain total-extension nullity 80 per environment.
The 15-edge excitation basis and postfit state/event correspondences are
trusted-controller inputs.

The official third checkpoint is preserved under
`phase3_t2_opaque_active_discovery/`. It retains the supplied finite grammar,
legality mask, passive rows, and unordered candidate pool, but autonomously
selects the causal excitation sequence. Its two controls and eight omissions
used 112 membership responses / 224 labels, made eight answer-free structural
inferences, and left 64 candidates sealed. All 440 legal rows and 120 long/path
programs were exact, all 460 undefined pairs were rejected, and five postfit
gauge certificates passed. The T1-first-14 reuse control remained
nonidentified. The truth-aware postfit teaching control accounted for 104
counterfactual queries, 16 singleton inferences, 64 unopened candidates, 208
labels, and zero new membership calls; it is noncausal and selection-
ineligible.

The official protocol/source-runtime/terminal/report record SHA-256 values are
`7c5ee8bcee72e0af5ac2d8404f54b479e1b7d1b1200922ec40caf66483c04292`,
`514ebb445d3eb00e456095bf3377bf4f7eb2e15a4282e0765ca307b5203e5e90`,
`31b3849c7a469ed380c68502193287e4917ee07170bbbd9c3c95d837055c1352`,
and
`9993913e5f60a73ce41fb08803c7dab24511165a4dc636e89b065d68be59c40f`.
The terminal and report raw-file SHA-256 values are
`e5e2524285c970eff1e45474d43aee43cf26615ce8ae89aa19b43dd0aa5b0819`
and
`89b1fa8fa2d5f21fd001143d8275fcde94acdac964169828d55f566386ae5bd9`.
`SUMMARY.json` has record/file SHA-256 values
`d70a52a187b8341bdfc072043fdba629803f5177385181109dd49ac191c6929b`
and
`d44abe1afb0e175fb160efccabfe084390018b1c87a994ecdbacfae74a68c7a3`;
the 14-row `EVIDENCE.sha256` manifest has raw SHA-256
`1ca40ffb794134599f39ec3ce6db25b9b58f00d2430658d938df91554f3dba00`.
All three checkpoints remain honest-code synthetic rehearsals, not process-
isolated total-WFA, factorization, natural-language, global-query-minimality,
or assumption-free representation evidence.
