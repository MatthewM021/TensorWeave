# V3 Validation and Campaign Closure Protocol

V3 is complete only when every applicable item below has a machine-readable record.

## A. Core correctness

- [ ] Streaming and parallel/tree reductions agree on deterministic fixtures.
- [ ] Prefix logits are unchanged by modifying future tokens.
- [ ] Prefix routes are unchanged by modifying future tokens.
- [ ] Padding after the valid prefix is invariant.
- [ ] Evaluation-only true route labels cannot affect model outputs.
- [ ] Global/null route semantics are tested.
- [ ] Branch permutations are either equivariant or correctly aligned for metrics.
- [ ] Model parameter count does not grow with configured maximum context length.
- [ ] State serialization and resume are deterministic.
- [ ] Checkpoints replay on exact recorded test splits.

## B. Routing audits

- [ ] Oracle, curriculum, and fully latent modes are separate configurations.
- [ ] Curriculum guidance schedule is recorded per step/epoch.
- [ ] Autonomous evaluation uses no oracle route input.
- [ ] Operational routing audit confirms the selected branch is the branch actually updated.
- [ ] Route metric is computed after optimal permutation where labels are symmetric.
- [ ] Route collapse/load imbalance is reported.
- [ ] Document-local remapping prevents global token lookup.
- [ ] Unseen symbol/binding combinations are included.

## C. Compact-export audits

- [ ] Hard exported ranks are recorded.
- [ ] Exported tensors are physically smaller.
- [ ] Dense-selected and compact outputs agree within tolerance.
- [ ] Discarded-channel perturbations cannot affect retained outputs.
- [ ] Inactive scale-signal channels cannot affect retained gates or outputs.
- [ ] Fresh-process load and inference pass.
- [ ] Serialized bytes, parameters, operation proxy, RSS, and wall time all reported.
- [ ] The 16 previously bug-affected pruned conditions are regenerated; old values are marked invalid.

## D. Scientific controls

- [ ] At least three paired strong seeds for headline architecture comparisons.
- [ ] Model seed and all data split seeds are shared across paired variants.
- [ ] MERA/TTN comparison is parameter- and optimizer-controlled.
- [ ] Strong GRU and cached Transformer opponents are included.
- [ ] Mixed-length training and true longer-update evaluation are included.
- [ ] Padding-only length tests are identified and not presented as extrapolation.
- [ ] Quality, route recovery, ranks, state size, memory, and wall time are all reported.
- [ ] Matched-quality comparisons are separated from nominal-length scaling hints.

## E. Campaign completion inherited from the interrupted V3 attempt

- [ ] Third strong dynamic-binding seed complete.
- [ ] Corrected causal TTN/MERA controls complete.
- [ ] Strong-opponent tournament complete.
- [ ] 16 corrected pruned/export runs regenerated.
- [ ] Clean 22-run smoke campaign complete with zero failures.
- [ ] 2,048-token scaling/audit run complete.
- [ ] Operational routing audit complete.
- [ ] Evaluation-label independence audit complete.
- [ ] Full checkpoint replay complete.
- [ ] Final test suite passes from a clean environment.

## F. Release integrity

- [ ] Every headline number links to a run record and checkpoint.
- [ ] Invalidated runs remain recorded but are excluded from aggregates.
- [ ] Executed configs and hashes are stored.
- [ ] Environment and code commit are stored.
- [ ] JSON is strict and finite where required.
- [ ] Checksums verify all release files.
- [ ] `V3_REPORT.md` distinguishes established results, indications, and failures.
- [ ] Repository tag/release points to the exact verified commit.
