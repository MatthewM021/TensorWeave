# Read This First — Self-Contained Work Handoff

This repository contains the actual V2 project. It does not rely on a ChatGPT sandbox link, a missing prior repository, or an unstated download.

## What is physically present

- `v2_baseline/` — the complete extracted V2 repository: source, tests, configurations, checkpoints, raw results, plots, reports, replay scripts, and validation manifests.
- `source_archives/mera_language_direction_demo_v2.zip` — the original V2 archive embedded verbatim as a second source of truth.
- `HANDOFF.md` — project history, verified findings, interrupted V3 state, and continuation order.
- `docs/` — V3 recovery and validation specifications.
- `v3_recovery/V3_RECOVERY_STATUS.yaml` — machine-readable record of what was and was not recovered from the interrupted V3 attempt.
- `WORK_START_PROMPT.md` — the instruction to paste into ChatGPT Work after this whole repository has been pushed to GitHub.
- `VERIFY_REPOSITORY.py` — verifies that every V2 file is present and byte-identical to the included manifest.

## The one required transfer step

ChatGPT Work cannot see files merely because they were linked in another ChatGPT conversation. Extract this archive and push **the entire extracted directory** to GitHub. Work must then be opened on that GitHub repository.

Do not create an empty repository containing only `HANDOFF.md` or only paste `WORK_START_PROMPT.md`; that would omit the code and evidence.

See `GITHUB_IMPORT.md` for exact commands.
