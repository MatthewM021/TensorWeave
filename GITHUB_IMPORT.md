# Put the Complete Handoff into GitHub

1. Download and extract `tnlm_work_repo_self_contained_2026-08-13.zip`.
2. Enter the extracted `tnlm_work_repo_self_contained_2026-08-13` directory.
3. Connect that entire directory to the empty GitHub repository you created:

```bash
git init
git add .
git commit -m "import complete V2 baseline and V3 recovery handoff"
git branch -M main
git remote add origin <YOUR_EMPTY_REPOSITORY_REMOTE>
git push -u origin main
git tag v2-handoff-baseline
git push origin v2-handoff-baseline
git switch -c v3-recovery
git push -u origin v3-recovery
```

Then open that GitHub repository in ChatGPT Work and paste the contents of `WORK_START_PROMPT.md`.

The critical point is that `git add .` must be run from the extracted handoff root. The `v2_baseline/` directory and its results/checkpoints must appear in GitHub before Work begins.
