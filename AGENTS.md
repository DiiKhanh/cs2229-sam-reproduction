# AGENTS.md

Instructions for AI agents working in this repository. This file is the source of truth; `CLAUDE.md` imports it.

## Project

A four-member CS2229 course project that reproduces part of *Sharpness-Aware Minimization for Efficiently Improving Generalization* (Foret et al., ICLR 2021): SGD vs. SAM-SGD on CIFAR-10 with WideResNet-28-10, 200 epochs, `rho = 0.05`, seeds 0-3. Report deadline: 2026-11-30.

Authoritative sources, in order:

1. `docs/2026-10-23-sam-experiment-work-plan-vi.md` - scope, run matrix, member roles, artifact contract, schedule.
2. `notebooks/sam_reproduction_vi.ipynb` - the single canonical training pipeline. There is no separate Python package.
3. `README.md` - team-facing Colab run and resume guide.

## Execution model

- Runs execute only on Google Colab GPUs. No local GPU is available.
- Each member uses their own Google Drive with `MyDrive/CS2229-SAM/data/cifar-10-python.tar.gz` (uploaded manually) and `MyDrive/CS2229-SAM/results/` (written by the notebook).
- **Run all** is the only supported workflow, for both the first run and every resume. `run_planned_runs()` skips completed runs, resumes partial runs from `last_checkpoint.pt`, and starts missing ones.
- The notebook fails closed: no Drive mount, missing or corrupt dataset, unwritable results directory, or a config that differs from an existing checkpoint all stop it before training.

## Branch model

- `main` holds the canonical notebook with `RUN_PLAN = "smoke"`.
- `run/core-s<k>` (k = 0..3) differs from `main` only in the branch section of the notebook's config cell: `RUN_PLAN = "core"` and `SEED = k`. Keep that diff to those two lines so `main` can be merged in cleanly.
- Make pipeline, recipe and documentation changes on `main`, then merge `main` into the run branches.
- Do not merge a `NOTEBOOK_VERSION` change into a run branch while that branch's runs are in progress: resume compares the full config, including the version, and refuses to continue. Coordinate with the team first.
- Extension runs (rho ablation, SGD 400 epochs, label noise) get their own branches after the core matrix is complete.

## Changing the notebook

- Any change that affects training, evaluation, data, or artifact content requires bumping `NOTEBOOK_VERSION`, and runs produced by older versions must not be aggregated with newer ones.
- Edit the notebook as JSON through `nbformat` or a notebook editor; keep `nbformat` 4.0 without cell ids, and strip outputs before committing.
- Never write large files on Drive with temp-file-then-rename or delete them in a loop. Drive moves deleted and rename-overwritten files to Trash, which keeps counting against quota for 30 days. Checkpoints (~280 MB each for WRN-28-10) therefore use `save_checkpoint()`, which overwrites in place after writing a fixed `.backup` copy, and `load_checkpoint()`, which falls back to that copy. Keep that pattern.
- Keep the artifact contract in the work plan (Section 6) and the notebook's `REQUIRED_ARTIFACTS`, `REQUIRED_SUMMARY_FIELDS` and `REQUIRED_HISTORY_COLUMNS` consistent.

## Verifying a change

Run the notebook end-to-end on CPU against a simulated Drive (needs PyTorch, torchvision, pandas, matplotlib, and a local copy of `cifar-10-python.tar.gz`):

```bash
mkdir -p /tmp/sam-drive/MyDrive/CS2229-SAM/data
cp cifar-10-python.tar.gz /tmp/sam-drive/MyDrive/CS2229-SAM/data/
python tools/run_notebook_locally.py notebooks/sam_reproduction_vi.ipynb \
    --drive /tmp/sam-drive --runtime /tmp/sam-runtime --plan smoke --seed 0
```

It must print `NOTEBOOK_OK`. For changes touching checkpointing or resume, also kill the process mid-run, rerun the same command, and confirm `history.csv` and `summary.json` match an uninterrupted run with the same seed. The smoke plan uses WRN-16-1 on a 1,024-image subset; it validates the pipeline, not WRN-28-10 runtime or memory. State plainly when a change has not been verified on a Colab GPU.

## Research integrity

- Select checkpoints only by validation accuracy; never tune or select on the test set.
- Report every seed, including unfavorable ones; never keep only the best run.
- Never fabricate or estimate results. Every number in docs or slides must trace to a `summary.json` or `history.csv` with its run ID.
- Treat completed run directories as immutable evidence. Do not overwrite, rename, or delete them; start a new run ID instead.
- Distinguish the paper's reported numbers from the team's own results.

## Conventions

- Team-facing documents and notebook text are in Vietnamese; keep established technical terms (SGD, SAM, checkpoint, validation, batch) in English. Code identifiers and this file are in English.
- Dated notes use the `YYYY-MM-DD-` prefix in `docs/`.
- Commits follow Conventional Commits, for example `feat(notebook): ...`, `docs: ...`, `chore(run): ...`.
- Never commit datasets or checkpoints (`*.pt`, `*.pt.backup`); see `.gitignore`.
