# AGENTS.md

## Mission
Work iteratively until the task is actually completed, not merely scaffolded.

## Operating mode
For any non-trivial task, follow this loop:

1. Inspect current repo state.
2. Form up to 3 hypotheses for the main blocker, ranked by probability.
3. Pick the highest-priority testable hypothesis.
4. Apply the minimum viable change.
5. Execute the relevant command(s).
6. Observe the real output.
7. Update the hypothesis ranking based on evidence.
8. Repeat until acceptance criteria are met or a hard external blocker is reached.

## Evidence policy
Never claim success without showing evidence from executed commands.
Do not stop at “files created”.
Always distinguish between:
- code written
- code executed
- code validated

## Failure policy
If a command fails:
- quote the exact failing command
- quote the key error line
- explain the most likely cause
- propose the minimum next fix
- retry

If two attempts fail for the same hypothesis, demote it and move to the next one.

## Environment policy
Before running Python commands:
1. Detect the actual Python executable path.
2. Print:
   - `where python`
   - `python --version`
   - the exact interpreter path used for execution
3. Prefer the project-local environment if available.
4. This repo uses a conda environment "chucaw-sagemaker"


## Repo-specific data contract
The local dataset for first execution is:
`data/20260409180000-6h-scda-fc.parquet`

The Athena source of truth is:
- database: `silverlayer`
- table: `forecast_parquet`

Use only the filtered subset:
- `variable = 't'`
- `isobaricinhpa = 1000.0`

Both LOCAL and ATHENA flows must converge to columns:
- `latitude`
- `longitude`
- `value`

## SageMaker architecture
Prefer SageMaker Batch Transform for FourCastNet.
Athena is extraction-only, never part of online inference.
The baseline model is deterministic nearest-grid lookup, not real ML.

## SageMaker cost safety policy (strict)
- Agents must not create real-time endpoints unless the user explicitly requests it in the same session.
- Agents must prefer Batch Transform for FourCastNet.
- Agents must run `scripts/check_no_sagemaker_always_on_compute.py` before any AWS write operation.
- Agents must show expected cost mode before creating resources.
- Agents must tag created resources with:
  - `Project=SbnAI`
  - `Component=FourCastNet`
  - `CostMode=batch-only`
  - `Owner=Fabian`
  - `Environment=dev`
- Agents must refuse to use `.deploy()` for FourCastNet unless the user explicitly accepts real-time endpoint cost risk.

## Task not done until
A task is not complete until the relevant acceptance commands have run successfully.

For this repo, completion requires evidence for:
1. local extraction from `data/20260409180000-6h-scda-fc.parquet`
2. `model.joblib` created
3. `model.tar.gz` created with correct structure
4. local smoke inference succeeds
5. deployment command is ready and validated as far as credentials/resources allow

## Reporting format
At the end of each work cycle, report:
- current hypothesis
- command executed
- observed result
- next concrete step

If ultrawork its active, close a task with a summary then do a last check of it and compare it to the initial prompt in order to understand if goal its actually completed, if not, proceed to compile the plan for the new task of fixing the weak and remaining points.
