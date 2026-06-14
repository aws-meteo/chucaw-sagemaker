# FourCastNet CPU Batch Transform — Repo Cleanliness Review + Pre-Mortem

REALTIME_ENDPOINT_COST_RISK

This document has two parts:

1. A quick repo-cleanliness review (current state of the repo, June 2026).
2. A **pre-mortem** for the planned use case: run FourCastNet on SageMaker Batch Transform
   (CPU instances, no endpoints), feeding `input.npy` tensors from S3 to the model **one
   after another**. The pre-mortem assumes the run has already gone wrong and works
   backward to the likely causes, so they can be fixed on paper before any AWS spend.

It is meant to be read alongside, not instead of, the existing operational docs:
`docs/fourcastnet_cpu_runbook.md`, `docs/fourcastnet_batch_transform.md`, and
`docs/sagemaker_cost_safety.md`.

---

## 1. Repo cleanliness review

**Overall verdict: clean.** The repo is ~1.6 MB, the working tree is clean, and there is
no incidental clutter:

- No `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, `*.bak`/`*_old`/`*copy*` files, or
  stray logs.
- `.gitignore` already excludes data, model artifacts, and debug output.
- No file in git history is larger than ~15 KB; model weights and `.npy` tensors are
  correctly kept out of version control.

**Dependency management is intentionally segmented**, not duplicated by accident:
- `environment.yml` — top-level conda env (Python 3.10, boto3, sagemaker, scikit-learn).
- `src/fourcastnet/serving/requirements.txt` — Phase 1 (`metadata_only`) container: numpy,
  torch, boto3 only.
- `src/fourcastnet/serving/requirements-forward.txt` — Phase 2 (`forward`) container: adds
  `timm`, `einops`.
- `training/requirements.txt`, `inference/requirements.txt` — domain-specific, minimal.

**`experimental/realtime_endpoint_dangerous/`** is not dead code — it is a deliberately
quarantined record of the May 2026 real-time-endpoint cost incident
(`docs/sagemaker_cost_safety.md`), kept isolated and out of the normal execution path.

**Minor, non-blocking observations** (no action required, just flagged for awareness):
- Root-level parquet-inspection scripts and `chucaw_preprocessor_extracted/` (an extracted
  wheel) are working/inspection artifacts rather than library code. They don't hurt
  anything, but if they're no longer needed for day-to-day work they're candidates to
  delete or move under a `tools/` or `scratch/` directory in a future cleanup pass.
- No CI/CD config (`.github/workflows/`). This is consistent with the repo's
  guardrail-script-driven, manual-execution SageMaker workflow
  (`docs/team_runbook.md`), but means the safety checks
  (`scripts/check_no_sagemaker_always_on_compute.py`,
  `scripts/check_no_fourcastnet_endpoints.py`, `tests/test_run_fourcastnet_batch_transform.py`)
  only run when a human/agent remembers to run them.

**Bottom line:** nothing here needs cleaning up before doing the FourCastNet batch
transform work. The risk surface for the planned run is entirely in the SageMaker/data
pipeline, covered below.

---

## 2. Pre-mortem: "the sequential `input.npy` batch transform run failed — why?"

Each section below is a failure-mode category: **scenario → evidence → severity/likelihood
→ mitigation**.

### A. Packaging / model-artifact drift

| | |
|---|---|
| **Scenario** | The Batch Transform job runs against a `model.tar.gz` that doesn't contain the current handler, or is missing required assets entirely. |
| **Evidence** | `scripts/package_fourcastnet_model.py` (L82-96) hard-requires `backbone.ckpt`, `global_means.npy`, `global_stds.npy` plus `inference.py` and a requirements file — it fails fast locally if any are missing, which is good. But `docs/fourcastnet_cpu_runbook.md` gap #4 notes the **S3 artifact at `model_data_url` may predate the merged handler** — repackaging is a manual step that's easy to forget after a handler change. |
| **Severity / likelihood** | High impact (silently runs stale code, e.g. an old `_attempt_forward` without the NVlabs AFNONet fix), medium likelihood (only triggers after a handler edit that isn't followed by repackage+reupload). |
| **Mitigation** | Before any real run, repackage with `--layout self-contained` and re-upload to `model_data_url`, even if "it was uploaded before." Two separate artifacts are needed — `model.tar.gz` (Phase 1, minimal requirements) and `model-forward.tar.gz` (Phase 2, `requirements-forward.txt`) — and two separate SageMaker Models should point at them (`sbnai-fourcastnet-fcn-v1-cpu-poc` vs. `...-cpu-forward`). Using the Phase‑1 model with `mode=forward` in the manifest doesn't crash — it returns a "controlled" `backend_unavailable` result — but it will look like a failure if you forget which model is which. |

### B. IAM / S3 permission gaps

| | |
|---|---|
| **Scenario** | The Transform job's execution role can't read one of the `input_s3_uri` objects, or can't write to `output_s3_uri`, or can't pull `model_data_url`. |
| **Evidence** | `configs/fourcastnet_batch_cpu_poc_v1.json` pins `execution_role_arn` to `AmazonSageMakerAdminIAMExecutionRole`. Inside the container, `_load_npy_from_s3` and `_write_json_to_s3` (`src/fourcastnet/serving/inference.py` L53-71) use the default boto3 credential chain (i.e. this role). |
| **Severity / likelihood** | High impact if it happens (the whole record fails), low likelihood given the admin role — but worth distinguishing failure modes. |
| **Mitigation** | Two different failure shapes to recognize: (1) a job-level failure before any record is processed (bad `model_data_url`, image pull failure, role can't be assumed) — visible via `describe-transform-job` `FailureReason`; (2) a per-record `ok=false` with `result=prediction_failed`, `error_type=ClientError`, `error_message` containing `AccessDenied` — visible inside the JSON report for that record. (2) is the one that matters for "one after another" processing: if every `.npy` lives under the same prefix as the canary tensor, permissions proven by the Phase‑1 canary should carry over; if real inputs live under a *different* prefix (e.g. `ecmwf/fourcastnet/...` vs `sagemaker/fourcastnet/fcn-v1/input/...`), re-check read access to that prefix specifically. |

### C. "One after another" sequencing mechanics

This is the section most directly tied to the stated goal, and it surfaces the biggest
open question.

**Critical open question — resolve before building anything:**
"One after another" can mean two very different things:

1. **N independent snapshots** (e.g. different forecast init times), each of which can be
   described by one line in a JSONL manifest and processed sequentially within a *single*
   Transform job. SageMaker already supports this natively via
   `MaxConcurrentTransforms=1` + `BatchStrategy=SingleRecord` + `SplitType=Line` — exactly
   the contract documented in `docs/fourcastnet_batch_transform.md` and configured in
   `configs/fourcastnet_batch_cpu_poc_v1.json`. **Nothing new is required at the AWS API
   level for this case** — only a manifest with N lines instead of 1.

2. **An autoregressive rollout**, where the model's output for file *N* must become
   `input.npy` for file *N+1* (e.g. 6h, 12h, 18h, 24h forecast steps chained together).
   **Batch Transform cannot do this within a single job** — there is no mechanism for one
   record's output to feed the next record's input mid-job. This would require external
   orchestration: N separate `CreateTransformJob` calls, run sequentially, each waiting for
   the previous job's output before being created — extending the pattern already sketched
   in `workflows/fourcastnet_cpu_batch_transform_poc_v2.yaml` into a loop.

These two interpretations lead to completely different tooling (a manifest generator vs.
an orchestration loop), so this should be settled first. Everything else in this section
assumes interpretation **(1)**.

| | |
|---|---|
| **Scenario** | A multi-line manifest is needed, but the only manifest-generation tooling that exists (`scripts/create_fourcastnet_canary_payload.py`) hardcodes a single synthetic `(1,10,10)` canary tensor and writes exactly **one** JSONL line. |
| **Evidence** | `scripts/create_fourcastnet_canary_payload.py` L23-39 — one tensor, one `payload`, one line written to `canary_manifest.jsonl`. |
| **Severity / likelihood** | Blocking for the stated goal — this is the literal gap between "what exists" and "process N `.npy` files from S3." |
| **Mitigation** | **Out of scope for this document** (per explicit decision), but flagged as the #1 follow-up: a script that lists/accepts N S3 `.npy` URIs and emits one JSONL line per file (each with its own `input_s3_uri`, `output_s3_uri`, `mode`, `max_runtime_guard`), reusing the existing per-record schema verbatim. |

| | |
|---|---|
| **Scenario** | Running N separate Transform jobs (one per `.npy` file) instead of one job with an N-line manifest. |
| **Evidence** | `model_fn` (`inference.py` L346-363) runs once per job/worker and is reused for every record `predict_fn` processes afterward. A fresh job pays full container startup + `torch.load(checkpoint_path, ...)` again. |
| **Severity / likelihood** | Medium-high cost/time impact if N is more than a handful — checkpoint load + container init is likely the single largest fixed cost per invocation on CPU. |
| **Mitigation** | Prefer **one Transform job with an N-line manifest** over N jobs. This also means "one after another" happens for free via `MaxConcurrentTransforms=1` — SageMaker itself serializes record processing on the single worker. |

| | |
|---|---|
| **Scenario** | Confusion between the per-record JSON report and SageMaker's own batch output file when N > 1. |
| **Evidence** | `_write_json_to_s3` (`inference.py` L62-71) writes `inference_report_<UTC-second-timestamp>.json` to `output_s3_uri` for *each record*, when `output_s3_uri` ends in `/`. Separately, with `AssembleWith=Line` (per `docs/fourcastnet_batch_transform.md` L80-91), SageMaker concatenates **all N** `output_fn` return values into a single `<manifest_key>.out` file at `S3OutputPath`. |
| **Severity / likelihood** | Low severity, but easy to misread results — e.g. expecting N `.out` files and finding one, or vice versa. |
| **Mitigation** | Treat the `.out` file as the authoritative "did every record run" check (run `scripts/validate_fourcastnet_batch_output.py` against it — it already iterates line-by-line, so it scales to N records unmodified). Treat the per-record `inference_report_*.json` files (or per-record distinct `output_s3_uri` prefixes, e.g. `output/<input-stem>/`) as the per-file artifacts. With `MaxConcurrentTransforms=1`, sequential records are seconds apart so timestamp-based filenames won't collide — but there's still no *deterministic* input→output filename mapping; correlate via the `input_s3_uri` field embedded in each report. |

### D. Phase 2 (`forward` mode) feasibility on CPU

| | |
|---|---|
| **Scenario** | `max_runtime_guard` is disabled "because the runbook says the tensor is too big," when it may not need to be. |
| **Evidence** | `MAX_GUARD_ELEMENTS = 50_000_000` (`inference.py` L22). A single `(1, 20, 720, 1440)` float32 tensor = 20,736,000 elements — **under** the 50M guard. `docs/fourcastnet_cpu_runbook.md` gap #7 frames the "real ~1GB tensor" as needing `max_runtime_guard=false`; ~1GB likely refers to the checkpoint/activations footprint, not necessarily the input array itself. |
| **Severity / likelihood** | Low-medium — disabling the guard removes a real OOM safety net unnecessarily if the actual input is a single `(1,20,720,1440)` sample. |
| **Mitigation** | Before setting `max_runtime_guard=false`, run a `metadata_only` pass on the real `input.npy` (cheap, no backend needed) and read `input_tensor.shape`/`size` from the report. Only disable the guard if the real shape genuinely exceeds 50M elements (e.g. a stacked multi-timestep batch). |

| | |
|---|---|
| **Scenario** | A `forward` invocation on CPU takes longer than the per-invocation timeout, and the record fails (or the whole job runs far longer/costlier than expected across N records). |
| **Evidence** | AFNONet config is `embed_dim=768`, `depth=12`, `img_size=(720,1440)`, `patch_size=8` → 16,200 patches per layer (`docs/fourcastnet_cpu_runbook.md` Phase 2). `scripts/run_fourcastnet_batch_transform.py`'s `create_transform_job` payload (L121-147) does **not** set `ModelClientConfig.InvocationsTimeoutInSeconds`, so the SageMaker default applies to every record. CPU forward latency for this architecture has not been measured in-repo. |
| **Severity / likelihood** | High — for N records, total job wall-clock ≈ N × per-record latency, and this directly drives Batch Transform cost (billed per instance-second). An unexpectedly slow per-record latency multiplies across the whole manifest. |
| **Mitigation** | Run a **single-record** Phase 2 timing test on `ml.m5.2xlarge` first. Use the measured latency to (a) decide whether `ModelClientConfig.InvocationsTimeoutInSeconds` needs to be set explicitly and (b) extrapolate total job cost for N files before submitting the full manifest. Keep `MaxConcurrentTransforms=1` regardless — it's both a cost guardrail and the mechanism that makes "one after another" sequential. |

| | |
|---|---|
| **Scenario** | Phase 2 container fails to install `timm`/`einops` at startup. |
| **Evidence** | `requirements-forward.txt` is bundled as `code/requirements.txt` and pip-installed at container start (`docs/fourcastnet_cpu_runbook.md` gap #2), which requires outbound internet access. |
| **Severity / likelihood** | Low today (no `VpcConfig` is set on the Transform job per current configs) — flagged only as a **future** risk if a VPC-restricted/private-subnet Transform job is introduced later. |
| **Mitigation** | None needed now; if VPC isolation is added later, either provide a NAT/VPC endpoint for PyPI or pre-bake `timm`/`einops` into a custom container image. |

### E. Correctness / silent-wrong-answer risks

| | |
|---|---|
| **Scenario** | The job reports `ok=true, fourcastnet_proven=true` but the forecast values are numerically wrong (e.g. swapped channels, mismatched normalization stats). |
| **Evidence** | `fourcastnet_proven` (`inference.py` L319) is set from `len(load_info["missing_keys"]) == 0` — i.e. it only certifies that the checkpoint's `state_dict` loaded cleanly into the `AFNONet` module. It says nothing about whether `global_means.npy`/`global_stds.npy` correspond to the same 20-channel ordering as the input tensor, or whether units match what the model was trained on. |
| **Severity / likelihood** | High impact if wrong (results look plausible but are scientifically invalid), low-medium likelihood given the channel order has been verified to match. |
| **Verification already done** | Chucaw's `FOURCASTNET_REQUIRED_CHANNELS` (`chucaw_preprocessor_extracted/chucaw_preprocessor/fourcastnet.py` L16-37) — `u10, v10, t2m, sp, msl, t850, u1000, v1000, z1000, u850, v850, z850, u500, v500, z500, t500, z50, r500, r850, tcwv` — matches the canonical NVlabs FourCastNet 20-channel ordering. Good. |
| **Mitigation** | `fourcastnet_proven=true` is necessary but not sufficient. Add a manual sanity check on `output_stats` (min/max/mean per the report) against known climatology ranges for each variable (e.g. `t2m` mean should be ~250-300K, not ~0 or ~1e6) for the first real run, especially if `global_means.npy`/`global_stds.npy` come from a different asset bundle than the one the channel ordering above was validated against. |

| | |
|---|---|
| **Scenario** | The S3 prefix that's supposed to contain N `input.npy` files for "one after another" processing is empty, partial, or stale — and the root cause is *upstream* of SageMaker entirely. |
| **Evidence** | `EXPECTED_LATS = np.arange(90.0, -90.0, -0.25)` (`chucaw_preprocessor_extracted/chucaw_preprocessor/fourcastnet.py` L12) produces exactly 720 latitudes, matching AFNONet's `img_h=720` — consistent. However, `platinum_parquet_to_fourcastnet.py` defaults `LATITUDE_POLICY` to `"fail"`. If the source platinum parquet has **721** distinct latitudes (common for 0.25° ERA5/ECMWF grids that include both poles, -90 and +90), `build_fourcastnet_tensor`/`resolve_expected_lats` (`fourcastnet.py` L91-109) will raise rather than apply `drop_south_pole`/`drop_north_pole` — and no `input.npy` gets produced for that partition at all. |
| **Severity / likelihood** | Medium — only matters if/when the upstream Glue job is (re)run to populate new `input.npy` files for the queue; doesn't affect already-existing tensors. |
| **Mitigation** | If the input queue is unexpectedly short or empty, check the upstream Glue job's `LATITUDE_POLICY` argument and the source parquet's latitude cardinality before debugging anything on the SageMaker side. |

### F. Cost guardrails

| | |
|---|---|
| **Scenario** | A multi-file ("one after another") Phase 2 run on `ml.m5.2xlarge`+ exceeds the project's SageMaker budget. |
| **Evidence** | `budget/budget-sagemaker.json` caps SageMaker spend at **$10/month**; `budget/budget-account.json` caps the whole account at **$30/month**. `scripts/run_fourcastnet_batch_transform.py` has solid pre-execute guardrails (dry-run by default, rejects placeholder URIs, blocks `application/x-npy` without `--allow-large-direct-payload`, blocks `MaxConcurrentTransforms>1` without `--allow-concurrency`, blocks `MultiRecord` for JSON content types) — but Batch Transform jobs have **no native max-runtime stopping condition** in the `create_transform_job` API. |
| **Severity / likelihood** | Medium-high if N is large and per-record latency (Section D) is underestimated — a job that runs much longer than expected keeps billing for its full duration regardless of budget alerts (which are informational, not enforcing). |
| **Mitigation** | Start with N=1-2 real files, not the full queue. Use the single-record timing from Section D to extrapolate: `estimated_cost ≈ N × per_record_seconds × instance_hourly_rate / 3600`. Compare against the $10/month SageMaker budget *before* submitting an N-file manifest. Monitor in-progress jobs with `aws sagemaker describe-transform-job` and be prepared to call `StopTransformJob` manually if a run is clearly hung — the budget notifications in `budget/budget-notifications.json` (50%/80% thresholds, emailed to `fbientrigo@gmail.com`) are after-the-fact signals, not preventive ones. |

---

## 3. Recommended action checklist (before the next real run)

1. **Resolve the sequencing question (Section C)**: are the N `input.npy` files independent
   snapshots (one N-line manifest suffices) or an autoregressive chain (needs orchestrated
   sequential jobs)? This determines what tooling to build next.
2. **Re-run the Phase 1 `metadata_only` canary** end-to-end per
   `docs/fourcastnet_cpu_runbook.md` to (re)prove IAM + S3 + container wiring — especially
   if it's been a while or the handler has changed since the last successful run.
3. **Check real `input.npy` shape/size** via a `metadata_only` report before deciding
   whether `max_runtime_guard=false` is actually necessary (Section D).
4. **Time one Phase 2 `forward` record** on `ml.m5.2xlarge` to get a real per-record CPU
   latency, and use it to estimate total cost for N files against the $10/month SageMaker
   budget (Sections D and F).
5. **(Follow-up, separate task)** Build the multi-file JSONL manifest generator — list/accept
   N S3 `.npy` URIs and emit one manifest line per file using the existing per-record schema
   (`input_s3_uri`, `output_s3_uri`, `mode`, `max_runtime_guard`). Only do this after step 1
   confirms the "independent snapshots, one manifest" interpretation.
