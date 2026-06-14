# FourCastNet on CPU Batch Transform — Runbook

How to actually run FourCastNet via SageMaker **CPU** Batch Transform, and what was
missing to get there. GPU transform quota is `0`; CPU (`ml.m5.large`) quota exists, so
this reuses the original GPU `model.tar.gz` under a **CPU** PyTorch image.

- **Region:** `us-east-1` · **Profile:** `sbnai-725`
- **Account/bucket:** `725644097028` · `chucaw-data-platinum-processed-725644097028-us-east-1-an`
- **CPU image:** `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310`
- **No endpoints, ever.** Batch Transform only (see `docs/sagemaker_cost_safety.md`).

This work is **phased**. Phase 1 is a hard gate for Phase 2.

| Phase | Mode | Needs backend? | Cost | Proves |
|-------|------|----------------|------|--------|
| 1 | `metadata_only` | No | ~$0 (seconds on `ml.m5.large`) | IAM + S3 r/w + CPU container init end-to-end |
| 2 | `forward` | Yes (FourCastNet) | Higher (slow on CPU, bigger instance) | Real neural-net inference (`fourcastnet_proven=true`) |

---

## Gap analysis — what was lacking

**Fixed in the repo:**
1. **Workflows used the wrong contract.** `workflows/fourcastnet_cpu_batch_transform_poc*.yaml`
   sent `application/x-npy` + `SplitType: None`, which forces `mode=forward` in
   `src/fourcastnet/serving/inference.py` (`input_fn`). On the stock CPU image that
   returns `ok=false, reason=backend_unavailable`. Now patched to the JSONL S3-pointer
   `metadata_only` contract (`application/jsonlines` + `SplitType: Line` +
   `AssembleWith: Line`), matching `configs/fourcastnet_batch_cpu_poc_v1.json`.
2. **No forward backend, no way to ship one.** `serving/requirements.txt` installs only
   `numpy/torch/boto3`. Added `serving/requirements-forward.txt` (with the FourCastNet
   backend) and a `--requirements-file` flag on `scripts/package_fourcastnet_model.py`
   so the backend is bundled **only** into a forward build — keeping the cheap Phase-1
   canary container minimal.

**Still operational (you run these on AWS — see steps below):**
3. Model assets (`backbone.ckpt`, `global_means.npy`, `global_stds.npy`) are not in the
   repo and there is no built `artifacts/fourcastnet/build/model.tar.gz`. Fetch assets
   from S3 to (re)package.
4. The `model.tar.gz` at the config's `model_data_url` must contain the **current**
   `code/inference.py`. If the existing S3 artifact predates the merged handler,
   repackage (self-contained) and re-upload.
5. The JSONL manifest at the config's `input_s3_uri`
   (`.../fcn-v1/input/canary_manifest.jsonl`) must be uploaded to S3.
6. The CPU SageMaker model must be created from the **CPU** config — the create-model
   script defaults to the GPU config, so pass `--config configs/fourcastnet_batch_cpu_poc_v1.json`.

**Phase-2 feasibility caveats:**
7. `MAX_GUARD_ELEMENTS = 50_000_000` blocks the real ~1GB tensor unless the manifest
   sets `"max_runtime_guard": false`.
8. `ml.m5.large` (8 GB) is likely too small for a 1GB tensor + model + backend on CPU;
   move to `ml.m5.2xlarge`/`4xlarge`. CPU forward is slow.
9. **Backend identity (resolved).** `backbone.ckpt` is the **NVlabs/FourCastNet**
   pretrained AFNO backbone (Pathak et al. 2022, BSD-3), shipped with `global_means.npy`
   + `global_stds.npy` as the standard NVlabs asset triad. The checkpoint is
   **NVlabs-format, not Modulus/PhysicsNeMo-format**:
   - `torch.load(...)` → dict keyed `'model_state'` (not `'model_state_dict'`)
   - weight keys are prefixed `module.` (strip the first 7 chars)
   - architecture is `AFNONet` (NVlabs `networks/afnonet.py`):
     `img_size=(720,1440)`, `patch_size=(8,8)`, `in_chans=20`, `out_chans=20`,
     `embed_dim=768`, `depth=12`
   Consequently `nvidia-physicsnemo`'s `FourCastNet` class cannot load this checkpoint
   directly, and the handler's current `_probe_backend` + `load_from_checkpoint` design
   (which targets Modulus/PhysicsNeMo classes) will not match it — it would fall back to
   an untrained `model_cls()`. Real forward inference requires vendoring the NVlabs
   AFNONet network code into the serving bundle (see Phase 2). The only extra pip deps
   are `timm` + `einops` (already in `requirements-forward.txt`).

---

## Pre-flight (every time)

Cost safety — confirm no always-on compute exists before any AWS write:

```bash
python scripts/check_no_sagemaker_always_on_compute.py --region us-east-1 --profile sbnai-725
python scripts/check_no_fourcastnet_endpoints.py --region us-east-1 --profile sbnai-725
```

---

## Phase 1 — `metadata_only` canary (do this first)

### 1. Package the handler into `model.tar.gz` and upload
Fetch the model assets from S3 into a local dir, then package self-contained so the
**current** handler ships inside `code/`:

```bash
# assets dir must contain: backbone.ckpt, global_means.npy, global_stds.npy
python scripts/package_fourcastnet_model.py \
  --assets-dir <local_assets_dir> \
  --serving-dir src/fourcastnet/serving \
  --layout self-contained
# -> artifacts/fourcastnet/build/model.tar.gz   (code/requirements.txt = minimal set)

aws s3 cp artifacts/fourcastnet/build/model.tar.gz \
  s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v1/model/model.tar.gz \
  --profile sbnai-725 --region us-east-1
```

### 2. Create the CPU SageMaker model (≈$0 — just a metadata record)
```bash
python scripts/describe_or_create_fourcastnet_model.py \
  --config configs/fourcastnet_batch_cpu_poc_v1.json \
  --region us-east-1 --profile sbnai-725
```

### 3. Generate + upload the one-line JSONL manifest
```bash
python scripts/create_fourcastnet_canary_payload.py \
  --input-s3-uri  s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v1/input/ \
  --output-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/batch-transform/fourcastnet-poc/2026-06-06-18z/ \
  --execute-upload --profile sbnai-725
```
The manifest line is `mode=metadata_only`, `max_runtime_guard=true`, and must land at
the config's `input_s3_uri` (`.../fcn-v1/input/canary_manifest.jsonl`).

### 4. Run the Batch Transform — pick one route

**Route A — repo runner (recommended, guardrails, dry-run by default):**
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --config configs/fourcastnet_batch_cpu_poc_v1.json \
  --model-name sbnai-fourcastnet-fcn-v1-cpu-poc \
  --input-s3-uri  s3://.../sagemaker/fourcastnet/fcn-v1/input/canary_manifest.jsonl \
  --output-s3-uri s3://.../sagemaker/batch-transform/fourcastnet-poc/2026-06-06-18z/ \
  --region us-east-1 --profile sbnai-725
# review the dry-run plan, then add --execute to create the TransformJob
```

**Route B — MWAA workflow:** `workflows/fourcastnet_cpu_batch_transform_poc_v2.yaml` is
already on the JSONL `metadata_only` contract. Upload it to your MWAA DAGs prefix and
trigger a run; it waits for the manifest, runs the transform, and verifies output.

### 5. Validate (the gate)
```bash
python scripts/validate_fourcastnet_batch_output.py \
  --uri s3://.../fourcastnet-poc/2026-06-06-18z/canary_manifest.jsonl.out \
  --profile sbnai-725
```
Expect `ok=true` → verdict **`PASSED_CANARY`**. This confirms IAM + S3 read/write + CPU
container init. **Do not proceed to Phase 2 until this passes.**

---

## Phase 2 — real `forward` inference on CPU

> **Prerequisite (handler work).** The shipped `backbone.ckpt` is the NVlabs AFNONet
> checkpoint (gap #9). The current handler's `_probe_backend`/`load_from_checkpoint`
> path targets Modulus/PhysicsNeMo classes and will **not** load it, so before Phase 2
> can prove `fourcastnet_proven=true` the serving handler must gain native AFNONet
> support:
> 1. Vendor `networks/afnonet.py` and `utils/img_utils.py` from NVlabs/FourCastNet into
>    `src/fourcastnet/serving/` (BSD-3 — retain the copyright header).
> 2. In `inference.py`, add a forward path that builds `AFNONet` with
>    `img_size=(720,1440), patch_size=(8,8), in_chans=20, out_chans=20, embed_dim=768,
>    depth=12`, loads `checkpoint['model_state']` with the `module.` prefix stripped,
>    normalizes the input with `global_means/global_stds` (20-channel order), runs the
>    forward pass, and denormalizes.
> `requirements-forward.txt` already pins the needed deps (`timm`, `einops`).

### 1. Build and upload a forward-capable artifact
```bash
python scripts/package_fourcastnet_model.py \
  --assets-dir <local_assets_dir> \
  --requirements-file requirements-forward.txt \
  --layout self-contained
# code/requirements.txt now includes timm + einops (heavier startup).
# NOTE: package_fourcastnet_model.py currently bundles only inference.py + the chosen
# requirements file. Extend it (small follow-up) to also copy the vendored AFNONet
# modules (afnonet.py, img_utils.py) into code/ so they ship in model.tar.gz.

aws s3 cp artifacts/fourcastnet/build/model.tar.gz \
  s3://.../sagemaker/fourcastnet/fcn-v1/model/model-forward.tar.gz \
  --profile sbnai-725 --region us-east-1
```
Create a **new** model (e.g. `sbnai-fourcastnet-fcn-v1-cpu-forward`) pointing at the
forward artifact — keep the canary model intact.

### 2. Manifest for real inference
Set `"mode": "forward"` and `"max_runtime_guard": false` (the real tensor exceeds the
50M-element guard), pointing at the real S3 tensor.

### 3. Run on a larger CPU instance
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --config configs/fourcastnet_batch_cpu_poc_v1.json \
  --model-name sbnai-fourcastnet-fcn-v1-cpu-forward \
  --instance-type ml.m5.2xlarge \
  --input-s3-uri  s3://.../forward_manifest.jsonl \
  --output-s3-uri s3://.../forward-out/ \
  --region us-east-1 --profile sbnai-725 --execute
```

### 4. Validate
Success signal is `fourcastnet_proven=true` in the output report. If load fell back to
`model_cls()`, the weights are untrained — revisit the backend pin (gap #9).

---

## Related docs
- `docs/fourcastnet_batch_transform.md` — Batch Transform vs endpoints, contract audit.
- `docs/team_runbook.md` — env vars, Athena source, profile/region.
- `docs/sagemaker_cost_safety.md` — the no-endpoint cost policy.
