REALTIME_ENDPOINT_COST_RISK

# SageMaker Real-time Endpoint Risk Audit

## Scope

Inspected default repository paths requested by the cleanup task:

- `scripts/`
- `src/`
- `notebooks/`
- `configs/`
- `examples/`
- `docs/`
- `README.md`
- `AGENTS.md`

Also inspected `experimental/realtime_endpoint_dangerous/` because quarantined endpoint code is allowed only there.

## Exact Search Commands Used

```powershell
rg -n -S --glob '!*.pyc' --glob '!*.png' --glob '!*.jpg' --glob '!*.parquet' --glob '!model.tar.gz' -e '\.deploy\(' -e '\bdeploy\(' -e 'create_endpoint' -e 'CreateEndpoint' -e 'create_endpoint_config' -e 'CreateEndpointConfig' -e 'EndpointConfig' -e 'Predictor\(' -e 'RealTimePredictor' -e 'sagemaker\.predictor' -e 'SKLearnModel\.deploy' -e 'PyTorchModel\.deploy' -e 'TensorFlowModel\.deploy' -e 'HuggingFaceModel\.deploy' -e 'Estimator\.deploy' -e 'endpoint_name' -e 'create-endpoint' -e 'create-endpoint-config' -e 'instance_type\s*=\s*["'']ml\.(g4dn|p|g|c|m)' scripts src notebooks configs examples docs README.md AGENTS.md experimental
```

```powershell
rg -n -S --glob '!*.pyc' --glob '!*.png' --glob '!*.jpg' --glob '!*.parquet' --glob '!model.tar.gz' -e 'invoke_endpoint' -e 'invoke-endpoint' -e 'invoke_endpoint_async' -e 'invoke-endpoint-async' -e 'sagemaker-runtime' scripts src notebooks configs examples docs README.md AGENTS.md experimental
```

```powershell
rg -n -S --glob '!*.pyc' --glob '!*.png' --glob '!*.jpg' --glob '!*.parquet' --glob '!model.tar.gz' -e 'create_notebook_instance' -e 'CreateNotebookInstance' -e 'create_app' -e 'CreateApp' -e 'delete_app' -e 'DeleteApp' -e 'NotebookInstance' -e 'Studio' -e 'UnifiedStudio' -e 'create_training_job' -e 'create_processing_job' -e 'create_transform_job' scripts src notebooks configs examples docs README.md AGENTS.md
```

## Files Inspected

File inventory command:

```powershell
rg --files scripts src notebooks configs examples docs README.md AGENTS.md
```

The command returned 91 text/code/documentation paths. The audit targeted every match in those paths plus the quarantined `experimental/realtime_endpoint_dangerous/` directory.

## Findings

| File | Line | Snippet | Classification | Action |
|---|---:|---|---|---|
| `src/deploy_endpoint.py` | 18 | `Real-time endpoints are disabled for FourCastNet by default. Use Batch Transform.` | `DANGEROUS_DEFAULT` before cleanup | Replaced endpoint deploy entrypoint with hard-fail guard. |
| `src/invoke_endpoint.py` | n/a | real-time endpoint invocation entrypoint | `DANGEROUS_DEFAULT` before cleanup | Deleted from the default FourCastNet path. Batch Transform jobs write results to S3 instead. |
| `src/fourcastnet/prepare_async_endpoint_payloads.py` | 17 | `Real-time endpoints are disabled for FourCastNet by default. Use Batch Transform.` | `LEGACY_DELETE_OR_QUARANTINE` | Default path now hard-fails; endpoint payload generator remains only under experimental path. |
| `src/fourcastnet/invoke_async_from_notebook.py` | 8, 10 | `async endpoint invocation helpers are disabled` / `Use Batch Transform.` | `DANGEROUS_DEFAULT` before cleanup | Replaced async endpoint invocation helper with hard-fail guard. |
| `src/fourcastnet/prepare_async_invoke_payload.py` | 8, 10 | `async endpoint payload helpers are disabled` / `Use Batch Transform.` | `DANGEROUS_DEFAULT` before cleanup | Replaced async endpoint command generator with hard-fail guard. |
| `src/fourcastnet/prepare_async_autoscaling_payloads.py` | 8, 10 | `async endpoint autoscaling helpers are disabled` / `Use Batch Transform.` | `DANGEROUS_DEFAULT` before cleanup | Replaced endpoint autoscaling command generator with hard-fail guard. |
| `scripts/experimental_deploy_realtime_endpoint.py` | 16 | `Real-time endpoints are disabled for FourCastNet by default.` | `LEGACY_DELETE_OR_QUARANTINE` | Default script path is a hard-fail stub. |
| `experimental/realtime_endpoint_dangerous/experimental_deploy_realtime_endpoint.py` | 4, 109, 125 | `REALTIME_ENDPOINT_COST_RISK` / `create_endpoint_config` / `create_endpoint` | `EXPERIMENTAL_ALLOWED_BUT_MUST_BE_GATED` | Quarantined experimental path. Requires `--allow-realtime-endpoint`, `--i-understand-this-can-cost-money`, explicit `--endpoint-name`, and explicit `--instance-type`. |
| `experimental/realtime_endpoint_dangerous/prepare_async_endpoint_payloads.py` | 2, 135, 139 | `REALTIME_ENDPOINT_COST_RISK` / `create-endpoint-config` / `create-endpoint` | `EXPERIMENTAL_ALLOWED_BUT_MUST_BE_GATED` | Quarantined experimental path. Requires dual danger flags and explicit endpoint/instance settings. |
| `scripts/generate_sagemaker_cleanup_commands.py` | 183 | `delete-endpoint --endpoint-name` | `SAFE_DOC_ONLY` equivalent local generator | Generates a local PowerShell cleanup script; no deletion occurs unless generated script is run with `-Execute`. |
| `scripts/deploy_quicksight_abrupt_batch_model.py` | 54 | `instance_type="ml.m5.large"` | `UNKNOWN_REQUIRES_HUMAN_REVIEW` | SageMaker Model creation for batch use; no endpoint creation call found. |
| `scripts/register_quicksight_abrupt_model_package.py` | 69 | `instance_type="ml.m5.large"` | `UNKNOWN_REQUIRES_HUMAN_REVIEW` | Model Registry metadata; no endpoint creation call found. |
| `scripts/run_fourcastnet_batch_transform.py` | 141 | `create_transform_job` | `SAFE_DOC_ONLY` for this endpoint audit | Batch Transform job creation is not endpoint hosting and is gated by `--execute`; not run during this cleanup. |
| `src/run_batch_transform_smoketest.py` | 178 | `create_transform_job` | `UNKNOWN_REQUIRES_HUMAN_REVIEW` | Batch Transform smoke script can create an ephemeral transform job; not endpoint hosting. |
| `docs/fourcastnet_batch_inference.md` | 3, 26 | `REALTIME_ENDPOINT_COST_RISK` / `Do not use .deploy()` | `SAFE_DOC_ONLY` | Documentation warning only. |
| `docs/fourcastnet_batch_transform.md` | 3, 61 | `REALTIME_ENDPOINT_COST_RISK` / `strictly forbids .deploy(), create_endpoint` | `SAFE_DOC_ONLY` | Documentation warning only. |
| `docs/llm_wiki/README.md` | 1, 43 | `REALTIME_ENDPOINT_COST_RISK` / `Do not call create_endpoint` | `SAFE_DOC_ONLY` | Documentation warning only. |
| `docs/sagemaker_cost_safety.md` | 1, 21 | `REALTIME_ENDPOINT_COST_RISK` / forbidden default mentions endpoint APIs | `SAFE_DOC_ONLY` | Project policy document created/updated. |
| `docs/reports/fourcastnet_async_endpoint_manual_commands.md` | 1, 26, 31 | warning marker / `create-endpoint-config` / `create-endpoint` | `DANGEROUS_EXAMPLE` | Historical report retained only under docs with warning marker. |
| `docs/reports/fourcastnet_async_scale_to_zero_ready_report.md` | 1, 53, 56 | warning marker / endpoint creation commands | `DANGEROUS_EXAMPLE` | Historical report retained only under docs with warning marker. |
| `docs/reports/fourcastnet_sagemaker_batch_audit.md` | 1, 25, 38 | warning marker / endpoint API snippets | `DANGEROUS_EXAMPLE` | Historical audit retained only under docs with warning marker. |
| `README.md` | 3, 9-11, 178 | warning marker / batch-only policy / endpoint commands intentionally fail | `SAFE_DOC_ONLY` | Updated from endpoint workflow to batch-only safety guidance. |
| `docs/training_baseline.md` | 3, 5-9, 29-30 | warning marker / endpoint sequence disabled / batch command | `SAFE_DOC_ONLY` | Updated from endpoint workflow to batch-only safety guidance. |
| `docs/team_runbook.md` | 3, 7, 120 | warning marker / batch-only baseline / endpoint commands intentionally fail | `SAFE_DOC_ONLY` | Updated from endpoint workflow to batch-only safety guidance. |
| `AGENTS.md` | 65-80 | strict SageMaker cost safety policy | `SAFE_DOC_ONLY` | Agent instructions include no real-time endpoints by default and preflight-before-write policy. |
| `tests/test_no_realtime_endpoint_defaults.py` | 9-26, 102-104 | dangerous endpoint patterns and warning marker requirement | `SAFE_DOC_ONLY` | Static guard added; fails on endpoint creation/invocation/default notebook/app creation patterns in default paths. |

## Paths That Could Recreate a Real-time Endpoint

Only quarantined experimental paths can create or generate creation commands:

- `experimental/realtime_endpoint_dangerous/experimental_deploy_realtime_endpoint.py:109`
- `experimental/realtime_endpoint_dangerous/experimental_deploy_realtime_endpoint.py:125`
- `experimental/realtime_endpoint_dangerous/prepare_async_endpoint_payloads.py:135`
- `experimental/realtime_endpoint_dangerous/prepare_async_endpoint_payloads.py:139`

These paths carry `REALTIME_ENDPOINT_COST_RISK` and require explicit danger flags.

## Files Safe As Documentation Only

- `README.md:3`
- `docs/training_baseline.md:3`
- `docs/team_runbook.md:3`
- `docs/fourcastnet_batch_inference.md:3`
- `docs/fourcastnet_batch_transform.md:3`
- `docs/llm_wiki/README.md:1`
- `docs/sagemaker_cost_safety.md:1`
- `docs/reports/fourcastnet_async_endpoint_manual_commands.md:1`
- `docs/reports/fourcastnet_async_scale_to_zero_ready_report.md:1`
- `docs/reports/fourcastnet_sagemaker_batch_audit.md:1`

## Files Changed

- `src/deploy_endpoint.py`
- `src/fourcastnet/prepare_async_endpoint_payloads.py`
- `src/fourcastnet/invoke_async_from_notebook.py`
- `src/fourcastnet/prepare_async_invoke_payload.py`
- `src/fourcastnet/prepare_async_autoscaling_payloads.py`
- `scripts/experimental_deploy_realtime_endpoint.py`
- `scripts/check_no_sagemaker_always_on_compute.py`
- `scripts/generate_sagemaker_cleanup_commands.py`
- `tests/test_no_realtime_endpoint_defaults.py`
- `docs/sagemaker_cost_safety.md`
- `docs/reports/sagemaker_realtime_endpoint_risk_audit.md`
- `README.md`
- `docs/training_baseline.md`
- `docs/team_runbook.md`
- `AGENTS.md`

## Unresolved

- `scripts/deploy_quicksight_abrupt_batch_model.py:54` and `scripts/register_quicksight_abrupt_model_package.py:69` still reference `ml.m5.large` for non-endpoint batch/model-registry image metadata. They do not create endpoints, but are left classified as `UNKNOWN_REQUIRES_HUMAN_REVIEW`.
- Batch Transform implementation remains a future step. This cleanup intentionally did not launch training, processing, transform, notebook, Studio, endpoint, or endpoint-config resources.
