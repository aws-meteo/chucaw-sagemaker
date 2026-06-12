REALTIME_ENDPOINT_COST_RISK

# SageMaker Cost Safety Policy

## Incident summary
- On May 27-29, 2026, endpoint `sbnai-fourcastnet-fcn-v0-2026-05-27-19-10-47-634` (`ml.g4dn.xlarge`) stayed active for about 41.8 hours.
- Observed cost was about 30 USD in Cost Explorer.
- FourCastNet inputs are about 1 GB, so real-time endpoint defaults are not acceptable.

## Resource types and billing behavior
- SageMaker Model: metadata reference to container + artifact. No idle compute billing.
- EndpointConfig: deployment config for endpoint variants. No idle compute billing by itself.
- Endpoint: always-on hosting surface for real-time or async endpoints. Can bill while idle.
- Batch Transform Job: ephemeral offline inference job. Bills while the job runs.
- Notebook Instance: managed VM. Bills while running.
- Studio App: managed app runtime (JupyterServer/KernelGateway/etc.). Can keep compute and storage active.
- Studio gp3 volume: persistent storage that can keep billing (`Studio:VolumeUsage.gp3`, `UnifiedStudio:VolumeUsage.gp3`) even without active inference.

## Required operating pattern for FourCastNet
- Allowed default: `CreateModel` + `CreateTransformJob` (batch-only).
- Forbidden default: `.deploy()` and endpoint creation workflows (`CreateEndpoint`, `CreateEndpointConfig`).
- Do not use real-time endpoints unless explicitly approved for an experiment with cost acknowledgment.

## Minimum verification commands
```bash
aws sagemaker list-endpoints
aws sagemaker list-notebook-instances
aws sagemaker list-apps
aws sagemaker list-training-jobs --status-equals InProgress
aws sagemaker list-processing-jobs --status-equals InProgress
aws sagemaker list-transform-jobs --status-equals InProgress
```

## Cost Explorer checks
- Look for `USE1-Host:ml.g4dn.xlarge / RunInstance`.
- Look for `Studio:VolumeUsage.gp3`.
- Look for `UnifiedStudio:VolumeUsage.gp3`.

## Safe workflow reminder
- Athena is extraction-only.
- Online hosting is not the default path for FourCastNet.
- Before any AWS write operation, run `scripts/check_no_sagemaker_always_on_compute.py`.
