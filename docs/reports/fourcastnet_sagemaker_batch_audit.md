REALTIME_ENDPOINT_COST_RISK

# FourCastNet SageMaker Batch Audit

## Files inspected

- `scripts/experimental_deploy_realtime_endpoint.py`
- `src/deploy_endpoint.py`
- `scripts/describe_or_create_fourcastnet_model.py`
- `scripts/run_fourcastnet_batch_transform.py`
- `scripts/check_no_fourcastnet_endpoints.py`
- `src/run_batch_transform_smoketest.py`
- `scripts/run_quicksight_abrupt_batch_smoke.py`
- `scripts/deploy_quicksight_abrupt_batch_model.py`
- `docs/fourcastnet_batch_inference.md`
- `configs/fourcastnet_batch_v0.json`
- Notebooks: `notebooks/*.ipynb` (no `.deploy()` or `create_endpoint` found via search)

## Endpoint creation evidence (cost-risk paths)

**Experimental real-time endpoint script**:

```
scripts/experimental_deploy_realtime_endpoint.py:L110-L121
    response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
            }
        ],
    )

scripts/experimental_deploy_realtime_endpoint.py:L125-L129
    response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
```

**Default endpoint script disabled**:

```
src/deploy_endpoint.py:L8-L15
    print("ERROR: Real-time endpoint deployment is experimental and disabled here.", file=sys.stderr)
    print(
        f"Use: python {experimental_script} --allow-realtime-endpoint",
        file=sys.stderr,
    )
```

## Batch transform evidence (existing + new)

**Existing smoke test**:

```
src/run_batch_transform_smoketest.py:L148-L195
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={...},
        Tags=tags,
    )
    ...
    sm_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=model_name,
        ...
    )
```

**QuickSight batch smoke**:

```
scripts/run_quicksight_abrupt_batch_smoke.py:L55-L75
    sm_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=args.model_name,
        TransformInput={...},
        TransformOutput={...},
        TransformResources={...},
        Tags=[...],
    )
```

## Current recommended execution path (batch-first)

1. `python scripts/describe_or_create_fourcastnet_model.py --config configs/fourcastnet_batch_v0.json`
2. `python scripts/run_fourcastnet_batch_transform.py --model-name <name> --input-s3-uri <s3> --output-s3-uri <s3> --region us-east-1 --profile sbnai-725 --wait`
3. `python scripts/check_no_fourcastnet_endpoints.py --region us-east-1 --profile sbnai-725`

## Current cost-risk paths

- `scripts/experimental_deploy_realtime_endpoint.py` (creates real-time endpoints; gated by `--allow-realtime-endpoint`).
- Any direct use of SageMaker `.deploy()` in custom notebooks (none found in repo search at audit time).

## What changed in this audit

- Added batch-first scripts: `describe_or_create_fourcastnet_model.py`, `run_fourcastnet_batch_transform.py`, `check_no_fourcastnet_endpoints.py`.
- Added config: `configs/fourcastnet_batch_v0.json`.
- Added documentation: `docs/fourcastnet_batch_inference.md`.
- Added static guard test: `tests/test_no_realtime_endpoint_creation.py`.
- Moved endpoint deployment to experimental-only script and disabled default `src/deploy_endpoint.py`.

