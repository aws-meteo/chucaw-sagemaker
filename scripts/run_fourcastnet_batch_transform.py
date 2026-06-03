#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime

DEFAULT_PROFILE = "sbnai-725"
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_CONTENT_TYPE = "application/jsonlines"
DEFAULT_ACCEPT = "application/jsonlines"


def _apply_config_defaults(args, config_data):
    if not args.model_name and "model_name" in config_data:
        args.model_name = config_data["model_name"]
    if args.profile == DEFAULT_PROFILE and "profile" in config_data:
        args.profile = config_data["profile"]
    if args.region == DEFAULT_REGION and "region" in config_data:
        args.region = config_data["region"]
    if args.instance_type == DEFAULT_INSTANCE_TYPE and "default_transform_instance_type" in config_data:
        args.instance_type = config_data["default_transform_instance_type"]
    if args.instance_count == DEFAULT_INSTANCE_COUNT and "default_transform_instance_count" in config_data:
        args.instance_count = int(config_data["default_transform_instance_count"])
    if args.content_type == DEFAULT_CONTENT_TYPE and "content_type" in config_data:
        args.content_type = config_data["content_type"]
    if args.accept == DEFAULT_ACCEPT and "accept" in config_data:
        args.accept = config_data["accept"]
    return config_data.get("tags") or {}


def main():
    parser = argparse.ArgumentParser(description="Run SageMaker Batch Transform for FourCastNet")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--model-name", type=str, help="Required if not in config")
    parser.add_argument("--input-s3-uri", type=str, required=True)
    parser.add_argument("--output-s3-uri", type=str, required=True)
    parser.add_argument("--instance-type", type=str, default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--instance-count", type=int, default=DEFAULT_INSTANCE_COUNT)
    parser.add_argument("--job-name-prefix", type=str, default="fcn-batch")
    parser.add_argument("--content-type", type=str, default=DEFAULT_CONTENT_TYPE)
    parser.add_argument("--accept", type=str, default=DEFAULT_ACCEPT)
    parser.add_argument("--strategy", type=str, default="SingleRecord")
    parser.add_argument("--split-type", type=str, default="Line")
    parser.add_argument("--assemble-with", type=str, default="Line")
    parser.add_argument("--max-concurrent-transforms", type=int, default=1)
    parser.add_argument("--max-payload-mb", type=int, default=100)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--tags", type=str, help="Additional tags as JSON string")
    parser.add_argument("--execute", action="store_true", help="Actual AWS job creation requires --execute")
    parser.add_argument("--allow-large-direct-payload", action="store_true", help="Allow x-npy without S3 pointer")
    parser.add_argument("--allow-concurrency", action="store_true", help="Allow concurrent transforms")
    args = parser.parse_args()

    config_tags = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config_data = json.load(f)
                config_tags = _apply_config_defaults(args, config_data)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.model_name:
        parser.error("--model-name is required (either via CLI or config file)")

    # Absolute blocker for endpoint APIs
    if "deploy" in sys.argv or "endpoint" in " ".join(sys.argv):
        print("ERROR: Endpoint commands are forbidden.", file=sys.stderr)
        sys.exit(1)

    # Stronger safety checks
    if args.execute:
        if "PLACEHOLDER" in args.input_s3_uri or "PLACEHOLDER" in args.output_s3_uri:
            print("ERROR: Placeholder S3 URIs cannot be used in --execute mode.", file=sys.stderr)
            sys.exit(1)
        if args.content_type == "application/x-npy" and not args.allow_large_direct_payload:
            print("ERROR: application/x-npy requires --allow-large-direct-payload.", file=sys.stderr)
            sys.exit(1)
        if args.max_concurrent_transforms > 1 and not args.allow_concurrency:
            print("ERROR: Concurrency > 1 requires --allow-concurrency.", file=sys.stderr)
            sys.exit(1)
            
    if "json" in args.content_type.lower() and args.strategy == "MultiRecord":
        print("ERROR: BatchStrategy=MultiRecord is not supported for JSON/JSONLines.", file=sys.stderr)
        sys.exit(1)

    print("WARNING: Ensure max_runtime_guard=true in canary JSONL to prevent OOM/cost spikes unless specifically running 1 GB tensors.")

    job_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    job_name = f"{args.job_name_prefix}-{job_id}"

    tags = [
        {"Key": "Project", "Value": "SbnAI"},
        {"Key": "Component", "Value": "FourCastNet"},
        {"Key": "CostMode", "Value": "batch-only"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Owner", "Value": "Fabian"}
    ]
    if isinstance(config_tags, dict):
        for k, v in config_tags.items():
            tags = [tag for tag in tags if tag["Key"] != k]
            tags.append({"Key": k, "Value": str(v)})
    if args.tags:
        try:
            extra_tags = json.loads(args.tags)
            if isinstance(extra_tags, dict):
                for k, v in extra_tags.items():
                    tags = [tag for tag in tags if tag["Key"] != k]
                    tags.append({"Key": k, "Value": v})
            elif isinstance(extra_tags, list):
                tags.extend(extra_tags)
        except Exception as e:
            print(f"Error parsing --tags: {e}", file=sys.stderr)
            sys.exit(1)

    payload = {
        "TransformJobName": job_name,
        "ModelName": args.model_name,
        "MaxConcurrentTransforms": args.max_concurrent_transforms,
        "MaxPayloadInMB": args.max_payload_mb,
        "BatchStrategy": args.strategy,
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": args.input_s3_uri
                }
            },
            "ContentType": args.content_type,
            "SplitType": args.split_type
        },
        "TransformOutput": {
            "S3OutputPath": args.output_s3_uri,
            "Accept": args.accept,
            "AssembleWith": args.assemble_with
        },
        "TransformResources": {
            "InstanceType": args.instance_type,
            "InstanceCount": args.instance_count
        },
        "Tags": tags
    }

    print("=== Planned Transform Job Payload ===")
    print(json.dumps(payload, indent=2))
    print("=====================================")

    if not args.execute:
        print("\n[DRY RUN] Job not created. Use --execute to actually call create-transform-job.")
        sys.exit(0)

    print("\n[EXECUTE] Creating Transform Job...")
    try:
        import boto3
    except ModuleNotFoundError as exc:
        print("ERROR: boto3 is required only for --execute mode.", file=sys.stderr)
        raise SystemExit(1) from exc

    session_kwargs = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    
    session = boto3.Session(**session_kwargs)
    client = session.client("sagemaker")
    
    try:
        response = client.create_transform_job(**payload)
        print(f"Job created! ARN: {response.get('TransformJobArn')}")
    except client.exceptions.ClientError as e:
        print(f"Error creating transform job: {e}", file=sys.stderr)
        sys.exit(1)

    if args.wait:
        print(f"Waiting for job {job_name} to complete...")
        waiter = client.get_waiter("transform_job_completed_or_stopped")
        try:
            waiter.wait(TransformJobName=job_name)
            print("Job finished.")
        except Exception as e:
            print(f"Error waiting for job: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
