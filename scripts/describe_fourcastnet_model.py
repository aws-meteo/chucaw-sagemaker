#!/usr/bin/env python3
import argparse
import json
import sys
import boto3

def main():
    parser = argparse.ArgumentParser(description="Describe FourCastNet SageMaker Model")
    parser.add_argument("--profile", type=str, default="sbnai-725")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--json-output", action="store_true")
    args = parser.parse_args()

    session_kwargs = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    
    session = boto3.Session(**session_kwargs)
    client = session.client("sagemaker")
    
    try:
        response = client.describe_model(ModelName=args.model_name)
        if args.json_output:
            print(json.dumps(response, default=str, indent=2))
        else:
            print(f"ModelName: {response.get('ModelName')}")
            print(f"ExecutionRoleArn: {response.get('ExecutionRoleArn')}")
            container = response.get('PrimaryContainer', {})
            print(f"PrimaryContainer.Image: {container.get('Image')}")
            print(f"PrimaryContainer.ModelDataUrl: {container.get('ModelDataUrl')}")
            print(f"PrimaryContainer.Environment: {container.get('Environment')}")
            print(f"CreationTime: {response.get('CreationTime')}")
    except client.exceptions.ClientError as e:
        print(f"Error describing model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
