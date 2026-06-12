#!/usr/bin/env python3
import argparse
import json
import uuid
import sys
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate canary payload for FourCastNet")
    parser.add_argument("--output-dir", default="canary_payloads")
    parser.add_argument("--input-s3-uri", required=True)
    parser.add_argument("--output-s3-uri", required=True)
    parser.add_argument("--max-runtime-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", default="metadata_only")
    parser.add_argument("--execute-upload", action="store_true")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    tensor = np.zeros((1, 10, 10), dtype=np.float32)
    tensor_path = outdir / "canary_tensor.npy"
    np.save(tensor_path, tensor, allow_pickle=False)

    tensor_s3_uri = args.input_s3_uri.rstrip("/") + "/canary_tensor.npy"
    
    payload = {
        "request_id": str(uuid.uuid4()),
        "mode": args.mode,
        "input_s3_uri": tensor_s3_uri,
        "output_s3_uri": args.output_s3_uri,
        "max_runtime_guard": args.max_runtime_guard
    }

    manifest_path = outdir / "canary_manifest.jsonl"
    with open(manifest_path, "w") as f:
        f.write(json.dumps(payload) + "\n")
    
    print(f"Created local canary artifacts in {outdir}")

    if args.execute_upload:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            print("ERROR: boto3 is required only for --execute-upload.", file=sys.stderr)
            raise SystemExit(1) from exc

        print("Uploading to S3...")
        session_kwargs = {}
        if args.profile:
            session_kwargs["profile_name"] = args.profile
        s3 = boto3.Session(**session_kwargs).client("s3")
        
        bucket, _, key = tensor_s3_uri[5:].partition("/")
        s3.upload_file(str(tensor_path), bucket, key)
        
        man_bucket, _, man_key = args.input_s3_uri[5:].partition("/")
        man_key = man_key.rstrip("/") + "/canary_manifest.jsonl"
        s3.upload_file(str(manifest_path), man_bucket, man_key)
        print("Uploaded successfully.")
    else:
        print("Skipped upload. Use --execute-upload to upload.")

if __name__ == "__main__":
    main()
