#!/usr/bin/env python3
import argparse
import json
import sys
import boto3
from urllib.parse import urlparse
from pathlib import Path

def validate_json(data):
    if not isinstance(data, dict):
        return "FAIL", "Invalid format (not an object)"
    
    if data.get("ok"):
        return "SUCCESS", "Execution completed normally"
    
    reason = data.get("reason", "")
    err_type = data.get("error_type", "")
    if "backend_unavailable" in reason or "backend_model_instantiation_failed" in reason:
        return "CONTROLLED_ERROR", f"Expected backend failure: {reason}"
    if err_type:
        return "ERROR", f"Failed with {err_type}: {data.get('error_message')}"
    return "FAIL", f"Unknown failure: {data}"

def main():
    parser = argparse.ArgumentParser(description="Validate FourCastNet batch output")
    parser.add_argument("--uri", required=True, help="S3 URI or local file path")
    parser.add_argument("--profile", default="")
    args = parser.parse_args()

    lines = []
    if args.uri.startswith("s3://"):
        session_kwargs = {}
        if args.profile:
            session_kwargs["profile_name"] = args.profile
        s3 = boto3.Session(**session_kwargs).client("s3")
        parsed = urlparse(args.uri)
        try:
            resp = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
            content = resp["Body"].read().decode("utf-8")
            lines = [l for l in content.split("\n") if l.strip()]
        except Exception as e:
            print(f"Error fetching S3 object: {e}")
            sys.exit(1)
    else:
        path = Path(args.uri)
        if not path.exists():
            print("Local file not found.")
            sys.exit(1)
        with open(path, "r") as f:
            lines = [l for l in f.read().split("\n") if l.strip()]

    if not lines:
        print("No records found.")
        sys.exit(1)

    print(f"Validating {len(lines)} records...")
    verdicts = []
    for line in lines:
        try:
            data = json.loads(line)
            verdict, msg = validate_json(data)
            print(f"Record: {verdict} - {msg}")
            verdicts.append(verdict)
        except Exception as e:
            print(f"Record: FAIL - invalid JSON: {e}")
            verdicts.append("FAIL")

    if all(v in ["SUCCESS", "CONTROLLED_ERROR"] for v in verdicts):
        print("FINAL VERDICT: PASSED_CANARY")
    else:
        print("FINAL VERDICT: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
