#!/usr/bin/env python3
import argparse
import io
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import pandas as pd
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
import os


DEFAULT_YEAR = "2026"
DEFAULT_MONTH = "04"
DEFAULT_DAY = "09"
DEFAULT_HOUR = "18"
EXPECTED_COLUMNS = {"latitude", "longitude", "value"}


def parse_args():
    parser = argparse.ArgumentParser(description="Query Athena and materialize t2m snapshot CSV")
    parser.add_argument("--year", default=DEFAULT_YEAR, help="Partition year (default: 2026)")
    parser.add_argument("--month", default=DEFAULT_MONTH, help="Partition month (default: 04)")
    parser.add_argument("--day", default=DEFAULT_DAY, help="Partition day (default: 09)")
    parser.add_argument("--hour", default=DEFAULT_HOUR, help="Partition hour (default: 18)")
    return parser.parse_args()


def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_query(database, table, year, month, day, hour):
    return f"""
SELECT latitude, longitude, value
FROM {database}.{table}
WHERE variable      = 't'
  AND isobaricInhPa = 1000.0
  AND year  = '{year}'
  AND month = '{month}'
  AND day   = '{day}'
  AND hour  = '{hour}'
""".strip()


def parse_s3_uri(s3_uri):
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    database = required_env("ATHENA_DATABASE")
    table = required_env("ATHENA_TABLE")
    output_bucket = required_env("ATHENA_OUTPUT_BUCKET")
    output_prefix = required_env("ATHENA_OUTPUT_PREFIX").strip("/")

    print(
        f"Executing Athena extraction for partitions: year={args.year}, month={args.month}, "
        f"day={args.day}, hour={args.hour}"
    )
    output_location = f"s3://{output_bucket}/{output_prefix}/"
    query = build_query(database, table, args.year, args.month, args.day, args.hour)

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    athena = session.client("athena")
    s3 = session.client("s3")

    try:
        start = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": output_location},
        )
        query_id = start["QueryExecutionId"]
        print(f"Athena QueryExecutionId: {query_id}")

        final_state = None
        failure_reason = None
        for _ in range(300):
            execution = athena.get_query_execution(QueryExecutionId=query_id)
            status = execution["QueryExecution"]["Status"]
            final_state = status["State"]
            if final_state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                failure_reason = status.get("StateChangeReason")
                break
            time.sleep(2)

        if final_state != "SUCCEEDED":
            raise RuntimeError(
                f"Athena query failed with state={final_state}. "
                f"Reason: {failure_reason or 'not provided'}"
            )

        result_uri = execution["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        result_bucket, result_key = parse_s3_uri(result_uri)
        response = s3.get_object(Bucket=result_bucket, Key=result_key)
        body = response["Body"].read()
        df = pd.read_csv(io.BytesIO(body))
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"Athena extraction error for partitions "
            f"{args.year}-{args.month}-{args.day} {args.hour}:00: {exc}"
        ) from exc

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Athena result missing required columns: "
            + ", ".join(missing_columns)
            + ". Expected: latitude, longitude, value"
        )

    df = df[["latitude", "longitude", "value"]].dropna()
    if df.empty:
        raise ValueError(
            "Athena query returned zero usable rows after filtering/selecting latitude, longitude, value"
        )

    out_path = repo_root / "data" / "t2m_snapshot.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Rows materialized: {len(df)}")
    print(f"Snapshot written to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
