import argparse
import sys


def resolve_args(required: list[str], optional: list[str] | None = None) -> dict[str, str]:
    """Resolve Glue job arguments, supporting both required and optional parameters.

    ``getResolvedOptions`` treats every key in its list as required, so we only
    pass truly-required args to it and handle optional args separately via
    ``parse_known_args``, which safely ignores Glue's internal injected arguments
    (``--JOB_ID``, ``--JOB_RUN_ID``, etc.) that would otherwise crash ``parse_args``.
    """
    optional = optional or []

    try:
        from awsglue.utils import getResolvedOptions  # type: ignore

        # getResolvedOptions marks every key it receives as REQUIRED.
        # Only pass the truly required subset; optional args are handled below.
        result: dict[str, str] = {}
        if required:
            values = getResolvedOptions(sys.argv, required)
            result = {k: values[k] for k in required}

        # Parse optional args with parse_known_args so Glue's injected internal
        # args (--JOB_ID, --enable-metrics, etc.) are silently ignored.
        if optional:
            opt_parser = argparse.ArgumentParser()
            for name in optional:
                opt_parser.add_argument(f"--{name}", required=False, default="")
            parsed_opt, _ = opt_parser.parse_known_args(sys.argv[1:])
            result.update(vars(parsed_opt))

        return result

    except Exception:
        # Fallback for local / non-Glue environments.
        # Use parse_known_args to tolerate any extra flags on the command line.
        parser = argparse.ArgumentParser()
        for name in required:
            parser.add_argument(f"--{name}", required=True)
        for name in optional:
            parser.add_argument(f"--{name}", required=False, default="")
        args, _ = parser.parse_known_args()
        return vars(args)
