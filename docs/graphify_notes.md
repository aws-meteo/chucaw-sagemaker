# Graphify Notes

Run Graphify only after baseline cleanup/documentation is done.

## Scope

Focus Graphify analysis on:

- `src/`
- `inference/`
- `training/`
- `docs/`

Exclude generated/non-source paths:

- `artifacts/`
- `tmp_model/`
- `__pycache__/`
- `.ipynb_checkpoints/`

## Suggested questions for the repo graph

1. How does data move from Athena to endpoint?
2. Which scripts create S3 artifacts?
3. Which environment variables control deployment?
4. Which files are diagnostic-only?
5. Which files are operational baseline?

Graphify should document dependencies and flow clarity, not add runtime complexity.
