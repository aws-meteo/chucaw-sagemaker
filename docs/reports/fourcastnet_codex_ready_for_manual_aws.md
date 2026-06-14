# FourCastNet Codex Ready for Manual AWS

## Scope Outcome
- FourCastNet SageMaker preparation path is now local-first and manual-AWS-ready.
- No AWS commands were executed by Codex.
- No SageMaker job was created by Codex.

## Files Added or Updated
- Added: `src/fourcastnet/prepare_model_registry_payloads.py`
- Added: `src/fourcastnet/prepare_processing_job_payload.py`
- Updated: `src/fourcastnet/run_processing_gpu_dryrun.py`
- Updated: `src/fourcastnet/build_fcn_hosting_artifact.py`
- Updated: `docs/reports/fourcastnet_execution_wave1_runbook.md`
- Added: `docs/reports/fourcastnet_manual_aws_commands.md`

## Local Validations Run
- Ran `py_compile` with `C:\ProgramData\miniconda3\envs\aws_backend\python.exe` for all `src/fourcastnet/**/*.py`.
- Result: pass.
- Parsed generated JSON payloads under `artifacts/fourcastnet/aws_payloads/`.
- Result: pass for:
  - `model_package_group.json`
  - `model_package.json`
  - `processing_job.json`
- Checked required docs existence:
  - `docs/reports/fourcastnet_manual_aws_commands.md` exists.
  - `docs/reports/fourcastnet_execution_wave1_runbook.md` exists.

## Payload Generation Status
- Generated local payload files:
  - `artifacts/fourcastnet/aws_payloads/model_package_group.json`
  - `artifacts/fourcastnet/aws_payloads/model_package.json`
  - `artifacts/fourcastnet/aws_payloads/processing_job.json`
- Payloads were generated locally only; no AWS API call was made by these scripts.

## Baseline Diff Check
- Pre-existing baseline modifications were detected before FCN hardening work and left untouched:
  - `scripts/register_quicksight_abrupt_model_package.py`
  - `src/build_hosting_artifact.py`
- Additional unrelated pre-existing modifications also remain in the working tree and were not reverted.

## Command Sheet Status
- Manual command sheet present at:
  - `docs/reports/fourcastnet_manual_aws_commands.md`

## Next Human Command (Run First)
```powershell
aws sso login --profile sbnai-725
```
