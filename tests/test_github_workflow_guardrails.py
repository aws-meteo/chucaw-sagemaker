from pathlib import Path


WORKFLOW = Path(".github/workflows/fourcastnet-model-deploy.yml")


def test_workflow_keeps_deploy_batch_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "create-endpoint" not in lowered
    assert "create_endpoint" not in lowered
    assert ".deploy(" not in text
    assert "--execute" not in text
    assert "check_no_sagemaker_always_on_compute.py" in text
    assert "check_no_fourcastnet_endpoints.py" in text
    assert "Expected cost mode: batch-only" in text
    for needle in (
        "vars.",
        "secrets.AWS_ROLE_TO_ASSUME",
        "secrets.FCN_ASSETS_S3_URI",
        "secrets.FCN_MODEL_S3_PREFIX",
        "secrets.FCN_TEST_INPUT_S3_URI",
        '--model-name "$MODEL_NAME"',
    ):
        assert (needle in text) is (needle != "vars.")


def test_workflow_deploys_only_after_manual_main_dispatch():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "push:" not in text
    assert "pull_request:" not in text
    assert "github.event_name == 'push'" not in text
