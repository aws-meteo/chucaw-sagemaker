from pathlib import Path
import importlib.util


def _load_register_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "register_quicksight_abrupt_model_package.py"
    spec = importlib.util.spec_from_file_location("register_quicksight_abrupt_model_package", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_inference_spec_matches_csv_contract():
    module = _load_register_module()
    image_uri = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    spec = module.build_inference_spec_for_image(
        image_uri=image_uri,
        model_artifact_s3_uri="s3://bucket/path/quicksight_abrupt_model.tar.gz",
    )

    container = spec["Containers"][0]
    assert container["Image"] == image_uri
    assert container["ModelDataUrl"] == "s3://bucket/path/quicksight_abrupt_model.tar.gz"
    assert container["Environment"]["SAGEMAKER_PROGRAM"] == "inference.py"
    assert container["Environment"]["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"
    assert spec["SupportedContentTypes"] == ["text/csv"]
    assert spec["SupportedResponseMIMETypes"] == ["text/csv"]


def test_build_customer_metadata_properties_no_commas():
    module = _load_register_module()
    metadata = module.build_customer_metadata_properties()

    assert metadata["Project"] == "chucaw"
    assert metadata["Component"] == "quicksight-sagemaker-augmentation"
    assert metadata["ModelKind"] == "csv-abrupt-temperature-classifier"
    assert metadata["InferenceMode"] == "BatchTransform"
    assert metadata["EndpointRequired"] == "false"

    # Verify no commas in metadata values
    for key, value in metadata.items():
        assert "," not in value, f"Comma found in metadata key {key}: {value}"

    assert "/" in metadata["InputColumns"]
    assert "/" in metadata["OutputColumns"]
    assert metadata["InputColumns"] == "lat/lon/t2m"
    assert metadata["OutputColumns"] == "abrupt_temp_change_label/abrupt_temp_change_score"
