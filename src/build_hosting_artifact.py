#!/usr/bin/env python3
"""Build hosting artifacts from SageMaker training output.

Official baseline: separate-source serving bundle.

- model artifact (`model.tar.gz`) contains only `model.joblib`
- source artifact (`source.tar.gz`) contains serving code (`inference.py`) and
  minimal runtime requirements (`requirements.txt`)
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import joblib
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


MODEL_TAR_NAME = "model.tar.gz"
SOURCE_TAR_NAME = "source.tar.gz"


def parse_args():
    """
    Analiza los argumentos de la línea de comandos para la construcción de artefactos.

    Returns
    -------
    argparse.Namespace
        Argumentos analizados, incluyendo URIs de S3 o rutas locales.
    """
    parser = argparse.ArgumentParser(
        description="Build separate-source SageMaker hosting artifacts from training artifact"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--training-artifact-s3-uri",
        help="S3 URI to training artifact model.tar.gz produced by a SageMaker training job",
    )
    input_group.add_argument(
        "--training-artifact-local",
        help="Local path to training artifact model.tar.gz produced by a SageMaker training job",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/build",
        help="Local output directory (default: artifacts/build)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload model.tar.gz and source.tar.gz to S3 after local build",
    )
    parser.add_argument(
        "--bucket",
        default="",
        help="Optional bucket used for both uploads when --model-bucket/--source-bucket are not set",
    )
    parser.add_argument("--model-bucket", default="", help="S3 bucket for model artifact upload")
    parser.add_argument("--model-prefix", default="", help="S3 key prefix for model artifact upload")
    parser.add_argument("--source-bucket", default="", help="S3 bucket for source artifact upload")
    parser.add_argument("--source-prefix", default="", help="S3 key prefix for source artifact upload")
    return parser.parse_args()


def required_env(name: str) -> str:
    """
    Obtiene una variable de entorno obligatoria.

    Parameters
    ----------
    name : str
        Nombre de la variable.

    Returns
    -------
    str
        Valor de la variable.

    Raises
    ------
    ValueError
        Si la variable no está definida o está vacía.
    """
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_s3_uri(s3_uri: str):
    """
    Descompone una URI de S3 en bucket y clave.

    Parameters
    ----------
    s3_uri : str
        URI de S3 (ej. s3://bucket/path/to/key).

    Returns
    -------
    tuple[str, str]
        (bucket, key).

    Raises
    ------
    ValueError
        Si la URI no es válida.
    """
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def make_boto3_session(region: str, profile: str):
    """
    Crea una sesión de boto3.

    Parameters
    ----------
    region : str
        Región de AWS.
    profile : str
        Nombre del perfil de AWS (opcional).

    Returns
    -------
    boto3.Session
        Sesión configurada.
    """
    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    return boto3.Session(**session_kwargs)


def download_training_artifact(session, s3_uri: str, destination: Path):
    """
    Descarga el artefacto de entrenamiento desde S3.

    Parameters
    ----------
    session : boto3.Session
        Sesión de AWS.
    s3_uri : str
        Ruta en S3 del archivo model.tar.gz.
    destination : Path
        Ruta local donde se guardará el archivo.

    Raises
    ------
    RuntimeError
        Si ocurre un error durante la descarga.
    """
    bucket, key = parse_s3_uri(s3_uri)
    s3_client = session.client("s3")
    try:
        s3_client.download_file(bucket, key, str(destination))
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to download training artifact from {s3_uri}: {exc}") from exc


def extract_tarball(tar_path: Path, output_dir: Path):
    """
    Extrae un archivo comprimido .tar.gz.

    Parameters
    ----------
    tar_path : Path
        Ruta al archivo comprimido.
    output_dir : Path
        Directorio de destino para la extracción.

    Raises
    ------
    RuntimeError
        Si la extracción falla.
    """
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
    except tarfile.TarError as exc:
        raise RuntimeError(f"Failed to extract tar.gz file {tar_path}: {exc}") from exc


def validate_training_artifact(extracted_dir: Path) -> Path:
    """
    Valida que el artefacto extraído contenga los archivos necesarios.

    Parameters
    ----------
    extracted_dir : Path
        Directorio donde se extrajo el modelo.

    Returns
    -------
    Path
        Ruta al archivo model.joblib validado.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra model.joblib.
    RuntimeError
        Si el archivo no se puede cargar.
    """
    model_path = extracted_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Invalid training artifact: model.joblib not found at artifact root ({model_path})"
        )
    try:
        joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Invalid training artifact: joblib.load failed for {model_path}: {exc}") from exc
    return model_path


def read_tar_entries(tar_path: Path) -> set[str]:
    """
    Lista las entradas (archivos/carpetas) dentro de un tarball.

    Parameters
    ----------
    tar_path : Path
        Ruta al archivo tar.gz.

    Returns
    -------
    set[str]
        Conjunto de nombres de las entradas.
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        return {name.rstrip("/") for name in tar.getnames()}


def validate_model_tar(model_tar_path: Path):
    """
    Valida que el artefacto del modelo para hosting sea correcto.

    Parameters
    ----------
    model_tar_path : Path
        Ruta al nuevo model.tar.gz generado.

    Raises
    ------
    RuntimeError
        Si faltan archivos o si tiene una estructura incompatible (ej. carpeta code/).
    """
    entries = read_tar_entries(model_tar_path)
    required_entries = {"model.joblib"}
    missing = sorted(required_entries - entries)
    if missing:
        raise RuntimeError(
            "Malformed model artifact: missing entries " + ", ".join(missing)
        )
    disallowed_prefix = "code/"
    if any(entry.startswith(disallowed_prefix) for entry in entries):
        raise RuntimeError(
            "Malformed model artifact: unexpected `code/` entry. "
            "Use separate-source serving bundle."
        )


def validate_source_tar(source_tar_path: Path):
    """
    Valida que el artefacto de código fuente para hosting sea correcto.

    Parameters
    ----------
    source_tar_path : Path
        Ruta al archivo source.tar.gz generado.

    Raises
    ------
    RuntimeError
        Si faltan archivos obligatorios (inference.py, requirements.txt).
    """
    entries = read_tar_entries(source_tar_path)
    required_entries = {"inference.py", "requirements.txt"}
    missing = sorted(required_entries - entries)
    if missing:
        raise RuntimeError(
            "Malformed source artifact: missing entries " + ", ".join(missing)
        )


def build_model_tar(trained_model_path: Path, model_tar_path: Path):
    """
    Empaqueta el archivo del modelo en un tarball model.tar.gz.

    Parameters
    ----------
    trained_model_path : Path
        Ruta al archivo model.joblib entrenado.
    model_tar_path : Path
        Ruta de salida para el tarball.
    """
    model_tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(trained_model_path, arcname="model.joblib")
    validate_model_tar(model_tar_path)


def build_source_tar(repo_root: Path, source_tar_path: Path):
    """
    Empaqueta el código de inferencia y dependencias en source.tar.gz.

    Parameters
    ----------
    repo_root : Path
        Raíz del repositorio.
    source_tar_path : Path
        Ruta de salida para el tarball.

    Raises
    ------
    FileNotFoundError
        Si no se encuentran los archivos de inferencia.
    """
    inference_path = repo_root / "inference" / "inference.py"
    requirements_path = repo_root / "inference" / "requirements.txt"
    if not inference_path.exists():
        raise FileNotFoundError(f"Required serving file missing: {inference_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"Required serving file missing: {requirements_path}")

    source_tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-bundle-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        shutil.copy2(inference_path, temp_dir / "inference.py")
        shutil.copy2(requirements_path, temp_dir / "requirements.txt")
        with tarfile.open(source_tar_path, "w:gz") as tar:
            tar.add(temp_dir / "inference.py", arcname="inference.py")
            tar.add(temp_dir / "requirements.txt", arcname="requirements.txt")

    validate_source_tar(source_tar_path)


def upload_artifact(session, local_path: Path, bucket: str, prefix: str, artifact_name: str):
    """
    Sube un artefacto a S3.

    Parameters
    ----------
    session : boto3.Session
        Sesión de AWS.
    local_path : Path
        Ruta al archivo local.
    bucket : str
        Bucket de destino.
    prefix : str
        Prefijo (carpeta) en S3.
    artifact_name : str
        Nombre del archivo en S3.

    Returns
    -------
    str
        URI de S3 del archivo subido.

    Raises
    ------
    RuntimeError
        Si la subida falla.
    """
    normalized_prefix = prefix.strip("/")
    key = f"{normalized_prefix}/{artifact_name}" if normalized_prefix else artifact_name
    s3_uri = f"s3://{bucket}/{key}"
    s3_client = session.client("s3")
    try:
        s3_client.upload_file(str(local_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload artifact to {s3_uri}: {exc}") from exc
    return s3_uri


def resolve_upload_targets(args):
    """
    Resuelve los buckets y prefijos de S3 para la subida de artefactos.

    Parameters
    ----------
    args : argparse.Namespace
        Argumentos de entrada.

    Returns
    -------
    tuple[str, str, str, str]
        (model_bucket, model_prefix, source_bucket, source_prefix).

    Raises
    ------
    ValueError
        Si falta algún destino obligatorio.
    """
    shared_bucket = args.bucket.strip()
    model_bucket = args.model_bucket.strip() or os.getenv("MODEL_S3_BUCKET", "").strip() or shared_bucket
    model_prefix = args.model_prefix.strip() or os.getenv("MODEL_S3_PREFIX", "").strip()
    source_bucket = (
        args.source_bucket.strip()
        or os.getenv("SERVING_SOURCE_S3_BUCKET", "").strip()
        or shared_bucket
    )
    source_prefix = (
        args.source_prefix.strip()
        or os.getenv("SERVING_SOURCE_S3_PREFIX", "").strip()
    )

    if not model_bucket:
        raise ValueError("Missing model bucket: set --model-bucket/--bucket or MODEL_S3_BUCKET")
    if not model_prefix:
        raise ValueError("Missing model prefix: set --model-prefix or MODEL_S3_PREFIX")
    if not source_bucket:
        raise ValueError(
            "Missing source bucket: set --source-bucket/--bucket or SERVING_SOURCE_S3_BUCKET"
        )
    if not source_prefix:
        raise ValueError(
            "Missing source prefix: set --source-prefix or SERVING_SOURCE_S3_PREFIX"
        )

    return model_bucket, model_prefix, source_bucket, source_prefix


def main():
    """
    Punto de entrada principal para la construcción de artefactos de hosting.
    """
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path.cwd() / ".env")

    region = os.getenv("REGION", "").strip()
    profile = os.getenv("AWS_PROFILE", "").strip()

    session = None
    if args.training_artifact_s3_uri or args.upload:
        if not region:
            raise ValueError("REGION is required for S3 download/upload operations")
        session = make_boto3_session(region=region, profile=profile)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()

    model_output_local = output_dir / MODEL_TAR_NAME
    source_output_local = output_dir / SOURCE_TAR_NAME

    with tempfile.TemporaryDirectory(prefix="training-artifact-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        local_training_artifact_path = temp_dir / "training-artifact.tar.gz"
        extracted_training_dir = temp_dir / "extracted-training"
        extracted_training_dir.mkdir(parents=True, exist_ok=True)

        if args.training_artifact_s3_uri:
            input_source = args.training_artifact_s3_uri
            download_training_artifact(session, args.training_artifact_s3_uri, local_training_artifact_path)
        else:
            input_source = str(Path(args.training_artifact_local).resolve())
            source_path = Path(args.training_artifact_local)
            if not source_path.is_absolute():
                source_path = (Path.cwd() / source_path).resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Training artifact local file not found: {source_path}")
            shutil.copy2(source_path, local_training_artifact_path)

        extract_tarball(local_training_artifact_path, extracted_training_dir)
        trained_model_path = validate_training_artifact(extracted_training_dir)
        build_model_tar(trained_model_path=trained_model_path, model_tar_path=model_output_local)
        build_source_tar(repo_root=repo_root, source_tar_path=source_output_local)

    print(f"Input training artifact: {input_source}")
    print(f"Output model artifact (local): {model_output_local}")
    print(f"Output source artifact (local): {source_output_local}")

    if args.upload:
        model_bucket, model_prefix, source_bucket, source_prefix = resolve_upload_targets(args)
        model_uri = upload_artifact(
            session=session,
            local_path=model_output_local,
            bucket=model_bucket,
            prefix=model_prefix,
            artifact_name=MODEL_TAR_NAME,
        )
        source_uri = upload_artifact(
            session=session,
            local_path=source_output_local,
            bucket=source_bucket,
            prefix=source_prefix,
            artifact_name=SOURCE_TAR_NAME,
        )
        print(f"Output model artifact (S3): {model_uri}")
        print(f"Output source artifact (S3): {source_uri}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
