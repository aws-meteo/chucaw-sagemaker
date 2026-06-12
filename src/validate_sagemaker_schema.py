"""
Validador para AWS QuickSight SageMaker Schema JSON.
Basado en: https://docs.aws.amazon.com/quick/latest/userguide/sagemaker-integration.html
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple


# Tipos válidos según AWS QuickSight
VALID_TYPES = {"INTEGER", "STRING", "DECIMAL"}
VALID_CONTENT_TYPES = {"CSV"}  # AWS docs show "CSV" not "text/csv" despite description
VALID_INSTANCE_TYPES = {
    "ml.m4.xlarge", "ml.m4.2xlarge", "ml.m4.4xlarge", "ml.m4.10xlarge", "ml.m4.16xlarge",
    "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge", "ml.m5.12xlarge", "ml.m5.24xlarge",
    "ml.m6i.large", "ml.m6i.xlarge", "ml.m6i.2xlarge", "ml.m6i.4xlarge", "ml.m6i.8xlarge", "ml.m6i.12xlarge", "ml.m6i.16xlarge", "ml.m6i.24xlarge", "ml.m6i.32xlarge",
}


def validate_schema(schema_path: str) -> Tuple[bool, List[str]]:
    """
    Valida un archivo de esquema (JSON) de SageMaker para su integración con QuickSight.

    Verifica la presencia de campos obligatorios, la validez de los tipos de datos,
    el formato de contenido y las políticas de instancias según la documentación de AWS.

    Parameters
    ----------
    schema_path : str
        Ruta al archivo JSON del esquema.

    Returns
    -------
    Tuple[bool, List[str]]
        Una tupla (is_valid, list_of_errors). Si no hay errores, is_valid es True.
    """
    errors = []
    
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON inválido: {e}"]
    except FileNotFoundError:
        return False, [f"Archivo no encontrado: {schema_path}"]
    
    # Campos requeridos
    required_fields = ["inputContentType", "outputContentType", "input", "output", "version"]
    for field in required_fields:
        if field not in schema:
            errors.append(f"❌ Campo requerido ausente: '{field}'")
    
    # Validar inputContentType
    if "inputContentType" in schema:
        if schema["inputContentType"] not in VALID_CONTENT_TYPES:
            errors.append(f"❌ inputContentType inválido: '{schema['inputContentType']}'. Debe ser 'text/csv'")
    
    # Validar outputContentType
    if "outputContentType" in schema:
        if schema["outputContentType"] not in VALID_CONTENT_TYPES:
            errors.append(f"❌ outputContentType inválido: '{schema['outputContentType']}'. Debe ser 'text/csv'")
    
    # Validar input
    if "input" in schema:
        if not isinstance(schema["input"], list):
            errors.append(f"❌ 'input' debe ser una lista")
        else:
            for i, field in enumerate(schema["input"]):
                col_errors = validate_column(field, i, "input")
                errors.extend(col_errors)
    
    # Validar output
    if "output" in schema:
        if not isinstance(schema["output"], list):
            errors.append(f"❌ 'output' debe ser una lista")
        else:
            for i, field in enumerate(schema["output"]):
                col_errors = validate_column(field, i, "output")
                errors.extend(col_errors)
    
    # Validar version
    if "version" in schema:
        if not isinstance(schema["version"], str):
            errors.append(f"❌ 'version' debe ser string")
    
    # Validar instanceTypes (si está presente)
    if "instanceTypes" in schema:
        if not isinstance(schema["instanceTypes"], list):
            errors.append(f"❌ 'instanceTypes' debe ser una lista")
        else:
            for itype in schema["instanceTypes"]:
                if itype not in VALID_INSTANCE_TYPES:
                    errors.append(f"⚠️ instanceType no es estándar: '{itype}'. Tipos válidos: {sorted(VALID_INSTANCE_TYPES)}")
    
    # Validar defaultInstanceType
    if "defaultInstanceType" in schema:
        if "instanceTypes" in schema:
            if schema["defaultInstanceType"] not in schema["instanceTypes"]:
                errors.append(f"❌ 'defaultInstanceType' ({schema['defaultInstanceType']}) no está en 'instanceTypes'")
    
    # Validar instanceCount (opcional pero si está presente)
    if "instanceCount" in schema:
        if not isinstance(schema["instanceCount"], int) or schema["instanceCount"] < 1:
            errors.append(f"❌ 'instanceCount' debe ser un entero positivo")
    
    # Validar description (límite 1000 caracteres)
    if "description" in schema:
        if len(schema["description"]) > 1000:
            errors.append(f"⚠️ 'description' excede 1000 caracteres ({len(schema['description'])})")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_column(field: Dict, index: int, section: str) -> List[str]:
    """
    Valida la definición de una columna individual dentro de las secciones input u output.

    Parameters
    ----------
    field : Dict
        Objeto que define la columna (name, type, etc.).
    index : int
        Índice de la columna en la lista para facilitar la identificación del error.
    section : str
        Nombre de la sección ("input" o "output").

    Returns
    -------
    List[str]
        Lista de errores encontrados para esta columna.
    """
    errors = []
    
    if not isinstance(field, dict):
        errors.append(f"❌ {section}[{index}] debe ser un objeto")
        return errors
    
    # Validar name
    if "name" not in field:
        errors.append(f"❌ {section}[{index}] falta 'name'")
    else:
        if len(field["name"]) > 100:
            errors.append(f"⚠️ {section}[{index}].name excede 100 caracteres")
    
    # Validar type
    if "type" not in field:
        errors.append(f"❌ {section}[{index}] falta 'type'")
    else:
        if field["type"] not in VALID_TYPES:
            errors.append(
                f"❌ {section}[{index}].type = '{field['type']}' es inválido. "
                f"Tipos válidos: {sorted(VALID_TYPES)}"
            )
    
    # Campos no permitidos
    valid_field_keys = {"name", "type", "nullable"}
    for key in field.keys():
        if key not in valid_field_keys:
            errors.append(f"⚠️ {section}[{index}] contiene campo no estándar: '{key}'")
    
    return errors


if __name__ == "__main__":
    schema_path = Path(__file__).parent.parent / "inference" / "schema.json"
    
    is_valid, errors = validate_schema(str(schema_path))
    
    print("=" * 60)
    print("VALIDACIÓN DE SCHEMA SAGEMAKER PARA QUICKSIGHT")
    print("=" * 60)
    
    if is_valid:
        print("✅ Schema válido - Listo para cargar en QuickSight")
    else:
        print("❌ Schema contiene errores:\n")
        for error in errors:
            print(f"  {error}")
    
    print("=" * 60)
    exit(0 if is_valid else 1)
