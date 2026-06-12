#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["latitude", "longitude", "value"]


def parse_args():
    """
    Analiza los argumentos de la línea de comandos.

    Returns
    -------
    argparse.Namespace
        Objeto con los argumentos analizados (ej. --input).
    """
    parser = argparse.ArgumentParser(description="Build trivial nearest-neighbor lookup model artifact")
    parser.add_argument("--input", required=True, help="CSV path with latitude,longitude,value")
    return parser.parse_args()


def main():
    """
    Punto de entrada para el entrenamiento del modelo trivial.

    Lee datos desde un CSV, valida las columnas requeridas (latitude, longitude, value),
    limpia valores nulos y guarda el modelo (un diccionario con arrays de numpy)
    como un archivo .joblib compatible con contenedores de SageMaker.

    Raises
    ------
    FileNotFoundError
        Si el archivo CSV de entrada no existe.
    ValueError
        Si faltan columnas obligatorias o si el dataset queda vacío tras la limpieza.
    """
