"""
Módulo de Ingestión y Limpieza de Datos.

En un sistema de recomendación (ej. películas, videojuegos, libros), este script se encargará de:
1. Cargar el corpus de datos (ej. un CSV con reseñas de usuarios, descripciones de ítems).
2. Limpiar el texto (eliminar ruido, caracteres especiales, normalizar minúsculas).
3. Fragmentar (chunking) descripciones largas si fuera necesario para los embeddings.
"""

from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

def load_data(file_path: str):
    """
    Carga los datos desde la ruta especificada.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                return pd.DataFrame(json.loads(line) for line in f if line.strip())
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Formato no soportado: {suffix or os.path.basename(file_path)}")

def clean_text(text: str) -> str:
    """
    Aplica técnicas de limpieza (regex, normalización de unicode).
    """
    if text is None:
        return ""

    value = html.unescape(str(text))
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ.,;:!?()/%+\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()

def chunk_text(text: str, chunk_size: int) -> list:
    """
    Divide textos muy largos en fragmentos más pequeños.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que 0")

    words = clean_text(text).split()
    if not words:
        return []

    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
