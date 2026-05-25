"""
Módulo de Resumen Extractivo.

Utiliza TextRank (como en la Práctica 3) para generar resúmenes cortos de reseñas largas
o descripciones de ítems. 
Es útil para mostrar al usuario un resumen rápido de por qué se le recomienda algo,
sin saturar la ventana de contexto del LLM.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

from modules.ingestion import clean_text


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.split()) >= 3]


def _sentence_tokens(sentence: str) -> list[str]:
    return [token for token in clean_text(sentence).split() if len(token) > 2]


def _cosine_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    if not counts_a or not counts_b:
        return 0.0

    common = set(counts_a) & set(counts_b)
    numerator = sum(counts_a[token] * counts_b[token] for token in common)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _pagerank(similarity_matrix: np.ndarray, damping: float = 0.85, iterations: int = 50) -> np.ndarray:
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])

    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    transition = np.divide(
        similarity_matrix,
        row_sums,
        out=np.full_like(similarity_matrix, 1.0 / n, dtype=float),
        where=row_sums != 0,
    )

    scores = np.full(n, 1.0 / n)
    teleport = np.full(n, (1.0 - damping) / n)
    for _ in range(iterations):
        scores = teleport + damping * transition.T.dot(scores)
    return scores

def extract_key_sentences(text: str, top_n: int = 3) -> list:
    """
    Aplica TextRank para devolver las 'top_n' oraciones más representativas del texto.
    """
    if top_n <= 0:
        return []

    sentences = _split_sentences(text)
    if len(sentences) <= top_n:
        return sentences

    tokenized = [_sentence_tokens(sentence) for sentence in sentences]
    similarity = np.zeros((len(sentences), len(sentences)), dtype=float)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = _cosine_similarity(tokenized[i], tokenized[j])
            similarity[i, j] = score
            similarity[j, i] = score

    scores = _pagerank(similarity)
    selected = sorted(np.argsort(scores)[-top_n:])
    return [sentences[index] for index in selected]

def summarize_reviews(reviews: list) -> str:
    """
    Dada una lista de reseñas de un mismo ítem, extrae un resumen general.
    """
    clean_reviews = [str(review).strip() for review in reviews if str(review).strip()]
    if not clean_reviews:
        return "No hay reseñas disponibles para resumir."

    combined = " ".join(clean_reviews)
    key_sentences = extract_key_sentences(combined, top_n=3)
    return " ".join(key_sentences) if key_sentences else clean_reviews[0]
