"""
Módulo de Resumen Extractivo.

Utiliza TextRank (como en la Práctica 3) para generar resúmenes cortos de reseñas largas
o descripciones de ítems. 
Es útil para mostrar al usuario un resumen rápido de por qué se le recomienda algo,
sin saturar la ventana de contexto del LLM.
"""

def extract_key_sentences(text: str, top_n: int = 3) -> list:
    """
    Aplica TextRank para devolver las 'top_n' oraciones más representativas del texto.
    """
    pass

def summarize_reviews(reviews: list) -> str:
    """
    Dada una lista de reseñas de un mismo ítem, extrae un resumen general.
    """
    pass
