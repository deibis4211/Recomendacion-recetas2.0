"""
Módulo de Agentes y Enrutamiento.

Basado en el DAG Multi-Agente de la Práctica 4.
El Router decidirá qué acción tomar en función del mensaje del usuario:
1. 'recomendar': Llama al retriever (RAG) para buscar ítems.
2. 'web': Si el usuario pregunta por un dato que no está en la DB (ej. "¿Cuándo sale la secuela de este juego?"), busca en Internet.
3. 'resumir': Llama al summarizer para resumir de qué trata un ítem en particular.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from modules.summarizer import summarize_reviews


VALID_ACTIONS = ("recomendar", "resumir", "web")


def _heuristic_route(query: str) -> Optional[str]:
    normalized = query.lower()
    if any(word in normalized for word in ("resume", "resumen", "resumir", "opiniones", "reseñas", "reviews")):
        return "resumir"
    if any(word in normalized for word in ("web", "internet", "online", "actual", "últim", "ultimo", "noticia", "quien", "quién", "presidente", "qué es")):
        return "web"
    return None

def _llm_route(query: str, llm_model, tokenizer) -> Optional[str]:
    if llm_model is None or tokenizer is None:
        return None

    from modules.llm import generate_response

    prompt = (
        "Clasifica la consulta del usuario en una de las siguientes categorías: recomendar, resumir, web.\n"
        "Devuelve ÚNICAMENTE la palabra exacta.\n\n"
        "Ejemplos:\n"
        "- Consulta: dime cómo preparar una tarta de manzana fácil\nCategoría: recomendar\n"
        "- Consulta: quiero una receta de pollo al horno con patatas\nCategoría: recomendar\n"
        "- Consulta: dime qué dice la gente sobre el plato 9032\nCategoría: resumir\n"
        "- Consulta: de qué trata el plato 'id'\nCategoría: resumir\n"
        "- Consulta: resume las reseñas del plato 'id'\nCategoría: resumir\n"
        "- Consulta: cuál es la capital de Francia\nCategoría: web\n"
        "- Consulta: necesito ideas para una cena romántica sin gluten\nCategoría: recomendar\n"
        "- Consulta: búscame las últimas noticias sobre la NASA\nCategoría: web\n"
        "- Consulta: busca en internet el ganador del ultimo mundial de futbol\nCategoría: web\n"
        "- Consulta: quiero un desayuno rápido con avena y plátano\nCategoría: recomendar\n\n"
        f"- Consulta: {query}\n"
        "Categoría:"
    )
    output = generate_response(prompt, llm_model, tokenizer, max_new_tokens=4).lower()
    for action in VALID_ACTIONS:
        if action in output:
            return action
    return None

def _rewrite_query(query: str, llm_model, tokenizer) -> str:
    if llm_model is None or tokenizer is None:
        return query

    from modules.llm import generate_response

    prompt = (
        "Extrae los ingredientes y restricciones de esta consulta y tradúcelos al INGLÉS como palabras clave.\n"
        "Devuelve SOLO las palabras clave en inglés separadas por espacios. No escribas frases completas.\n"
        "Ejemplo: 'quiero una receta rápida de pollo sin horno' -> 'chicken fast no oven'\n\n"
        f"Consulta: {query}\n"
        "Palabras clave (inglés):"
    )
    output = generate_response(prompt, llm_model, tokenizer, max_new_tokens=20).strip()
    
    # Limpieza por si el LLM devuelve saltos de línea o comillas
    output = output.replace("\n", " ").replace('"', '').replace("'", "")
    
    print(f"[Agente] Query reescrita para RAG: '{output}'")
    return output if output else query

def route_query(query: str, llm_model=None, tokenizer=None) -> dict:
    """
    Usa el LLM para decidir la acción, o cae en la heurística si el LLM falla.
    """
    action = _llm_route(query, llm_model, tokenizer) or _heuristic_route(query) or "recomendar"
    
    # Si vamos a buscar recetas, traducimos la frase a palabras clave
    arguments = query
    if action == "recomendar":
        arguments = _rewrite_query(query, llm_model, tokenizer)
        
    return {"action": action, "arguments": arguments}

def execute_tool(
    action: str,
    arguments: str,
    retriever=None,
    interactions_df=None,
    top_k: int = 5,
):
    """
    Ejecuta la herramienta correspondiente y devuelve el contexto a inyectar al LLM.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"Herramienta no válida: {action}")

    if action == "recomendar":
        if retriever is None:
            raise ValueError("La herramienta recomendar necesita un RecipeRetriever inicializado.")
        results = retriever.hybrid_search(arguments, top_k=top_k)
        return _format_recipe_results(results)

    if action == "resumir":
        if interactions_df is None:
            return "No se ha cargado el dataset de interacciones para resumir reseñas."
        recipe_id = _extract_recipe_id(arguments)
        if recipe_id is None:
            return "Indica el id de la receta para resumir sus reseñas."
        reviews = interactions_df.loc[
            interactions_df["recipe_id"].astype(str) == str(recipe_id),
            "review",
        ].dropna().tolist()
        summary = summarize_reviews(reviews)
        return f"Resumen de opiniones de la receta {recipe_id}:\n{summary}"

    return _web_fallback(arguments)


def _extract_recipe_id(text: str) -> Optional[str]:
    match = re.search(r"\b(?:id|receta)\s*[:#]?\s*(\d+)\b", text.lower())
    if match:
        return match.group(1)
    match = re.search(r"\b\d{2,}\b", text)
    return match.group(0) if match else None


def _format_recipe_results(results: list[Dict[str, Any]]) -> str:
    if not results:
        return "No se encontraron recetas relevantes en la base local."

    lines = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        name = metadata.get("name", "Receta sin nombre")
        minutes = metadata.get("minutes", "?")
        topic = metadata.get("topic", "?")
        score = float(result.get("score", 0.0))
        text = str(result.get("text", ""))
        lines.append(
            f"{index}. {name} (id: {result.get('id')}, {minutes} min, tópico {topic}, score {score:.3f})\n"
            f"   {text}"
        )
    return "\n".join(lines)


def _web_fallback(query: str, max_results: int = 3) -> str:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        return f"Error: La librería ddgs no está instalada. Ejecuta: pip install ddgs"

    results = []
    try:
        with DDGS() as ddgs:
            for idx, item in enumerate(ddgs.text(query, max_results=max_results), start=1):
                title = item.get("title", "(sin titulo)")
                body = item.get("body", "")
                results.append(f"- {title}: {body}")
    except Exception as exc:
        return f"No se pudo consultar la web en este entorno: {exc}"

    if not results:
        return "No se encontraron resultados web útiles."
    
    return "Resultados web encontrados:\n" + "\n".join(results)
