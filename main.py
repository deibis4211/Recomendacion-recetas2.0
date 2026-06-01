"""
Punto de Entrada del Sistema (Bucle Online).

Es la interfaz interactiva con el usuario.
Flujo:
1. Carga el LLM y los índices generados por offline_pipeline.py.
2. Bucle infinito: Pide input al usuario.
3. El Agente decide la herramienta.
4. Se ejecuta la herramienta (RAG, Web, Resumen).
5. El LLM sintetiza la respuesta final basándose en el contexto.
"""

import os
import pandas as pd
from modules.agent import execute_tool, route_query
from modules.llm import build_prompt, generate_response, load_llm
from modules.retriever import DEFAULT_MODEL, RecipeRetriever, _resolve_hf_model

def _default_llm_model():
    gemma_cache = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots")
    if not os.path.isdir(gemma_cache):
        return "Qwen/Qwen2.5-1.5B-Instruct"

    snapshots = [
        os.path.join(gemma_cache, name)
        for name in os.listdir(gemma_cache)
        if os.path.isdir(os.path.join(gemma_cache, name))
    ]
    if not snapshots:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    return max(snapshots, key=os.path.getmtime)

def _load_interactions(path: str = "datasets/Processed_interactions.csv"):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def _load_retriever():
    retriever = RecipeRetriever()
    if not retriever.load_bm25():
        raise RuntimeError("No se encontró el índice BM25. Ejecuta primero offline_pipeline.py.")

    try:
        from bertopic import BERTopic
        for model_path in ("models/bertopic_recipes", "bertopic_recipes"):
            if os.path.exists(model_path):
                topic_model = BERTopic.load(model_path, embedding_model=_resolve_hf_model(DEFAULT_MODEL))
                retriever.set_topic_model(topic_model)
                break
    except Exception as exc:
        print(f"Aviso: no se pudo cargar BERTopic para topic-aware ranking: {exc}")

    return retriever

if __name__ == "__main__":
    print("Cargando Asistente Recomendador...")
    retriever = _load_retriever()
    print("Cargando interacciones (puede tardar un poco)...")
    interactions_df = _load_interactions()

    model_id = os.getenv("CULINARYRAG_MODEL", "") or _default_llm_model()
    model, tokenizer = load_llm(model_id) if model_id else (None, None)
    if model is None:
        print("Aviso: CULINARYRAG_MODEL no está definido; se mostrará el contexto recuperado sin síntesis LLM.")
    
    print("Asistente listo. Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break

        try:
            routed = route_query(user_input, model, tokenizer)
            context = execute_tool(
                routed["action"],
                routed["arguments"],
                retriever=retriever,
                interactions_df=interactions_df,
                top_k=1,
            )

            prompt = build_prompt(user_input, context, routed["action"])
            print("\nGenerando respuesta...")
            answer = generate_response(prompt, model, tokenizer, max_new_tokens=400)
        except Exception as exc:
            answer = f"No he podido completar la consulta: {exc}"

        print(f"\nAsistente: {answer}")
