"""
Módulo de Agentes y Enrutamiento.

Basado en el DAG Multi-Agente de la Práctica 4.
El Router decidirá qué acción tomar en función del mensaje del usuario:
1. 'recomendar': Llama al retriever (RAG) para buscar ítems.
2. 'web': Si el usuario pregunta por un dato que no está en la DB (ej. "¿Cuándo sale la secuela de este juego?"), busca en Internet.
3. 'resumir': Llama al summarizer para resumir de qué trata un ítem en particular.
"""

def route_query(query: str, llm_model, tokenizer) -> dict:
    """
    Usa decodificación restringida para forzar al LLM a elegir una herramienta válida
    ('recomendar', 'web', 'resumir') y extraer los argumentos.
    """
    pass

def execute_tool(action: str, arguments: str):
    """
    Ejecuta la herramienta correspondiente y devuelve el contexto a inyectar al LLM.
    """
    pass
