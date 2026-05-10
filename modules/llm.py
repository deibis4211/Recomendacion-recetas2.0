"""
Módulo de Modelo de Lenguaje (LLM).

Encapsula la carga y ejecución del LLM local (ej. Gemma-3-1b-it).
Se encarga de inyectar el contexto recuperado en el prompt (Prompt Engineering)
y generar la respuesta final natural para el usuario.
"""

def load_llm(model_id: str):
    """
    Carga el modelo y el tokenizador en la VRAM disponible (CUDA).
    """
    pass

def generate_response(prompt: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    """
    Genera la respuesta del LLM dado un prompt preparado.
    """
    pass
