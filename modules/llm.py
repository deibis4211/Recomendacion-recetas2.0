"""
Módulo de Modelo de Lenguaje (LLM).

Encapsula la carga y ejecución del LLM local (ej. Gemma-3-1b-it).
Se encarga de inyectar el contexto recuperado en el prompt (Prompt Engineering)
y generar la respuesta final natural para el usuario.
"""

from __future__ import annotations

import importlib.util

def load_llm(model_id: str):
    """
    Carga el modelo y el tokenizador en la VRAM disponible (CUDA).
    """
    if not model_id:
        return None, None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Para usar el LLM instala transformers y torch, o ejecuta sin modelo LLM."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    kwargs = {"device_map": "auto"} if has_accelerate else {}
    if torch.cuda.is_available():
        kwargs["dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not has_accelerate:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

def generate_response(prompt: str, model, tokenizer, max_new_tokens: int = 128) -> str:
    """
    Genera la respuesta del LLM dado un prompt preparado.
    """
    if model is None or tokenizer is None:
        return prompt

    import torch

    if getattr(tokenizer, "chat_template", None):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_prompt(user_query: str, context: str) -> str:
    """
    Construye el prompt final para sintetizar una respuesta con el contexto recuperado.
    """
    return (
        "Eres CulinaryRAG, un asistente experto culinario. "
        "Responde en español basándote ÚNICAMENTE en la información del contexto recuperado (que puede estar en inglés, tradúcela si es necesario). "
        "Puedes hacer deducciones lógicas evidentes a partir del contexto (ej: si el contexto habla de la victoria de un equipo, puedes deducir que ganó). "
        "NO inventes datos o ingredientes que no estén respaldados por el texto original. "
        "Si el usuario pide una receta, recomiéndala basándote en el contexto. Si pide un resumen de opiniones, resúmelas. Si hace otra pregunta, contéstala con los datos dados. "
        "Si el contexto no tiene información para responder a la pregunta, indícalo.\n\n"
        f"Consulta del usuario:\n{user_query}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Respuesta en español:"
    )
