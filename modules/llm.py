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
        kwargs["dtype"] = torch.bfloat16

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
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def build_prompt(user_query: str, context: str) -> str:
    """
    Construye el prompt final para sintetizar una respuesta con el contexto recuperado.
    """
    # Limpiamos frases que asustan al LLM (safety filters)
    clean_query = user_query.lower()
    for trigger in ["busca en internet", "busca en la web", "busca online", "buscar en internet"]:
        clean_query = clean_query.replace(trigger, "").strip()
    
    if not clean_query:
        clean_query = user_query

    return (
        "You are CulinaryRAG, a helpful assistant. You must answer strictly in SPANISH.\n\n"
        "RULES:\n"
        "1. If the context contains a recipe, TRANSLATE all ingredients and steps to Spanish.\n"
        "2. Format your response with a bulleted list for 'INGREDIENTES:' and a numbered list for 'PASOS:'.\n"
        "3. DIETARY RESTRICTIONS: If the user asks for a VEGAN (vegano) recipe, you MUST replace 'eggs' with 'lino', 'butter' with 'margarina vegetal', and 'milk' with 'leche vegetal'.\n"
        "4. EQUIPMENT RESTRICTIONS: If the user says they have no oven ('sin horno'), you MUST replace any baking/oven steps with 'Refrigerar en la nevera por 2 horas'.\n"
        "5. If the context contains a summary of reviews, present it enthusiastically.\n"
        "6. If the context contains web search results, just answer the question directly.\n"
        "7. NEVER refuse to answer, never apologize, and never say you don't have internet. Trust the context.\n\n"
        f"User Query:\n{clean_query}\n\n"
        f"Context:\n{context}\n\n"
        "Response (in Spanish):"
    )
