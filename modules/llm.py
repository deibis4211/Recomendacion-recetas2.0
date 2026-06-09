import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_id: str):
    """
    Carga el LLM y su tokenizador de forma unificada.
    """
    print(f"Cargando LLM: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer

def generate_response(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 300
) -> str:
    """
    Genera texto a partir de un prompt usando el LLM cargado.
    """
    if model is None or tokenizer is None:
        return ""
        
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, outputs)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def build_prompt(user_query: str, context: str, action: str = "recomendar") -> str:
    """
    Construye el prompt específico según la acción (Agente de Múltiples Prompts).
    """
    import re
    
    clean_query = user_query.lower()
    for trigger in ["busca en internet", "busca en la web", "busca online", "buscar en internet"]:
        clean_query = clean_query.replace(trigger, "").strip()
    
    if not clean_query:
        clean_query = user_query

    if action == "recomendar":
        return (
            "Eres un chef encargado de adaptar y transcribir recetas. Responde siempre en español.\n\n"
            "INSTRUCCIONES ESTRICTAS:\n"
            "1. Lee la receta del [CONTEXTO RECUPERADO] y NO TE INVENTES ingredientes ni datos que no aparezcan en ella.\n"
            "2. Propon versiones alternativas de la receta SOLO SI el usuario pide restricciones:\n"
            "Por ej. sin horno, sin gluten, sin lactosa, vegana, etc.\n"
            "3. Escribe el resultado final usando este formato exacto:\n\n"
            "**[NOMBRE DE LA RECETA]**\n"
            "**INGREDIENTES:**\n"
            "- [Ingrediente 1]\n\n"
            "**PASOS DE PREPARACIÓN:**\n"
            "1. [Paso 1]\n\n"
            f"[CONTEXTO RECUPERADO]:\n{context}\n\n"
            f"[CONSULTA DEL USUARIO]:\n{clean_query}\n\n"
            "Respuesta:"
        )
        
    elif action == "resumir":
        return (
            "Eres un analista de opiniones culinarias. Tu única tarea es resumir los comentarios de los usuarios.\n\n"
            "INSTRUCCIONES:\n"
            "- Lee los comentarios del [CONTEXTO].\n"
            "- Escribe un resumen breve en español sobre lo que piensa la gente.\n"
            "- PROHIBIDO: No inventes ingredientes ni pasos de cocina. Limítate a resumir lo que dice la gente.\n\n"
            f"[CONTEXTO]:\n{context}\n\n"
            f"[CONSULTA DEL USUARIO]:\n{clean_query}\n\n"
            "Resumen de opiniones:"
        )
        
    elif action == "web":
        # Limpieza de fonéticas tipo [ˈpeðɾo ˈsantʃeθ] para no marear al modelo
        clean_context = re.sub(r'\[.*?\]', '', context)
        
        return (
            "[SISTEMA]\n"
            "Eres un extractor de datos estricto. Tu memoria interna ha sido borrada. "
            "SOLO puedes usar la información que aparece entre las etiquetas <CONTEXTO>.\n\n"
            "[REGLAS CRÍTICAS]\n"
            "1. Si la respuesta no está en el <CONTEXTO>, responde exactamente: 'Lo siento, la información no está disponible en los documentos.'\n"
            "2. Prohibido usar conocimientos previos.\n"
            "3. Si el contexto dice 'A' y tú crees que es 'B', responde 'A' obligatoriamente.\n"
            "4. Sé breve.\n\n"
            f"[CONSULTA]\n{clean_query}\n\n"
            f"<CONTEXTO>\n{clean_context}\n\n"
            "[PROCESAMIENTO]\n"
            "Analiza el contexto paso a paso y extrae el dato exacto para la consulta.\n"
            "Respuesta directa:\n"
        )
    
    else:
        # Fallback genérico
        return f"Responde a {clean_query} usando: {context}"
