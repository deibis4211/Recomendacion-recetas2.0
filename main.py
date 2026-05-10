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

if __name__ == "__main__":
    print("Cargando Asistente Recomendador...")
    # 1. load_llm()
    # 2. Cargar índice vectorial
    
    print("¡Asistente listo! Escribe 'salir' para terminar.")
    while True:
        user_input = input("\nTú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
            
        # action = route_query(user_input)
        # context = execute_tool(action)
        # prompt = build_prompt(user_input, context)
        # answer = generate_response(prompt)
        
        # print(f"\nAsistente: {answer}")
