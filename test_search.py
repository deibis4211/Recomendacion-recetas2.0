from modules.retriever import RecipeRetriever
import os

def probar_busqueda():
    print("=== TEST DE BÚSQUEDA HÍBRIDA (RAG) ===")
    
    # 1. Inicializar el buscador (Carga ChromaDB)
    retriever = RecipeRetriever()
    
    # 2. Cargar el índice de palabras clave (BM25)
    if not retriever.load_bm25():
        print("ERROR: No se encontró el índice BM25.")
        return

    # 3. Cargar el modelo de tópicos para el Re-ranking (Topic-Aware)
    from bertopic import BERTopic
    model_path = "models/bertopic_recipes"
    if os.path.exists(model_path):
        print(f"Cargando modelo de tópicos para re-ranking desde {model_path}...")
        topic_model = BERTopic.load(model_path, embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        retriever.set_topic_model(topic_model)

    # 3. Lista de consultas para probar la potencia del sistema
    consultas = [
        "creamy pasta",
        "chicken spicy",
        "chocolate no oven",
        "breakfast"
    ]

    for query in consultas:
        print(f"\n>>> Buscando: '{query}'...")
        
        # Hacemos la búsqueda híbrida (RRF)
        resultados = retriever.hybrid_search(query, top_k=3)

        # Mostrar resultados
        for i, res in enumerate(resultados, 1):
            m = res['metadata']
            print(f"   {i}. {m['name']} (Tópico: {m['topic']})")
            print(f"      ID: {res['id']} | Tiempo: {m['minutes']} min")
            print(f"      Contexto: {res['text'][:120]}...")
        print("-" * 60)

if __name__ == "__main__":
    probar_busqueda()
