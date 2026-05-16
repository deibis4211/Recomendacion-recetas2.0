import os
import pandas as pd
from bertopic import BERTopic
from modules.retriever import RecipeRetriever

def run_offline_pipeline():
    print("=== Iniciando Pipeline de Procesamiento Offline ===")
    
    # 1. Cargar datos procesados
    path_recipes = "datasets/Processed_recipes.csv"
    if not os.path.exists(path_recipes):
        print(f"ERROR: No se encuentra {path_recipes}. Ejecuta primero datasets/preprocesado.py")
        return

    print("Cargando recetas...")
    df = pd.read_csv(path_recipes)
    
    # 2. Cargar modelo de tópicos
    # Usamos la ruta donde guardamos el modelo el otro día
    model_path = "bertopic_recipes" 
    if not os.path.exists(model_path):
        model_path = "models/bertopic_recipes"
        
    print(f"Cargando modelo de tópicos desde {model_path}...")
    try:
        topic_model = BERTopic.load(model_path, embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    except Exception as e:
        print(f"Aviso: No se pudo cargar el modelo de tópicos ({e}). Se indexará sin tópicos.")
        topic_model = None

    # 3. Inicializar e Indexar el Retriever
    retriever = RecipeRetriever()
    
    # Ejecutamos la indexación masiva
    # Esto guardará automáticamente los vectores en ChromaDB y el índice BM25
    retriever.index_recipes(df, topic_model=topic_model)
    
    print("\n=== Pipeline completada con éxito ===")
    print("Sistema listo para realizar búsquedas híbridas.")

if __name__ == "__main__":
    run_offline_pipeline()
