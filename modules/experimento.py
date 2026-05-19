import os
import pandas as pd
from topics import train_topic_model, visualize_model, calculate_coherence_score

# Configuramos nuevas rutas para el experimento
MODEL_DIR_NO_SEEDS = "models/bertopic_no_seeds"
VIS_DIR_NO_SEEDS = "visualizations_no_seeds"

if __name__ == "__main__":
    print("Iniciando experimento SIN SEMILLAS...")
    
    # 1. Cargamos y preparamos los mismos datos (estratificados)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PATH_RECIPES = os.path.join(base_dir, "datasets", "Processed_recipes.csv")
    df_recipes = pd.read_csv(PATH_RECIPES)
    
    df_recipes['minutes_q'] = pd.qcut(df_recipes['minutes'], q=4, labels=['Q1','Q2','Q3','Q4'])
    df_sample = df_recipes.groupby('minutes_q', group_keys=False).apply(lambda x: x.sample(min(len(x), 2500), random_state=42))
    
    corpus = (df_sample['name'].fillna('') + " " + df_sample['tags'].fillna('')).tolist()

    # 2. ENTRENAMOS SIN SEMILLAS (seed_topics=None)
    # Pasamos una ruta de guardado distinta
    print(f"Entrenando modelo puramente no supervisado...")
    modelo_puro = train_topic_model(corpus, seed_topics=None)
    
    # 3. Guardamos en la nueva ruta (usando safetensors para guardar como carpeta)
    os.makedirs(MODEL_DIR_NO_SEEDS, exist_ok=True)
    modelo_puro.save(MODEL_DIR_NO_SEEDS, serialization="safetensors")
    
    # 4. Generamos visualizaciones específicas
    visualize_model(modelo_puro, output_dir=VIS_DIR_NO_SEEDS)
    
    # Calcular e imprimir coherencia
    coherence_score = calculate_coherence_score(modelo_puro, corpus)
    
    print(f"\n¡Experimento terminado!")
    print(f"Modelo en: {MODEL_DIR_NO_SEEDS}")
    print(f"Coherencia de tópicos calculada (C_v): {coherence_score:.4f}")
    print(f"Visualización en: {VIS_DIR_NO_SEEDS}/mapa_interactivo.html")

