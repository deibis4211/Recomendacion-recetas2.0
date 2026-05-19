"""
Módulo de Modelado de Tópicos (Procesamiento Offline).

Para un recomendador, este módulo usará BERTopic sobre el
corpus de recetas para descubrir las temáticas subyacentes.
Ejemplo: Agrupar recetas en "postres veganos", "comida rápida", "cenas ligeras".

Estos tópicos se pueden almacenar y usar luego para filtrar recomendaciones o 
enriquecer el contexto del LLM.
"""

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import pandas as pd
import os

MODEL_DIR = "models/bertopic_recipes"

def train_topic_model(corpus: list, seed_topics: list = None):
    """
    Entrena el modelo BERTopic sobre el corpus limpio.
    Permite usar 'seed_topics' para forzar temáticas conocidas (ej. vegan, dessert).
    """
    from umap import UMAP
    import hdbscan

    print(f"Iniciando entrenamiento con {len(corpus)} documentos...")
    
    # 1. Definimos el modelo de embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Configuración UMAP (reducción de dimensionalidad)
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )

    # 3. Configuración HDBSCAN (clustering)
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=50,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # 4. Configuramos BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        seed_topic_list=seed_topics,
        calculate_probabilities=False,
        verbose=True
    )
    
    # 5. Entrenamos
    topics, probs = topic_model.fit_transform(corpus)
    
    # 4. Guardamos el modelo para usarlo en el RAG online
    os.makedirs("models", exist_ok=True)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)
    print(f"Modelo entrenado y guardado en {MODEL_DIR}")
    
    return topic_model

def get_item_topics(item_id: str, df: pd.DataFrame, topic_model=None):
    """
    Devuelve los tópicos principales asociados a un ítem concreto.
    """
    if topic_model is None:
        try:
            topic_model = BERTopic.load(MODEL_DIR)
        except Exception:
            return "Modelo no encontrado. Entrénalo primero."
            
    # Buscamos la receta en el dataframe
    recipe = df[df['id'] == int(item_id)]
    if recipe.empty:
        return "Receta no encontrada"
        
    # Preparamos el texto igual que en el entrenamiento
    texto = str(recipe['name'].values[0]) + " " + str(recipe['tags'].values[0])
    
    # Predecimos a qué tópico pertenece
    topics, _ = topic_model.transform([texto])
    topic_id = topics[0]
    
    # Si el tópico es -1, significa que BERTopic lo considera "ruido" o no categorizado
    if topic_id == -1:
        return "Sin categorizar (Ruido)"
        
    topic_info = topic_model.get_topic(topic_id)
    # topic_info es una lista de tuplas (palabra, probabilidad)
    palabras_clave = [word for word, prob in topic_info[:5]]
    
    return f"Tópico {topic_id}: " + ", ".join(palabras_clave)

def visualize_model(topic_model, output_dir="visualizations"):
    """
    Genera visualizaciones interactivas y las guarda como archivos HTML.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generando visualizaciones interactiva...")
    
    # 1. Mapa de distancia inter-tópico (El mapa de "círculos")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(output_dir, "mapa_interactivo.html"))
    
    # 2. Gráfico de barras de palabras clave por tópico
    fig_barchart = topic_model.visualize_barchart(top_n_topics=15)
    fig_barchart.write_html(os.path.join(output_dir, "palabras_clave.html"))
    
    print(f"Visualizaciones guardadas en la carpeta '{output_dir}/'")
    print("Puedes abrirlas con cualquier navegador para explorar los datos.")

def calculate_coherence_score(topic_model, corpus: list) -> float:
    """
    Calcula la coherencia de tópicos (C_v) usando Gensim.
    """
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    import re

    print("Calculando la coherencia de tópicos (C_v) con Gensim...")
    
    # 1. Obtener los tópicos y extraer las palabras (excluyendo el tópico -1 de ruido)
    topics = topic_model.get_topics()
    topic_words = []
    for topic_id, words in topics.items():
        if topic_id == -1:
            continue
        words_list = [w[0] for w in words[:10]]
        topic_words.append(words_list)
        
    if not topic_words:
        print("No se encontraron tópicos estructurados (todos se consideraron ruido). Coherencia: 0.0")
        return 0.0

    # 2. Tokenizar los documentos del corpus
    tokenized_docs = [re.sub(r"[^a-z0-9áéíóúñü\s]", " ", doc.lower()).split() for doc in corpus]
    
    # 3. Crear el diccionario y calcular coherencia
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    
    cv_score = coherence_model.get_coherence()
    print(f"Coherencia de tópicos (C_v): {cv_score:.4f}")
    return float(cv_score)


if __name__ == "__main__":
    # SCRIPT DE PRUEBA / ENTRENAMIENTO OFFLINE
    print("Cargando dataset...")
    # Usamos una ruta absoluta relativa a este script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_path = os.path.join(base_dir, 'datasets', 'Processed_recipes.csv')
    df_recipes = pd.read_csv(df_path)
    
    # ---------------------------------------------------------
    # MUESTREO ESTRATIFICADO: Garantizar representatividad
    # ---------------------------------------------------------
    # Dividimos las recetas en 4 grupos (cuartiles) según lo que tardan en cocinarse
    # Así nos aseguramos de tener recetas rápidas, normales y lentas en la misma proporción que el dataset original
    df_recipes['time_category'] = pd.qcut(df_recipes['minutes'], q=4, labels=['muy_rapida', 'rapida', 'media', 'lenta'], duplicates='drop')
    
    # Extraemos la muestra manteniendo la proporción exacta de cada grupo
    sample_size = 10000
    df_sample = df_recipes.groupby('time_category', group_keys=False).apply(
        lambda x: x.sample(int(len(x) / len(df_recipes) * sample_size), random_state=42)
    ).copy()
    
    print(f"Muestra estratificada generada: {len(df_sample)} recetas.")
    # ---------------------------------------------------------
    
    # Creamos el corpus juntando el nombre y las etiquetas
    df_sample['text_for_topic'] = df_sample['name'].fillna('') + " " + df_sample['tags'].fillna('')
    corpus_entrenamiento = df_sample['text_for_topic'].tolist()
    
    # Definimos unas "semillas" para guiar al modelo
    semillas = [
        ["vegan", "vegetarian", "plant-based"],
        ["dessert", "cake", "sweet", "chocolate"],
        ["breakfast", "morning", "eggs", "pancakes"],
        ["quick", "easy", "fast", "15-minutes"]
    ]
    
    # Entrenamos
    modelo = train_topic_model(corpus_entrenamiento, seed_topics=semillas)
    
    # Generamos el mapa interactivo
    visualize_model(modelo)
    
    # Calcular la coherencia de tópicos (C_v)
    coherence_score = calculate_coherence_score(modelo, corpus_entrenamiento)
    print(f"\nCoherencia de tópicos calculada: {coherence_score:.4f}")
    
    # Mostramos los tópicos generados en consola
    print("\nResumen de tópicos descubiertos:")
    print(modelo.get_topic_info().head(10))

