"""
Módulo de Modelado de Tópicos (Procesamiento Offline).

Para un recomendador, este módulo usará BERTopic (como en la Práctica 3) sobre el
corpus de reseñas/descripciones para descubrir las temáticas subyacentes.
Ejemplo: Agrupar reseñas de videojuegos en "buenos gráficos", "historia inmersiva", "bugs".

Estos tópicos se pueden almacenar y usar luego para filtrar recomendaciones o 
enriquecer el contexto del LLM.
"""

def train_topic_model(corpus: list, seed_topics: dict = None):
    """
    Entrena el modelo BERTopic sobre el corpus limpio.
    Permite usar 'seed_topics' para forzar temáticas conocidas (ej. géneros).
    """
    pass

def get_item_topics(item_id: str):
    """
    Devuelve los tópicos principales asociados a un ítem concreto.
    """
    pass
