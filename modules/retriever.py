"""
Módulo Retriever (RAG).

Implementa la búsqueda híbrida vista en la Práctica 4.
Combina búsqueda semántica (SentenceTransformers) y léxica (BM25) usando 
fusión RRF (Reciprocal Rank Fusion).

En el recomendador, el usuario hace una consulta ("Quiero un juego de mundo abierto relajante")
y este módulo recupera los ítems que mejor hagan 'match' con esa descripción.
"""

def build_index(corpus: list):
    """
    Genera los embeddings semánticos y el índice BM25 de los ítems.
    """
    pass

def hybrid_search(query: str, top_k: int = 5) -> list:
    """
    Ejecuta la búsqueda semántica y léxica, aplica RRF y devuelve los mejores candidatos.
    """
    pass
