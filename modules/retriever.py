import os
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from bertopic import BERTopic
from typing import List, Dict, Tuple
import re
import numpy as np
import pickle
import os

# Configuración idéntica a la Práctica 4
RRF_K = 60
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "data/chroma_db"

class RecipeRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Cargando modelo de embeddings: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Re-Ranker: El "juez" que decide el orden final (Opción B)
        print("Cargando Re-Ranker (Cross-Encoder)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Cliente persistente para no re-indexar 184k recetas cada vez
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name="recipes_hybrid",
            metadata={"hnsw:space": "cosine"} # Usamos similitud coseno como en la práctica
        )
        
        self.bm25 = None
        self.recipe_ids = [] # Para mapear el índice de BM25 con los IDs reales
        self.bm25_path = os.path.join(DB_PATH, "bm25_model.pkl")
        self.topic_model = None # Modelo BERTopic opcional para re-ranking

    def _normalize(self, text: str) -> List[str]:
        """Normalización léxica (igual que en la Práctica 4)"""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9áéíóúñü\s]", " ", text)
        return [t for t in text.split() if t]

    def index_recipes(self, df: pd.DataFrame, topic_model=None):
        """
        Indexa las recetas en ChromaDB (Semántico) y BM25 (Léxico).
        df: DataFrame con las recetas procesadas.
        topic_model: El modelo BERTopic entrenado para añadir tópicos como metadatos.
        """
        print(f"Preparando indexación de {len(df)} recetas...")
        
        # 1. Preparar textos para embeddings (Nombre + Descripción + Ingredientes)
        # Usamos una combinación rica para que el RAG sea preciso
        texts_to_embed = (
            df['name'].fillna('') + " " + 
            df['description'].fillna('') + " " + 
            df['ingredients'].fillna('')
        ).tolist()

        # 2. Obtener tópicos si el modelo está disponible
        topics = [-1] * len(df)
        if topic_model:
            print("Calculando tópicos para las recetas...")
            # Aquí usamos el modelo ya entrenado para predecir el tópico de cada receta
            topics, _ = topic_model.transform(texts_to_embed)

        # 3. Indexar en ChromaDB (Semántico)
        # Nota: En producción con 184k, esto se haría por batches.
        print("Generando embeddings e indexando en ChromaDB (esto puede tardar)...")
        
        # Para evitar colapsar la RAM, indexamos en bloques de 5000
        batch_size = 5000
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            batch_texts = texts_to_embed[i : i + batch_size]
            batch_topics = topics[i : i + batch_size]
            
            ids = batch_df['id'].astype(str).tolist()
            
            # Usamos enumerate para tener un índice (j) relativo al lote actual (0 a 4999)
            metadatas = []
            for j, (idx, row) in enumerate(batch_df.iterrows()):
                metadatas.append({
                    "name": row['name'], 
                    "topic": int(batch_topics[j]),
                    "minutes": int(row['minutes'])
                })
            
            self.collection.add(
                documents=batch_texts,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Indexados {i + len(batch_df)} / {len(df)} recetas...")

        # 4. Preparar BM25 (Léxico)
        print("Preparando índice léxico BM25...")
        tokenized_corpus = [self._normalize(t) for t in texts_to_embed]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.recipe_ids = df['id'].astype(str).tolist()
        
        # Guardar BM25 para no tener que re-tokenizar 184k textos
        self.save_bm25()
        print("¡Indexación completada!")

    def save_bm25(self):
        """Guarda el índice léxico en disco"""
        os.makedirs(DB_PATH, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "ids": self.recipe_ids}, f)
        print(f"Índice BM25 guardado en {self.bm25_path}")

    def load_bm25(self):
        """Carga el índice léxico desde disco"""
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.recipe_ids = data["ids"]
            print("Índice BM25 cargado correctamente.")
            return True
        return False

    def set_topic_model(self, model: BERTopic):
        """Asigna un modelo BERTopic para mejorar la precisión de búsqueda"""
        self.topic_model = model
        print("Modelo de tópicos cargado en el Retriever para re-ranking.")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Búsqueda híbrida usando RRF + Topic-Aware Reranking.
        """
        if self.bm25 is None:
            raise ValueError("El buscador no ha sido indexado. Llama a index_recipes() o load_bm25().")

        # 0. Identificar el tópico de la consulta (Topic-Aware)
        query_topic = -1
        if self.topic_model:
            # Predecimos el tópico de la pregunta del usuario
            topics, _ = self.topic_model.transform([query])
            query_topic = int(topics[0])
            print(f"[IA] Tópico detectado en consulta: {query_topic}")

        # 1. Ranking Semántico (ChromaDB)
        results_sem = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2 # Pedimos más para la fusión
        )
        
        # Mapeamos IDs a su posición en el ranking (1-based)
        rank_sem = {id_: idx for idx, id_ in enumerate(results_sem['ids'][0], start=1)}

        # 2. Ranking Léxico (BM25)
        tokenized_query = self._normalize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Ordenamos los IDs por score de BM25
        top_indices_lex = np.argsort(bm25_scores)[::-1][:top_k * 2]
        rank_lex = {self.recipe_ids[idx]: i for i, idx in enumerate(top_indices_lex, start=1)}

        # 3. Fusión RRF con Bonus por Tópico
        all_ids = set(rank_sem.keys()) | set(rank_lex.keys())
        
        # Recuperamos metadatos de los candidatos para saber sus tópicos
        # Pedimos los metadatos de todos los candidatos detectados
        candidate_data = self.collection.get(ids=list(all_ids))
        id_to_topic = {id_: meta['topic'] for id_, meta in zip(candidate_data['ids'], candidate_data['metadatas'])}

        rrf_scores = []
        for doc_id in all_ids:
            score = 0.0
            if doc_id in rank_sem:
                score += 1.0 / (RRF_K + rank_sem[doc_id])
            if doc_id in rank_lex:
                score += 1.0 / (RRF_K + rank_lex[doc_id])
            
            # BONUS DE TÓPICO: Si el tópico coincide con la consulta, damos un empujón
            # Esto evita que salgan recetas de bacon (Tópico X) en búsquedas de chocolate (Tópico Y)
            if query_topic != -1 and id_to_topic.get(doc_id) == query_topic:
                score *= 1.5 # Bonus del 50%
                
            rrf_scores.append((doc_id, score))
        
        # 4. Recuperar datos para el Re-ranking
        final_top_ids = [doc[0] for doc in rrf_scores[:top_k * 3]] # Cogemos el top 15 para re-rankear
        candidate_results = self.collection.get(ids=final_top_ids)
        
        # 5. RE-RANKING (Cross-Encoder)
        # El Cross-Encoder compara la query con cada documento y da una puntuación real
        pairs = [[query, doc] for doc in candidate_results['documents']]
        cross_scores = self.reranker.predict(pairs)
        
        # Unimos IDs con sus nuevas puntuaciones
        scored_results = []
        for i in range(len(candidate_results['ids'])):
            scored_results.append({
                "id": candidate_results['ids'][i],
                "text": candidate_results['documents'][i],
                "metadata": candidate_results['metadatas'][i],
                "score": cross_scores[i]
            })
        
        # Ordenamos por la puntuación del Re-Ranker
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results[:top_k]

if __name__ == "__main__":
    # Script de prueba rápido
    print("Probando inicialización del Retriever...")
    retriever = RecipeRetriever()
    print("Retriever listo.")
