# Arquitectura técnica de CulinaryRAG

CulinaryRAG es un asistente conversacional de recetas construido para cumplir el proyecto final MITEX. La arquitectura separa el procesamiento offline del corpus y el bucle online de interacción con el usuario.

Fuente de datos: Food.com Recsys Dataset de Kaggle.

## 1. Vista global

```text
CSV Food.com
    ↓
datasets/preprocesado.py
    ↓
modules/topics.py ─────────────┐
    ↓                           │
offline_pipeline.py             │
    ↓                           │
ChromaDB + BM25 + tópicos       │
    ↓                           │
main.py → agent.py → retriever.py / summarizer.py / web fallback
    ↓
llm.py
    ↓
Respuesta en español
```

## 2. Pipeline offline

### 2.1 Preprocesado

Archivo: `datasets/preprocesado.py`

- Carga `Filtered_recipes.csv` y `Filtered_interactions.csv`.
- Elimina interacciones sin reseña textual.
- Conserva usuarios con al menos 8 reseñas.
- Cruza interacciones con recetas existentes.
- Exporta `Processed_recipes.csv` y `Processed_interactions.csv`.

### 2.2 Tópicos

Archivo: `modules/topics.py`

- Entrena BERTopic sobre nombre y etiquetas de recetas.
- Usa `SentenceTransformer`, UMAP y HDBSCAN.
- Aplica muestreo estratificado por cuartiles de `minutes`.
- Permite semillas culinarias para orientar categorías como vegano, postres, desayuno o recetas rápidas.
- Guarda el modelo en `models/bertopic_recipes`.

### 2.3 Indexación

Archivo: `offline_pipeline.py`

- Carga las recetas procesadas.
- Carga BERTopic si está disponible.
- Usa `RecipeRetriever.index_recipes()` para:
  - generar embeddings de `name + description + ingredients`;
  - persistirlos en ChromaDB;
  - construir y guardar el índice BM25;
  - guardar metadatos como nombre, minutos y tópico.

## 3. Pipeline online

### 3.1 Entrada principal

Archivo: `main.py`

- Carga el retriever y el índice BM25.
- Carga BERTopic para `topic-aware ranking` si existe.
- Carga interacciones para resúmenes.
- Carga un LLM local mediante Transformers.
- Ejecuta el bucle conversacional.

### 3.2 Agente

Archivo: `modules/agent.py`

Acciones disponibles:

- `recomendar`: búsqueda local RAG.
- `resumir`: TextRank sobre reseñas de una receta.
- `web`: búsqueda externa básica cuando la consulta pide actualidad o información no local.

El router intenta clasificar con el LLM y valida la salida contra el conjunto cerrado de acciones. Si no hay LLM o la salida no es válida, usa una heurística determinista.

### 3.3 Recuperación híbrida

Archivo: `modules/retriever.py`

Componentes:

- Embeddings con `paraphrase-multilingual-MiniLM-L12-v2`.
- ChromaDB persistente para búsqueda semántica.
- BM25 para búsqueda léxica.
- Query expansion español-inglés para mejorar consultas sobre corpus en inglés.
- Reciprocal Rank Fusion con `RRF_K = 60`.
- Bonus de tópico del 50% si BERTopic clasifica consulta y receta en el mismo tópico.
- Re-ranking final con `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Penalización o bonus por restricciones, por ejemplo `sin horno`.

### 3.4 Resumen extractivo

Archivo: `modules/summarizer.py`

Implementa TextRank:

- división en oraciones;
- tokenización y limpieza;
- similitud de coseno entre oraciones;
- PageRank;
- selección de frases centrales.

### 3.5 LLM

Archivo: `modules/llm.py`

- Carga modelos de lenguaje causales compatibles con Hugging Face Transformers.
- Construye el prompt final con la consulta y el contexto recuperado.
- Genera la respuesta en español usando generación determinista.

## 4. Trazabilidad MITEX

| Requisito | Archivo principal |
| --- | --- |
| Embeddings | `modules/retriever.py` |
| Modelado de tópicos | `modules/topics.py` |
| Resumen extractivo | `modules/summarizer.py` |
| LLM | `modules/llm.py` |
| RAG | `main.py`, `modules/retriever.py` |
| Agentes | `modules/agent.py` |
| Offline/online | `offline_pipeline.py`, `main.py` |
| Limpieza y preprocesado | `datasets/preprocesado.py` |

## 5. Validación recomendada

- Ejecutar `python datasets/preprocesado.py` y comprobar que se generan los CSV procesados.
- Ejecutar `python modules/topics.py` y revisar tópicos/visualizaciones.
- Ejecutar `python offline_pipeline.py` para construir ChromaDB y BM25.
- Ejecutar `python test_search.py` para validar recuperación sin cargar el LLM completo.
- Ejecutar `python main.py` y probar consultas como:
  - `quiero una receta rápida de pollo sin horno`;
  - `resume las reseñas de la receta 12345`;
  - `busca en internet una tendencia actual de recetas veganas`.
