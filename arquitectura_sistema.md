# Documentación Técnica: Arquitectura del Sistema Recomendador

DATASET EN: www.kaggle.com/datasets/iamnotwhale/food-com-recsys-dataset?resource=download

Este documento describe la arquitectura técnica propuesta para el Proyecto Final MITEX. El sistema es un **asistente interactivo especializado en recomendaciones** (el dominio exacto: películas, videojuegos, libros, etc., es configurable).

## 1. Diseño Arquitectónico Global

Para cumplir con los requisitos del proyecto (separación de módulos offline/online) y garantizar la escalabilidad y respuesta en tiempo real, la arquitectura se divide en dos grandes bloques:

### A. Pipeline Offline (Pre-procesamiento)
Este bloque (`offline_pipeline.py`) se encarga de ingerir y preparar los datos de forma asíncrona antes de que el usuario interactúe con el sistema. 

## 2. Pipeline de Datos (ETL)

### 2.1. Ingestión y Limpieza (`ingestion.py`)
*   **Fuentes**: `Processed_recipes.csv` y `Processed_interactions.csv`.
*   **Campos Clave**: 
    *   Recetas: `name`, `steps`, `description`, `ingredients`, `tags`.
    *   Interacciones: `review`, `corrected_rating`.
*   **Proceso**: Unión de datasets por `recipe_id` y normalización de textos (limpieza de HTML, minúsculas).

### 2.2. Procesamiento Offline (`offline_pipeline.py`)
1.  **Topic Modeling (BERTopic)**: Agrupamiento de recetas basado en `tags` e `ingredients` para identificar estilos de cocina automáticamente.
2.  **Indexing**: Generación de embeddings de la combinación `name + description + ingredients` y almacenamiento en base de datos vectorial (ChromaDB).
3.  **Pre-resumen**: Aplicación de **TextRank** a las recetas con exceso de reseñas para generar un "consenso" inicial.

## 3. Arquitectura del Sistema Online

### 3.1. Agente Orquestador (`agent.py`)
Utiliza un modelo local con **decodificación restringida** para actuar como router.
*   **Herramientas disponibles**:
    *   `recomendador`: Búsqueda híbrida (RRF) en el índice de recetas.
    *   `resumidor`: Uso de **TextRank** en tiempo real para sintetizar reseñas de una receta específica.
    *   `calculadora`: Filtrado por metadatos (`minutes`, `calories`).

### 3.2. Recuperación Híbrida (`retriever.py`)
Combina dos rankings mediante **Reciprocal Rank Fusion (RRF)**:
1.  **Búsqueda Semántica**: Distancia de coseno sobre los embeddings.
2.  **Búsqueda Léxica (BM25)**: Búsqueda exacta de ingredientes específicos.

### 3.3. Generación (LLM)
*   **Contexto**: Se inyectan los `steps` de la receta y el resumen de las `reviews`.
*   **Prompting**: Diseño de sistema para que el modelo no solo dé la receta, sino que justifique por qué encaja con los gustos del usuario.

---

## 4. Cumplimiento de Requisitos del Proyecto

La arquitectura cubre holgadamente todos los requerimientos estipulados:

| Requisito | Implementación en la Arquitectura |
| :--- | :--- |
| **Embeddings** | Generados con `SentenceTransformers` en el módulo *Retriever*. |
| **Modelado de Tópicos** | Módulo offline *Topics* usando `BERTopic` para agrupar ítems/reseñas. |
| **Resumen Extractivo** | Módulo *Summarizer* aplicando algoritmo `TextRank` (grafos). |
| **LLM** | Corazón del sistema (*Router* y *Sintetizador*). Modelo local (ej. Gemma). |
| **RAG** | Módulo *Retriever* integrando búsqueda híbrida y RRF para inyección de contexto. |
| **Agentes** | Enrutador con decodificación restringida que decide qué módulo llamar (RAG, Web, Summarizer). |
| **Arquitectura** | Clara división lógica entre `offline_pipeline.py` y `main.py`. |

---

## 3. Consideraciones de Diseño
- **No se usan frameworks cerrados**: Toda la lógica de enrutamiento (como el Router DAG) y el pipeline de RAG (RRF) se implementa directamente en código Python (evitando LangChain o LlamaIndex), evidenciando el dominio técnico profundo requerido por la rúbrica de evaluación.
- **Eficiencia de Contexto**: El uso combinado de *Resumen Extractivo* previo a la inyección y de *RRF* garantiza que el LLM solo reciba los datos de máxima calidad, evitando alucinaciones y desbordamiento de tokens de contexto.
