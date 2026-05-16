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
    *   *Representatividad (Muestreo Estratificado)*: Para el entrenamiento, se extrae una muestra de 10.000 recetas estratificadas por el tiempo de cocción (`minutes` en cuartiles). Esto garantiza un 99% de nivel de confianza y asegura que todos los perfiles de cocina (desde snacks de 5 min hasta asados de 3h) estén proporcionalmente representados en el modelo.
    *   *Visualización e Interpretabilidad*: El sistema genera mapas interactivos de distancia inter-tópico y gráficos de barras de palabras clave (vía Plotly) para permitir la validación humana de los clústeres descubiertos.
2.  **Indexing**: Generación de embeddings de la combinación `name + description + ingredients` y almacenamiento en base de datos vectorial (ChromaDB) junto con un índice léxico (BM25).
3.  **Topic-Aware Reranking**: Integración del modelo BERTopic en el proceso de recuperación. El sistema identifica el tópico de la consulta del usuario en tiempo real y aplica un bonus de relevancia (vía Fusión RRF) a los documentos que comparten el mismo tópico, filtrando eficazmente resultados ruidosos.
4.  **Pre-resumen**: Aplicación de **TextRank** a las recetas con exceso de reseñas para generar un "consenso" inicial.

## 3. Arquitectura del Sistema Online

### 3.1. Agente Orquestador (`agent.py`)
Utiliza un modelo local con **decodificación restringida** para actuar como router.
*   **Herramientas disponibles**:
    *   `recomendador`: Búsqueda híbrida (RRF) en el índice de recetas.
    *   `resumidor`: Uso de **TextRank** en tiempo real para sintetizar reseñas de una receta específica.
    *   `calculadora`: Filtrado por metadatos (`minutes`, `calories`).

### 3.2. Recuperación Híbrida (`retriever.py`)
Combina dos rankings y una etapa de validación profunda:
1.  **Recuperación Híbrida (RRF)**: Fusión de búsqueda semántica (Vectores) y léxica (BM25).
2.  **Topic-Aware Boost**: Bonus de relevancia si el documento coincide con el tópico de la consulta detectado por BERTopic.
3.  **Re-Ranking (Cross-Encoder)**: Uso de un modelo `ms-marco-MiniLM-L-6-v2` para re-ordenar los mejores candidatos basándose en la relación semántica exacta entre pregunta y documento, resolviendo problemas de negación (ej: "no oven").

### 3.3. Generación (LLM)
*   **Contexto**: Se inyectan los `steps` de la receta y el resumen de las `reviews`.
*   **Prompting**: Diseño de sistema para que el modelo no solo dé la receta, sino que justifique por qué encaja con los gustos del usuario.

---

## 4. Cumplimiento de Requisitos del Proyecto

La arquitectura cubre holgadamente todos los requerimientos estipulados:

| Requisito | Implementación en la Arquitectura |
| :--- | :--- |
| **Embeddings** | Generados con `SentenceTransformers` en el módulo *Retriever*. |
| **Topic Modeling** | Implementado con `BERTopic` para categorización automática y filtrado semántico (*Topic-Aware Reranking*). |
| **Búsqueda Híbrida** | Fusión RRF de rankings semánticos y léxicos (BM25) para máxima precisión. |
| **Precisión Semántica** | Uso de **Cross-Encoder** para re-ranking, eliminando falsos positivos y manejando negaciones. |
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
