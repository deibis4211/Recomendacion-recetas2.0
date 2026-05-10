# Documentación Técnica: Arquitectura del Sistema Recomendador

Este documento describe la arquitectura técnica propuesta para el Proyecto Final MITEX. El sistema es un **asistente interactivo especializado en recomendaciones** (el dominio exacto: películas, videojuegos, libros, etc., es configurable).

## 1. Diseño Arquitectónico Global

Para cumplir con los requisitos del proyecto (separación de módulos offline/online) y garantizar la escalabilidad y respuesta en tiempo real, la arquitectura se divide en dos grandes bloques:

### A. Pipeline Offline (Pre-procesamiento)
Este bloque (`offline_pipeline.py`) se encarga de ingerir y preparar los datos de forma asíncrona antes de que el usuario interactúe con el sistema. 

**Flujo:**
1. **Adquisición y Limpieza (`ingestion.py`)**: Carga del corpus bruto (JSON/CSV). Se aplican técnicas de limpieza de texto, normalización de caracteres y división en chunks si los documentos son demasiado extensos.
2. **Modelado de Tópicos (`topics.py`)**: Utilizando `BERTopic` + `UMAP` + `HDBSCAN`, se agrupa el corpus para descubrir temáticas latentes. Estos tópicos se precalculan y se asocian como metadatos a cada ítem de recomendación.
3. **Indexación Vectorial (`retriever.py`)**: Se calculan los embeddings semánticos (`SentenceTransformers`) y los índices léxicos (`BM25`) de todos los ítems, persistiendo esta información localmente (ej. ChromaDB o archivos serializados).

### B. Bucle Online (Interacción en Tiempo Real)
Este bloque (`main.py`) mantiene vivo el sistema y gestiona la conversación con el usuario apoyándose en una arquitectura basada en **Agentes y Enrutamiento**.

**Flujo:**
1. **Router Multi-Agente (`agent.py`)**: El modelo de lenguaje (LLM) evalúa la intención del usuario usando *decodificación restringida* para garantizar la validez de la acción.
   * *Acción RAG*: Si el usuario busca recomendaciones.
   * *Acción Web*: Si pregunta por datos fuera del corpus (ej. actualidad).
   * *Acción Resumen*: Si pide información muy detallada sobre un ítem específico.
2. **Ejecución de Herramientas**:
   * **RAG Híbrido (`retriever.py`)**: Ejecuta una búsqueda semántica y léxica, fusionando resultados mediante **RRF (Reciprocal Rank Fusion)** para evitar la pérdida de información central (LitM).
   * **Summarizer (`summarizer.py`)**: Usa **TextRank** (grafos) para extraer las sentencias clave de reseñas largas y pasarlas como contexto condensado al LLM.
3. **Sintetizador (`llm.py`)**: El contexto resultante de las herramientas se inyecta en un *prompt* dinámico. El LLM (ej. Gemma-3-1b-it) genera una respuesta conversacional y natural.

---

## 2. Cumplimiento de Requisitos del Proyecto

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
