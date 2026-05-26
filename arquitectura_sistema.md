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
4.  **Pre-resumen (`summarizer.py`)**: Uso de **TextRank** a las reseñas históricas para precomputar o extraer consensos iniciales.

## 3. Arquitectura del Sistema Online

### 3.1. Agente Orquestador (`agent.py`)
Utiliza un modelo local con **decodificación restringida** para actuar como router.
*   **Herramientas disponibles**:
    *   `recomendador`: Búsqueda híbrida (RRF) en el índice de recetas.
    *   `resumidor`: Uso de **TextRank** en tiempo real para sintetizar reseñas de una receta específica.
    *   `calculadora`: Filtrado por metadatos (`minutes`, `calories`).

### 3.2. Recuperación Híbrida (`retriever.py`)
Combina múltiples estrategias y validaciones profundas para extraer las recetas más idóneas:
1.  **Query Expansion y Detección de Restricciones**: La consulta original del usuario se expande para mejorar la cobertura, y se analizan explícitamente restricciones lógicas (ej: intolerancias o negaciones).
2.  **Recuperación Híbrida (RRF)**: Fusión de búsqueda semántica (Vectores) y léxica (BM25) usando la consulta expandida.
3.  **Topic-Aware Boost**: Bonus de relevancia si el documento coincide con el tópico de la consulta.
4.  **Re-Ranking (Cross-Encoder)**: Uso de un modelo `ms-marco-MiniLM-L-6-v2` para re-ordenar los candidatos. Además, la puntuación se ajusta matemáticamente (`constraint_adjustment`) penalizando o premiando según el cumplimiento estricto de las restricciones detectadas.

### 3.3. Generación (`llm.py`)
*   **Gestión del LLM Local**: Carga y prepara el modelo (ej. Gemma) asegurando la inferencia en GPU.
*   **Prompting y Contexto**: Inyecta los `steps` de la receta recuperada y el resumen de reseñas al sistema de prompts para generar una respuesta que recomiende y justifique por qué encaja con los gustos del usuario.

### 3.4. Resumen Extractivo (`summarizer.py`)
*   Implementación pura del algoritmo de grafos **TextRank** usando similitud del coseno y PageRank. Extrae de forma dinámica las oraciones (consenso general) más representativas de todas las reviews, condensando la información y evitando sobrecargar el contexto del modelo de lenguaje.

---

## 4. Cumplimiento de Requisitos del Proyecto

La arquitectura cubre holgadamente todos los requerimientos estipulados:

| Requisito | Implementación en la Arquitectura |
| :--- | :--- |
| **Embeddings** | Generados con `SentenceTransformers` en el módulo *Retriever*. |
| **Topic Modeling** | Implementado con `BERTopic` para categorización automática y filtrado semántico (*Topic-Aware Reranking*). |
| **Búsqueda Híbrida** | Fusión RRF de rankings semánticos y léxicos (BM25) para máxima precisión. |
| **Precisión Semántica** | Expansión de consultas y uso de **Cross-Encoder** para re-ranking con ajuste por restricciones. |
| **Modelado de Tópicos** | Módulo offline *Topics* usando `BERTopic` para agrupar ítems/reseñas. |
| **Resumen Extractivo** | Módulo `summarizer.py` aplicando algoritmo matemático `TextRank` (grafos y PageRank). |
| **LLM** | Corazón del sistema (`llm.py` y `agent.py`). Modelo local. |
| **RAG** | Módulo `retriever.py` integrando búsqueda híbrida y RRF para inyección de contexto. |
| **Agentes** | Enrutador (`agent.py`) con decodificación restringida que decide qué módulo llamar. |
| **Arquitectura** | Clara división lógica entre `offline_pipeline.py` y el agente online. |

---

## 5. Pruebas y Validación (`test_search.py`)
Para garantizar la calidad de la recuperación (RAG) sin tener que iniciar todo el agente de lenguaje natural, el sistema incorpora el script `test_search.py`. Este módulo de pruebas aísla el comportamiento del `retriever.py` y lanza consultas complejas contra el índice local para verificar de primera mano cómo actúan la fusión RRF, el Topic-Aware Reranking y la penalización por restricciones lógicas.

---

## 6. Consideraciones de Diseño
- **No se usan frameworks cerrados**: Toda la lógica de enrutamiento (como el Router DAG) y el pipeline de RAG (RRF) se implementa directamente en código Python (evitando LangChain o LlamaIndex), evidenciando el dominio técnico profundo requerido por la rúbrica de evaluación.
- **Eficiencia de Contexto**: El uso combinado de *Resumen Extractivo* previo a la inyección y de *RRF* garantiza que el LLM solo reciba los datos de máxima calidad, evitando alucinaciones y desbordamiento de tokens de contexto.
