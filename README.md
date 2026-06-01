# CulinaryRAG: Sistema Inteligente de Recomendación de Recetas

CulinaryRAG es un sistema avanzado de soporte conversacional y recomendación de recetas desarrollado como proyecto final para el Máster en Innovación Tecnológica (MITEX). 

Este sistema emplea una arquitectura híbrida que combina técnicas clásicas de minería de textos con modelos neuronales modernos. Integra representaciones vectoriales densas (embeddings), modelado de tópicos (BERTopic), búsqueda léxica (BM25), Fusión de Rangos Recíprocos (RRF), resumen extractivo (TextRank), Generación Aumentada por Recuperación (RAG), un Modelo de Lenguaje de Gran Escala (LLM) local y un enrutador agéntico de herramientas para proporcionar una experiencia de usuario robusta y precisa.

---

## Arquitectura y Cumplimiento de Requisitos

El sistema ha sido diseñado para cumplir estrictamente con los requisitos técnicos y arquitectónicos especificados en las directrices del proyecto:

| Requisito Técnico | Detalles de Implementación |
| :--- | :--- |
| **Embeddings Densos** | El componente `modules/retriever.py` genera representaciones vectoriales densas usando `SentenceTransformer` y las persiste en una base de datos vectorial local `ChromaDB`. |
| **Modelado de Tópicos** | El componente `modules/topics.py` entrena un modelo `BERTopic` offline usando muestreo estratificado para asignar tópicos latentes a las recetas. |
| **Resumen Extractivo** | El componente `modules/summarizer.py` implementa el algoritmo `TextRank` para extraer las oraciones más relevantes de las reseñas de los usuarios. |
| **Modelo de Lenguaje (LLM)** | El componente `modules/llm.py` sirve el LLM local e implementa una arquitectura de Múltiples Prompts Agénticos, seleccionando dinámicamente el prompt según la tarea (recetas, resúmenes o web). |
| **Generación Aumentada (RAG)** | El controlador `main.py` recupera el contexto local y lo inyecta en el prompt específico para la síntesis final, aplicando transformaciones estructurales. |
| **Enrutador Agéntico** | El componente `modules/agent.py` emplea un clasificador LLM basado en *Few-Shot Prompting* para inferir la intención del usuario, respaldado por heurísticas de contingencia. |
| **Separación Offline/Online** | `offline_pipeline.py` prepara y serializa los índices, mientras que `main.py` ejecuta el bucle interactivo de baja latencia. |
| **Recuperación Avanzada (Extra)**| Incluye un índice léxico (BM25), fusión RRF, Re-Ranking mediante Cross-Encoder y ajustes heurísticos de restricciones. |

---

## Estructura del Repositorio

```text
.
├── main.py                     # Bucle interactivo principal para el asistente online
├── offline_pipeline.py         # Construcción y serialización de los índices persistentes
├── test_search.py              # Suite de pruebas aisladas para el módulo de recuperación
├── modules/
│   ├── agent.py                # Enrutador de acciones y lógica de ejecución de herramientas
│   ├── ingestion.py            # Limpieza y normalización de texto
│   ├── llm.py                  # Carga del LLM y generación de respuestas
│   ├── retriever.py            # Recuperación híbrida (ChromaDB, BM25, RRF, Cross-Encoder)
│   ├── summarizer.py           # Implementación de TextRank para reseñas
│   ├── topics.py               # Entrenamiento e inferencia de BERTopic
│   └── experimento.py          # Comparación de modelado de tópicos sin semillas
├── datasets/
│   └── preprocesado.py         # Construcción y filtrado de los archivos CSV procesados
└── docu/
    ├── memoria.tex             # Memoria técnica principal (código fuente LaTeX)
    └── memoria.pdf             # Memoria técnica compilada
```

---

## Instalación y Configuración

Se recomienda estrictamente ejecutar este proyecto dentro de un entorno virtual aislado para evitar conflictos de dependencias.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Gestión de Modelos
Los modelos de Hugging Face (embeddings, re-rankers y LLMs) se descargan automáticamente en la primera ejecución y se almacenan en la caché local. 
- El LLM por defecto es `Qwen/Qwen2.5-1.5B-Instruct` para asegurar compatibilidad sin requerir tokens de autenticación. 
- Si se detecta una versión en caché de `Gemma-3-1b-it`, el sistema la priorizará automáticamente.

---

## Flujo de Datos

El sistema espera que los conjuntos de datos originales de Food.com se encuentren en la siguiente estructura:

```text
datasets/original/Filtered_recipes.csv
datasets/original/Filtered_interactions.csv
```

Para ejecutar la canalización de preprocesamiento de datos:

```bash
python datasets/preprocesado.py
```

Este script limpia el texto, maneja valores faltantes y genera los conjuntos de datos procesados requeridos por la canalización offline:

```text
datasets/Processed_recipes.csv
datasets/Processed_interactions.csv
```

---

## Flujo de Ejecución

1. **Entrenamiento de Modelado de Tópicos**: Entrena el modelo BERTopic para extraer metadatos.
   ```bash
   python modules/topics.py
   ```

2. **Generación de Índices**: Construye y serializa los índices ChromaDB y BM25.
   ```bash
   python offline_pipeline.py
   ```

3. **Pruebas Aisladas (Opcional)**: Verifica el subsistema de recuperación sin cargar el LLM.
   ```bash
   python test_search.py
   ```

4. **Lanzar Asistente**: Inicia el bucle interactivo principal.
   ```bash
   python main.py
   ```

*(Opcional)* Para forzar el uso de un modelo específico de Hugging Face, establezca la variable de entorno:
```bash
CULINARYRAG_MODEL="ruta/al/modelo/huggingface" python main.py
```

---

## Documentación Técnica

El informe técnico exhaustivo, que incluye el diseño arquitectónico, justificaciones de diseño, matriz de cumplimiento, validación modular, limitaciones y trabajo futuro, se encuentra en `docu/memoria.tex`.

Para compilar el código fuente LaTeX en un documento PDF:

```bash
cd docu
pdflatex memoria.tex
pdflatex memoria.tex
```
