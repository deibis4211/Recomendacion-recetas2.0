# CulinaryRAG

Sistema de soporte inteligente y recomendación conversacional de recetas para el proyecto final MITEX.

El asistente combina minería de textos clásica y modelos neuronales: embeddings, BERTopic, BM25, RRF, TextRank, RAG, LLM local y un router agéntico de herramientas.

## Cumplimiento del enunciado

| Requisito MITEX | Implementación |
| --- | --- |
| Embeddings | `modules/retriever.py` genera vectores con `SentenceTransformer` y los persiste en ChromaDB. |
| Modelado de tópicos | `modules/topics.py` entrena BERTopic offline con muestreo estratificado. |
| Resumen extractivo | `modules/summarizer.py` implementa TextRank sobre reseñas. |
| LLM | `modules/llm.py` carga un modelo local compatible con Transformers. |
| RAG | `main.py` recupera contexto local y lo inyecta en el prompt final. |
| Agentes | `modules/agent.py` enruta entre recomendación, resumen y búsqueda web. |
| Arquitectura offline/online | `offline_pipeline.py` prepara índices; `main.py` ejecuta el bucle interactivo. |
| Extra opcional | `datasets/preprocesado.py` limpia y filtra el corpus; el retriever añade BM25, RRF, Cross-Encoder y restricciones. |

## Estructura

```text
.
├── main.py                     # Bucle online del asistente
├── offline_pipeline.py         # Creación de índices persistentes
├── test_search.py              # Prueba aislada del recuperador
├── modules/
│   ├── agent.py                # Router y ejecución de herramientas
│   ├── ingestion.py            # Limpieza de texto
│   ├── llm.py                  # Carga y generación con LLM
│   ├── retriever.py            # ChromaDB, BM25, RRF, Cross-Encoder
│   ├── summarizer.py           # TextRank
│   ├── topics.py               # BERTopic
│   └── experimento.py          # Comparación de tópicos sin semillas
├── datasets/
│   └── preprocesado.py         # Construcción de CSV procesados
└── docu/
    ├── memoria.tex             # Informe LaTeX
    └── memoria.pdf             # Informe compilado
```

## Instalación

Se recomienda usar un entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los modelos de Hugging Face se descargan al primer uso si no existen en caché. Para Gemma puede ser necesario tener acceso aceptado y token configurado; si no se detecta Gemma local, el sistema usa `Qwen/Qwen2.5-1.5B-Instruct`.

## Datos

El proyecto espera los CSV originales del dataset Food.com en:

```text
datasets/original/Filtered_recipes.csv
datasets/original/Filtered_interactions.csv
```

Preprocesado:

```bash
python datasets/preprocesado.py
```

Esto genera:

```text
datasets/Processed_recipes.csv
datasets/Processed_interactions.csv
```

## Ejecución

Entrenar tópicos:

```bash
python modules/topics.py
```

Crear índices ChromaDB y BM25:

```bash
python offline_pipeline.py
```

Probar solo la búsqueda:

```bash
python test_search.py
```

Lanzar el asistente:

```bash
python main.py
```

También puede forzarse otro LLM con:

```bash
CULINARYRAG_MODEL="ruta/o/modelo/huggingface" python main.py
```

## Documentación

La memoria principal está en `docu/memoria.tex`. Para compilarla:

```bash
cd docu
pdflatex memoria.tex
pdflatex memoria.tex
```

El informe incluye la arquitectura, justificación de diseño, matriz de cumplimiento del enunciado, validación por módulos, limitaciones y trabajo futuro.
