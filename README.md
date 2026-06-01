# CulinaryRAG: Intelligent Recipe Recommendation System

CulinaryRAG is an advanced conversational support system and recipe recommender developed as the final project for the Master in Technological Innovation (MITEX). 

This system leverages a hybrid architecture combining classical text mining techniques with modern neural models. It integrates dense embeddings, topic modeling (BERTopic), lexical search (BM25), Reciprocal Rank Fusion (RRF), extractive summarization (TextRank), Retrieval-Augmented Generation (RAG), a local Large Language Model (LLM), and an agentic tool router to provide a robust and accurate user experience.

---

## Architecture and Requirements Fulfillment

The system has been designed to strictly fulfill the architectural and technical requirements specified in the project guidelines:

| Technical Requirement | Implementation Details |
| :--- | :--- |
| **Dense Embeddings** | The `modules/retriever.py` component generates dense vector representations using `SentenceTransformer` and persists them in a local `ChromaDB` vector database. |
| **Topic Modeling** | The `modules/topics.py` component trains a `BERTopic` model offline using stratified sampling to assign latent topics to recipes. |
| **Extractive Summarization** | The `modules/summarizer.py` component implements the `TextRank` algorithm to extract the most relevant sentences from user reviews. |
| **Language Model (LLM)** | The `modules/llm.py` component loads and serves a local LLM compatible with the Hugging Face `transformers` library. |
| **Retrieval-Augmented Gen.** | The `main.py` controller retrieves the local context and injects it into a dynamic prompt for final synthesis. |
| **Agentic Router** | The `modules/agent.py` component acts as a deterministic and heuristic router, deciding between recipe recommendation, summarization, or web fallback. |
| **Offline/Online Separation** | `offline_pipeline.py` prepares and serializes the indexes, while `main.py` executes the low-latency interactive loop. |
| **Advanced Retrieval (Extra)**| Includes a lexical index (BM25), RRF fusion, Cross-Encoder Re-Ranking, and heuristic constraint adjustments. |

---

## Repository Structure

```text
.
├── main.py                     # Main interactive loop for the online assistant
├── offline_pipeline.py         # Construction and serialization of persistent indexes
├── test_search.py              # Isolated testing suite for the retrieval module
├── modules/
│   ├── agent.py                # Action router and tool execution logic
│   ├── ingestion.py            # Text cleaning and normalization
│   ├── llm.py                  # LLM loading and response generation
│   ├── retriever.py            # Hybrid retrieval (ChromaDB, BM25, RRF, Cross-Encoder)
│   ├── summarizer.py           # TextRank implementation for reviews
│   ├── topics.py               # BERTopic training and inference
│   └── experimento.py          # Seedless topic modeling comparison
├── datasets/
│   └── preprocesado.py         # Construction and filtering of processed CSVs
└── docu/
    ├── memoria.tex             # Main technical report (LaTeX source)
    └── memoria.pdf             # Compiled technical report
```

---

## Installation and Setup

It is strictly recommended to run this project within an isolated virtual environment to prevent dependency conflicts.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Model Management
Hugging Face models (embeddings, re-rankers, and LLMs) are downloaded automatically upon first execution and stored in the local cache. 
- The default LLM is `Qwen/Qwen2.5-1.5B-Instruct` to ensure compatibility without requiring authentication tokens. 
- If a cached version of `Gemma-3-1b-it` is detected, the system will prioritize it.

---

## Data Pipeline

The system expects the original Food.com dataset files in the following directory structure:

```text
datasets/original/Filtered_recipes.csv
datasets/original/Filtered_interactions.csv
```

To execute the data preprocessing pipeline:

```bash
python datasets/preprocesado.py
```

This script cleans the text, handles missing values, and generates the processed datasets required by the offline pipeline:

```text
datasets/Processed_recipes.csv
datasets/Processed_interactions.csv
```

---

## Execution Workflow

1. **Topic Modeling Training**: Train the BERTopic model to extract metadata.
   ```bash
   python modules/topics.py
   ```

2. **Index Generation**: Build and serialize the ChromaDB and BM25 indexes.
   ```bash
   python offline_pipeline.py
   ```

3. **Isolated Testing (Optional)**: Verify the retrieval subsystem without loading the LLM.
   ```bash
   python test_search.py
   ```

4. **Launch Assistant**: Start the main interactive loop.
   ```bash
   python main.py
   ```

*(Optional)* To force the usage of a specific Hugging Face model, set the environment variable:
```bash
CULINARYRAG_MODEL="path/or/huggingface/model_id" python main.py
```

---

## Technical Documentation

The comprehensive technical report, including the architectural design, design justifications, compliance matrix, modular validation, limitations, and future work, is located in `docu/memoria.tex`.

To compile the LaTeX source code into a PDF document:

```bash
cd docu
pdflatex memoria.tex
pdflatex memoria.tex
```
