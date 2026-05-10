"""
Pipeline de Procesamiento Offline.

Este script se ejecuta UNA SOLA VEZ (o cuando el corpus cambie) antes de iniciar el asistente.
Flujo:
1. Ingestión: Lee el CSV de datos.
2. Limpieza: Preprocesa el texto.
3. Tópicos: Entrena BERTopic y asigna tópicos a los ítems.
4. Indexación: Genera los embeddings y el índice BM25 y los guarda a disco.
"""

if __name__ == "__main__":
    print("Iniciando procesamiento offline...")
    # 1. load_data()
    # 2. clean_text()
    # 3. train_topic_model()
    # 4. build_index()
    print("Procesamiento completado y datos persistidos.")
