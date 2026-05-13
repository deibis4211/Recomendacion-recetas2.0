import os
from bertopic import BERTopic
from topics import visualize_model

# Calculamos la ruta absoluta al modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# El modelo está un nivel por encima de 'modules', en la raíz de proyecto_grande
MODEL_PATH = os.path.join(BASE_DIR, "..", "bertopic_recipes")

if not os.path.exists(MODEL_PATH):
    # Por si acaso se guardó dentro de 'models'
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "bertopic_recipes")

print(f"Cargando modelo desde: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    model = BERTopic.load(MODEL_PATH)
    visualize_model(model)
else:
    print(f"ERROR: No se encuentra el modelo en {MODEL_PATH}")
