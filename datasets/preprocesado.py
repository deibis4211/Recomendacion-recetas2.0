import pandas as pd
import os

# Directorio base donde se encuentra este script (datasets)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de entrada (Originales en original)
PATH_RAW_RECIPES = os.path.join(BASE_DIR, 'original', 'Filtered_recipes.csv')
PATH_RAW_INTERACTIONS = os.path.join(BASE_DIR, 'original', 'Filtered_interactions.csv')

# Rutas de salida
PATH_OUT_RECIPES = os.path.join(BASE_DIR, 'Processed_recipes.csv')
PATH_OUT_INTERACTIONS = os.path.join(BASE_DIR, 'Processed_interactions.csv')

print("Iniciando preprocesado de datos...")

## 1. CARGA DE DATOS
df_recipes = pd.read_csv(PATH_RAW_RECIPES)
df_interactions = pd.read_csv(PATH_RAW_INTERACTIONS)

## 2. PROCESAMIENTO DE INTERACCIONES
# Mantenemos 'review' para el módulo de Summarizer (TextRank)
# Mantenemos 'corrected_rating' para recomendaciones de calidad
cols_inter = ['user_id', 'recipe_id', 'rating', 'review', 'corrected_rating']
df_interactions = df_interactions[cols_inter]

# Limpieza: Eliminar interacciones sin texto (crucial para minería de textos)
df_interactions = df_interactions.dropna(subset=['review'])

# Filtro de calidad: Usuarios con al menos 8 reseñas
reviews_por_user = df_interactions['user_id'].value_counts()
minimo = 8
usuarios_def = reviews_por_user[reviews_por_user >= minimo].index
df_interactions_filtrado = df_interactions[df_interactions['user_id'].isin(usuarios_def)].copy()

# Cruce con recetas para asegurar que la receta existe
df_interactions_filtrado = df_interactions_filtrado.merge(df_recipes[['id']], left_on='recipe_id', right_on='id', how='inner')

## 3. PROCESAMIENTO DE RECETAS
# Mantenemos 'steps' para el RAG (instrucciones de cocina)
# Mantenemos 'description' y 'tags' para BERTopic y Embeddings
cols_recep = ['name', 'id', 'minutes', 'ingredients', 'steps', 'description', 'tags', 'calories (#)']
df_recipes = df_recipes[cols_recep]

# Limpieza: Eliminar recetas con campos críticos vacíos
df_recipes = df_recipes.dropna(subset=['name', 'steps', 'ingredients'])

# Cruce con interacciones filtradas (solo recetas que tengan reviews de calidad)
df_recetas_filtrado = df_recipes.merge(df_interactions_filtrado[['recipe_id']], left_on='id', right_on='recipe_id', how='inner')
df_recetas_filtrado = df_recetas_filtrado.drop_duplicates(subset=['id'])

## 4. GUARDADO
df_interactions_filtrado.to_csv(PATH_OUT_INTERACTIONS, index=False)
df_recetas_filtrado.to_csv(PATH_OUT_RECIPES, index=False)

print(f"--- Proceso Finalizado ---")
print(f"Recetas finales: {len(df_recetas_filtrado)}")
print(f"Interacciones finales: {len(df_interactions_filtrado)}")
print(f"Archivos guardados en la carpeta 'datasets/'")