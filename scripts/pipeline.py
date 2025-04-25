import sys
from pathlib import Path
import os
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Добавляем корень проекта в Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Импорты из пользовательских модулей
try:
    from config.helpers import make_data_set, read_pkl, save_pkl, make_recipes, filter_recipe
    from config.dish import Recipe, RecipesProject
    from config.vector_db import build_chroma_db, request_chroma_db
    from config.food_graph import (
        make_tags_list, lemmatize_tags, build_knowledge_graph,
        lemmatize, lemmatize_sentance, make_one_word_tags_list,
        enreach_query_with_relative_tags, save_graph
    )
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    sys.exit(1)

# Пути к данным
MAIN_DIR = "/content/drive/MyDrive/demo_500_recipes/"
FILE_NAME = "/content/drive/MyDrive/llm_kaggle/llm_kaggle/dataset/demo_500/recipes_"

def load_pickle():
    """
    Загружает рецепты из файлов .pickle.
    Возвращает список рецептов или пустой список в случае ошибок.
    """
    povar_recipes = []
    try:
        # Загружаем файлы с 1 по 3 (можно изменить диапазон)
        for i in range(1, 4):
            file_path = f"{FILE_NAME}{i}.pickle"
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found!")
                continue
            try:
                _povar_recipes = read_pkl(file_path)
                povar_recipes.extend(_povar_recipes)
                logger.info(f"Loaded {len(_povar_recipes)} recipes from {file_path}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                continue
        logger.info(f"Total loaded {len(povar_recipes)} recipes")
    except Exception as e:
        logger.error(f"Unexpected error in load_pickle: {e}")
    return povar_recipes

def create_db(recipes_list):
    """
    Создаёт векторное хранилище для списка рецептов.
    Возвращает vector_store и tokenizer или (None, None) в случае ошибки.
    """
    if not recipes_list:
        logger.error("No recipes provided - cannot create vector store")
        return None, None

    try:
        # Определяем устройство (GPU/CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Создаём векторное хранилище
        vector_store, tokenizer = build_chroma_db(recipes_list)
        logger.info("Vector store and tokenizer created successfully")
        return vector_store, tokenizer
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None, None

def main():
    """
    Основная функция для обработки рецептов, создания векторного хранилища и графа знаний.
    """
    # Создаём выходную папку
    output_dir = "/content/drive/MyDrive/llm_kaggle/llm_kaggle/dataset"
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory {output_dir} created or already exists")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return

    # Загружаем рецепты
    logger.info("Loading recipes...")
    povar_recipes = load_pickle()

    if not povar_recipes:
        logger.error("No recipes loaded! Check your pickle files and paths.")
        return

    # Присваиваем povar_recipes напрямую, так как они уже объекты Recipe
    logger.info("Using loaded recipes...")
    recipes_list = povar_recipes
    logger.info(f"Loaded {len(recipes_list)} recipes")

    # Выводим статистику для первых 5 рецептов
    for recipe in recipes_list[:5]:
        logger.info(f"Recipe {recipe.id}: steps={len(recipe.steps)}, time={recipe.standard_time}, rating={recipe.ratingValue}, votes={recipe.ratingCount}")

    # Фильтруем рецепты
    try:
        recipes_list = filter_recipe(
            recipes_list,
            max_steps=20,  # Смягчено
            max_min=240,   # Смягчено
            min_rating=3,  # Смягчено
            min_votes=1    # Смягчено
        )
        logger.info(f"Filtered to {len(recipes_list)} recipes")
    except Exception as e:
        logger.error(f"Error filtering recipes: {e}")
        return

    if not recipes_list:
        logger.error("No recipes after filtering! Adjust filter parameters.")
        return

    # Сохраняем отфильтрованные рецепты
    try:
        output_path = os.path.join(output_dir, "Recipe_final.pickle")
        save_pkl(recipes_list, output_path)
        logger.info(f"Saved filtered recipes to {output_path}")
    except Exception as e:
        logger.error(f"Error saving filtered recipes: {e}")
        return

    # Проверяем и выводим page_content первого рецепта
    if recipes_list:
        if hasattr(recipes_list[0], 'document') and recipes_list[0].document is not None:
            logger.info(f"First recipe page_content: {recipes_list[0].document.page_content}")
        else:
            logger.error("First recipe has no document")
    else:
        logger.error("recipes_list is empty after filtering")

    # Создаём векторное хранилище
    logger.info("Creating vector store...")
    vector_store, tokenizer = create_db(recipes_list)
    if vector_store is None or tokenizer is None:
        logger.error("Failed to create vector store or tokenizer. Exiting.")
        return

    # Создаём теги
    logger.info("Creating tags...")
    try:
        tags = make_tags_list(recipes_list)
        lemmatize_tags(tags)
        one_word_tags = make_one_word_tags_list(tags)
        logger.info(f"Created {len(tags)} tags and {len(one_word_tags)} one-word tags")
    except Exception as e:
        logger.error(f"Error creating tags: {e}")
        return

    # Создаём граф знаний
    logger.info("Creating knowledge graph...")
    try:
        graph = build_knowledge_graph(recipes_list, tags)
        logger.info("Knowledge graph created successfully")
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        return

    # Создаём проект
    logger.info("Creating RecipesProject...")
    try:
        rp = RecipesProject(
            recipes=recipes_list,
            knowledgeGraph=graph,
            tags=tags,
            vectorStore=vector_store,
            oneWordTags=one_word_tags
        )
        logger.info("RecipesProject created successfully")
    except Exception as e:
        logger.error(f"Error creating RecipesProject: {e}")
        return

    # Сохраняем компоненты проекта
    try:
        save_pkl(rp.recipes, os.path.join(output_dir, "recipes.pickle"))
        save_pkl(rp.knowledgeGraph, os.path.join(output_dir, "knowledgeGraph.pickle"))
        save_pkl(rp.tags, os.path.join(output_dir, "tags.pickle"))
        save_pkl(rp.oneWordTags, os.path.join(output_dir, "oneWordTags.pickle"))
        logger.info("All project components saved successfully")
    except Exception as e:
        logger.error(f"Error saving project components: {e}")
        return

    logger.info("Project successfully created!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        sys.exit(1)