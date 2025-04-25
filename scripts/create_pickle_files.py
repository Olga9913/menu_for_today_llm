import os
import sys
from pathlib import Path
import logging
from typing import List

# Добавляем корень проекта в Python path
project_root = Path(__file__).resolve().parents[2]  # На два уровня вверх от scripts/
# sys.path.append(str(project_root))
sys.path.append("/content/drive/MyDrive/llm_kaggle/llm_kaggle")

from config.helpers import make_data_set, make_recipes, save_pkl
from config.dish import Recipe

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('create_pickle_files.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Пути к данным
HTML_DIR = "/content/drive/MyDrive/demo_500_recipes/"  # Укажите путь к папке с HTML-файлами
OUTPUT_DIR = "/content/drive/MyDrive/llm_kaggle/llm_kaggle/dataset/demo_500/"
FILE_PREFIX = "recipes_"

# Размер части для каждого файла .pickle
CHUNK_SIZE = 166  # Количество рецептов в одном файле .pickle

def create_pickle_files(html_dir: str, output_dir: str, file_prefix: str, chunk_size: int) -> None:
    """
    Обрабатывает HTML-файлы, создаёт объекты Recipe и сохраняет их в файлы .pickle.

    Args:
        html_dir (str): Путь к папке с HTML-файлами.
        output_dir (str): Путь к папке для сохранения .pickle файлов.
        file_prefix (str): Префикс для имен файлов .pickle (например, 'recipes_').
        chunk_size (int): Количество рецептов в одном файле .pickle.
    """
    try:
        # Проверяем существование входной директории
        if not os.path.exists(html_dir):
            logger.error(f"Input directory {html_dir} does not exist")
            raise FileNotFoundError(f"Directory {html_dir} not found")

        # Создаём выходную директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory {output_dir} created or already exists")

        # Обрабатываем HTML-файлы
        logger.info(f"Processing HTML files from {html_dir}...")
        raw_recipes = make_data_set(html_dir)
        logger.info(f"Extracted {len(raw_recipes)} raw recipes from HTML files")

        if not raw_recipes:
            logger.error("No recipes extracted from HTML files")
            return

        # Преобразуем в объекты Recipe
        logger.info("Converting raw recipes to Recipe objects...")
        recipes_list = make_recipes(raw_recipes)
        logger.info(f"Created {len(recipes_list)} Recipe objects")

        if not recipes_list:
            logger.error("No Recipe objects created")
            return

        # Разбиваем рецепты на части и сохраняем в .pickle файлы
        logger.info(f"Saving recipes to .pickle files with chunk size {chunk_size}...")
        for i in range(0, len(recipes_list), chunk_size):
            chunk = recipes_list[i:i + chunk_size]
            file_number = (i // chunk_size) + 1
            output_file = os.path.join(output_dir, f"{file_prefix}{file_number}.pickle")
            try:
                save_pkl(chunk, output_file)
                logger.info(f"Saved {len(chunk)} recipes to {output_file}")
            except Exception as e:
                logger.error(f"Error saving {output_file}: {e}")
                continue

        logger.info("All .pickle files created successfully")

    except Exception as e:
        logger.error(f"Unexpected error in create_pickle_files: {e}")

if __name__ == "__main__":
    try:
        create_pickle_files(HTML_DIR, OUTPUT_DIR, FILE_PREFIX, CHUNK_SIZE)
    except Exception as e:
        logger.error(f"Failed to run create_pickle_files: {e}")
        exit(1)