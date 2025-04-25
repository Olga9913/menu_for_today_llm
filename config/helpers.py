import os
import re
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from config.dish import Recipe

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_recipe_html(file_name: str) -> Dict[str, Any]:
    """Парсит HTML файл рецепта и возвращает словарь с данными"""
    try:
        with open(file_name, 'r', encoding='utf-8') as site:
            soup = BeautifulSoup(site, 'html5lib')
        
        res = {}
        
        # Извлечение данных
        def extract_text(tag: str, attrs: Dict) -> List[str]:
            elements = soup.find_all(tag, attrs=attrs)
            return [x.get_text().strip() for x in elements if x.get_text().strip()]

        res['description'] = extract_text('span', {'class': 'detailed_full'})
        res['type'] = extract_text('span', {'itemprop': 'itemListElement'})
        
        # Обработка тегов
        tags_div = soup.find('div', class_='detailed_tags')
        res['tags'] = re.split(r"\n|/", tags_div.get_text().strip()) if tags_div else []
        res['tags'] = [t.strip() for t in res['tags'] if t.strip()]
        
        # Обработка ингредиентов
        res['ingridients'] = [
            [p.strip() for p in re.split(r"\n|/", ing.get_text().strip()) if p.strip()]
            for ing in soup.find_all('li', class_='ingredient')
        ]
        
        res['yield_value'] = extract_text('span', {'class': 'yield value'})
        
        # Обработка калорий
        calories_div = soup.find('div', class_='calories_info')
        if calories_div:
            res['calories_info'] = [c.strip() for c in re.split(r"\n|/", calories_div.get_text().strip()) if c.strip()]
        
        res['ratingValue'] = extract_text('span', {'itemprop': 'ratingValue'})
        res['ratingCount'] = extract_text('span', {'itemprop': 'ratingCount'})
        res['steps'] = extract_text('div', {'class': 'detailed_step_description_big'})
        res['time'] = extract_text('span', {'class': 'duration'})
        
        return res
    
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}")
        return {}

def make_data_set(dir_name: str) -> List[Dict[str, Any]]:
    """Обрабатывает все HTML файлы в директории"""
    try:
        files = [f for f in os.listdir(dir_name) if f.endswith('.html')]
        return [process_recipe_html(os.path.join(dir_name, f)) for f in tqdm(files)]
    except Exception as e:
        logger.error(f"Error in make_data_set: {str(e)}")
        return []

def save_pkl(data: Any, file_name: str) -> None:
    """Сохраняет данные в pickle файл"""
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.error(f"Error saving {file_name}: {str(e)}")
        raise

def read_pkl(file_name: str) -> Any:
    """Читает данные из pickle файла"""
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        try:
            with open(file_name, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e:
            logger.error(f"Error reading {file_name} with latin1: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error reading {file_name}: {str(e)}")
        raise

def make_recipes(recipes: List[Dict[str, Any]], output_file: Optional[str] = None) -> List[Recipe]:
    """Создает объекты Recipe из сырых данных"""
    result = []
    for idx, data in enumerate(tqdm(recipes)):
        try:
            recipe = Recipe(data)
            recipe.id = f"recipe_{idx}"
            recipe.standardize_time()
            recipe.make_document()
            result.append(recipe)
        except Exception as e:
            logger.warning(f"Skipping recipe {idx}: {str(e)}")
    
    if output_file:
        save_pkl(result, output_file)
    
    return result

def util_select_recipe(recipe: Recipe, max_steps: int = 10, max_min: int = 120,
                     min_rating: float = 4, min_votes: int = 5) -> bool:
    """Проверяет, соответствует ли рецепт критериям"""
    return all([
        bool(recipe.name),
        bool(recipe.ingridients),
        0 < len(recipe.steps) <= max_steps,
        recipe.ratingValue >= min_rating,
        recipe.ratingCount >= min_votes,
        recipe.standard_time <= max_min
    ])

def filter_recipe(recipes: List[Recipe], max_steps: int = 10, max_min: int = 120,
                min_rating: float = 4, min_votes: int = 5) -> List[Recipe]:
    """Фильтрует рецепты по заданным критериям"""
    filtered = [r for r in recipes if util_select_recipe(r, max_steps, max_min, min_rating, min_votes)]
    logger.info(f"Filtered {len(filtered)}/{len(recipes)} recipes")
    return filtered