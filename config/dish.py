import re
import uuid
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from config.food_graph import query_graph, ATTRIBUTES_ORDER, enreach_query_with_relative_tags
import logging

# Настройка логирования
logger = logging.getLogger(__name__)

def make_str(obj: Any) -> str:
    """
    Преобразует объект в строку, объединяя элементы списка через запятые.

    Args:
        obj: Объект (строка, список или другой тип).

    Returns:
        Строковое представление объекта.
    """
    res = ''
    if isinstance(obj, list):
        for obj_ in obj:
            if res == "":
                res = make_str(obj_)
            else:
                res += ", " + make_str(obj_)
    else:
        res = str(obj)
    return res

class Recipe:
    def __init__(self, recipe: Optional[Dict] = None):
        """
        Инициализирует объект Recipe.

        Args:
            recipe: Словарь с данными рецепта (по умолчанию None).
        """
        self.make_keys_dict()
        for key in self.keys_dict:
            setattr(self, key, None)
        if recipe is not None:
            self.add_features(recipe)
            self.clean_tags()
            self.add_tags()
            self.standardize_time()

    def make_document(self) -> None:
        """
        Создаёт атрибуты display и document для рецепта.
        """
        self.display = self.make_str_recipe()
        self.document = Document(
            page_content=self.make_str_recipe_db(),
            metadata={'id': self.id, 'name': self.name}
        )
        logger.debug(f"Document created for recipe: {self.name}")

    def display(self) -> None:
        """
        Выводит все атрибуты рецепта.
        """
        for attr in self.__dict__.keys():
            print(f"{attr}: {getattr(self, attr)}")

    def clean_tags(self) -> None:
        """
        Очищает теги, удаляя ненужные слова (например, 'кухня', 'рецепты').
        """
        for attr in ATTRIBUTES_ORDER:
            if attr in self.__dict__ and getattr(self, attr) is not None:
                tags = getattr(self, attr)
                if attr == 'ingridients':
                    new_tags = [
                        (
                            tag[0].replace('кухня', '').replace('рецепты', '')
                            .replace('питание', '').replace('для', '')
                            .replace('блюда', '').strip(),
                            tag[1]
                        )
                        for tag in tags
                    ]
                else:
                    new_tags = [
                        tag.replace('кухня', '').replace('рецепты', '')
                        .replace('питание', '').replace('для', '')
                        .replace('блюда', '').strip()
                        for tag in tags
                    ]
                setattr(self, attr, new_tags)
                logger.debug(f"Cleaned tags for attribute {attr}")

    def add_tags(self) -> None:
        """
        Добавляет дополнительные теги на основе существующих (например, 'детский' для 'для детей').
        """
        for attr in ATTRIBUTES_ORDER:
            if attr == 'ingridients':
                continue
            if attr in self.__dict__ and getattr(self, attr) is not None:
                tags = getattr(self, attr)
                new_tags = tags.copy()
                if 'для детей' in set(tags):
                    new_tags.append('детский')
                if 'пп' in set(tags):
                    new_tags.extend(['полезный', 'здоровый'])
                if 'на скорую руку' in set(tags):
                    new_tags.append('быстрый')
                setattr(self, attr, new_tags)
                logger.debug(f"Added tags for attribute {attr}")

    def standardize_time(self) -> None:
        """
        Преобразует время приготовления в минуты и сохраняет в standard_time.
        """
        if self.time is None:
            self.standard_time = 0
            return

        s = self.time.replace('ч.', 'ч')
        days = hours = minutes = 0
        if "дн." in s:
            days, right = s.split("дн.", 1)
        else:
            right = s
        if "ч" in right:
            hours, right = right.split("ч", 1)
        if "мин" in right:
            minutes = right.split("мин")[0]
        try:
            days = int(days.strip()) if days.strip() else 0
            hours = int(hours.strip()) if hours.strip() else 0
            minutes = int(minutes.strip()) if minutes.strip() else 0
            self.standard_time = days * 60 * 24 + hours * 60 + minutes
        except ValueError as e:
            logger.error(f"Error standardizing time for recipe {self.name}: {e}")
            self.standard_time = 0
        logger.debug(f"Standardized time for recipe {self.name}: {self.standard_time} minutes")

    def make_str_recipe(self) -> str:
        """
        Формирует полное строковое представление рецепта для отображения.

        Returns:
            Строковое представление рецепта.
        """
        res = ""
        if self.name is not None:
            res += self.name + "\n\n"
        if self.recipeYield is not None:
            res += self.keys_dict["recipeYield"] + ": " + make_str(self.recipeYield) + ".\n"
        if self.ingridients is not None:
            res += self.keys_dict["ingridients"] + "\n"
            for ing in self.ingridients:
                res += "\t" + ing[0] + ": " + ing[1] + ".\n"
        if self.steps is not None:
            res += "\n" + self.keys_dict["steps"] + "\n"
            for n, step in enumerate(self.steps):
                res += "\t" + str(n + 1) + ". " + step + ".\n"
        for key in self.keys_dict_order[5:10]:
            if key in self.__dict__ and getattr(self, key) is not None:
                res += self.keys_dict[key] + ": " + make_str(getattr(self, key)) + ".\n"
        return res

    def make_str_recipe_db(self) -> str:
        """
        Формирует строковое представление рецепта для базы данных.

        Returns:
            Строковое представление рецепта (название, ингредиенты, шаги).
        """
        res = ""
        if self.name is not None:
            res += self.name + " "
        if self.ingridients is not None:
            res += ", ".join([ing[0] for ing in self.ingridients])
        if self.steps is not None:
            res += ", ".join(self.steps)
        return res.strip()

    def add_features(self, recipe: Dict) -> None:
        """
        Извлекает и устанавливает атрибуты рецепта из словаря.

        Args:
            recipe: Словарь с данными рецепта.
        """
        # Название и тип блюда
        if len(recipe.get('type', [])) > 0:
            self.name = recipe['type'][-1]
            self.meal = recipe['type'][2:-1]
        else:
            self.name = None

        # Описание
        if len(recipe.get('description', [])) == 2:
            self.description = [recipe['description'][0], recipe['description'][1].split(":")[1].strip()]

        # Ингредиенты
        self.ingridients = [None] * len(recipe.get('ingridients', []))
        for n, ing_set in enumerate(recipe.get('ingridients', [])):
            if '       ' in ing_set[1]:
                ing_set_1 = [d.strip() for d in ing_set[1].split('       ') if d]
                self.ingridients[n] = (ing_set[0].lower(), ing_set_1[0].lower(), ing_set_1[1].lower() if len(ing_set_1) > 1 else None)
            else:
                self.ingridients[n] = (ing_set[0].lower(), ing_set_1[0].lower(), None)

        # Маппинг атрибутов
        attr_name = {
            'yield_value': 'recipeYield',
            'Калорийность': 'calories',
            'Жиры': 'fatContent',
            'Углеводы': 'carbohydrateContent',
            'Белки': 'proteinContent',
            'Назначение': 'occasions',
            'Диета': 'diet',
            'Основной ингредиент': 'mainIngridients',
            'География кухни': 'geography',
            'Блюдо': 'type_',
        }

        # Теги
        _tags = {}
        if 'tags' in recipe:
            i = 0
            key = None
            while i < len(recipe['tags']):
                if ":" in recipe['tags'][i]:
                    key = recipe['tags'][i].replace(":", "").strip()
                    _tags[key] = []
                else:
                    _tags[key].append(recipe['tags'][i])
                i += 1
            for key in _tags:
                setattr(self, attr_name[key], [t.lower() for t in _tags[key]])

        # Рейтинг и количество оценок
        self.ratingValue = float(recipe.get('ratingValue', [0])[0]) if recipe.get('ratingValue') else 0.0
        self.ratingCount = float(recipe.get('ratingCount', [0])[0]) if recipe.get('ratingCount') else 0

        # Время
        self.time = recipe.get('time', [None])[0]

        # Количество порций
        if recipe.get('yield_value', []):
            setattr(self, attr_name['yield_value'], float(recipe['yield_value'][0]))

        # Калорийность
        if 'calories_info' in recipe:
            for i in range(0, len(recipe['calories_info']), 2):
                val, key = recipe['calories_info'][i], recipe['calories_info'][i + 1]
                setattr(self, attr_name[key], val)

        # Шаги
        if 'steps' in recipe:
            self.steps = [re.sub(r'^\d+\.?\s*', '', s.split("\t")[-1].strip()) for s in recipe['steps']]

        logger.debug(f"Features added for recipe: {self.name}")

    def make_keys_dict(self) -> None:
        """
        Создаёт словари keys_dict и keys_dict_order для атрибутов рецепта.
        """
        self.keys_dict = {
            'name': 'Название',
            'description': 'Описание',
            'recipeYield': 'Количество порций',
            'ingridients': 'Ингридиенты',
            'steps': 'Способ приготовления',
            'calories': 'Калории',
            'proteinContent': 'Белки',
            'fatContent': 'Жиры',
            'carbohydrateContent': 'Углеводы',
            'time': 'Время приготовления',
            'meal': 'Тип блюда',
            'occasions': 'Назначение',
            'diet': 'Диета',
            'mainIngridients': 'Основные ингредиенты',
            'geography': 'География кухни',
            'ratingValue': 'Средняя оценка',
            'ratingCount': 'Количество оценок',
            'standard_time': 'standard_time'
        }

        self.keys_dict_order = [
            'name', 'description', 'recipeYield', 'ingridients', 'steps',
            'calories', 'proteinContent', 'fatContent', 'carbohydrateContent',
            'time', 'meal', 'occasions', 'diet', 'mainIngridients', 'geography',
            'ratingValue', 'ratingCount'
        ]

class RecipesProject:
    def __init__(self, recipes: Optional[List[Recipe]] = None, knowledgeGraph: Optional[Any] = None,
                 tags: Optional[List[str]] = None, vectorStore: Optional[Any] = None,
                 oneWordTags: Optional[List[str]] = None):
        """
        Инициализирует объект RecipesProject.

        Args:
            recipes: Список объектов Recipe.
            knowledgeGraph: Граф знаний.
            tags: Список тегов.
            vectorStore: Векторное хранилище (например, ChromaDB).
            oneWordTags: Список однословных тегов.
        """
        self.recipes = recipes if recipes is not None else []
        self.knowledgeGraph = knowledgeGraph
        self.tags = tags if tags is not None else []
        self.vectorStore = vectorStore
        self.oneWordTags = oneWordTags if oneWordTags is not None else []
        self.model = None
        self.tokenizer = None
        logger.info("RecipesProject initialized")

    def add_recipes_list(self, recipes: List[Recipe]) -> None:
        """
        Добавляет список рецептов в проект.

        Args:
            recipes: Список объектов Recipe.
        """
        if not isinstance(recipes, list):
            logger.error("Recipes must be a list")
            raise ValueError("Recipes must be a list")
        self.recipes = recipes
        logger.info(f"Added {len(recipes)} recipes to RecipesProject")

    def add_knowledge_graph(self, knowledgeGraph: Any) -> None:
        """
        Добавляет граф знаний в проект.

        Args:
            knowledgeGraph: Граф знаний.
        """
        self.knowledgeGraph = knowledgeGraph
        logger.info("Knowledge graph added to RecipesProject")

    def add_tags(self, tags: List[str]) -> None:
        """
        Добавляет список тегов в проект.

        Args:
            tags: Список тегов.
        """
        if not isinstance(tags, list):
            logger.error("Tags must be a list")
            raise ValueError("Tags must be a list")
        self.tags = tags
        logger.info(f"Added {len(tags)} tags to RecipesProject")

    def add_vector_store(self, vectorStore: Any) -> None:
        """
        Добавляет векторное хранилище в проект.

        Args:
            vectorStore: Векторное хранилище (например, ChromaDB).
        """
        self.vectorStore = vectorStore
        logger.info("Vector store added to RecipesProject")

    def add_one_word_tags(self, oneWordTags: List[str]) -> None:
        """
        Добавляет список однословных тегов в проект.

        Args:
            oneWordTags: Список однословных тегов.
        """
        if not isinstance(oneWordTags, list):
            logger.error("OneWordTags must be a list")
            raise ValueError("OneWordTags must be a list")
        self.oneWordTags = oneWordTags
        logger.info(f"Added {len(oneWordTags)} one-word tags to RecipesProject")

    def enrich_query_with_tags(self, query: str, verbose: bool = False) -> str:
        """
        Обогащает запрос дополнительными тегами из oneWordTags.

        Args:
            query: Исходный текстовый запрос.
            verbose: Если True, выводит информацию о добавленных тегах.

        Returns:
            Обогащённый запрос.
        """
        new_tags = enreach_query_with_relative_tags(query, self.oneWordTags)
        new_tags_str = " ".join(new_tags)
        if verbose:
            if new_tags_str:
                print(f"New tags are added: {new_tags_str}")
            else:
                print("New tags are not added")
        res = f"{query} {new_tags_str}".strip()
        logger.debug(f"Enriched query: {res}")
        return res

    def invoke(self, query: str, verbose: bool = False) -> str:
        """
        Обрабатывает запрос, используя граф знаний и теги.

        Args:
            query: Текстовый запрос.
            verbose: Если True, выводит дополнительную информацию.

        Returns:
            Ответ на запрос.
        """
        enriched_query = self.enrich_query_with_tags(query, verbose=verbose)
        answer = query_graph(enriched_query, self.knowledgeGraph, self.tags, verbose=verbose)
        logger.info(f"Processed query: {query}, Answer: {answer}")
        return answer