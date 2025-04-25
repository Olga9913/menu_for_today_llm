from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import logging
import os

logger = logging.getLogger(__name__)

def build_chroma_db(recipes_list: List['Recipe'], model_name: str = "all-MiniLM-L6-v2") -> Tuple[Optional[Chroma], Optional[AutoTokenizer]]:
    """
    Создаёт векторное хранилище для списка рецептов.

    Args:
        recipes_list: Список объектов Recipe.
        model_name: Название модели для эмбеддингов (например, 'all-MiniLM-L6-v2' или 'ai-forever/sbert_large_nlu_ru').

    Returns:
        Кортеж (vector_store, tokenizer) или (None, None) в случае ошибки.
    """
    if not recipes_list:
        logger.error("No recipes provided - cannot create vector store")
        return None, None

    try:
        # Инициализируем модель SentenceTransformer
        logger.info(f"Loading SentenceTransformer: {model_name}")
        embedding_function = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding function: {model_name}")

        # Создаём документы для ChromaDB
        documents = [recipe.document for recipe in recipes_list if hasattr(recipe, 'document') and recipe.document is not None]
        if not documents:
            logger.error("No valid documents found in recipes")
            return None, None

        # Инициализируем ChromaDB
        logger.info("Creating ChromaDB collection: recipes_collection")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            collection_name="recipes_collection",
            persist_directory="./chroma_db"
        )
        logger.info(f"Created ChromaDB collection: recipes_collection")

        # Загружаем токенизатор
        logger.info(f"Loading tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer for {model_name}")

        # Сохраняем векторное хранилище
        vector_store.persist()
        logger.info(f"Added {len(documents)} recipes to vector store")
        return vector_store, tokenizer

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None, None

def request_chroma_db(vector_store: Chroma, query: str, tokenizer: AutoTokenizer, top_k: int = 5) -> List[Document]:
    """
    Выполняет запрос к векторному хранилищу.

    Args:
        vector_store: Векторное хранилище Chroma.
        query: Текстовый запрос.
        tokenizer: Токенизатор для обработки запроса.
        top_k: Количество возвращаемых результатов.

    Returns:
        Список документов, наиболее релевантных запросу.
    """
    try:
        results = vector_store.similarity_search(query, k=top_k)
        logger.info(f"Retrieved {len(results)} results for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error querying vector store: {e}")
        return []