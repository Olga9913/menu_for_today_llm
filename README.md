# menu_for_today_llm

## Описание
Генератор по созданию рецептов по запросам пользователей

Проект использует LLM (Mistral 7B) и RAG для подбора рецептов на основе:
- Диетических ограничений (вегетарианское, безглютеновое и т.д.).
- Доступных ингредиентов.
- Времени приготовления.

## Запуск проекта

1. **Установка окружения**:
   ```bash
   python -m venv healthyai_venv
   source healthyai_venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download ru_core_news_sm
2. **Запуск Ollama (Mistral 7B)**:
    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull mistral
    ollama serve &
3. **Подготовка данных**:

- Разархивируйте файлы в data/recipes_*.zip.

- Запустите обработку данных:

    ```bash
    python scripts/app.py --prepare-data
4. **Запуск RAG и графа знаний**:

    python config/chroma_setup.py  # Создание базы векторов
    python config/graph_builder.py # Построение графа
5. **Запуск приложения**:

    python scripts/app.py --query "Вегетарианская паста"


### **Особенности проекта**  
 
- **Персонализация**:  
  - Подбор рецептов на основе **калорийности, баланса БЖУ и времени приготовления**.  
  - Учет диетических ограничений: **веганские, безглютеновые, низкоуглеводные** варианты.  
- **ЗОЖ-метрики**:  
  - Оценка рецептов по **полезности** (например, низкое содержание сахара, высокое содержание белка).  
  - Возможность **замены ингредиентов** на более здоровые альтернативы.  
- **Гибкость и масштабируемость**:  
  - Поддержка **FastAPI** для будущего развития REST API.  
  - Четкая структура кода (`config/`, `scripts/`, `notebooks/`) для удобной доработки.  

#### **Технические улучшения**  
- Использование **Mistral 7B** через Ollama для быстрого развертывания без обучения.  
- Оптимизированная обработка данных (**30–60 минут** для 1000+ рецептов).  
- Векторный поиск (Chroma DB) + граф знаний для точных рекомендаций.  