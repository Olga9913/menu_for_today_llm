from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

# Initialize chat history
def create_chat_history():
    chat_history = ChatMessageHistory()
    return chat_history

# Convert chat history to a formatted string
def formatted_chat_history(chat_history):
    res = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in chat_history.messages]
        )
    return res

def formatted_rag(rag):
    res = "\n".join(rag)
    return res
    
# Define the prompt template
TEMPLATE = """
Вы — опытный кулинарный помощник. Ваша цель — давать полезные, креативные и простые в использовании советы по приготовлению пищи.
У вас есть доступ к большой базе рецептов, хранящейся в ChromaDB, которая позволяет вам извлекать соответствующие рецепты и предлагать
персонализированные идеи приготовления пищи на основе предпочтений пользователя, доступных ингредиентов, диетических ограничений и стиля приготовления.

Используйте историю чата, чтобы сохранять контекст и избегать повторений. Если пользователь упомянул определенные ингредиенты,
предпочтения или антипатии ранее в разговоре, учтите это при предоставлении совета.

Используйте выбранные рецепты из ChromaDB в качестве руководства для создания индивидуальных предложений. Вы можете комбинировать элементы из разных рецептов или адаптировать их на основе предпочтений пользователя.

Четко форматируйте ответы, включая:
1. **Название**: название рецепта или предложения.
2. **Ингредиенты**: перечислите ингредиенты с количеством.
3. **Инструкция**: пошаговый процесс.
4. **Советы/Альтернативы**: предоставьте полезные советы или альтернативы, если определенные ингредиенты недоступны.

---

**История чата:**  
{chat_history}

**Запрос:**  
{user_input}

- **Выбранные рецепты из ChromaDB:**  
{retrieved_recipes}

---

Дайте понятный, полезный и дружелюбный ответ на запрос пользователя.
Отвечайте на русском языке
"""

# Create the Langchain prompt
prompt = PromptTemplate(
    input_variables=[
        "chat_history", 
        "user_input", 
        "available_ingredients", 
        "dietary_preferences", 
        "retrieved_recipes"
    ],
    template=TEMPLATE
)