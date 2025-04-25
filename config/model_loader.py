from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

MODEL_NAME = "mistral"  # Change to "phi3" or "gemma:2b" if needed

# Initialize Ollama LLM

def build_model(model=MODEL_NAME):
    llm = OllamaLLM(
        model=MODEL_NAME,
        keep_alive=-1,
        repeat_last_n=0,
        top_k=40,
        top_p=0.6,
        temperature=0.5,
        # format="json",
    )
    return llm

SYSTEM_PROMPT = """Ты — кулинарный помощник, который рекомендует рецепты на основе запросов пользователей. "
            "Используй доступные ингредиенты чтобы предложить лучший рецепт. "
            "Выбери рецепты, в которые входят указанные пользователем ингридиенты. "
            "Опиши рецепт кратко: укажи название, основные ингредиенты и способ приготовления. "
            "Отвечай только на русском языке."""

def llm_invoke(model, request_str):
    
    # Create a properly formatted prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", request_str)
    ])
    
    # Format the final prompt before passing it to `llm.invoke()`
    formatted_prompt = prompt.format_messages()
    
    # Get response from LLM
    response = model.invoke(formatted_prompt)
    
    # Print the response
    return response 
