from llms import OpenAILLM, LlamaLLM, OllamaLlamaLLM
from dotenv import load_dotenv
import os
import logging

# Configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def use_openai_generate():
    """
    Demonstrates text generation using OpenAI's Completion API.
    """
    try:
        openai_llm = OpenAILLM(
            model="text-davinci-003",
            api_key="your-openai-api-key",  # Replace with your actual API key
            max_tokens=100,
            temperature=0.5
        )
        prompt = "Once upon a time in a land far, far away,"
        logger.info("Generating text using OpenAI Completion API...")
        response = openai_llm.generate(prompt)
        print("OpenAI Completion Response:")
        print(response)
    except Exception as e:
        logger.error(f"Error in use_openai_generate: {e}")

def use_openai_chat():
    """
    Demonstrates chat functionality using OpenAI's Chat API.
    """
    try:
        openai_chat_llm = OpenAILLM(
            model="text-davinci-003",        # Completion model
            chat_model="gpt-3.5-turbo",      # Chat model
            api_key="your-openai-api-key",   # Replace with your actual API key
            max_tokens=100,
            temperature=0.5
        )
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"}
        ]
        logger.info("Generating chat response using OpenAI Chat API...")
        chat_response = openai_chat_llm.chat(messages)
        print("\nOpenAI Chat Response:")
        print(chat_response)
    except Exception as e:
        logger.error(f"Error in use_openai_chat: {e}")

def use_llama_generate():
    """
    Demonstrates text generation using LLaMA via Hugging Face Transformers.
    """
    try:
        llama_llm = LlamaLLM(
            model_name="huggyllama/llama-7b",  # Ensure this model is available locally or via Hugging Face
            max_tokens=100,
            temperature=0.5
        )
        prompt = "Once upon a time in a land far, far away,"
        logger.info("Generating text using LLaMA via Hugging Face...")
        response = llama_llm.generate(prompt)
        print("\nLLaMA Generate Response:")
        print(response)
    except Exception as e:
        logger.error(f"Error in use_llama_generate: {e}")

def use_llama_chat():
    """
    Demonstrates chat functionality using LLaMA via Hugging Face Transformers.
    """
    try:
        llama_llm = LlamaLLM(
            model_name="huggyllama/llama-7b",
            max_tokens=100,
            temperature=0.5
        )
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well! How can I help you today?"}
        ]
        logger.info("Generating chat response using LLaMA via Hugging Face...")
        chat_response = llama_llm.chat(messages)
        print("\nLLaMA Chat Response:")
        print(chat_response)
    except Exception as e:
        logger.error(f"Error in use_llama_chat: {e}")

def use_ollama_llama_generate():
    """
    Demonstrates text generation using LLaMA via Ollama locally.
    """
    try:
        ollama_llm = OllamaLlamaLLM(
            model_name=os.getenv("OLLAMA_MODEL_NAME"),
            api_url=os.getenv("OLLAMA_GENERATE_ENDPOINT"),
            max_tokens=100,
            temperature=0.5
        )
        prompt = "Once upon a time in a land far, far away,"
        logger.info("Generating text using LLaMA via Ollama...")
        response = ollama_llm.generate(prompt)
        print("\nOllama LLaMA Generate Response:")
        print(response)
    except Exception as e:
        logger.error(f"Error in use_ollama_llama_generate: {e}")

def use_ollama_llama_chat():
    """
    Demonstrates chat functionality using LLaMA via Ollama locally.
    """
    try:
        ollama_llm = OllamaLlamaLLM(
            model_name=os.getenv("OLLAMA_MODEL_NAME"),
            api_url=os.getenv("OLLAMA_CHAT_ENDPOINT"),  # Adjust based on your Ollama setup
            max_tokens=100,
            temperature=0.5
        )
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well! How can I help you today?"}
        ]
        logger.info("Generating chat response using LLaMA via Ollama...")
        chat_response = ollama_llm.chat(messages)
        print("\nOllama LLaMA Chat Response:")
        print(chat_response)
    except Exception as e:
        logger.error(f"Error in use_ollama_llama_chat: {e}")

def main():
    """
    Main function to execute all LLM functionalities.
    """
    # use_openai_generate()
    # use_openai_chat()
    # use_llama_generate()
    # use_llama_chat()
    # use_ollama_llama_generate()
    use_ollama_llama_chat()

if __name__ == "__main__":
    main()
