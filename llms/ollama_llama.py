import requests
from typing import Any, Dict, Optional, List
import os
from dotenv import load_dotenv

from .base import LLM

load_dotenv()

class OllamaLlamaLLM(LLM):
    """
    Implementation of the LLM interface for LLaMA models using Ollama locally.
    """

    def __init__(
        self,
        model_name: str = os.getenv("OLLAMA_MODEL_NAME"),
        api_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the Ollama LLaMA LLM.

        Args:
            model_name (str): The name of the LLaMA model to use.
            api_url (str): The base URL for Ollama's API endpoint.
            **kwargs: Additional parameters.
        """
        self.model_name = model_name
        self.api_url = api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.parameters = kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using Ollama's local API.

        Args:
            prompt (str): The input text prompt.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens or self.parameters.get("max_tokens", 150),
            "temperature": temperature or self.parameters.get("temperature", 0.7),
            **kwargs
        }

        response = requests.post(self.api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("generated_text", "").strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Simulate a chat by concatenating messages and generating a response via Ollama.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content'.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        # Concatenate messages into a single prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
            else:
                prompt += f"{role.capitalize()}: {content}\n"

        prompt += "Assistant: "  # Prompt the model to continue as assistant

        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def tokenize(self, text: str) -> int:
        """
        Tokenize the input text and return the number of tokens.

        Args:
            text (str): The input text.

        Returns:
            int: Number of tokens.
        """
        # Placeholder: Implement tokenization based on Ollama's tokenizer or use a compatible library
        import tiktoken  # Ensure this library is compatible

        # Example: Adjust encoding based on the model if needed
        encoding = tiktoken.get_encoding("gpt2")  # Replace with appropriate encoding
        tokens = encoding.encode(text)
        return len(tokens)

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Set or update parameters for the Ollama LLM.

        Args:
            **kwargs: Parameters to set.
        """
        self.parameters.update(kwargs)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve current parameters of the Ollama LLM.

        Returns:
            Dict[str, Any]: Current parameters.
        """
        return self.parameters.copy()
