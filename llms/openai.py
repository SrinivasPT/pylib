import os
import openai
from typing import Any, Dict, Optional, List

from .base import LLM

class OpenAILLM(LLM):
    """
    Implementation of the LLM interface for OpenAI's GPT models.
    Supports both Completion and Chat endpoints.
    """

    def __init__(
        self,
        model: str = "text-davinci-003",
        chat_model: Optional[str] = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model (str): The OpenAI completion model to use.
            chat_model (Optional[str]): The OpenAI chat model to use.
            api_key (Optional[str]): OpenAI API key.
            **kwargs: Additional parameters.
        """
        self.completion_model = model
        self.chat_model = chat_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as a parameter or via the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key
        self.parameters = kwargs

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using OpenAI's Completion API.

        Args:
            prompt (str): The input text prompt.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            **kwargs: Additional OpenAI API parameters.

        Returns:
            str: The generated text.
        """
        response = openai.Completion.create(
            model=self.completion_model,
            prompt=prompt,
            max_tokens=max_tokens or self.parameters.get("max_tokens", 150),
            temperature=temperature or self.parameters.get("temperature", 0.7),
            **kwargs
        )
        return response.choices[0].text.strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using OpenAI's Chat API.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries with 'role' and 'content'.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            **kwargs: Additional OpenAI API parameters.

        Returns:
            str: The generated text.
        """
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens or self.parameters.get("max_tokens", 150),
            temperature=temperature or self.parameters.get("temperature", 0.7),
            **kwargs
        )
        return response.choices[0].message['content'].strip()

    def tokenize(self, text: str) -> int:
        """
        Estimate the number of tokens in the input text using OpenAI's tokenizer.

        Args:
            text (str): The input text.

        Returns:
            int: Number of tokens.
        """
        import tiktoken

        # Select appropriate encoding based on model
        if self.completion_model.startswith("gpt-3.5") or self.completion_model.startswith("gpt-4"):
            encoding = tiktoken.encoding_for_model(self.completion_model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")  # Default

        tokens = encoding.encode(text)
        return len(tokens)

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Set or update parameters for the OpenAI LLM.

        Args:
            **kwargs: Parameters to set.
        """
        self.parameters.update(kwargs)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve current parameters of the OpenAI LLM.

        Returns:
            Dict[str, Any]: Current parameters.
        """
        return self.parameters.copy()
