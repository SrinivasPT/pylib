import os
from typing import Any, Dict, Optional, List

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .base import LLM

class LlamaLLM(LLM):
    """
    Implementation of the LLM interface for LLaMA models using Hugging Face Transformers.
    Supports both generate and simulated chat via context management.
    """

    def __init__(
        self,
        model_name: str = "huggyllama/llama-7b",
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize the LLaMA LLM.

        Args:
            model_name (str): Hugging Face model identifier.
            tokenizer_name (Optional[str]): Hugging Face tokenizer identifier. Defaults to model_name.
            device (Optional[str]): Device to run the model on ('cpu', 'cuda'). Defaults to 'cuda' if available.
            **kwargs: Additional parameters.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parameters = kwargs

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        ).to(self.device)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text using the LLaMA model.

        Args:
            prompt (str): The input text prompt.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        max_length = (input_ids.shape[1] + (max_tokens or self.parameters.get("max_tokens", 150)))

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature or self.parameters.get("temperature", 0.7),
            **kwargs
        )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """
        Simulate a chat by concatenating messages and generating a response.

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
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Set or update parameters for the LLaMA LLM.

        Args:
            **kwargs: Parameters to set.
        """
        self.parameters.update(kwargs)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve current parameters of the LLaMA LLM.

        Returns:
            Dict[str, Any]: Current parameters.
        """
        return self.parameters.copy()
