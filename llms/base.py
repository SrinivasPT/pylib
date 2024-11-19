from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class LLM(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response from the language model based on the given prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response based on a sequence of messages for conversational context."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> int:
        """
        Tokenize the input text and return the number of tokens.

        Args:
            text (str): The input text.

        Returns:
            int: Number of tokens.
        """
        pass

    @abstractmethod
    def set_parameters(self, **kwargs: Any) -> None:
        """
        Set or update parameters for the LLM.

        Args:
            **kwargs: Parameters to set.
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve current parameters of the LLM.

        Returns:
            Dict[str, Any]: Current parameters.
        """
        pass

