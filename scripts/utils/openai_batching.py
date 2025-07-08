import math
from typing import List, Tuple
import tiktoken
from config.config import MODEL_NAME

DEFAULT_MODEL_NAME = MODEL_NAME


class OpenAIConfig:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 1,
        max_tokens_per_request: int = 8000,
        max_output_tokens: int = 3000,
        temperature: float = 0.7,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens_per_request = max_tokens_per_request
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        # Set up tokenizer for the model
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text_into_chunks(self, text: str, max_tokens: int) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.decode(chunk) for chunk in chunks]

    def is_prompt_too_large(self, prompt: str) -> bool:
        return (
            self.count_tokens(prompt)
            > self.max_tokens_per_request - self.max_output_tokens
        )

    def batch_items(self, items: List, batch_size: int = None) -> List[List]:
        b_size = batch_size or self.batch_size
        return [items[i : i + b_size] for i in range(0, len(items), b_size)]

    def calculate_total_batches(self, num_items: int) -> int:
        return math.ceil(num_items / self.batch_size)
