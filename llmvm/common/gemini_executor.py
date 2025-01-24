import os
from typing import cast

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.openai_executor import OpenAIExecutor

logging = setup_logging()

class GeminiExecutor(OpenAIExecutor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('GEMINI_API_KEY')),
        default_model: str = 'gemini-1.5-pro',
        api_endpoint: str = 'https://generativelanguage.googleapis.com/v1beta',
        default_max_input_len: int = 2097152,
        default_max_output_len: int = 8192,
        max_images: int = 20,
    ):
        super().__init__(
            api_key=api_key,
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_input_len=default_max_input_len,
            default_max_output_len=default_max_output_len,
            max_images=max_images,
        )

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def scratchpad_token(self) -> str:
        return 'scratchpad'

    def name(self) -> str:
        return 'gemini'
