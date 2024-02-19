import asyncio
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import google.generativeai as genai

from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content, Executor,
                                  Message, TokenStopNode, User, awaitable_none)
from llmvm.common.perf import TokenPerf, TokenPerfWrapper

logging = setup_logging()

class GeminiExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('GOOGLE_API_KEY')),
        # from the docs:
        # Note: The vision model gemini-pro-vision is not optimized for multi-turn chat.
        default_model: str = 'gemini-pro',
        api_endpoint: str = '',
        default_max_token_len: int = 32768,
        default_max_completion_len: int = 2048,
    ):
        self.api_key = api_key
        self.default_model = default_model
        self.api_endpoint = api_endpoint
        self.default_max_token_len = default_max_token_len
        self.default_max_completion_len = default_max_completion_len
        genai.configure(api_key=api_key)

        self.aclient = genai.GenerativeModel(self.default_model)

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def max_tokens(self, model: Optional[str]) -> int:
        model = model if model else self.default_model
        match model:
            case 'gemini-pro':
                return 34748
            case _:
                logging.warning(f'max_tokens() is not implemented for model {model}. Returning {self.default_max_token_len}')
                return self.default_max_token_len

    def set_default_model(self, default_model: str):
        self.default_model = default_model

    def get_default_model(self):
        return self.default_model

    def set_default_max_tokens(self, default_max_tokens: int):
        self.default_max_token_len = default_max_tokens

    def max_prompt_tokens(
        self,
        completion_token_len: Optional[int] = None,
        model: Optional[str] = None,
    ) -> int:
        return self.max_tokens(model) - (completion_token_len if completion_token_len else self.default_max_completion_len)

    def max_completion_tokens(
        self,
        model: Optional[str] = None,
    ):
        return self.default_max_completion_len

    def count_tokens(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        extra_str: str = '',
        model: Optional[str] = None,
    ) -> int:
        model_str = model if model else self.default_model

        # obtained from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        def num_tokens_from_messages(messages, model: str):
            if model in {
                "gemini-pro",
            }:
                tokens_per_message = 3
            else:
                logging.error(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa: E501
                tokens_per_message = 3
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for _, value in message.items():
                    num_tokens += self.aclient.count_tokens(value).total_tokens
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], Message):
            dict_messages = [Message.to_dict(m) for m in messages]  # type: ignore
            dict_messages.append(Message.to_dict(User(Content(extra_str))))
            return num_tokens_from_messages(dict_messages, model=model_str)
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            return num_tokens_from_messages(messages, model=model_str)
        elif isinstance(messages, str):
            return num_tokens_from_messages([Message.to_dict(User(Content(messages)))], model=model_str)
        else:
            raise ValueError('cannot calculate tokens for messages: {}'.format(messages))

    def name(self) -> str:
        return 'gemini'

    def __dict_message_to_gemini_message(
        self,
        message: Dict[str, str]
    ) -> Dict:
        role = 'user'
        if message['role'] == 'assistant':
            role = 'model'

        return {
            'role': role,
            'parts': [message['content']],
        }

    async def aexecute_direct(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_completion_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        model = model if model else self.default_model

        # only works if profiling or LLMVM_PROFILING is set to true
        message_tokens = self.count_tokens(messages, model=model)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens, model=model):
            raise Exception('Prompt too long, message tokens: {}, completion tokens: {} total tokens: {}, available tokens: {}'
                            .format(message_tokens,
                                    max_completion_tokens,
                                    message_tokens + max_completion_tokens,
                                    self.max_tokens(model)))

        token_trace = TokenPerf('aexecute_direct', 'gemini', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        gemini_messages = [self.__dict_message_to_gemini_message(m) for m in messages]

        for message in gemini_messages:
            if not message['parts'] or message['parts'] == '' or message['parts'] == b'':
                gemini_messages.remove(message)

        # the messages API also doesn't allow for multiple User or Assistant messages in a row, so we're
        # going to add an Assistant message in between two User messages, and a User message between two Assistant.
        messages_list: List[Dict[str, Any]] = []

        for i in range(len(gemini_messages)):
            if i > 0 and gemini_messages[i]['role'] == gemini_messages[i - 1]['role']:
                if gemini_messages[i]['role'] == 'user':
                    messages_list.append({'role': 'model', 'parts': ['Thanks. I am ready for your next message.']})
                elif gemini_messages[i]['role'] == 'model':
                    messages_list.append({'role': 'user', 'parts': ['Thanks. I am ready for your next message.']})
            messages_list.append(gemini_messages[i])

        response = await self.aclient.generate_content_async(
            contents=messages_list,
            stream=True
        )
        return TokenPerfWrapper(response, token_trace)  # type: ignore

    async def aexecute(
        self,
        messages: List[Message],
        max_completion_tokens: int = 4096,
        temperature: float = 0.2,
        model: Optional[str] = None,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model

        # fresh message list
        messages_list: List[Dict[str, str]] = []

        for message in [m for m in messages if m.role() != 'system']:
            messages_list.append(Message.to_dict(message))

        chat_response = self.aexecute_direct(
            messages_list,
            max_completion_tokens=max_completion_tokens,
            model=model if model else self.default_model,
            temperature=temperature,
        )

        text_response = ''

        async for chunk in await chat_response:  # type: ignore
            s = chunk.text or ''
            await stream_handler(Content(s))
            text_response += s
        await stream_handler(TokenStopNode())

        messages_list.append({'role': 'assistant', 'content': text_response})
        conversation: List[Message] = [Message.from_dict(m) for m in messages_list]

        assistant = Assistant(
            message=conversation[-1].message,
            messages_context=conversation
        )

        return assistant

    def execute(
        self,
        messages: List[Message],
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        model: Optional[str] = None,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_completion_tokens, temperature, model, stream_pipe))
