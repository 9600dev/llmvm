import asyncio
import datetime as dt
import os
from typing import (Any, Awaitable, Callable, Dict, Generator, Generic, List,
                    Optional, Sequence, Tuple, TypeVar, Union, cast)

from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic
from tokenizers import Tokenizer

from container import Container
from helpers.logging_helpers import TimedLogger, response_writer, setup_logging
from objects import (Assistant, AstNode, Content, Executor, Message, System,
                     TokenStopNode, User, awaitable_none)

logging = setup_logging()

class AnthropicExecutor(Executor):
    def __init__(
        self,
        api_key: str,
        default_model: str = 'claude-2',
        api_endpoint: str = 'https://api.anthropic.com',
        default_max_tokens: int = 200000,
    ):
        self.default_max_tokens = default_max_tokens
        self.default_model = default_model
        self.client = AsyncAnthropic(api_key=api_key, base_url=api_endpoint)

    def user_token(self):
        return HUMAN_PROMPT.replace(':', '').replace('\n', '')

    def assistant_token(self) -> str:
        return AI_PROMPT.replace(':', '').replace('\n', '')

    def append_token(self) -> str:
        return AI_PROMPT

    def max_tokens(self, model: Optional[str]) -> int:
        model = model if model else self.default_model
        match model:
            case _:
                return self.default_max_tokens

    def set_default_model(self, default_model: str):
        self.default_model = default_model

    def get_default_model(self):
        return self.default_model

    def set_default_max_tokens(self, default_max_tokens: int):
        self.default_max_tokens = default_max_tokens

    def max_prompt_tokens(
        self,
        completion_token_count: int = 2048,
        model: Optional[str] = None,
    ) -> int:
        return self.max_tokens(model) - completion_token_count

    def calculate_tokens(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        extra_str: str = '',
        model: Optional[str] = None,
    ) -> int:
        model_str = model if model else self.default_model

        async def tokenizer_len(text: str) -> int:
            return len((await self.client.get_tokenizer()).encode(text))

        def num_tokens_from_messages(messages, model: str):
            # this is inexact, but it's a reasonable approximation
            if model in {
                'claude-2',
                'claude-2.0',
                'claude-2.1',
                'claude-instant-1.2',
                'claude-instant-1',
            }:
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                logging.error(f"""num_tokens_from_messages() is not implemented for model {model}.""")  # noqa: E501
                tokens_per_message = 3
                tokens_per_name = 1
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += asyncio.run(tokenizer_len(value))
                    if key == "name":
                        num_tokens += tokens_per_name
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
        return 'anthropic'

    def aexecute_direct(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, str]] = [],
        model: Optional[str] = None,
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        model = model if model else self.default_model

        def __format_prompt(messages: List[Dict[str, str]]) -> str:
            prompt = ''
            for message in messages:
                if message['role'] == 'assistant':
                    prompt += f"""{AI_PROMPT} {message['content']}\n\n"""
                elif message['role'] == 'user':
                    prompt += f"""{HUMAN_PROMPT} {message['content']}\n\n"""

            prompt += f"""{AI_PROMPT}"""
            return prompt

        if functions:
            raise NotImplementedError('functions are not implemented for ClaudeExecutor')

        message_tokens = self.calculate_tokens(messages)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens):
            raise Exception('Prompt too long, message tokens: {}, completion tokens: {} total tokens: {}, available tokens: {}'
                            .format(message_tokens,
                                    max_completion_tokens,
                                    message_tokens + max_completion_tokens,
                                    self.max_tokens(model)))

        stream = self.client.completions.create(
            max_tokens_to_sample=max_completion_tokens,
            model=model,
            stream=True,
            temperature=temperature,
            prompt=__format_prompt(messages),
        )
        return stream

    async def aexecute(
        self,
        messages: List[Message],
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        model: Optional[str] = None,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model

        def last(predicate, iterable):
            result = [x for x in iterable if predicate(x)]
            if result:
                return result[-1]
            return None

        # find the system message and append to the front
        system_message = last(lambda x: x.role() == 'system', messages)

        if not system_message:
            system_message = System(Content('You are a helpful assistant.'))

        # fresh message list
        messages_list: List[Dict[str, str]] = []

        messages_list.append(Message.to_dict(system_message))
        for message in [m for m in messages if m.role() != 'system']:
            messages_list.append(Message.to_dict(message))

        stream = self.aexecute_direct(
            messages_list,
            max_completion_tokens=max_completion_tokens,
            model=model,
            temperature=temperature,
        )

        text_response = ''
        async for completion in await stream:  # type: ignore
            s = completion.completion
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
