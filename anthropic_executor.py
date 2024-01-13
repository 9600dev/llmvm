import asyncio
import datetime as dt
from typing import Any, Awaitable, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from helpers.logging_helpers import setup_logging
from objects import (Assistant, AstNode, Content, Executor, Message, System,
                     TokenStopNode, User, awaitable_none)
from perf import TokenPerf, TokenPerfWrapperAnthropic

logging = setup_logging()


class AnthropicExecutor(Executor):
    def __init__(
        self,
        api_key: str,
        default_model: str = 'claude-2.1',
        api_endpoint: str = 'https://api.anthropic.com',
        default_max_tokens: int = 200000,
    ):
        self.default_max_tokens = default_max_tokens
        self.default_model = default_model
        self.client = AsyncAnthropic(api_key=api_key, base_url=api_endpoint)

    def user_token(self):
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

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
        completion_token_count: int = 4000,
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

    async def aexecute_direct(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, str]] = [],
        model: Optional[str] = None,
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
    ):
        model = model if model else self.default_model

        if functions:
            raise NotImplementedError('functions are not implemented for ClaudeExecutor')

        message_tokens = self.calculate_tokens(messages=messages, model=model)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens, model=model):
            raise Exception('Prompt too long, message tokens: {}, completion tokens: {} total tokens: {}, available tokens: {}'
                            .format(message_tokens,
                                    max_completion_tokens,
                                    message_tokens + max_completion_tokens,
                                    self.max_tokens(model)))

        # the messages API does not accept System messages, only User and Assistant messages.
        # get the system message from the dictionary, and remove it from the list
        system_message: str = ''
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                messages.remove(message)

        # anthropic disallows empty messages, so we're going to remove any Message that doesn't contain content
        for message in messages:
            if not message['content'] or message['content'] == '' or message['content'] == b'':
                messages.remove(message)

        # the messages API also doesn't allow for multiple User or Assistant messages in a row, so we're
        # going to add an Assistant message in between two User messages, and a User message between two Assistant.
        messages_list: List[Dict[str, str]] = []

        for i in range(len(messages)):
            if i > 0 and messages[i]['role'] == messages[i - 1]['role']:
                if messages[i]['role'] == 'user':
                    messages_list.append({'role': 'assistant', 'content': 'Thanks. I am ready for your next message.'})
                elif messages[i]['role'] == 'assistant':
                    messages_list.append({'role': 'user', 'content': 'Thanks. I am ready for your next message.'})
            messages_list.append(messages[i])

        token_trace = TokenPerf('aexecute_direct', 'openai', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        # AsyncStreamManager[AsyncMessageStream]
        stream = self.client.beta.messages.stream(
            max_tokens=max_completion_tokens,
            messages=messages_list,  # type: ignore
            model=model,
            system=system_message,
            temperature=temperature,
        )

        return TokenPerfWrapperAnthropic(stream, token_trace)

    async def aexecute(
        self,
        messages: List[Message],
        max_completion_tokens: int = 4000,
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

        async with await stream as stream_async:
            async for text in stream_async.text_stream:  # type: ignore
                await stream_handler(Content(text))
                text_response += text
            await stream_handler(TokenStopNode())

        _ = await stream_async.get_final_message()

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
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
        model: Optional[str] = None,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_completion_tokens, temperature, model, stream_pipe))
