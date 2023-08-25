import datetime as dt
import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import openai
import tiktoken

from helpers.logging_helpers import response_writer, setup_logging
from objects import Assistant, Content, Executor, Message, System, User
from persistent_cache import PersistentCache

logging = setup_logging()


class OpenAIExecutor(Executor):
    def __init__(
        self,
        openai_key: str = cast(str, os.environ.get('OPENAI_API_KEY')),
        max_function_calls: int = 5,
        model: str = 'gpt-3.5-turbo-16k-0613',
        verbose: bool = True,
        cache: PersistentCache = PersistentCache(),
    ):
        self.openai_key = openai_key
        self.verbose = verbose
        self.model = model
        self.max_function_calls = max_function_calls
        self.cache: PersistentCache = cache

    def max_tokens(self) -> int:
        match self.model:
            case 'gpt-3.5-turbo-16k-0613':
                return 16385
            case 'gpt-3.5-turbo-16k':
                return 16385
            case _:
                return 4096

    def max_prompt_tokens(self, completion_token_count: int = 2048) -> int:
        return self.max_tokens() - completion_token_count - 256

    def calculate_tokens(
            self,
            messages: List[Message] | List[Dict[str, str]] | str, extra_str: str = ''
    ) -> int:
        # obtained from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
            """Return the number of tokens used by a list of messages."""
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                print("Warning: model not found. Using cl100k_base encoding.")
                encoding = tiktoken.get_encoding("cl100k_base")
            if model in {
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                }:
                tokens_per_message = 3
                tokens_per_name = 1
            elif model == "gpt-3.5-turbo-0301":
                tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
                tokens_per_name = -1  # if there's a name, the role is omitted
            elif "gpt-3.5-turbo" in model:
                print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
                return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
            elif "gpt-4" in model:
                print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
                return num_tokens_from_messages(messages, model="gpt-4-0613")
            else:
                raise NotImplementedError(
                    f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
                )
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], Message):
            dict_messages = [Message.to_dict(m) for m in messages]  # type: ignore
            dict_messages.append(Message.to_dict(User(Content(extra_str))))
            return num_tokens_from_messages(dict_messages, model=self.model)
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            return num_tokens_from_messages(messages, model=self.model)
        elif isinstance(messages, str):
            return num_tokens_from_messages([Message.to_dict(User(Content(messages)))], model=self.model)
        else:
            raise ValueError('cannot calculate tokens for messages: {}'.format(messages))

    def name(self) -> str:
        return 'openai'

    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k-0613',
        max_completion_tokens: int = 1024,
        temperature: float = 0.2,
        chat_format: bool = True,
        stream: bool = False,
    ) -> Dict:
        message_tokens = self.calculate_tokens(messages)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens):
            raise Exception('Prompt too long, message tokens: {}, completion tokens: {} total tokens: {}, available tokens: {}'
                            .format(message_tokens,
                                    max_completion_tokens,
                                    message_tokens + max_completion_tokens,
                                    self.max_tokens()))

        if not chat_format and len(functions) > 0:
            raise Exception('Functions are not supported in non-chat format')

        if chat_format:
            if functions:
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    functions=functions,
                    messages=messages,
                    stream=stream,
                )
            else:
                # for whatever reason, [] functions generates an InvalidRequestError
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    messages=messages,
                    stream=stream,
                )
            return response  # type: ignore
        else:
            response = openai.Completion.create(
                model=model,
                temperature=temperature,
                max_tokens=max_completion_tokens,
                messages=messages,
            )
            return response  # type: ignore

    def execute(
        self,
        messages: List[Message],
        temperature: float = 0.2,
        max_completion_tokens: int = 2048,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> Assistant:
        def last(predicate, iterable):
            result = [x for x in iterable if predicate(x)]
            if result:
                return result[-1]
            return None

        if self.cache and self.cache.has_key(messages):
            return cast(Assistant, self.cache.get(messages))

        # find the system message and append to the front
        system_message = last(lambda x: x.role() == 'system', messages)

        if not system_message:
            system_message = System(Content('You are a helpful assistant.'))

        start_time = dt.datetime.now()
        logging.debug('OpenAIExecutor.execute() user_message={}'.format(str(messages[-1])[0:25]))

        # fresh message list
        messages_list: List[Dict[str, str]] = []

        messages_list.append(Message.to_dict(system_message))
        for message in [m for m in messages if m.role() != 'system']:
            messages_list.append(Message.to_dict(message))

        chat_response = self.execute_direct(
            messages_list,
            max_completion_tokens=max_completion_tokens,
            chat_format=True,
            temperature=temperature,
            stream=True if stream_handler else False,
        )
        logging.debug('OpenAIExecutor.execute() finished in {}ms'.format((dt.datetime.now() - start_time).microseconds / 1000))

        # handle the stream stuff
        if stream_handler:
            text_response = ''
            for chunk in chat_response:
                s = chunk.choices[0].delta.get('content', '')
                stream_handler(s)
                text_response += s

            stream_handler('\n')

            messages_list.append({'role': 'assistant', 'content': text_response})
            conversation: List[Message] = [Message.from_dict(m) for m in messages_list]

            assistant = Assistant(
                message=conversation[-1].message,
                messages_context=conversation
            )

            if self.cache: self.cache.set(messages, assistant)
            return assistant

        if len(chat_response) == 0:
            return Assistant(
                message=Content('The model could not execute the query.'),
                error=True,
                messages_context=[Message.from_dict(m) for m in messages_list],
                system_context=system_message,
            )

        messages_list.append(chat_response['choices'][0]['message'])

        conversation: List[Message] = [Message.from_dict(m) for m in messages_list]

        assistant = Assistant(
            message=conversation[-1].message,
            messages_context=conversation
        )

        if self.cache: self.cache.set(messages, assistant)
        return assistant
