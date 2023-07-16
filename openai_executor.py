from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import openai

from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import setup_logging
from objects import (Assistant, Content, ExecutionFlow, Executor, Message,
                     NaturalLanguage, System, User)

logging = setup_logging()


class OpenAIExecutor(Executor):
    def __init__(
        self,
        openai_key: str,
        max_function_calls: int = 5,
        model: str = 'gpt-3.5-turbo-16k',
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
            case 'gpt-3.5-turbo-16k':
                return 16385
            case _:
                return 4096

    def name(self) -> str:
        return 'openai'

    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
        max_completion_tokens: int = 256,
        temperature: float = 1.0,
        chat_format: bool = True,
    ) -> Dict:
        total_tokens = Helpers.calculate_tokens(messages)
        if total_tokens + max_completion_tokens > self.max_tokens():
            raise Exception(
                'Prompt too long, calculated user tokens: {}, completion tokens: {} total model tokens: {}'
                .format(total_tokens, max_completion_tokens, self.max_tokens())
            )

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
                )
            else:
                # for whatever reason, [] functions generates an InvalidRequestError
                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_completion_tokens,
                    messages=messages,
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

    def execute_with_agents(
        self,
        call: NaturalLanguage,
        agents: List[Callable],
    ) -> Assistant:
        if self.cache and self.cache.has_key(call.messages):
            return cast(Assistant, self.cache.get(call.messages))

        logging.debug('execute_with_agents')

        user_message: User = cast(User, call.messages[-1])
        messages = []
        message_results = []

        functions = [Helpers.get_function_description_flat(f) for f in agents]

        prompt = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/tool_execution_prompt.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': str(user_message),
            }
        )

        # todo, we probably need to figure out if we should pass in the
        # entire message history or not.
        messages.append({'role': 'system', 'content': prompt['system_message']})
        messages.append({'role': 'user', 'content': prompt['user_message']})

        chat_response = self.execute_direct(
            model=self.model,
            temperature=1.0,
            max_completion_tokens=2048,  # todo: calculate this properly
            messages=messages,
            chat_format=True,
        )

        chat_response = chat_response['choices'][0]['message']  # type: ignore
        message_results.append(chat_response)

        if len(chat_response) == 0:
            return Assistant(Content('The model could not execute the query.'), error=True)
        else:
            logging.debug('OpenAI Assistant Response: {}'.format(chat_response['content']))

            assistant = Assistant(
                message=Content(chat_response['content']),
                error=False,
                messages_context=[Message.from_dict(m) for m in messages],
                system_context=prompt['system_message'],
                llm_call_context=call,
            )

            if self.cache: self.cache.set(call.messages, assistant)
            return assistant

    def execute(
        self,
        messages: List[Message],
    ) -> Assistant:
        if self.cache and self.cache.has_key(messages):
            return cast(Assistant, self.cache.get(messages))

        # find the system message
        system_message = Helpers.first(lambda x: x.role() == 'system', messages)
        user_messages = Helpers.filter(lambda x: x.role() == 'user', messages)

        if not system_message:
            system_message = System(Content('You are a helpful assistant.'))

        logging.debug('OpenAIExecutor.execute system_message={} user_messages={}'
                      .format(system_message, user_messages))

        messages_list: List[Dict[str, str]] = []

        messages_list.append(Message.to_dict(system_message))
        for message in user_messages:
            messages_list.append(Message.to_dict(message))

        chat_response = self.execute_direct(
            messages_list,
            max_completion_tokens=2048,  # tood: calculate this properly
            chat_format=True,
        )

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
