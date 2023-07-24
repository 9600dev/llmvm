from abc import ABC, abstractmethod
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

import openai
import tiktoken

from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import response_writer, setup_logging
from objects import (Assistant, Content, ExecutionFlow, Executor, LLMCall,
                     Message, System, User)

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

    def max_prompt_tokens(self, completion_token_count: int = 2048) -> int:
        return self.max_tokens() - completion_token_count - 256

    def calculate_tokens(self, messages: List[Message], extra_str: str = '') -> int:
        content = '\n'.join([str(message.message) for message in messages])
        content += extra_str
        return len(tiktoken.encoding_for_model(self.model).encode(content))

    def name(self) -> str:
        return 'openai'

    def execute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: str = 'gpt-3.5-turbo-16k',
        max_completion_tokens: int = 1024,
        temperature: float = 1.0,
        chat_format: bool = True,
    ) -> Dict:
        message_tokens = Helpers.calculate_tokens(messages)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens):
            raise Exception('Prompt too long, calculated message tokens: {}, max completion tokens: {} total tokens: {}, available model tokens: {}'
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
        call: LLMCall,
        agents: List[Callable],
        temperature: float = 1.0,
    ) -> Assistant:
        if self.cache and self.cache.has_key((call.message, call.supporting_messages)):
            return cast(Assistant, self.cache.get((call.message, call.supporting_messages)))

        encoded_m = str(call.message).encode('utf-8', errors='ignore')[:100]
        logging.debug('OpenAIExecutor.execute_with_agents: {}'.format(encoded_m))

        user_message: User = cast(User, call.message)
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
            temperature=temperature,
            max_completion_tokens=2048,  # todo: calculate this properly
            messages=messages,
            chat_format=True,
        )

        chat_response = chat_response['choices'][0]['message']  # type: ignore
        message_results.append(chat_response)

        if len(chat_response) == 0:
            return Assistant(Content('The model could not execute the query.'), error=True)
        else:
            encoded_m = str(chat_response['content']).encode('utf-8', errors='ignore')[:100]
            logging.debug('OpenAI Assistant Response: {}'.format(encoded_m))

            assistant = Assistant(
                message=Content(chat_response['content']),
                error=False,
                messages_context=[Message.from_dict(m) for m in messages],
                system_context=prompt['system_message'],
                llm_call_context=call,
            )

            if self.cache: self.cache.set((call.message, call.supporting_messages), assistant)
            return assistant

    def execute(
        self,
        messages: List[Message],
        temperature: float = 1.0,
        max_completion_tokens: int = 2048,
    ) -> Assistant:

        if self.cache and self.cache.has_key(messages):
            return cast(Assistant, self.cache.get(messages))

        # find the system message
        system_message = Helpers.first(lambda x: x.role() == 'system', messages)
        user_messages = Helpers.filter(lambda x: x.role() == 'user', messages)

        if not system_message:
            system_message = System(Content('You are a helpful assistant.'))

        logging.debug('OpenAIExecutor.execute system_message={} user_messages={}'
                      .format(system_message, str(user_messages).encode('utf-8', errors='ignore')[:100]))

        messages_list: List[Dict[str, str]] = []

        messages_list.append(Message.to_dict(system_message))
        for message in user_messages:
            messages_list.append(Message.to_dict(message))

        chat_response = self.execute_direct(
            messages_list,
            max_completion_tokens=max_completion_tokens,
            chat_format=True,
            temperature=temperature,
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
