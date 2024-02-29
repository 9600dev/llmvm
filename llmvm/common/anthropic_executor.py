import asyncio
import base64
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic

from llmvm.common.container import Container
from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content, Executor,
                                  FileContent, ImageContent, Message,
                                  PdfContent, System, TokenStopNode, User,
                                  awaitable_none)
from llmvm.common.perf import (TokenPerf, TokenStreamManager)

logging = setup_logging()


class AnthropicExecutor(Executor):
    def __init__(
        self,
        api_key: str,
        default_model: str = 'claude-2.1',
        api_endpoint: str = 'https://api.anthropic.com',
        default_max_token_len: int = 200000,
        default_max_completion_len: int = 4096,
        beta: bool = True,
    ):
        self.default_max_token_len = default_max_token_len
        self.default_max_completion_len = default_max_completion_len
        self.default_model = default_model
        self.client = AsyncAnthropic(api_key=api_key, base_url=api_endpoint)
        self.beta = beta

    def messages_trace(self, executor_name: str, message: List[Dict[str, str]]):
        if Container.get_config_variable('LLMVM_EXECUTOR_TRACE', default=''):
            with open(os.path.expanduser(Container.get_config_variable('LLMVM_EXECUTOR_TRACE')), 'a+') as f:
                for m in message:
                    f.write(f"{m['role'].capitalize()}: {m['content']}\n\n")

    def from_dict(self, message: Dict[str, Any]) -> 'Message':
        role = message['role']
        message_content = message['content']

        url = message['url'] if 'url' in message else ''
        content_type = message['content_type'] if 'content_type' in message else ''

        # when converting from MessageModel, there can be an embedded image
        # in the content parameter that needs to be converted back to bytes
        if (
            isinstance(message_content, list)
            and len(message_content) > 0
            and 'type' in message_content[0]
            and message_content[0]['type'] == 'image'
        ):
            byte_content = base64.b64decode(message_content[0]['source']['data'])
            content = ImageContent(byte_content, message_content[0]['source']['data'])

        elif content_type == 'pdf':
            if url and not message_content:
                with open(url, 'rb') as f:
                    content = PdfContent(f.read(), url)
            else:
                content = PdfContent(FileContent.decode(str(message_content)), url)
        elif content_type == 'file':
            # if there's a url here, but no content, then it's a file local to the server
            if url and not message_content:
                with open(url, 'r') as f:
                    content = FileContent(f.read().encode('utf-8'), url)
            # else, it's been transferred from the client to server via b64
            else:
                content = FileContent(FileContent.decode(str(message_content)), url)
        else:
            content = Content(message_content)

        if role == 'user':
            return User(content)
        elif role == 'system':
            return System(content)
        elif role == 'assistant':
            return Assistant(content)
        raise ValueError(f'role not found or not supported: {message}')

    def wrap_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        def wrap_message(index: int, content: Content) -> str:
            if isinstance(content, FileContent):
                return f"<message number={index}><file url={content.url}>{content.get_content()}</file></message>"
            elif isinstance(content, PdfContent):
                return f"<message number={index}><pdf url={content.url}>{content.get_content()}</pdf></message>"
            else:
                return f"<message index={index}>{content.get_content()}</message>"

        wrapped = []

        # grab the last system message
        system_messages = [m for m in messages if m.role() == 'system']
        if len(system_messages) > 1:
            logging.debug('More than one system message in the message list. Using the last one.')

        system_message = ''
        if len(system_messages) > 0:
            system_message = system_messages[-1]
            wrapped.append({'role': 'system', 'content': system_message.message.get_content()})

        messages = [m for m in messages if m.role() != 'system']

        counter = 1
        for i in range(len(messages)):
            if isinstance(messages[i], User) and isinstance(messages[i].message, ImageContent):
                wrapped.append({
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image',
                            'source': {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(messages[i].message.sequence).decode('utf-8')  # type: ignore
                            }
                        }
                    ]
                })
            elif isinstance(messages[i], User) and i < len(messages) - 1:  # is not last message, context messages
                wrapped.append({'role': 'user', 'content': wrap_message(counter, messages[i].message)})
                counter += 1
            elif isinstance(messages[i], User) and i == len(messages) - 1:  # is the last message
                wrapped.append({'role': 'user', 'content': messages[i].message.get_content()})
            else:
                wrapped.append({'role': messages[i].role(), 'content': messages[i].message.get_content()})

        messages_list = []

        for i in range(len(wrapped)):
            if i > 0 and wrapped[i]['role'] == wrapped[i - 1]['role']:
                if wrapped[i]['role'] == 'user':
                    messages_list.append({'role': 'assistant', 'content': 'Thanks. I am ready for your next message.'})
                elif wrapped[i]['role'] == 'assistant':
                    messages_list.append({'role': 'user', 'content': 'Thanks. I am ready for your next message.'})
            messages_list.append(wrapped[i])

        return messages_list

    def user_token(self):
        if self.beta:
            return 'User'
        else:
            return HUMAN_PROMPT.replace(':', '').replace('\n', '')

    def assistant_token(self) -> str:
        if self.beta:
            return 'Assistant'
        else:
            return AI_PROMPT.replace(':', '').replace('\n', '')

    def append_token(self) -> str:
        if self.beta:
            return ''
        else:
            return AI_PROMPT

    def max_tokens(self, model: Optional[str]) -> int:
        model = model if model else self.default_model
        match model:
            case 'claude-2.1':
                return 200000
            case 'claude-2.0':
                return 200000
            case str(model) if model.startswith('claude-3'):
                return 200000
            case 'claude-instant-1.2':
                return 100000
            case _:
                logging.debug('max_tokens() is not implemented for model {}, returning 200000'.format(model))
                return 200000

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
        model: Optional[str] = None,
    ) -> int:
        model_str = model if model else self.default_model

        async def tokenizer_len(content: str | Dict) -> int:
            return await self.client.count_tokens(str(content))

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
            elif model.startswith('claude-3'):
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                logging.debug(f"num_tokens_from_messages() is not implemented for model {model}.")
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
            dict_messages = self.wrap_messages(cast(List[Message], messages))
            return num_tokens_from_messages(dict_messages, model=model_str)
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            return num_tokens_from_messages(messages, model=model_str)
        elif isinstance(messages, str):
            return num_tokens_from_messages(self.wrap_messages([User(Content(messages))]), model=model_str)
        else:
            raise ValueError('cannot calculate tokens for messages: {}'.format(messages))

    def name(self) -> str:
        return 'anthropic'

    def __format_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ''
        for message in messages:
            if message['role'] == 'assistant':
                prompt += f"""{AI_PROMPT} {message['content']}\n\n"""
            elif message['role'] == 'user':
                prompt += f"""{HUMAN_PROMPT} {message['content']}\n\n"""

        prompt += f"""{AI_PROMPT}"""
        return prompt

    async def __aexecute_direct(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, str]] = [],
        model: Optional[str] = None,
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> TokenStreamManager:
        model = model if model else self.default_model

        if functions:
            raise NotImplementedError('functions are not implemented for ClaudeExecutor')

        message_tokens = self.count_tokens(messages=messages, model=model)
        if message_tokens > self.max_prompt_tokens(max_completion_tokens, model=model):
            raise Exception('Prompt too long. prompt tokens: {}, completion tokens: {}, total: {}, max context window: {}'
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

        self.messages_trace(self.name(), messages_list)

        token_trace = TokenPerf('__aexecute_direct', 'openai', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        if self.beta:
            # AsyncStreamManager[AsyncMessageStream]
            stream = self.client.messages.stream(
                max_tokens=max_completion_tokens,
                messages=messages_list,  # type: ignore
                model=model,
                system=system_message,
                temperature=temperature,
            )
            return TokenStreamManager(stream, token_trace)
        else:
            stream = await self.client.completions.create(
                max_tokens_to_sample=max_completion_tokens,
                model=model,
                stream=True,
                temperature=temperature,
                prompt=self.__format_prompt(messages_list),
            )
            return TokenStreamManager(stream, token_trace)

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
        messages_list: List[Dict[str, str]] = self.wrap_messages(messages)

        stream = self.__aexecute_direct(
            messages_list,
            max_completion_tokens=max_completion_tokens,
            model=model,
            temperature=temperature,
        )

        text_response = ''

        async with await stream as stream_async:  # type: ignore
            async for text in stream_async:
                await stream_handler(Content(text))
                text_response += text
            await stream_handler(TokenStopNode())

        if self.beta:
            _ = await stream_async.get_final_message()  # type: ignore
        else:
            async for completion in await stream:  # type: ignore
                s = completion.completion
                await stream_handler(Content(s))
                text_response += s
            await stream_handler(TokenStopNode())

        messages_list.append({'role': 'assistant', 'content': text_response})
        conversation: List[Message] = [self.from_dict(m) for m in messages_list]

        messages_trace(messages_list)

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
