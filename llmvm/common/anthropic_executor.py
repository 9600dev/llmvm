import asyncio
import base64
import json
import jsonpickle
import os
from typing import Any, Awaitable, Callable, Optional, cast

from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic
from anthropic.types import ThinkingConfigParam, Message as AntMessage
from pydantic import BaseModel

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, BrowserContent, Content,
                                  Executor, FileContent, HTMLContent, ImageContent,
                                  MarkdownContent, Message, PdfContent, System,
                                  TextContent, TokenCountCache, TokenNode, TokenPerf,
                                  TokenStopNode, TokenThinkingNode, User, awaitable_none)
from llmvm.common.perf import TokenStreamManager

logging = setup_logging()

tokens128k = [
    'claude-3-7-sonnet-20250219',
]

class AnthropicExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('ANTHROPIC_API_KEY')),
        default_model: str = 'claude-sonnet-4-20250514',
        api_endpoint: str = 'https://api.anthropic.com',
        default_max_input_len: int = 200000,
        default_max_output_len: int = 64000,
        max_images: int = 20,
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_input_len=default_max_input_len,
            default_max_output_len=default_max_output_len,
        )
        self.client = AsyncAnthropic(api_key=api_key, base_url=api_endpoint)
        self.max_images = max_images
        self.token_count_cache: TokenCountCache = TokenCountCache()

    def user_token(self):
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def scratchpad_token(self) -> str:
        return 'thinking'

    def name(self) -> str:
        return 'anthropic'

    def to_dict(self, message: 'Message', model: Optional[str], server_serialization: bool = False) -> dict[str, Any]:
        content_list = []

        # maintain Assistant thinking blocks as they have hashes.
        if isinstance(message, Assistant) and message.underlying:
            cached_assistant_message: AntMessage = cast(AntMessage, Helpers.b64_to_dill(message.underlying))  # type: ignore
            return_message = {
                'role': message.role(),
                'content': []
            }

            for content in cached_assistant_message.content:
                if content.type == 'text':
                    return_message['content'].append(
                        {
                            'type': 'text',
                            'text': content.text.strip(),
                            **({'citations': content.citations} if content.citations else {})
                        }
                    )
                elif content.type == 'thinking':
                    return_message['content'].append(
                        {
                            'type': 'thinking',
                            'thinking': content.thinking,
                            'signature': content.signature
                        }
                    )
                else:
                    raise ValueError(f"Unknown content type: {content.type}")
            return return_message

        # otherwise, serialize the message as normal
        for content in message.message:
            if isinstance(content, ImageContent) and content.sequence:
                if 'image/unknown' not in Helpers.classify_image(content.get_bytes()):
                    content_list.append({
                        'type': 'image',
                        'source': {
                            "type": "base64",
                            "media_type": Helpers.classify_image(content.get_bytes()),
                            "data": base64.b64encode(Helpers.anthropic_resize(content.get_bytes())).decode('utf-8') # type: ignore
                        },
                        **({'cache_control': {'type': 'ephemeral'}} if message.prompt_cached and content is message.message[-1] else {}),
                        **({'url': content.url} if server_serialization else {}),
                        **({'content_type': 'image'} if server_serialization else {})
                    })
                else:
                    logging.warning(f"Image content type not supported for model {model}")
            elif isinstance(content, TextContent) and content.sequence:  # is not last message, context messages
                content_list.append({
                    'type': 'text',
                    'text': content.get_str(),
                    **({'cache_control': {'type': 'ephemeral'}} if message.prompt_cached and content is message.message[-1] else {}),
                    **({'url': content.url} if server_serialization else {}),
                    **({'content_type': 'text'} if server_serialization else {})
                })
            elif not content.sequence:
                logging.warning(f'Content inside message {message.to_json()} was empty.')
            else:
                raise ValueError(f"Cannot serialize unknown content type: {type(content)} in message {message.to_json()}")

        dict_message = {
            'role': message.role(),
            'content': content_list,
        }
        return dict_message

    def from_dict(self, message: dict[str, Any]) -> 'Message':
        # pull out Message related content
        role = message['role']
        message_content = message['content']

        # force content to be a list
        if not isinstance(message_content, list):
            message_content = [message_content]

        content_list: list[Content] = []

        for i in range(len(message_content)):
            if 'type' in message_content[i] and message_content[i]['type'] == 'text':
                url = message_content[i]['url'] if 'url' in message_content[i] else ''
                content_list.append(TextContent(message_content[i]['text'], url=url))
            elif 'type' in message_content[i] and message_content[i]['type'] == 'image':
                byte_content = base64.b64decode(message_content[i]['source']['data'])
                url = message_content[i]['url'] if 'url' in message_content[i] else ''
                content_list.append(ImageContent(byte_content, url=url))
            else:
                raise ValueError(f'Unknown content type: {message_content[i]}')

        if role == 'assistant':
            return Assistant(content_list[0])
        elif role == 'user':
            return User(content_list)
        elif role == 'system':
            return System(cast(TextContent, content_list[0]).get_str())
        else:
            raise ValueError(f'role not found or not supported: {message}')

    def unpack_and_wrap_messages(
        self,
        messages: list[Message],
        model: Optional[str] = None,
        raise_exception: bool = True,
    ) -> list[dict[str, str]]:
        wrapped: list[dict[str, str]] = []

        if not messages or not all(isinstance(m, Message) for m in messages):
            logging.error('Messages must be a list of Message objects.')
            for m in [m for m in messages if not isinstance(m, Message)]:
                logging.error(f'Invalid message: {m}, the type should be Message but its type is {type(m)}')
            if raise_exception: raise ValueError('Messages must be a list of Message objects.')

        # deal with the system message
        system_messages = cast(list[System], Helpers.filter(lambda m: m.role() == 'system', messages))
        if len(system_messages) > 1:
            logging.warning('More than one system message in the message list. Using the last one.')

        if len(system_messages) > 0:
            wrapped.append(self.to_dict(system_messages[-1], model, server_serialization=False))

        # expand the PDF, Markdown, BrowserContent, and FileContent messages
        expanded_messages: list[Message] = [m for m in messages if m.role() != 'system'].copy()

        if expanded_messages[0].role() != 'user':
            if raise_exception: raise ValueError('First message must be from User')

        for message in expanded_messages:
            for i in range(len(message.message) - 1, -1, -1):
                if isinstance(message.message[i], PdfContent):
                    content_list = cast(list[Content], ObjectTransformers.transform_pdf_to_content(cast(PdfContent, message.message[i]), self))
                    message.message[i:i+1] = content_list
                elif isinstance(message.message[i], MarkdownContent):
                    content_list = cast(list[Content], ObjectTransformers.transform_markdown_to_content(cast(MarkdownContent, message.message[i]), self))
                    message.message[i:i+1] = content_list
                elif isinstance(message.message[i], BrowserContent):
                    content_list = cast(list[Content], ObjectTransformers.transform_browser_to_content(cast(BrowserContent, message.message[i]), self))
                    message.message[i:i+1] = content_list
                elif isinstance(message.message[i], FileContent):
                    content_list = cast(list[Content], ObjectTransformers.transform_file_to_content(cast(FileContent, message.message[i]), self))
                    message.message[i:i+1] = content_list
                elif isinstance(message.message[i], HTMLContent):
                    content_list = cast(list[Content], ObjectTransformers.transform_html_to_content(cast(HTMLContent, message.message[i]), self))
                    message.message[i:i+1] = content_list

        # check to see if there are more than self.max_images images in the message list
        images = [c for c in Helpers.flatten([m.message for m in expanded_messages]) if isinstance(c, ImageContent)]
        image_count = len(images)

        # remove smaller images if there are too many
        if image_count >= self.max_images:
            logging.debug(f'Image count is {image_count}, filtering.')

            # get the top self.max_images ordered by byte array size, then remove the rest
            images.sort(key=lambda x: len(x.sequence), reverse=True)
            smaller_images = images[self.max_images:]

            for image in smaller_images:
                for i in range(len(expanded_messages)):
                    for j in range(len(expanded_messages[i].message)):
                        if expanded_messages[i].message[j] == image:
                            expanded_messages[i].message.pop(j)
                            break

        # now build the json dictionary and return
        for i in range(len(expanded_messages)):
            wrapped.append(self.to_dict(expanded_messages[i], model, server_serialization=False))

        # anthropic doesn't like content that is just non-whitespace
        for i in range(len(wrapped) - 1, -1, -1):
            if isinstance(wrapped[i]['content'], list):
                for j in range(len(wrapped[i]['content']) -1, -1, -1):
                    if wrapped[i]['content'][j]['type'] == 'text' and wrapped[i]['content'][j]['text'].strip() == '':  # type: ignore
                        wrapped[i]['content'].pop(j)  # type: ignore
            else:
                if wrapped[i]['content']['type'] == 'text' and wrapped[i]['content']['text'].strip() == '':  # type: ignore
                    wrapped.pop(i)

        return wrapped

    async def count_tokens(
        self,
        messages: list[Message],
    ) -> int:
        unpacked_messages: list[dict[str, Any]] = self.unpack_and_wrap_messages(messages=messages, raise_exception=False)
        return await self.count_tokens_dict(unpacked_messages)

    async def count_tokens_dict(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        # expensive call, so we're going to cache the results
        if self.token_count_cache.get(messages):
            return cast(int, self.token_count_cache.get(messages))

        system_message = [m for m in messages if m['role'] == 'system']
        rest_messages = [m for m in messages if m['role'] != 'system']

        num_tokens = await self.client.beta.messages.count_tokens(
            messages=rest_messages,  # type: ignore
            model=self.default_model,
            system=str(system_message[0]['content'] if system_message and system_message[0]['content'] else ''),
        )
        self.token_count_cache.put(messages, num_tokens.input_tokens)
        return num_tokens.input_tokens

    async def aexecute_direct(
        self,
        messages: list[dict[str, Any]],
        functions: list[dict[str, str]] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
        thinking: int = 0,
    ) -> TokenStreamManager:
        model = model if model else self.default_model

        if thinking > 0 and temperature != 1.0:
            temperature = 1.0

        if functions:
            raise NotImplementedError('functions are not implemented for ClaudeExecutor')

        message_tokens = await self.count_tokens_dict(messages=messages)

        if message_tokens > self.max_input_tokens(model=model):
            raise Exception('Prompt too long. input tokens: {}, requested output tokens: {}, total: {}, model: {} max context window is: {}'
                            .format(message_tokens,
                                    max_output_tokens,
                                    message_tokens + max_output_tokens,
                                    str(model),
                                    self.max_input_tokens(model=model)))

        # the messages API does not accept System messages, only User and Assistant messages.
        # get the system message from the dictionary, and remove it from the list
        system_message: str = ''
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                messages.remove(message)

        # the Anthropic messages API also doesn't allow for multiple User or Assistant messages in a row, so we're
        # going to add an Assistant message in between two User messages, and a User message between two Assistant.
        messages_list: list[dict[str, Any]] = []

        for i in range(len(messages)):
            if i > 0 and messages[i]['role'] == messages[i - 1]['role']:
                if messages[i]['role'] == 'user':
                    messages_list.append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'Thanks. I am ready for your next message.'}]})
                elif messages[i]['role'] == 'assistant':
                    messages_list.append({'role': 'user', 'content': [{'type': 'text', 'text': 'Thanks. I am ready for your next message.'}]})
            messages_list.append(messages[i])

        # todo, this is a busted hack. if a helper function returns nothing, then usually that
        # message get stripped away
        if messages_list[0]['role'] != 'system' and messages_list[0]['role'] != 'user':
            logging.warning('First message was not a system or user message. This is a bug. Adding a default empty message.')
            messages_list.insert(0, {'role': 'user', 'content': [{'type': 'text', 'text': 'None.'}]})

        # ugh, anthropic api can't have an assistant message with trailing whitespace...
        if messages_list[-1]['role'] == 'assistant':
            for j in range(len(messages_list[-1]['content'])):
                messages_list[-1]['content'][j]['text'] = messages_list[-1]['content'][j]['text'].rstrip()

        messages_trace([{'role': 'system', 'content': [{'type': 'text', 'text': system_message}]}] + messages_list)

        token_trace = TokenPerf('aexecute_direct', 'anthropic', model)  # type: ignore
        token_trace.start()

        thinking_block: ThinkingConfigParam = {
            'type': 'enabled',
            'budget_tokens': thinking,
        } if thinking else {'type': 'disabled'}

        try:
            # AsyncStreamManager[AsyncMessageStream]
            stream = self.client.messages.stream(
                max_tokens=max_output_tokens,
                messages=messages_list,  # type: ignore
                model=model,
                system=system_message,
                temperature=temperature,
                stop_sequences=stop_tokens,
                thinking=thinking_block,
                extra_headers={"anthropic-beta": "output-128k-2025-02-19"} if model in tokens128k else {},
            )
            return TokenStreamManager(stream, token_trace)
        except Exception as e:
            logging.error(e)
            raise

    async def aexecute(
        self,
        messages: list[Message],
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model

        # wrap and check message list
        messages_list: list[dict[str, Any]] = self.unpack_and_wrap_messages(messages, model)

        stream = self.aexecute_direct(
            messages_list,
            max_output_tokens=max_output_tokens,
            model=model,
            temperature=temperature,
            stop_tokens=stop_tokens,
            thinking=thinking,
        )

        text_response: str = ""
        thinking_response: str = ""
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for token in stream_async:
                if token.thinking:
                    await stream_handler(TokenThinkingNode(token.token))
                    thinking_response += token.token
                else:
                    await stream_handler(TokenNode(token.token))
                    text_response += token.token

            await stream_handler(TokenStopNode())
            perf = stream_async.perf

        _ = await stream_async.get_final_message()  # this forces an update to the perf object
        perf.log()

        assistant = Assistant(
            message=TextContent(text_response.strip()),
            thinking=TextContent(thinking_response.strip()),
            error=False,
            stop_reason=perf.stop_reason,
            stop_token=perf.stop_token,
            perf_trace=perf,
            total_tokens=perf.total_tokens,
            underlying=Helpers.dill_to_b64(perf.object)
        )

        if assistant.get_str() == '': logging.warning(f'Assistant message is empty. Returning empty message. {perf.request_id or ""}')
        return assistant

    def execute(
        self,
        messages: list[Message],
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_output_tokens, temperature, stop_tokens, model, thinking, stream_pipe))
