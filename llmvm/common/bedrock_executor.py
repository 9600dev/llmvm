import asyncio
import base64
import json
import os
from typing import Any, Awaitable, Callable, Coroutine, Optional, cast

import boto3

from llmvm.common.bedrock_helpers import get_image_token_count
from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, BrowserContent, Content,
                                  Executor, FileContent, ImageContent,
                                  MarkdownContent, Message, PdfContent, System,
                                  TextContent, TokenCountCache, TokenNode, TokenPerf,
                                  TokenStopNode, User, awaitable_none)
from llmvm.common.perf import TokenStreamManager

logging = setup_logging()

class BedrockExecutor(Executor):
    def __init__(
        self,
        api_key: str = '',
        default_model: str = 'amazon.nova-pro-v1:0',
        api_endpoint: str = '',
        default_max_input_len: int = 3000000,
        default_max_output_len: int = 4096,
        max_images: int = 20,
        region_name: str = 'us-east-1',
        client_name: str = 'bedrock-runtime',
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_input_len=default_max_input_len,
            default_max_output_len=default_max_output_len,
        )
        self.token_count_cache: TokenCountCache = TokenCountCache()
        self.api_key = api_key
        self.client = boto3.client(client_name, region_name=region_name)
        self.max_images = max_images

    def user_token(self):
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def scratchpad_token(self) -> str:
        return 'scratchpad'

    def name(self) -> str:
        return 'bedrock'

    def max_input_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        return self.default_max_input_len

    def max_output_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        return self.default_max_output_len

    def to_dict(self, message: 'Message', model: Optional[str], server_serialization: bool = False) -> dict[str, Any]:
        content_list = []
        for content in message.message:
            if isinstance(content, ImageContent) and content.sequence:
                if 'image/unknown' not in Helpers.classify_image(content.get_bytes()):
                    content_list.append({
                        'image': {
                            'format': Helpers.classify_image(content.get_bytes()).replace('image/', ''),
                            'source': {
                                'bytes': base64.b64encode(content.get_bytes()).decode('utf-8'),
                            },
                            **({'url': content.url} if server_serialization else {}),
                            **({'content_type': 'image'} if server_serialization else {}),
                        }
                    })
                else:
                    logging.warning(f"Image content type not supported for {model}")
            elif isinstance(content, TextContent) and content.sequence:  # is not last message, context messages
                content_list.append({
                    'text': content.get_str(),
                    **({'url': content.url} if server_serialization else {}),
                    **({'content_type': 'text'} if server_serialization else {})
                })
            elif not content.sequence:
                logging.warning(f'Content inside message {message.to_json()} was empty.')
            else:
                raise ValueError(f"Cannot serialize unknown content type: {type(content)} in message {message.to_json()}")

        # system messages can only be a single string
        if message.role() == 'system':
            return {
                'role': message.role(),
                'content': {'text': content_list[0]['text'] },
            }

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
            if 'text' in message_content[i]:
                url = message_content[i]['url'] if 'url' in message_content[i] else ''
                content_list.append(TextContent(message_content[i]['text'], url=url))
            elif 'image' in message_content[i]:
                byte_content = base64.b64decode(message_content[i]['image']['source']['bytes'])
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

    def unpack_and_wrap_messages(self, messages: list[Message], model: Optional[str] = None) -> list[dict[str, str]]:
        wrapped: list[dict[str, str]] = []

        if not messages or not all(isinstance(m, Message) for m in messages):
            logging.error('Messages must be a list of Message objects.')
            for m in [m for m in messages if not isinstance(m, Message)]:
                logging.error(f'Invalid message: {m}, the type should be Message but its type is {type(m)}')
            raise ValueError('Messages must be a list of Message objects.')

        # deal with the system message
        system_messages = cast(list[System], Helpers.filter(lambda m: m.role() == 'system', messages))
        if len(system_messages) > 1:
            logging.warning('More than one system message in the message list. Using the last one.')

        if len(system_messages) > 0:
            wrapped.append(self.to_dict(system_messages[-1], model, server_serialization=False))

        # expand the PDF, Markdown, BrowserContent, and FileContent messages
        expanded_messages: list[Message] = [m for m in messages if m.role() != 'system'].copy()

        if expanded_messages[0].role() != 'user':
            raise ValueError('First message must be from User')

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

        return wrapped

    async def count_tokens(
        self,
        messages: list[Message],
    ) -> int:
        unpacked_messages: list[dict[str, Any]] = self.unpack_and_wrap_messages(messages=messages)
        return await self.count_tokens_dict(unpacked_messages)

    async def count_tokens_dict(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        if self.token_count_cache.get(messages):
            return cast(int, self.token_count_cache.get(messages))

        num_tokens = 0
        json_accumulator = ''
        for message in messages:
            for content in message['content']:
                if 'image' in content and 'source' in content['image'] and 'bytes' in content['image']['source']:
                    b64data = content['image']['source']['bytes']
                    num_tokens += get_image_token_count(b64data)['estimated_tokens']
                else:
                    json_accumulator += json.dumps(content, indent=2)

        result = num_tokens + int(len(json_accumulator.split(' ')) * .90)
        self.token_count_cache.put(messages, result)
        return result

    async def aexecute_direct(
        self,
        messages: list[dict[str, Any]],
        functions: list[dict[str, str]] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
    ) -> TokenStreamManager:
        model = model if model else self.default_model

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
        system_message: dict = {'text': 'You are a helpful assistant.'}
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                messages.remove(message)

        # the messages API also doesn't allow for multiple User or Assistant messages in a row, so we're
        # going to add an Assistant message in between two User messages, and a User message between two Assistant.
        messages_list: list[dict[str, Any]] = []

        for i in range(len(messages)):
            if i > 0 and messages[i]['role'] == messages[i - 1]['role']:
                if messages[i]['role'] == 'user':
                    messages_list.append({'role': 'assistant', 'content': [{'text': 'Thanks. I am ready for your next message.'}]})
                elif messages[i]['role'] == 'assistant':
                    messages_list.append({'role': 'user', 'content': [{'text': 'Thanks. I am ready for your next message.'}]})
            messages_list.append(messages[i])

        # todo, this is a busted hack. if a helper function returns nothing, then usually that
        # message get stripped away
        if messages_list[0]['role'] != 'system' and messages_list[0]['role'] != 'user':
            logging.warning('First message was not a system or user message. This is a bug. Adding a default empty message.')
            messages_list.insert(0, {'role': 'user', 'content': [{'text': 'None.'}]})

        # ugh, anthropic api can't have an assistant message with trailing whitespace...
        if messages_list[-1]['role'] == 'assistant':
            for j in range(len(messages_list[-1]['content'])):
                messages_list[-1]['content'][j]['text'] = messages_list[-1]['content'][j]['text'].rstrip()

        messages_trace([{'role': 'system', 'content': [{'text': system_message}]}] + messages_list)

        token_trace = TokenPerf('aexecute_direct', 'bedrock', model)  # type: ignore
        token_trace.start()

        inf_params = {"max_new_tokens": max_output_tokens, "temperature": temperature, "stopSequences": stop_tokens}

        request_body = {
            "messages": messages_list,
            "system": [system_message],
            "inferenceConfig": inf_params,
        }

        try:
            response = self.client.invoke_model_with_response_stream(
                modelId=model, body=json.dumps(request_body)
            )

            stream = response.get("body")
            return TokenStreamManager(stream, token_trace, response_object=response)
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

        if thinking > 0:
            raise NotImplementedError('Thinking is not implemented for BedrockExecutor')

        # wrap and check message list
        messages_list: list[dict[str, Any]] = self.unpack_and_wrap_messages(messages, model)

        stream: Coroutine[Any, Any, TokenStreamManager] = self.aexecute_direct(
            messages_list,
            max_output_tokens=max_output_tokens,
            model=model,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )

        text_response: str = ''
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for token in stream_async:
                # nova will emit stop tokens before it stops, so we need to check for that
                if not token.token in stop_tokens:
                    await stream_handler(TokenNode(token.token))
                text_response += token.token
            await stream_handler(TokenStopNode())
            perf = stream_async.perf

            # EventStream and the response object doesn't give me access to the stopReason. sigh.
            stopped = [stop_token for stop_token in stop_tokens if text_response.endswith(stop_token)]
            if stopped:
                perf.stop_reason = 'stop'
                perf.stop_token = next(iter(stopped))
                text_response = text_response[:-len(perf.stop_token)]

        await stream_async.get_final_message()  # this forces an update to the perf object
        perf.log()

        assistant = Assistant(
            message=TextContent(text_response.strip()),
            error=False,
            stop_reason=perf.stop_reason,
            stop_token=perf.stop_token,
            perf_trace=perf,
            total_tokens=perf.total_tokens or await self.count_tokens(messages),
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
