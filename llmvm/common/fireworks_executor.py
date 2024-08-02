import asyncio
import base64
from io import BytesIO
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast
from PIL import Image

import tiktoken

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, Content, Executor, FileContent, ImageContent, MarkdownContent,
                                  Message, PdfContent, TokenStopNode, User, awaitable_none)
from llmvm.common.perf import TokenPerf, TokenStreamManager
from fireworks.client import AsyncFireworks

logging = setup_logging()

class FireworksExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('GOOGLE_API_KEY')),
        default_model: str = 'accounts/fireworks/models/llama-v3p1-405b-instruct',
        api_endpoint: str = '',
        default_max_token_len: int = 128000,
        default_max_output_len: int = 4096,
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_token_len=default_max_token_len,
            default_max_output_len=default_max_output_len,
        )
        self.client = AsyncFireworks(api_key=api_key)

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def count_tokens(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        model: Optional[str] = None,
    ) -> int:
        model_str = model if model else self.default_model
        encoding = tiktoken.get_encoding('cl100k_base')

        # obtained from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        def num_tokens_from_messages(messages, model: str):
            if model in {
                "llama3-400b",
                "accounts/fireworks/models/llama-v3p1-405b-instruct",
            }:
                tokens_per_message = 3
            else:
                logging.error(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")  # noqa: E501
                tokens_per_message = 3
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message

                for key, value in message.items():
                    if isinstance(value, list):
                        for list_item in value:
                            if 'type' in list_item and list_item['type'] == 'image_url' and 'image_url' in list_item:
                                if 'detail' in list_item['image_url'] and list_item['image_url']['detail'] == 'high':
                                    try:
                                        with Image.open(BytesIO(base64.b64decode(list_item['image_url']['url'].split(',')[1]))) as img:  # NOQA: E501
                                            width, height = img.size
                                            num_tokens += self.__calculate_image_tokens(width=width, height=height)
                                    except Exception as e:
                                        num_tokens += 85
                                else:
                                    num_tokens += 85
                    elif value:
                        try:
                            num_tokens += len(encoding.encode(value))
                        except Exception as e:
                            logging.error(f"Error encoding message: {e}")
                            continue
                        if key == "name":
                            num_tokens += 1
            num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], Message):
            dict_messages = [Message.to_dict(m) for m in messages]  # type: ignore
            return num_tokens_from_messages(dict_messages, model=model_str)
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            return num_tokens_from_messages(messages, model=model_str)
        elif isinstance(messages, str):
            return num_tokens_from_messages([Message.to_dict(User(Content(messages)))], model=model_str)
        else:
            raise ValueError('cannot calculate tokens for messages: {}'.format(messages))

    def name(self) -> str:
        return 'llama3-400b'

    def __calculate_image_tokens(self, width: int, height: int):
        from math import ceil

        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n
        return total

    def wrap_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        def wrap_message(index: int, content: Content) -> str:
            if isinstance(content, FileContent):
                return f"<file url={content.url}>{content.get_str()}</file>"
            elif isinstance(content, PdfContent):
                # return f"<pdf url={content.url}>{content.get_str()}</pdf>"
                return f"<pdf url={content.url}>{ObjectTransformers.transform_pdf_content(content, self)}</pdf>"
            elif isinstance(content, MarkdownContent):
                return f"{ObjectTransformers.transform_markdown_content(content, self)}"
            else:
                return f"{content.get_str()}"

        wrapped = []

        # grab the last system message
        system_messages = [m for m in messages if m.role() == 'system']
        if len(system_messages) > 1:
            logging.debug('More than one system message in the message list. Using the last one.')

        system_message = ''
        if len(system_messages) > 0:
            system_message = system_messages[-1]
            wrapped.append({'role': 'system', 'content': system_message.message.get_str()})

        messages = [m for m in messages if m.role() != 'system']

        # check to see if there are more than 20 images in the message list
        image_count = len([m for m in messages if isinstance(m, User) and isinstance(m.message, ImageContent)])
        if image_count >= 20:
            logging.debug(f'Image count is {image_count}, filtering.')

            # get the top 20 ordered by size, then remove the rest
            images = [m for m in messages if isinstance(m, User) and isinstance(m.message, ImageContent)]
            images.sort(key=lambda x: len(x.message.sequence), reverse=True)
            smaller_images = images[20:]

            # remove the images from the messages list and collapse the previous and next message
            # if there is no other images in between
            for image in smaller_images:
                index = messages.index(image)
                if (
                    index > 0
                    and not isinstance(messages[index - 1].message, ImageContent)
                    and not isinstance(messages[index + 1].message, ImageContent)
                ):
                    previous = messages[index - 1]
                    next = messages[index + 1]
                    previous.message = Content(previous.message.get_str() + next.message.get_str())
                    messages.remove(image)
                    messages.remove(next)
                else:
                    messages.remove(image)

        counter = 1
        for i in range(len(messages)):
            if isinstance(messages[i], User) and isinstance(messages[i].message, ImageContent):
                # figure out
                if 'image/unknown' not in Helpers.classify_image(messages[i].message.sequence):
                    b64data = base64.b64encode(Helpers.anthropic_resize(messages[i].message.sequence)).decode('utf-8')
                    wrapped.append({
                        'role': 'user',
                        'content': [{
                                'type': 'image_url',
                                'image_url': {
                                    "url": f"data:image/png;base64,{b64data}",
                                }
                        }]
                    })
            elif isinstance(messages[i], User) and i < len(messages) - 1:  # is not last message, context messages
                wrapped.append({'role': 'user', 'content': wrap_message(counter, messages[i].message)})
                counter += 1
            elif (
                isinstance(messages[i], User)
                and i == len(messages) - 1
                and (
                    isinstance(messages[i].message, PdfContent)
                )
            ):  # is the last message, and it's a pdf or image
                wrapped.append({'role': 'user', 'content': wrap_message(counter, messages[i].message)})
            elif isinstance(messages[i], User) and i == len(messages) - 1:  # is the last message
                wrapped.append({'role': 'user', 'content': messages[i].message.get_str()})
            else:
                wrapped.append({'role': messages[i].role(), 'content': messages[i].message.get_str()})

        return wrapped


    async def __aexecute_direct(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        model = model if model else self.default_model

        # only works if profiling or LLMVM_PROFILING is set to true
        message_tokens = self.count_tokens(messages, model=model)
        if message_tokens > self.max_input_tokens(max_output_tokens, model=model):
            raise Exception('Prompt too long. prompt tokens: {}, output tokens: {}, total: {}, max context window: {}'
                            .format(message_tokens,
                                    max_output_tokens,
                                    message_tokens + max_output_tokens,
                                    self.max_tokens(model)))

        token_trace = TokenPerf('__aexecute_direct', 'fireworks', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        response = self.client.chat.completions.acreate(
            self.default_model,
            messages=messages,
            stream=True,
            temperature=temperature,
        )  # type: ignore

        return TokenStreamManager(response, token_trace)  # type: ignore

    async def aexecute(
        self,
        messages: List[Message],
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model

        # fresh message list
        messages_list: List[Dict[str, str]] = self.wrap_messages(messages)

        stream = self.__aexecute_direct(
            messages_list,
            max_output_tokens=max_output_tokens,
            model=model if model else self.default_model,
            temperature=temperature,
        )

        text_response = ''
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for text in stream_async:  # type: ignore
                await stream_handler(Content(text))
                text_response += text
            await stream_handler(TokenStopNode())
            perf = stream_async.perf

        messages_list.append({'role': 'assistant', 'content': text_response})
        conversation: List[Message] = [Message.from_dict(m) for m in messages_list]

        assistant = Assistant(
            message=conversation[-1].message,
            messages_context=conversation
        )

        perf.log()
        messages_trace(messages_list)

        return assistant

    def execute(
        self,
        messages: List[Message],
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        model: Optional[str] = None,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_output_tokens, temperature, model, stream_pipe))
