import asyncio
import base64
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import google.generativeai as genai
from google.generativeai.protos import Part
from llmvm.common.helpers import Helpers

from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, BrowserContent, Content, Executor, FileContent, ImageContent, MarkdownContent,
                                  Message, PdfContent, System, TokenStopNode, User, awaitable_none)
from llmvm.common.perf import TokenPerf, TokenStreamManager

logging = setup_logging()

class GeminiExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('GEMINI_API_KEY')),
        default_model: str = 'gemini-1.5-pro-002',
        api_endpoint: str = '',
        default_max_token_len: int = 2097152,
        default_max_output_len: int = 8192,
        max_images: int = 20,
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_token_len=default_max_token_len,
            default_max_output_len=default_max_output_len,
        )
        genai.configure(api_key=api_key)
        self.max_images = max_images

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def name(self) -> str:
        return 'gemini'

    def from_dict(self, message: Dict[str, Any]) -> Message:
        role = message['role']
        message_content = message['parts']

        url = message['url'] if 'url' in message else ''
        content_type = message['content_type'] if 'content_type' in message else ''

        # when converting from MessageModel, there can be an embedded image
        # in the content parameter that needs to be converted back to bytes
        if (
            isinstance(message_content, list)
            and len(message_content) > 0
            and 'inline_data' in message_content[0]
        ):
            byte_content = base64.b64decode(message_content[0]['inline_data']['data'])
            content = ImageContent(byte_content, '')

        elif content_type == 'pdf':
            if url and not message_content:
                with open(url, 'rb') as f:
                    content = PdfContent(f.read(), url)
            else:
                content = PdfContent(FileContent.decode(str(message_content[0]['text'])), url)

        elif content_type == 'file':
            # if there's a url here, but no content, then it's a file local to the server
            if url and not message_content:
                with open(url, 'r') as f:
                    content = FileContent(f.read().encode('utf-8'), url)
            # else, it's been transferred from the client to server via b64
            else:
                content = FileContent(FileContent.decode(str(message_content[0]['text'])), url)

        elif isinstance(message_content, list):
            content = Content(' '.join(message_content))

        else:
            content = Content(message_content)

        if role == 'user':
            return User(content)
        elif role == 'system':
            return System(content)
        elif role == 'model':
            return Assistant(content)
        raise ValueError(f'role not found or not supported: {message}')

    def wrap_messages(self, model: Optional[str], messages: List[Message]) -> List[Dict[str, Any]]:
        wrapped = []

        # deal with the system message
        system_messages = [m for m in messages if m.role() == 'system']
        system_message: Optional[System] = None
        if len(system_messages) > 1:
            logging.debug('More than one system message in the message list. Using the last one.')
        if len(system_messages) >= 1:
            system_message = cast(System, system_messages[-1])

        # the system message will be sucked out of the dictionary
        # and added to the GenerativeModel.system_instruction
        messages = [m for m in messages if m.role() != 'system']

        # expand the PDF, Markdown, and BrowserContent messages
        expanded_messages = []
        for message in messages:
            if isinstance(message, User) and isinstance(message.message, PdfContent):
                expanded_messages.extend(ObjectTransformers.transform_pdf_content(message.message, self))
            elif isinstance(message, User) and isinstance(message.message, MarkdownContent):
                expanded_messages.extend(ObjectTransformers.transform_markdown_content(message.message, self))
            elif isinstance(message, User) and isinstance(message.message, BrowserContent):
                expanded_messages.extend(ObjectTransformers.transform_browser_content(message.message, self))
            else:
                expanded_messages.append(message)

        messages = expanded_messages
        if system_message: messages.insert(0, system_message)

        # check to see if there are more than self.max_images images in the message list
        image_count = len([m for m in messages if isinstance(m, User) and isinstance(m.message, ImageContent)])
        if image_count >= self.max_images:
            logging.debug(f'Image count is {image_count}, filtering.')

            # get the top self.max_images ordered by size, then remove the rest
            images = [m for m in messages if isinstance(m, User) and isinstance(m.message, ImageContent)]
            images.sort(key=lambda x: len(x.message.sequence), reverse=True)
            smaller_images = images[self.max_images:]

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

        # build out the gemini Dict[str, Any] dictionary
        counter = 1
        for i in range(len(messages)):
            if isinstance(messages[i], User) and isinstance(messages[i].message, ImageContent):
                wrapped.append({
                    'role': 'user',
                    'parts': [
                        {
                            'inline_data': {
                                'mime_type': Helpers.classify_image(messages[i].message.sequence),
                                'data': base64.b64encode(messages[i].message.sequence).decode('utf-8')  # type: ignore
                            }
                        }
                    ]
                })

            elif isinstance(messages[i], User) and i < len(messages) - 1:  # is not last message, context messages
                wrapped.append({
                    'role': 'user',
                    'parts': [messages[i].message.get_str()]
                })
                counter += 1

            elif (
                isinstance(messages[i], User)
                and i == len(messages) - 1
                and (
                    isinstance(messages[i].message, PdfContent)
                    or isinstance(messages[i].message, ImageContent)
                )
            ):  # is the last message, and it's a pdf or image
                wrapped.append({
                    'role': 'user',
                    'parts': [messages[i].message.get_str()]
                })

            elif isinstance(messages[i], User) and i == len(messages) - 1:  # is the last message
                wrapped.append({
                    'role': 'user',
                    'parts': [messages[i].message.get_str()]
                })

            elif isinstance(messages[i], System):
                wrapped.append({
                    'role': 'system',
                    'parts': [messages[i].message.get_str()]
                })

            else:
                wrapped.append({
                    'role': 'user' if messages[i].role() == 'user' else 'model',
                    'parts': [messages[i].message.get_str()]
                })

        return wrapped

    async def count_tokens(
        self,
        messages: List[Message] | List[Dict[str, str]] | str,
        model: Optional[str] = None,
    ) -> int:
        # https://ai.google.dev/gemini-api/docs/tokens?lang=python
        model_str = model if model else self.default_model

        messages_text = ''
        image_counter = 0

        # message case
        if isinstance(messages, list) and len(messages) > 0 and all(isinstance(message, Message) for message in messages):
            messages_text = '\n'.join([m.message.get_str() for m in messages])  # type: ignore
        # dict case
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            for m in messages:
                if m['parts'] and 'inline_data' in m['parts'][0]:  # type: ignore
                    image_counter += 258
                else:
                    messages_text += ' '.join(m['parts'])
        else:
            messages_text = str(messages)

        aclient = genai.GenerativeModel(model_str)
        token_count = aclient.count_tokens(messages_text).total_tokens
        token_count += image_counter
        return token_count

    async def aexecute_direct(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, str]] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
    ):
        model = model if model else self.default_model

        if functions:
            raise NotImplementedError('Functions are not supported for this method')

        # only works if profiling or LLMVM_PROFILING is set to true
        message_tokens = await self.count_tokens(messages, model=model)
        if message_tokens > self.max_input_tokens(max_output_tokens, model=model):
            raise Exception('Prompt too long. prompt tokens: {}, output tokens: {}, total: {}, max context window: {}'
                            .format(message_tokens,
                                    max_output_tokens,
                                    message_tokens + max_output_tokens,
                                    self.max_tokens(model)))

        # special case system message which will have role='system'
        system_message = Helpers.filter(lambda item: item['role'] == 'system', messages)
        system_prompt = system_message[0]['parts'][0] if system_message else None
        # remove the system prompt
        messages_list = Helpers.filter(lambda item: item['role'] != 'system', messages)

        token_trace = TokenPerf('aexecute_direct', 'gemini', model, prompt_len=message_tokens)  # type: ignore
        token_trace.start()

        config = genai.GenerationConfig(
            temperature=temperature,
            stop_sequences=stop_tokens,
            max_output_tokens=max_output_tokens,
        )

        aclient = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=config
        )

        response = await aclient.generate_content_async(
            contents=messages_list,  # type: ignore
            generation_config=config,
            stream=True,
        )

        return TokenStreamManager(response, token_trace)  # type: ignore

    async def aexecute(
        self,
        messages: List[Message],
        max_output_tokens: int = 8192,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: Optional[Dict[str, Any]] = None,
    ) -> Assistant:
        model = model if model else self.default_model

        messages_list = self.wrap_messages(model, messages)

        stream = self.aexecute_direct(
            messages=messages_list,
            max_output_tokens=max_output_tokens,
            model=model if model else self.default_model,
            temperature=temperature,
            stop_tokens=stop_tokens,
        )

        text_response = ''
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for text in stream_async:  # type: ignore
                await stream_handler(Content(text))
                text_response += text
            await stream_handler(TokenStopNode())
            perf = stream_async.perf

        messages_list.append({'role': 'model', 'parts': [text_response]})
        conversation: List[Message] = [self.from_dict(m) for m in messages_list]

        await stream_async.get_final_message()

        assistant = Assistant(
            message=conversation[-1].message,
            messages_context=conversation,
            stop_reason=stream_async.perf.stop_reason,
            stop_token=stream_async.perf.stop_token,
        )

        perf.log()

        messages_trace(messages_list)

        return assistant

    def execute(
        self,
        messages: List[Message],
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        stop_tokens: List[str] = [],
        model: Optional[str] = None,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
        template_args: Optional[Dict[str, Any]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(self.aexecute(messages, max_output_tokens, temperature, stop_tokens, model, stream_pipe, template_args))
