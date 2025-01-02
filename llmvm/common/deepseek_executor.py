import base64
import os
from typing import Any, Optional, cast

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import Assistant, BrowserContent, Content, FileContent, ImageContent, MarkdownContent, Message, PdfContent, System, TextContent, User
from llmvm.common.openai_executor import OpenAIExecutor

logging = setup_logging()

class DeepSeekExecutor(OpenAIExecutor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('DEEPSEEK_API_KEY')),
        default_model: str = 'deepseek-chat',

        api_endpoint: str = 'https://api.deepseek.com/v1',
        default_max_input_len: int = 128000,
        default_max_output_len: int = 4096,
        max_images: int = 0,
    ):
        if max_images > 0:
            raise ValueError('Deepseek does not support images. max_images must be 0.')

        super().__init__(
            api_key=api_key,
            default_model=default_model,
            api_endpoint=api_endpoint,
            default_max_input_len=default_max_input_len,
            default_max_output_len=default_max_output_len,
            max_images=max_images,
        )

    def user_token(self) -> str:
        return 'User'

    def assistant_token(self) -> str:
        return 'Assistant'

    def append_token(self) -> str:
        return ''

    def name(self) -> str:
        return 'deepseek'

    def to_dict(self, message: Message, model: Optional[str], server_serialization: bool = False) -> list[dict[str, Any]]:
        content_list: list[str] = []
        for content in message.message:
            if isinstance(content, TextContent) and content.sequence:
                content_list.append(content.get_str())
            elif not content.sequence:
                logging.warning(f'Content inside message {message.to_json()} was empty.')
            elif isinstance(content, ImageContent):
                raise ValueError(f'ImageContent is not supported for Deepseek.')
            else:
                raise ValueError(f"Cannot serialize unknown content type: {type(content)} in message {message.to_json()}")

        return [
            {
                'role': message.role(),
                'content': content
            }
            for content in content_list
        ]

    def from_dict(self, message: dict[str, Any]) -> 'Message':
        # pull out Message related content
        role = message['role']
        message_content = message['content']

        if role == 'assistant':
            return Assistant(message_content)
        elif role == 'user':
            return User(message_content)
        elif role == 'system':
            return System(cast(TextContent, message_content).get_str())
        else:
            raise ValueError(f'role not found or not supported: {message}')

    def unpack_and_wrap_messages(self, messages: list[Message], model: Optional[str] = None) -> list[dict[str, str]]:
        wrapped: list[dict[str, str]] = []

        if not messages or not all(isinstance(m, Message) for m in messages):
            logging.error('Messages must be a list of Message objects.')
            for m in [m for m in messages if not isinstance(m, Message)]:
                logging.error(f'Invalid message: {m}')
            raise ValueError('Messages must be a list of Message objects.')

        # deal with the system message
        system_messages = cast(list[System], Helpers.filter(lambda m: m.role() == 'system', messages))
        if len(system_messages) > 1:
            logging.warning('More than one system message in the message list. Using the last one.')

        if len(system_messages) > 0:
            wrapped.append(self.to_dict(system_messages[-1], model, server_serialization=False)[0])

        # expand the PDF, Markdown, BrowserContent, and FileContent messages
        expanded_messages: list[Message] = [m for m in messages if m.role() != 'system'].copy()

        if expanded_messages[0].role() != 'user':
            raise ValueError('First message must be from User')

        for message in expanded_messages:
            for i in range(len(message.message)):
                if isinstance(message.message[i], PdfContent):
                    message.message = cast(list[Content], ObjectTransformers.transform_pdf_to_content(cast(PdfContent, message.message[i]), self))
                elif isinstance(message.message[i], MarkdownContent):
                    message.message = cast(list[Content], ObjectTransformers.transform_markdown_to_content(cast(MarkdownContent, message.message[i]), self))
                elif isinstance(message.message[i], BrowserContent):
                    message.message = cast(list[Content], ObjectTransformers.transform_browser_to_content(cast(BrowserContent, message.message[i]), self))
                elif isinstance(message.message[i], FileContent):
                    message.message = cast(list[Content], ObjectTransformers.transform_file_to_content(cast(FileContent, message.message[i]), self))

        # check to see if there are more than self.max_images images in the message list
        images = [c for c in Helpers.flatten([m.message for m in expanded_messages]) if isinstance(c, ImageContent)]
        image_count = len(images)

        # remove smaller images if there are too many
        if image_count > 0 and image_count >= self.max_images:
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
            wrapped.extend(self.to_dict(expanded_messages[i], model, server_serialization=False))

        return wrapped


