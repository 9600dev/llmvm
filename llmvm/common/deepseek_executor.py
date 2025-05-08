import json
import os
from typing import Any, Awaitable, Callable, Optional, cast
from importlib import resources

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import Assistant, AstNode, BrowserContent, Content, FileContent, HTMLContent, ImageContent, MarkdownContent, Message, PdfContent, System, TextContent, User, awaitable_none
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.common.perf import TokenStreamManager

logging = setup_logging()

class DeepSeekExecutor(OpenAIExecutor):
    def __init__(
        self,
        api_key: str = cast(str, os.environ.get('DEEPSEEK_API_KEY')),
        default_model: str = 'deepseek-chat',

        api_endpoint: str = 'https://api.deepseek.com/v1',
        default_max_input_len: int = 64000,
        default_max_output_len: int = 4096,
        max_images: int = 0,
    ):
        if max_images > 0:
            raise ValueError('Deepseek does not support images. max_images must be 0.')

        if default_max_input_len > 64000:
            logging.debug('Deepseek does not support more than 64k tokens. default_max_input_len must be 64000 or less.')
            default_max_input_len = 64000

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

    def scratchpad_token(self) -> str:
        return 'scratchpad'

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

        # deepseek reasoner doesn't support multiple user or assistant messages in a row
        # so we should collapse multiple content blocks into a single block
        return [
            {
                'role': message.role(),
                'content': '\n\n'.join(content_list)
            }
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
        # deepseek right now doesn't support images, so it'll be zero.
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

        # deepseek reasoner doesn't support multiple user or assistant messages in a row
        expanded_messages_temp = []
        for i in range(len(expanded_messages)):
            if i > 0 and isinstance(expanded_messages[i], User) and isinstance(expanded_messages[i - 1], User):
                expanded_messages_temp.append(Assistant(TextContent('Okay, I am ready for your next message.')))
            elif i > 0 and isinstance(expanded_messages[i], Assistant) and isinstance(expanded_messages[i - 1], Assistant):
                raise ValueError('Two assistant messages in a row should not happen.')

            expanded_messages_temp.append(expanded_messages[i])
        expanded_messages = expanded_messages_temp

        # now build the json dictionary and return
        for i in range(len(expanded_messages)):
            wrapped.extend(self.to_dict(expanded_messages[i], model, server_serialization=False))

        return wrapped

    def max_input_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        # deepseek v3 API is limited to 64k tokens right now
        return self.default_max_input_len

    def max_output_tokens(
        self,
        model: Optional[str] = None,
    ) -> int:
        return self.default_max_output_len

    async def count_tokens(
        self,
        messages: list[Message],
    ) -> int:
        messages_list = self.unpack_and_wrap_messages(messages, self.default_model)
        return await self.count_tokens_dict(messages_list)

    async def count_tokens_dict(
        self,
        messages: list[dict[str, Any]],
    ) -> int:
        import transformers

        with resources.path("llmvm.common", "deepseek_tokenizer") as tokenizer_path:
            # Convert path to str just to be sure HF Transformers sees it as a string path
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_path), allow_remote_code=True)

            num_tokens = 0
            json_accumulator = ''
            for message in messages:
                if 'content' in message and isinstance(message['content'], str):
                    json_accumulator += message['content']
                elif 'content' in message and isinstance(message['content'], list):
                    for content in message['content']:
                        if 'image_url' in content['type'] and 'url' in content['image_url']:
                            b64data = content['image_url']['url']
                            num_tokens += Helpers.openai_image_tok_count(b64data.split(',')[1])
                        else:
                            json_accumulator += json.dumps(content, indent=2)

            token_count = len(tokenizer.encode(json_accumulator))
            return token_count

    async def aexecute_direct(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, str]] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
    ) -> TokenStreamManager:
        return await super().aexecute_direct(messages, functions, model, max_output_tokens, temperature, stop_tokens)

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
        return await super().aexecute(messages, max_output_tokens, temperature, stop_tokens, model, thinking, stream_handler)
