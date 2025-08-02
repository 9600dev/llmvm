import asyncio
import base64
import json
import os
from typing import Any, Awaitable, Callable, Optional, cast

import tiktoken
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import Function

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import messages_trace, setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (
    Assistant,
    AstNode,
    BrowserContent,
    Content,
    Executor,
    FileContent,
    HTMLContent,
    ImageContent,
    MarkdownContent,
    Message,
    PdfContent,
    System,
    TextContent,
    TokenNode,
    TokenStopNode,
    TokenThinkingNode,
    User,
    awaitable_none,
)
from llmvm.common.perf import O1AsyncIterator, TokenPerf, TokenStreamManager

logging = setup_logging()


class OpenAIExecutor(Executor):
    def __init__(
        self,
        api_key: str = cast(str, Container().get_config_variable("OPENAI_API_KEY")),
        default_model: str = "gpt-4.1",
        api_endpoint: str = Container().get_config_variable("OPENAI_API_BASE", "OPENAI_API_BASE", "https://api.openai.com/v1"),
        default_max_input_len: int = 1047576,
        default_max_output_len: int = 32768,
        max_images: int = 20,
    ):
        super().__init__(
            default_model=default_model,
            api_endpoint=api_endpoint,
            api_key=api_key,
            default_max_input_len=default_max_input_len,
            default_max_output_len=default_max_output_len,
        )
        self.aclient = AsyncOpenAI(api_key=api_key, base_url=self.api_endpoint)
        self.max_images = max_images if 'Llama' not in default_model else 8
        self.api_key = api_key

    def responses(self, model: Optional[str]) -> bool:
        if model and 'Llama' in model:
            return False

        if model and 'grok' in model:
            return False

        return self.does_not_stop(model)

    def user_token(self) -> str:
        return "User"

    def assistant_token(self) -> str:
        return "Assistant"

    def append_token(self) -> str:
        return ""

    def scratchpad_token(self) -> str:
        return "scratchpad"

    def name(self) -> str:
        return "openai"

    def to_dict(
        self,
        message: "Message",
        model: Optional[str],
        server_serialization: bool = False,
    ) -> dict[str, Any]:
        content_list = []
        for content in message.message:
            if isinstance(content, ImageContent) and content.sequence:
                if "image/unknown" not in Helpers.classify_image(content.get_bytes()):
                    content_list.append(
                        {
                            "type": "input_image" if self.responses(model) else "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64.b64encode(content.sequence).decode('utf-8')}" if self.responses(model) else {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(content.sequence).decode('utf-8')}",
                                "detail": "high",
                            },
                            **({"url": content.url} if server_serialization else {}),
                            **(
                                {"content_type": "image"}
                                if server_serialization
                                else {}
                            ),
                        }
                    )
                else:
                    logging.warning(f"Image content type not supported for {model}")
            elif isinstance(content, TextContent) and content.sequence:
                message_type = "output_text" if message.role() == "assistant" else "input_text"
                if not self.responses(model):
                    message_type = "text"
                content_list.append(
                    {
                        "type": message_type,
                        "text": content.get_str(),
                        **({"url": content.url} if server_serialization else {}),
                        **({"content_type": "text"} if server_serialization else {}),
                    }
                )
            elif not content.sequence:
                logging.warning(
                    f"Content inside message {message.to_json()} was empty."
                )
            else:
                raise ValueError(
                    f"Cannot serialize unknown content type: {type(content)} in message {message.to_json()}"
                )

        dict_message = {
            "role": message.role(),
            "content": content_list,
        }
        return dict_message

    def from_dict(self, message: dict[str, Any]) -> "Message":
        # pull out Message related content
        role = message["role"]
        message_content = message["content"]

        # force content to be a list
        if not isinstance(message_content, list):
            message_content = [message_content]

        content_list: list[Content] = []

        for i in range(len(message_content)):
            if "type" in message_content[i] and message_content[i]["type"] == "text":
                url = message_content[i]["url"] if "url" in message_content[i] else ""
                content_list.append(TextContent(message_content[i]["text"], url=url))
            elif (
                "type" in message_content[i]
                and message_content[i]["type"] == "image_url"
            ):
                byte_content = base64.b64decode(
                    message_content[i]["image_url"]["url"].split(",")[1]
                )
                url = message_content[i]["url"] if "url" in message_content[i] else ""
                content_list.append(ImageContent(byte_content, url=url))
            else:
                raise ValueError(f"Unknown content type: {message_content[i]}")

        if role == "assistant":
            return Assistant(content_list[0])
        elif role == "user":
            return User(content_list)
        elif role == "system":
            return System(cast(TextContent, content_list[0]).get_str())
        else:
            raise ValueError(f"role not found or not supported: {message}")

    def unpack_and_wrap_messages(
        self, messages: list[Message], model: Optional[str] = None
    ) -> list[dict[str, str]]:
        wrapped: list[dict[str, str]] = []

        if not messages or not all(isinstance(m, Message) for m in messages):
            logging.error("Messages must be a list of Message objects.")
            for m in [m for m in messages if not isinstance(m, Message)]:
                logging.error(f"Invalid message: {m}")
            raise ValueError("Messages must be a list of Message objects.")

        # deal with the system message
        system_messages = cast(
            list[System], Helpers.filter(lambda m: m.role() == "system", messages)
        )
        if len(system_messages) > 1:
            logging.warning(
                "More than one system message in the message list. Using the last one."
            )

        if len(system_messages) > 0:
            wrapped.append(
                self.to_dict(system_messages[-1], model, server_serialization=False)
            )

        # expand the PDF, Markdown, BrowserContent, and FileContent messages
        expanded_messages: list[Message] = [
            m for m in messages if m.role() != "system" and m.role() != "developer"
        ].copy()

        if expanded_messages[0].role() != "user":
            raise ValueError("First message must be from User")

        for message in expanded_messages:
            for i in range(len(message.message) - 1, -1, -1):
                if isinstance(message.message[i], PdfContent):
                    content_list = cast(
                        list[Content],
                        ObjectTransformers.transform_pdf_to_content(
                            cast(PdfContent, message.message[i]),
                            self,
                            xml_wrapper=False,
                        ),
                    )
                    message.message[i : i + 1] = content_list
                elif isinstance(message.message[i], MarkdownContent):
                    content_list = cast(
                        list[Content],
                        ObjectTransformers.transform_markdown_to_content(
                            cast(MarkdownContent, message.message[i]), self
                        ),
                    )
                    message.message[i : i + 1] = content_list
                elif isinstance(message.message[i], BrowserContent):
                    content_list = cast(
                        list[Content],
                        ObjectTransformers.transform_browser_to_content(
                            cast(BrowserContent, message.message[i]), self
                        ),
                    )
                    message.message[i : i + 1] = content_list
                elif isinstance(message.message[i], FileContent):
                    content_list = cast(
                        list[Content],
                        ObjectTransformers.transform_file_to_content(
                            cast(FileContent, message.message[i]),
                            self,
                            xml_wrapper=False,
                        ),
                    )
                    message.message[i : i + 1] = content_list
                elif isinstance(message.message[i], HTMLContent):
                    content_list = cast(
                        list[Content],
                        ObjectTransformers.transform_html_to_content(
                            cast(HTMLContent, message.message[i]), self
                        ),
                    )
                    message.message[i : i + 1] = content_list

        # check to see if there are more than self.max_images images in the message list
        images = [
            c
            for c in Helpers.flatten([m.message for m in expanded_messages])
            if isinstance(c, ImageContent)
        ]
        image_count = len(images)

        # remove smaller images if there are too many
        if image_count >= self.max_images:
            logging.debug(f"Image count is {image_count}, filtering.")

            # get the top self.max_images ordered by byte array size, then remove the rest
            images.sort(key=lambda x: len(x.sequence), reverse=True)
            smaller_images = images[self.max_images :]

            for image in smaller_images:
                for i in range(len(expanded_messages)):
                    for j in range(len(expanded_messages[i].message)):
                        if expanded_messages[i].message[j] == image:
                            expanded_messages[i].message.pop(j)
                            break

        # remove empty assistant messages
        expanded_messages = Helpers.remove(lambda m: m.role() == "assistant" and m.get_str() == "", expanded_messages)

        # now build the json dictionary and return
        for i in range(len(expanded_messages)):
            wrapped.append(
                self.to_dict(expanded_messages[i], model, server_serialization=False)
            )

        return wrapped

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
        num_tokens = 0
        json_accumulator = ""
        for message in messages:
            if "content" in message and isinstance(message["content"], str):
                json_accumulator += message["content"]
            elif "content" in message and isinstance(message["content"], list):
                for content in message["content"]:
                    if "image_url" in content["type"] and "url" in content["image_url"]:
                        b64data = content["image_url"]["url"]
                        num_tokens += Helpers.openai_image_tok_count(
                            b64data.split(",")[1]
                        )
                    elif "input_image" in content["type"] and self.responses(self.default_model):
                        b64data = content["image_url"]
                        num_tokens += Helpers.openai_image_tok_count(
                            b64data.split(",")[1]
                        )
                    else:
                        json_accumulator += json.dumps(content, indent=2)

        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(json_accumulator))
        return token_count

    async def aexecute_direct(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, str]] = [],
        model: Optional[str] = None,
        max_output_tokens: int = 16384,
        temperature: float = 1.0,
        stop_tokens: list[str] = [],
        thinking: int = 0,
    ) -> TokenStreamManager:
        model = model if model else self.default_model

        # only works if profiling or LLMVM_PROFILING is set to true
        message_tokens = await self.count_tokens_dict(messages)

        if message_tokens > self.max_input_tokens(model=model):
            raise Exception(
                "Prompt too long. input tokens: {}, requested output tokens: {}, total: {}, model: {} max context window is: {}".format(
                    message_tokens,
                    max_output_tokens,
                    message_tokens + max_output_tokens,
                    str(model),
                    self.max_input_tokens(model=model),
                )
            )

        if (
            model is not None
            and "deepseek-reasoner" in model
        ):
            messages = [m for m in messages if m["role"] != "system"]

        messages_cast = cast(list[ChatCompletionMessageParam], messages)
        functions_cast = cast(list[Function], functions)

        messages_trace(messages)

        token_trace = TokenPerf(
            "aexecute_direct", "openai", model, prompt_len=message_tokens
        )  # type: ignore
        token_trace.start()

        # reasoning (supported by gemini etc also)
        effort = None
        if thinking == 1:
            effort = "low"
        elif thinking == 2:
            effort = "medium"
        elif thinking == 3:
            effort = "high"
        elif thinking > 3 and 'gemini' not in model:
            raise ValueError(f'Invalid thinking value: {thinking}. Valid values are 1, 2, or 3 for low, medium, or high.')

        if model is not None and self.responses(model):
            # responses API
            instructions_message = "You are a helpful assistant."
            instructions_message_list = [message for message in messages_cast if message['role'] == 'developer' or message['role'] == 'system']
            if instructions_message_list:
                instructions_message = instructions_message_list[0]['content'][0]['text']  # type: ignore
            messages_cast = [message for message in messages_cast if message['role'] != 'developer' and message['role'] != 'system']

            reasoning = {
                'effort': effort,
                'summary': 'auto'
            }

            base_params = {
                "model": model if model else self.default_model,
                "instructions": instructions_message,
                "input": messages_cast,
                "temperature": temperature if not self.does_not_stop(model) else None,
                "max_output_tokens": max_output_tokens,
                "functions": functions_cast if functions else None,
                "reasoning": reasoning,
                "stream": True,
            }

            params = {k: v for k, v in base_params.items() if v is not None}
            response = await self.aclient.responses.create(**params)
            return TokenStreamManager(response, token_trace)  # type: ignore

        else:
            base_params = {
                "model": model if model else self.default_model,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "messages": messages_cast,
                "functions": functions_cast if functions else None,
                "stream_options": {"include_usage": True},
                "stream": True,
                **({'reasoning_effort': effort} if thinking and 'gemini' not in model else {}),
                **({'stop': stop_tokens} if stop_tokens and 'grok' not in model else {})
            }

            # weird bug in gemini OAI compat where it believes stop needs to be a list
            # so if not attached to llmvm server, this will fail
            if 'gemini' in model and 'stop' in base_params and not base_params['stop']:
                del base_params['stop']

            if thinking and 'gemini' in model:
                if thinking == 1:
                    thinking_budget = 2048
                elif thinking == 2:
                    thinking_budget = 4096
                elif thinking == 3:
                    thinking_budget = 8192
                else:
                    thinking_budget = thinking

                extra_body = {
                    'extra_body': {
                        "google": {
                        "thinking_config": {
                            "thinking_budget": thinking_budget,
                            "include_thoughts": True
                            }
                        }
                    }
                }
                base_params['extra_body'] = extra_body

            params = {k: v for k, v in base_params.items() if v is not None}
            response = await self.aclient.chat.completions.create(**params)
            return TokenStreamManager(response, token_trace)  # type: ignore

    async def aexecute(
        self,
        messages: list[Message],
        max_output_tokens: int = 16384,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
    ) -> Assistant:
        model = model if model else self.default_model
        max_stop_token_len = max([len(stop_token) for stop_token in stop_tokens]) if stop_tokens else 0

        # wrap and check message list
        messages_list: list[dict[str, Any]] = self.unpack_and_wrap_messages(
            messages, model
        )

        stream = self.aexecute_direct(
            messages_list,
            max_output_tokens=max_output_tokens,
            model=model,
            temperature=temperature,
            stop_tokens=stop_tokens,
            thinking=thinking
        )

        text_response: str = ""
        thinking_response: str = ""
        perf = None

        async with await stream as stream_async:  # type: ignore
            async for token in stream_async:  # type: ignore
                if token.thinking:
                    await stream_handler(TokenThinkingNode(token.token))
                    thinking_response += token.token
                else:
                    await stream_handler(TokenNode(token.token))
                    text_response += token.token

                # non-stopping models are a pain.
                if self.does_not_stop(model) and any(stop_token in text_response[-max_stop_token_len*10:] for stop_token in stop_tokens):
                    await stream_async.stream.close()
                    # figure out what stop token was in the response
                    max_index = max(enumerate([text_response.rfind(stop_token) for stop_token in stop_tokens]), key=lambda x: x[1])[0]
                    stop_token = stop_tokens[max_index]
                    text_response = text_response[0:text_response.rfind(stop_token)+len(stop_token)]
                    break

            await stream_handler(TokenStopNode())
            perf = stream_async.perf

        _ = await stream_async.get_final_message()
        perf.log()

        assistant = Assistant(
            message=TextContent(text_response.strip()),
            thinking=TextContent(thinking_response.strip()),
            error=False,
            stop_reason=perf.stop_reason,
            stop_token=perf.stop_token,
            perf_trace=perf,
            total_tokens=perf.total_tokens,
        )
        if assistant.get_str() == "":
            logging.warning(
                f"Assistant message is empty. Returning empty message. {perf.request_id or ''}"
            )
        return assistant

    def execute(
        self,
        messages: list[Message],
        max_output_tokens: int = 16384,
        temperature: float = 0.2,
        stop_tokens: list[str] = [],
        model: Optional[str] = None,
        thinking: int = 0,
        stream_handler: Optional[Callable[[AstNode], None]] = None,
    ) -> Assistant:
        async def stream_pipe(node: AstNode):
            if stream_handler:
                stream_handler(node)

        return asyncio.run(
            self.aexecute(
                messages,
                max_output_tokens,
                temperature,
                stop_tokens,
                model,
                thinking,
                stream_pipe,
            )
        )
