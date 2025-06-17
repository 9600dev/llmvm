import ast
import asyncio
import copy
import datetime
import math
# required for the prompt
import tzlocal
import random
from typing import Any, Awaitable, Callable, Optional, Tuple, cast

import pandas as pd

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import (
    no_indent_debug,
    response_writer,
    role_debug,
    setup_logging,
)
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (
    Answer,
    Assistant,
    AstNode,
    BrowserContent,
    Content,
    Controller,
    Executor,
    FileContent,
    FunctionCall,
    FunctionCallMeta,
    LLMCall,
    MarkdownContent,
    Message,
    PandasMeta,
    PdfContent,
    Statement,
    System,
    TextContent,
    TokenCompressionMethod,
    TokenNode,
    User,
    awaitable_none,
)
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.server.auto_global_dict import AutoGlobalDict

logging = setup_logging()


class ExecutionController(Controller):
    def __init__(
        self,
        executor: Executor,
        helpers: Callable[[], list[Callable]],
        exception_limit: int = 3,
        thread_id: int = 0,
    ):
        super().__init__()
        self.executor = executor
        self.helpers = helpers
        self.exception_limit = exception_limit
        self.thread_id = thread_id

    def __execution_prompt(self, executor: Executor, model: str, thinking: int) -> str:
        if (
            executor.name() == "openai"
            and cast(OpenAIExecutor, executor).does_not_stop(model)
        ) or (executor.name() == "anthropic" and thinking > 0):
            logging.debug(
                "ExecutionController.__execution_prompt() using python_continuation_execution_reasoning.prompt"
            )
            return "python_continuation_execution_reasoning.prompt"
        else:
            logging.debug(
                "ExecutionController.__execution_prompt() using python_continuation_execution.prompt"
            )
            return "python_continuation_execution.prompt"

    async def __execute_llm_call(
        self,
        llm_call: LLMCall,
    ) -> Assistant:
        # execute the call to check to see if the Answer satisfies the original query
        messages: list[Message] = copy.deepcopy(llm_call.context_messages)

        # don't append the user message if it's empty
        if llm_call.user_message.get_str().strip() != "":
            messages.append(llm_call.user_message)

        try:
            assistant: Assistant = await llm_call.executor.aexecute(
                messages,
                max_output_tokens=llm_call.completion_tokens_len,
                temperature=llm_call.temperature,
                model=llm_call.model,
                stop_tokens=llm_call.stop_tokens,
                thinking=llm_call.thinking,
                stream_handler=llm_call.stream_handler,
            )
            role_debug(
                logging, llm_call.prompt_name, "User", llm_call.user_message.get_str()
            )
            role_debug(logging, llm_call.prompt_name, "Assistant", assistant.get_str())
        except Exception as ex:
            role_debug(
                logging, llm_call.prompt_name, "User", llm_call.user_message.get_str()
            )
            raise ex
        response_writer(llm_call.prompt_name, assistant)
        return assistant

    async def aexecute_llm_call_simple(
        self,
        llm_call: LLMCall,
        template: dict[str, Any] = {},
    ) -> Assistant:
        if llm_call.prompt_name:
            prompt = Helpers.load_and_populate_prompt(
                prompt_name=llm_call.prompt_name,
                template=template,
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                scratchpad_token=self.get_executor().scratchpad_token(),
                append_token=self.get_executor().append_token(),
            )
            llm_call.user_message = User(TextContent(prompt["user_message"]))

        return await self.__execute_llm_call(llm_call)

    def execute_llm_call_simple(
        self,
        llm_call: LLMCall,
        template: dict[str, Any] = {},
    ) -> Assistant:
        return asyncio.run(
            self.aexecute_llm_call_simple(
                llm_call=llm_call,
                template=template,
            )
        )

    def __parse_template(
        self, message: Message, template_args: dict[str, Any]
    ) -> Message:
        # {{templates}}
        # check to see if any of the content nodes have {{templates}} in them, and if so, replace
        # with template_args.
        for content in message.message:
            if isinstance(content, TextContent):
                message_text = content.get_str()
                for key, value in template_args.items():
                    key_replace = "{{" + key + "}}"
                    if key_replace in message_text:
                        content.sequence = message_text.replace(key_replace, value)
        return message

    def __parse_message_template(
        self, messages: list[Message], template_args: dict[str, Any]
    ) -> list[Message]:
        new_messages = []
        for message in messages:
            # Assume each message has a 'message' attribute that is a list of contents.
            replaced = False
            for content in message.message:
                if (
                    isinstance(content, (TextContent, FileContent))
                    and "[system_message]" in content.get_str()
                    and "[user_message]" in content.get_str()
                ):
                    # Obtain the two prompts from the helper.
                    system, user = Helpers.get_prompts(
                        content.get_str(),
                        template_args,
                        self.get_executor().user_token(),
                        self.get_executor().assistant_token(),
                        self.get_executor().scratchpad_token(),
                        self.get_executor().append_token(),
                    )
                    # Append the parsed system and user messages.
                    new_messages.append(system)
                    new_messages.append(user)
                    replaced = True
                    break  # Found our marker in this message, so stop checking further.
            if not replaced:
                new_messages.append(message)
        return new_messages

    def __parse_function(self, message: Message, tools: list[Callable]) -> Message:
        for content in [
            c
            for c in message.message
            if isinstance(c, TextContent) or isinstance(c, FileContent)
        ]:
            content_text = content.get_str()
            if "{{functions}}" in content_text:
                new_content = content_text.replace(
                    "{{functions}}",
                    "\n".join(
                        [Helpers.get_function_description_flat(f) for f in tools]
                    ),
                )
                if isinstance(content, TextContent):
                    content.sequence = new_content
                else:
                    content.sequence = new_content.encode("utf-8")
        return message

    async def __similarity(
        self,
        llm_call: LLMCall,
        query: str,
    ) -> Assistant:
        tokens_per_message = (
            math.floor(
                (
                    llm_call.max_prompt_len
                    - await self.executor.count_tokens([llm_call.user_message])
                )
                / len(llm_call.context_messages)
            )  # noqa E501
        )
        write_client_stream(
            TextContent(
                f"Performing context window compression type: similarity with tokens per message {tokens_per_message}.\n"
            )
        )  # noqa E501

        # Simple truncation approach instead of vector similarity
        similarity_messages = []
        for i in range(len(llm_call.context_messages)):
            prev_message = llm_call.context_messages[i]

            # Basic truncation approach
            message_str = prev_message.get_str()
            # Use a portion at the beginning and end to maintain context
            trunc_size = tokens_per_message // 2
            start_part = message_str[: trunc_size * 4]  # Rough character estimation
            end_part = (
                message_str[-trunc_size * 4 :]
                if len(message_str) > trunc_size * 8
                else ""
            )

            if len(start_part) + len(end_part) > 0:
                truncated_message = start_part
                if end_part:
                    truncated_message += (
                        "\n\n... [content truncated] ...\n\n" + end_part
                    )
            else:
                truncated_message = message_str

            # check for the header of a statement_to_message. We need to keep this
            if "Result:\n" in message_str and not truncated_message.startswith(
                "Result:\n"
            ):
                truncated_message = "Result:\n" + truncated_message

            similarity_messages.append(User(TextContent(truncated_message)))

        total_similarity_tokens = sum(
            [await self.executor.count_tokens([m.message]) for m in similarity_messages]
        )
        if total_similarity_tokens > llm_call.max_prompt_len:
            logging.error(
                f"__similarity() total_similarity_tokens: {total_similarity_tokens} is greater than max_prompt_len: {llm_call.max_prompt_len}, will perform map/reduce."
            )  # noqa E501

        assistant_result = await self.__execute_llm_call(
            llm_call=LLMCall(
                user_message=llm_call.user_message,
                context_messages=similarity_messages,
                executor=llm_call.executor,
                model=llm_call.model,
                temperature=llm_call.temperature,
                max_prompt_len=llm_call.max_prompt_len,
                completion_tokens_len=llm_call.completion_tokens_len,
                prompt_name=llm_call.prompt_name,
                stream_handler=llm_call.stream_handler,
            ),
        )
        return assistant_result

    async def __map_reduce(
        self,
        query: str,
        original_query: str,
        llm_call: LLMCall,
    ) -> Assistant:
        prompt_len = await self.executor.count_tokens(
            llm_call.context_messages + [llm_call.user_message]
        )
        write_client_stream(
            TextContent(
                f"Performing context window compression type: map/reduce with token length {prompt_len}.\n"
            )
        )

        # collapse the context messages into single message
        context_message = User(
            TextContent("\n\n".join([m.get_str() for m in llm_call.context_messages]))
        )
        chunk_results = []

        # iterate over the data.
        map_reduce_prompt_tokens = await self.executor.count_tokens(
            [
                User(
                    TextContent(
                        Helpers.load_resources_prompt("map_reduce_map.prompt")[
                            "user_message"
                        ]
                    )
                )
            ]
        )

        chunk_size = (
            llm_call.max_prompt_len
            - map_reduce_prompt_tokens
            - await self.executor.count_tokens([llm_call.user_message])
            - 32
        )  # noqa E501

        # Simple text chunking implementation to replace vector_search.chunk
        def simple_chunk_text(text, chunk_size, overlap=0):
            # Approximate character count for token estimation (rough approximation)
            char_per_token = 4
            chars_per_chunk = chunk_size * char_per_token

            # Create chunks with basic sentences splitting
            chunks = []
            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + chars_per_chunk, text_len)

                # Try to end at a sentence boundary if possible
                if end < text_len:
                    for sentence_end in [". ", "! ", "? ", "\n\n"]:
                        pos = text.rfind(sentence_end, start, end)
                        if pos > start:
                            end = pos + len(sentence_end)
                            break

                chunks.append(text[start:end])
                start = end - (overlap * char_per_token)
                if start < 0:
                    start = 0

            return chunks

        chunks = simple_chunk_text(context_message.get_str(), chunk_size)

        for chunk in chunks:
            chunk_assistant = await self.aexecute_llm_call_simple(
                llm_call=LLMCall(
                    user_message=User(TextContent("")),
                    context_messages=[],
                    executor=llm_call.executor,
                    model=llm_call.model,
                    temperature=llm_call.temperature,
                    max_prompt_len=llm_call.max_prompt_len,
                    completion_tokens_len=llm_call.completion_tokens_len,
                    prompt_name="map_reduce_map.prompt",
                    stream_handler=llm_call.stream_handler,
                ),
                template={
                    "original_query": original_query,
                    "query": query,
                    "data": chunk,
                },
            )
            chunk_results.append(chunk_assistant.get_str())

        # perform the reduce
        map_results = "\n\n====\n\n" + "\n\n====\n\n".join(chunk_results)

        assistant_result = await self.aexecute_llm_call_simple(
            llm_call=LLMCall(
                user_message=User(TextContent("")),
                context_messages=[],
                executor=self.executor,
                model=llm_call.model,
                temperature=llm_call.temperature,
                max_prompt_len=llm_call.max_prompt_len,
                completion_tokens_len=llm_call.completion_tokens_len,
                prompt_name="map_reduce_reduce.prompt",
                stream_handler=llm_call.stream_handler,
            ),
            template={
                "original_query": original_query,
                "query": query,
                "map_results": map_results,
            },
        )
        return assistant_result

    async def __summary_map_reduce(
        self,
        llm_call: LLMCall,
    ) -> Assistant:
        header_text_token_len = 60
        # todo: this is a hack. we should check the length of all the messages, and if they're less
        # than the tokens per message, then we can add more tokens to other messages.
        tokens_per_message = (
            math.floor(
                (
                    llm_call.max_prompt_len
                    - await self.executor.count_tokens([llm_call.user_message])
                )
                / len(llm_call.context_messages)
            )  # noqa E501
        )
        tokens_per_message = tokens_per_message - header_text_token_len

        write_client_stream(
            TextContent(
                f"Performing context window compression type: summary map/reduce with tokens per message {tokens_per_message}.\n"
            )
        )  # noqa E501

        llm_call_copy = llm_call.copy()

        if (
            await llm_call.executor.count_tokens([llm_call.user_message])
            > tokens_per_message
        ):
            logging.debug(
                "__summary_map_reduce() user message is longer than the summary window, will try to cut."
            )
            llm_call_copy.user_message = User(
                TextContent(llm_call.user_message.get_str()[0:tokens_per_message])
            )

        # for all messages, do a similarity search
        summary_messages = []
        for i in range(len(llm_call.context_messages)):
            current_message = llm_call.context_messages[i]

            summary_str = ""
            # we want to summarize the message into the appropriate 'tokens per message' size
            if isinstance(current_message.message, PdfContent):
                summary_str = f"This message contains a summary of the PDF document at: {current_message.message.url}.\n\n"
            elif isinstance(current_message.message, FileContent):
                summary_str = f"This message contains a summary of the file at: {current_message.message.url}.\n\n"
            elif isinstance(current_message.message, Content):
                summary_str = f"This message contains a summary of some arbitrary content. A future prompt will contain the full message.\n\n"  # noqa E501

            summary_str += f"{current_message.get_str()[0:tokens_per_message]}"
            summary_messages.append(User(TextContent(summary_str)))

        # we should have all the same messages, but summarized.
        llm_call_copy.context_messages = summary_messages

        assistant_result = await self.__execute_llm_call(
            llm_call_copy,
        )
        return assistant_result

    async def __lifo(self, llm_call: LLMCall) -> Assistant:
        write_client_stream(
            TextContent(
                "Performing context window compression type: last-in-first-out.\n"
            )
        )
        lifo_messages = copy.deepcopy(llm_call.context_messages)

        prompt_context_messages = [llm_call.user_message]
        current_tokens = (
            await self.executor.count_tokens([llm_call.user_message])
            + llm_call.completion_tokens_len
        )

        # reverse over the messages, last to first
        for i in range(len(lifo_messages) - 1, -1, -1):
            if (
                current_tokens + await self.executor.count_tokens([lifo_messages[i]])
                < llm_call.max_prompt_len
            ):
                prompt_context_messages.append(lifo_messages[i])
                current_tokens += await self.executor.count_tokens([lifo_messages[i]])
            else:
                break

        new_call = llm_call.copy()
        new_call.context_messages = prompt_context_messages[::-1]

        assistant_result = await self.__execute_llm_call(llm_call=new_call)
        return assistant_result

    def statement_to_str(
        self,
        context: list[Statement] | Statement | str | list[Content] | Content,
    ):
        messages = self.statement_to_message(context)
        return "\n\n".join(
            [
                message.message.get_str()
                for message in messages
                if isinstance(message.message, Content)
            ]
        )

    def statement_to_message(
        self,
        context: list[Statement] | Statement | str | list[Content] | Content,
    ) -> list[Message]:
        from llmvm.server.base_library.function_bindable import FunctionBindable

        statement_result_prompts = {
            "answer": "answer_result.prompt",
            "assistant": "assistant_result.prompt",
            "function_call": "function_call_result.prompt",
            "function_meta": "functionmeta_result.prompt",
            "llm_call": "llm_call_result.prompt",
            "str": "str_result.prompt",
            "foreach": "foreach_result.prompt",
            "list": "list_result.prompt",
        }

        if (
            isinstance(context, list)
            and len(context) > 0
            and (
                isinstance(context[0], Content)
                or isinstance(context[0], Message)
                or isinstance(context[0], Statement)
            )
        ):
            return Helpers.flatten([self.statement_to_message(c) for c in context])

        if isinstance(context, str):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts["str"],
                template={
                    "str_result": context,
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                scratchpad_token=self.get_executor().scratchpad_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt["user_message"]))]

        elif isinstance(context, FunctionCall):
            # check to see if the return result is something we already know
            # that can be cohersed into a message
            if isinstance(context.result(), Content):
                return self.statement_to_message(cast(Content, context.result()))

            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts[context.token()],
                template={
                    "function_call": context.to_code_call(),
                    "function_signature": context.to_definition(),
                    "function_result": str(context.result()),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                scratchpad_token=self.get_executor().scratchpad_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt["user_message"]))]

        elif isinstance(context, FunctionCallMeta):
            # check to see if the return result is something we already know
            # that can be cohersed into a message
            if isinstance(context.result(), Content):
                return self.statement_to_message(cast(Content, context.result()))

            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts["function_meta"],
                template={
                    "function_callsite": context.callsite,
                    "function_result": str(context.result()),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                scratchpad_token=self.get_executor().scratchpad_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt["user_message"]))]

        elif isinstance(context, Assistant):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts["assistant"],
                template={
                    "assistant_result": str(context.message),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                scratchpad_token=self.get_executor().scratchpad_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt["user_message"]))]

        elif isinstance(context, PandasMeta):
            return [User(TextContent(context.df.to_csv()))]

        elif isinstance(context, MarkdownContent):
            return [
                User(
                    cast(
                        list[Content],
                        ObjectTransformers.transform_markdown_to_content(
                            context, self.executor
                        ),
                    )
                )
            ]

        elif isinstance(context, PdfContent):
            return [
                User(
                    cast(
                        list[Content],
                        ObjectTransformers.transform_pdf_to_content(
                            context, self.executor
                        ),
                    )
                )
            ]

        elif isinstance(context, BrowserContent):
            return [
                User(
                    cast(
                        list[Content],
                        ObjectTransformers.transform_browser_to_content(
                            context, self.executor
                        ),
                    )
                )
            ]

        elif isinstance(context, FileContent):
            return [
                User(
                    cast(
                        list[Content],
                        ObjectTransformers.transform_file_to_content(
                            context, self.executor
                        ),
                    )
                )
            ]

        elif isinstance(context, FunctionBindable):
            return [User(Content(context.result()))]  # type: ignore

        elif isinstance(context, User):
            return [context]

        elif isinstance(context, Content):
            return [User(context)]

        elif isinstance(context, list):

            def is_node(n: Any) -> bool:
                return (
                    isinstance(n, AstNode)
                    or isinstance(n, Statement)
                    or isinstance(n, Content)
                    or isinstance(n, Message)
                    or isinstance(n, FunctionBindable)
                )

            # lists can either be native lists, or they can be lists with
            # AstNodes, Content nodes, or Message nodes
            if all([is_node(c) for c in context]):
                return Helpers.flatten([self.statement_to_message(c) for c in context])
            else:
                result_prompt = Helpers.load_and_populate_prompt(
                    prompt_name=statement_result_prompts["list"],
                    template={"list_result": "\n".join([str(c) for c in context])},
                    user_token=self.get_executor().user_token(),
                    assistant_token=self.get_executor().assistant_token(),
                    scratchpad_token=self.get_executor().scratchpad_token(),
                    append_token=self.get_executor().append_token(),
                )
                return [User(TextContent(result_prompt["user_message"]))]

        logging.debug(
            f"statement_to_message() unusual type {type(context)}, context is: {str(context)}"
        )
        return [User(TextContent(str(context)))]

    def get_executor(self) -> Executor:
        return self.executor

    async def aexecute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> Assistant:
        """
        Internal function to execute an LLM call with prompt template and context messages.
        Deals with chunking, map/reduce, and other logic if the message/context messages
        are too long for the context window.
        """
        model = llm_call.model if llm_call.model else self.executor.default_model
        llm_call.model = model

        """
        Executes an LLM call on a prompt_message with a context of messages.
        Performs either a chunk_and_rank, or a map/reduce depending on the
        context relavence to the prompt_message.
        """

        # If we have a PdfContent here, we need to convert it into the appropriate format
        # before firing off the call.
        from llmvm.common.pdf import Pdf

        prompt_len = await self.executor.count_tokens(
            llm_call.context_messages + [llm_call.user_message]
        )
        max_prompt_len = self.executor.max_input_tokens(model=model)

        # I have either a message, or a list of messages. They might need to be map/reduced.
        # todo: we usually have a prepended message of context to help the LLM figure out
        # what to do with the message at a later stage. This is getting removed right now.
        if prompt_len <= max_prompt_len:
            # straight call
            return await self.__execute_llm_call(llm_call=llm_call)
        elif prompt_len > max_prompt_len and compression == TokenCompressionMethod.LIFO:
            return await self.__lifo(llm_call)
        elif (
            prompt_len > max_prompt_len
            and compression == TokenCompressionMethod.SUMMARY
        ):
            return await self.__summary_map_reduce(llm_call)
        elif (
            prompt_len > max_prompt_len
            and compression == TokenCompressionMethod.MAP_REDUCE
        ):
            return await self.__map_reduce(query, original_query, llm_call)
        elif (
            prompt_len > max_prompt_len
            and compression == TokenCompressionMethod.SIMILARITY
        ):
            return await self.__similarity(llm_call, query)
        else:
            # Default behaviour is to summarize the context with map/reduce
            write_client_stream(
                TextContent(
                    f"The message prompt length: {prompt_len} is bigger than the max prompt length: {max_prompt_len} for executor {llm_call.executor.name()}\n"
                )
            )
            write_client_stream(
                TextContent(
                    f"Context window compression strategy: {compression.name}.\n"
                )
            )
            write_client_stream(
                TextContent("Using simplified context window compression approach.\n")
            )

            return await self.__map_reduce(query, original_query, llm_call)

    def execute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> Assistant:
        llm_call.model = (
            llm_call.model if llm_call.model else self.executor.default_model
        )

        return asyncio.run(
            self.aexecute_llm_call(
                llm_call=llm_call,
                query=query,
                original_query=original_query,
                compression=compression,
            )
        )

    async def aexecute(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        model: Optional[str] = None,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        thinking: int = 0,
        template_args: dict[str, Any] = {},
    ) -> list[Message]:
        # [system_message] [user_message] pairs - deal with them in text content or file content
        messages = self.__parse_message_template(messages, template_args)

        # {{templates}} check to see if the messages have {{templates}} in them, and if so, replace with template_args
        messages = [self.__parse_template(m, template_args) for m in messages]

        llm_call = LLMCall(
            user_message=messages[-1],
            context_messages=messages[0:-1],
            executor=self.executor,
            model=model if model else self.executor.default_model,
            temperature=temperature,
            max_prompt_len=self.executor.max_input_tokens(),
            completion_tokens_len=self.executor.max_output_tokens(),
            prompt_name="",
            stop_tokens=[],
            thinking=thinking,
            stream_handler=stream_handler,
        )

        response: Assistant = await self.aexecute_llm_call(
            llm_call,
            query=messages[-1].get_str(),
            original_query="",
            compression=compression,
        )
        return messages + [response]

    async def aexecute_continuation(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: dict[str, Any] = {},
        helpers: list[Callable] = [],
        cookies: list[dict[str, Any]] = [],
        runtime_state: AutoGlobalDict = AutoGlobalDict({}, {}),
        thinking: int = 0,
    ) -> Tuple[list[Message], AutoGlobalDict]:
        # we're making a copy because we return the state at the end
        runtime_state = runtime_state.copy()

        if max_output_tokens == 0:
            max_output_tokens = None

        def parse_code_block_result(result) -> list[AstNode]:
            from matplotlib.pyplot import Figure  # type: ignore

            if isinstance(result, str):
                return [TextContent(result)]
            elif isinstance(result, (int, float, bool, dict, complex)):
                return [TextContent(str(result))]
            elif isinstance(result, Content):
                return [result]
            elif isinstance(result, Answer):
                return parse_code_block_result(result.result())  # type: ignore
            elif isinstance(result, FunctionCallMeta):
                return parse_code_block_result(result.result())  # type: ignore
            elif isinstance(result, list):
                results = [parse_code_block_result(r) for r in result]
                return Helpers.flatten(results)
            elif isinstance(result, Assistant):
                return cast(list[AstNode], result.message)
            elif isinstance(result, Statement):
                messages = self.statement_to_message(result)
                content: list[Content] = Helpers.flatten([m.message for m in messages])
                return cast(list[AstNode], content)
            elif isinstance(result, AstNode):
                return [result]
            elif isinstance(result, pd.DataFrame):
                if len(result.to_csv()) < 1024:
                    return [TextContent(result.to_csv())]
                else:
                    return [TextContent(result.head(10).to_csv()[0:1024])]
            elif isinstance(result, Figure):
                return [Helpers.matplotlib_figure_to_image_content(result)]
            elif isinstance(result, datetime.datetime):
                return [TextContent(str(result))]
            elif isinstance(result, tuple):
                results = [parse_code_block_result(r) for r in result]
                return Helpers.flatten(results)
            elif asyncio.iscoroutine(result):
                return [
                    TextContent(
                        "The result is an asyncio.coroutine. You can use asyncio.run(result) to get the result, or run multiple coroutines in parallel using asyncio.run(asyncio.gather(*coroutines))."
                    )
                ]
            elif Helpers.is_function(result):
                return [TextContent(Helpers.get_function_description_flat(result))]
            else:
                logging.error(f"Unknown content type: {type(result)}, returning shape")
                methods = f"The result is an instance of type {type(result)} with the following methods and static methods:\n"
                methods += f"\n".join(
                    [
                        Helpers.get_function_description_flat(f)
                        for f in Helpers.get_methods_and_statics(result)
                    ]
                )
                return [TextContent(methods)]

        from llmvm.server.python_runtime_host import PythonRuntimeHost

        python_runtime_host = PythonRuntimeHost(
            controller=self, answer_error_correcting=False, thread_id=self.thread_id
        )

        model = model if model else self.executor.default_model

        response: Assistant = Assistant(TextContent(""))

        # [system_message] [user_message] pairs - deal with them in text content or file content
        messages = self.__parse_message_template(messages, template_args)

        # {{templates}} check to see if the messages have {{templates}} in them, and if so, replace with template_args
        messages = [self.__parse_template(m, template_args) for m in messages]

        # {{functions}} template
        messages = [self.__parse_function(m, helpers) for m in messages]

        # bootstrap the continuation execution
        completed = False
        results: list[Statement] = []
        old_model = python_runtime_host.controller.get_executor().default_model
        if model:
            python_runtime_host.controller.get_executor().default_model = model

        # inject the python_continuation_execution.prompt prompt
        functions = [Helpers.get_function_description_flat(f) for f in helpers]

        system_message, tools_message = Helpers.prompts(
            prompt_name=self.__execution_prompt(self.get_executor(), model, thinking),
            template={
                "functions": "\n".join(functions),
                "context_window_tokens": str(self.get_executor().max_input_tokens()),
                "context_window_words": str(
                    int(self.get_executor().max_input_tokens() * 0.75)
                ),
                "context_window_bytes": str(
                    int(self.get_executor().max_input_tokens() * 4)
                ),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            scratchpad_token=self.get_executor().scratchpad_token(),
            append_token=self.get_executor().append_token(),
        )

        if self.get_executor().name() == "openai" and cast(
            OpenAIExecutor, self.get_executor()
        ).responses(model):
            # merge system and tools messages
            # https://platform.openai.com/docs/guides/reasoning-best-practices
            system_message = System(
                "Formatting re-enabled\n"
                + system_message.get_str()
                + "\n\n"
                + tools_message.get_str()
            )
            tools_message = User(TextContent(""))

        # anthropic prompt caching through the first three messages
        assistant_reply = Assistant(TextContent("Yes, I am ready."), hidden=True)
        assistant_reply.prompt_cached = True
        system_message.hidden = True
        tools_message.hidden = True

        # inject the tools_message into the messages, and make sure it's first.
        tools_message.pinned = (
            0  # pinned as the first message # todo: thread this through the system
        )
        messages_copy = []
        messages_copy.append(system_message)
        if not (
            self.get_executor().name() == "openai"
            and cast(OpenAIExecutor, self.get_executor()).responses(model)
        ):
            messages_copy.append(tools_message)
            messages_copy.append(assistant_reply)

        # strip prompt caching from all messages so we can only have it on the last message
        for message in messages:
            message.prompt_cached = False

        messages_copy.extend(copy.deepcopy(messages))

        # todo: hack
        browser_content_start_token = "<helpers_result>BrowserContent("

        # execute the continuation loop
        exception_counter = 0
        code_blocks_executed: list[str] = []
        while not completed and exception_counter <= self.exception_limit:
            # because BrowserContent can be overly verbose, we'll only allow the last BrowserContent
            # message to be used in the continuation loop
            last_message_str = messages_copy[-1].get_str()
            if last_message_str.count(browser_content_start_token) > 1:
                # remove all but the last BrowserContent message
                last_message_str = Helpers.keep_last_browser_content(last_message_str)
                messages_copy[-1] = Assistant(
                    TextContent(last_message_str),
                    underlying=messages_copy[-1].underlying,
                )

            # undo the last prompt cached flag, because we're moving the cache checkpoint to the end
            prompt_cache_counter = 0
            for message in reversed(messages_copy):
                if (
                    message.prompt_cached
                    and message != assistant_reply
                    and message != system_message
                    and message != tools_message
                ):
                    message.prompt_cached = False
                    prompt_cache_counter += 1
                if prompt_cache_counter >= 2:
                    break

            # this is an anthropic pattern
            messages_copy[-1].prompt_cached = True
            messages_copy[-2].prompt_cached = True

            llm_call = LLMCall(
                user_message=messages_copy[-1],
                context_messages=messages_copy[0:-1],
                executor=self.executor,
                model=model,
                temperature=temperature,
                max_prompt_len=self.executor.max_input_tokens(),
                completion_tokens_len=max_output_tokens
                or self.executor.max_output_tokens(),
                prompt_name="",
                stop_tokens=["</helpers>", "</complete>"],
                thinking=thinking,
                stream_handler=stream_handler,
            )

            response: Assistant = await self.aexecute_llm_call(
                llm_call,
                query=messages[-1].get_str(),
                original_query="",
                compression=compression,
            )

            # openai doesn't return the stop token, so we should check the last message for the <helpers> token
            if (
                "<helpers>" in response.get_str()
                and response.stop_token == ""
                and (
                    self.get_executor().name() == "openai"
                    or self.get_executor().name() == "gemini"
                    or self.get_executor().name() == "deepseek"
                    or self.get_executor().name() == "bedrock"
                )
            ):
                response.stop_token = "</helpers>"

            if (
                response.stop_reason == "stop"
                and response.stop_token != "</helpers>"
                and (
                    self.get_executor().name() == "openai"
                    or self.get_executor().name() == "gemini"
                    or self.get_executor().name() == "deepseek"
                    or self.get_executor().name() == "bedrock"
                )
            ):
                response.stop_token = "</complete>"

            if (
                "</complete>" in response.get_str()
                and response.stop_reason == ""
                and (
                    self.get_executor().name() == "openai"
                    or self.get_executor().name() == "gemini"
                    or self.get_executor().name() == "deepseek"
                    or self.get_executor().name() == "bedrock"
                )
            ):
                response.message = [
                    TextContent(response.get_str().replace("</complete>", ""))
                ]
                response.stop_token = "</complete>"

            if (
                response.stop_reason == "end_turn"
                and response.stop_token == ""
                and self.get_executor().name() == "anthropic"
                and "<helpers>" in response.get_str().strip()
                and not "</helpers>" in response.get_str().strip()
            ):
                # check to see if this is an unterminated <helpers> block
                # for whatever reason, anthropic seems to have regressed on this lately.
                if PythonRuntimeHost.get_helpers_code_blocks(
                    response.get_str().strip() + "\n</helpers>"
                ):
                    response.stop_token = "</helpers>"
                    response.stop_reason = "stop_sequence"
                    response.message = [
                        TextContent(response.get_str().strip() + "\n</helpers>")
                    ]

            # Two Assistant messages in a row: we don't want two assistant messages in a row (which is what happens if you're asking
            # for a continuation), so we remove the last Assistant message and replace it with the Assistant response
            # we just got, plus the previous Assistant response.
            if isinstance(messages_copy[-1], Assistant):
                previous_assistant = messages_copy.pop()
                response.message = [
                    TextContent(previous_assistant.get_str() + " " + response.get_str())
                ]

            # add the stop token to the response
            if (
                response.stop_token
                and response.stop_token == "</helpers>"
                and not response.get_str().strip().endswith("</helpers>")
            ):
                response.message = [
                    TextContent(response.get_str() + "\n" + response.stop_token)
                ]

            # extract any code blocks the Assistant wants to run
            assistant_response_str = (
                response.get_str().replace("Assistant:", "").strip()
            )
            code_blocks: list[str] = PythonRuntimeHost.get_helpers_code_blocks(
                assistant_response_str
            )
            code_blocks_remove = []

            # filter code_blocks we've already seen
            for code_block in code_blocks:
                for code_block_executed in code_blocks_executed:
                    if Helpers.compare_code_blocks(code_block, code_block_executed):
                        code_blocks_remove.append(code_block)
            code_blocks = [
                code_block
                for code_block in code_blocks
                if code_block not in code_blocks_remove
            ]

            # run code blocks we haven't run before
            if code_blocks and not response.stop_token == "</complete>":
                # emit some debugging
                code_block = "\n".join(code_blocks)
                if not (
                    self.get_executor().name() == "openai"
                    and cast(OpenAIExecutor, self.get_executor()).does_not_stop(model)
                ):
                    write_client_stream(TokenNode("</helpers>\n"))

                write_client_stream(
                    TextContent("Executing helpers code block locally.\n")
                )

                no_indent_debug(logging, "")
                no_indent_debug(
                    logging,
                    "** [bold yellow]Python Abstract Syntax Tree:[/bold yellow] **",
                )
                # debug out AST
                lines = Helpers.split_on_newline(
                    Helpers.escape_newlines_in_strings(code_block)
                )  # code_block.split('\n')
                line_counter = 1
                for line in lines:
                    line = line.replace("[", "\\[")
                    no_indent_debug(logging, f"{line_counter:02}  {line}")
                    line_counter += 1
                no_indent_debug(logging, "")

                # debug output
                response_writer("llm_call", code_block)

                # make sure the code block is valid and syntactically correct Python
                try:
                    _ = ast.parse(Helpers.escape_newlines_in_strings(code_block))
                except SyntaxError as ex:
                    logging.debug(
                        "ExecutionController.aexecute() SyntaxError trying to parse <helpers></helpers> code block: {}".format(
                            ex
                        )
                    )
                    code_block = PythonRuntimeHost.fix_python_parse_compile_error(
                        controller=self,
                        python_code=code_block,
                        error=str(ex),
                    )
                    if code_block.strip().startswith("```"):
                        code_block_extracted = Helpers.extract_code_blocks(
                            code_block.strip()
                        )
                        code_block = (
                            code_block_extracted[0]
                            if code_block_extracted
                            else code_block
                        )

                if cookies:
                    runtime_state["cookies"] = cookies

                code_execution_result: Optional[list[AstNode]] = []
                hidden = False
                answers = []

                try:
                    # here we're using the original messages list because messages() wouldn't work without
                    # the original. we append the response to this messages list later in the code.
                    answers = python_runtime_host.compile_and_execute_code_block(
                        python_code=code_block,
                        messages_list=messages_copy,
                        helpers=helpers,
                        runtime_state=runtime_state,
                    )

                    # Python was executed without exceptions, reset the exception counter
                    # and add any result() results to the results list
                    exception_counter = 0
                    code_blocks_executed.append(code_block)
                except Exception as ex:
                    # we call the assistant again, and the string will often contain the original code block
                    # even though we got an exception, we want to make sure we specify that we've executed it
                    code_blocks_executed.append(code_block)
                    hidden = True
                    # update the code_execution_result to expose the exception for the next iteration
                    logging.debug(
                        "ExecutionController.aexecute_continuation() Exception executing code block: {}".format(
                            ex
                        )
                    )
                    code_execution_result = parse_code_block_result(f"\n{str(ex)}")

                    exception_counter += 1
                    if exception_counter == self.exception_limit:
                        EXCEPTION_PROMPT = """We have reached our exception limit.
                        Running any of the previous code again won't work.
                        You have one more shot, try something very different. Feel free to just emit a natural language message instead of code.\n"""
                        code_execution_result = parse_code_block_result(
                            f"{Helpers.str_get_str(code_execution_result)}\n\n{EXCEPTION_PROMPT}"
                        )
                        logging.debug("aexecute() Exception limit reached)")

                # code_execution_result will be empty if the code block executed without exceptions
                # and will include the exception if it threw
                if answers and not code_execution_result:
                    # code block successfully executed, grab the result from any result() blocks and stuff them
                    # in the code_execution_result var
                    code_execution_result = parse_code_block_result(answers)
                elif not answers and code_execution_result:
                    # exception!
                    # code_execution_result = parse_code_block_result(code_execution_result)
                    pass
                elif not answers and not code_execution_result:
                    # sometimes dove doesn't generate an result() block, so we'll have to get the last assignment of the code and use that.
                    last_assignment = PythonRuntimeHost.get_last_assignment(
                        code_block, runtime_state
                    )
                    if last_assignment:
                        # todo: this forces a string, but we can deal with more than that these days
                        code_execution_result = parse_code_block_result(
                            last_assignment[1]
                        )
                    else:
                        last_statement = PythonRuntimeHost.get_last_statement(code_block, runtime_state)
                        if last_statement:
                            name, value = last_statement[0], parse_code_block_result(last_statement[1])
                            code_execution_result = parse_code_block_result(value)

                if not code_execution_result:
                    # no result() block, or last assignment was None
                    code_execution_result = parse_code_block_result(
                        f'No result() block found in code block, or last Python statement was "None".'
                    )

                assert (
                    isinstance(code_execution_result, list)
                    and len(code_execution_result) > 0
                )

                # todo: this should be AstNode or TextNode or ...
                # but for now we're just making it a str()
                code_execution_result_str = "\n".join(
                    [Helpers.str_get_str(c) for c in code_execution_result]
                )

                # we have a <helpers_result></helpers_result> block, push it to the cli client
                if len(code_execution_result_str) > 300 and not 'An exception occured' in code_execution_result_str and not '<ast>' in code_execution_result_str:
                    # grab the first and last 150 characters
                    write_client_stream(
                        TokenNode(
                            f"<helpers_result>{code_execution_result_str[:150]}\n\n ...excluded for brevity...\n\n{code_execution_result_str[-150:]}</helpers_result>\n\n"
                        )
                    )
                else:
                    write_client_stream(
                        TokenNode(
                            f"<helpers_result>{code_execution_result_str}</helpers_result>\n\n"
                        )
                    )

                # todo: we're using a string here to embed the result in the helpers_result
                # but I think we can probably have all sorts of stuff in here, including images.
                messages_copy.append(response)
                content_messages = []
                content_messages.append(TextContent(f"<helpers_result>"))
                for c in code_execution_result:
                    if isinstance(c, Content):
                        content_messages.append(c)
                    else:
                        logging.debug(
                            "code_execution_result item is not a Content: {}".format(c)
                        )
                        content_messages.append(TextContent(Helpers.str_get_str(c)))
                content_messages.append(TextContent("</helpers_result>"))
                completed_code_user_message = User(content_messages, hidden=hidden)
                # required to complete the continuation
                messages_copy.append(completed_code_user_message)

            # we have a stop token and there are no code blocks. Assistant is finished, so we can just append the response
            elif response.stop_token and response.stop_token == "</complete>":
                # oh gemini, why do you do this? returning </complete> as a string not as a stop token. sigh.
                if (
                    response.get_str().strip() != ""
                    and response.get_str().strip() != "</complete>"
                ):
                    results.append(Answer(result=response.get_str()))
                # empty assistant, maybe we got a </helpers_result> that was correct. this should be in the code_execution_result
                elif code_blocks and code_execution_result:
                    results.extend(
                        [
                            answer
                            for answer in cast(list, code_execution_result)
                            if isinstance(answer, Answer)
                        ]
                    )
                elif (
                    "</complete>" in response.get_str().strip()
                    or response.get_str().strip() == ""
                ):
                    # grab a previous assistant, as the reasoning models are forced to emit </complete> at the end of the response
                    last_assistant = Helpers.last(
                        lambda m: m.role() == "assistant", messages_copy
                    )
                    if last_assistant:
                        results.append(Answer(result=last_assistant.get_str()))
                completed = True
            # code_blocks was filtered out, and we have a stop token, so it's trying to repeat the code again
            elif not code_blocks and response.stop_token == "</helpers>":
                assistant_response_str = response.get_str()
                assistant_response_str += (
                    "\n\n"
                    + "I've repeated the same code block. I should try something different and not repeat the same code again."
                )
                messages_copy.append(
                    Assistant(
                        TextContent(assistant_response_str),
                        total_tokens=response.total_tokens,
                        stop_reason=response.stop_reason,
                        stop_token=response.stop_token,
                        perf_trace=response.perf_trace,
                        underlying=response.underlying,
                        hidden=True,
                    )
                )
            # stupid reasoning models
            elif (
                not code_blocks
                and response.stop_token == ""
                and self.get_executor().name() == "openai"
                and cast(OpenAIExecutor, self.get_executor()).does_not_stop(model)
            ):
                messages_copy.append(
                    Assistant(
                        TextContent(assistant_response_str),
                        total_tokens=response.total_tokens,
                        stop_reason=response.stop_reason,
                        stop_token=response.stop_token,
                        perf_trace=response.perf_trace,
                        underlying=response.underlying,
                        hidden=True,
                    )
                )
                messages_copy.append(
                    User(
                        TextContent(
                            """I didn't see a </complete> tag yet. If you're finished, emit the </complete> tag. If not, keep going. Use <helpers> if you need to!"""
                        ),
                        hidden=True,
                    )
                )
                completed = False
            else:
                # if there are no code blocks, we're done
                if response.get_str().strip() != "":
                    results.append(Answer(result=response.get_str()))
                    completed = True
                else:
                    logging.debug(
                        "ExecutionController.aexecute_continuation() Empty message returned. Pushing harder."
                    )
                    PUSH_HARDER_PROMPT = """
                    You've returned an empty message. I need you to return your final result or continue
                    to break down and solve the current problem.
                    """
                    messages_copy.append(
                        User(TextContent(PUSH_HARDER_PROMPT), hidden=True)
                    )
                    completed = False

        if model:
            python_runtime_host.controller.get_executor().default_model = old_model
        dedupped: list[Statement] = list(
            reversed(Helpers.remove_duplicates(results, lambda a: a.result()))
        )

        if messages_copy[-1].role() != "assistant":
            messages_copy.append(
                Assistant(
                    TextContent(
                        "\n".join(
                            [Helpers.str_get_str(statement) for statement in dedupped]
                        )
                    ),
                    total_tokens=response.total_tokens,
                    stop_reason=response.stop_reason,
                    stop_token=response.stop_token,
                    perf_trace=response.perf_trace,
                    underlying=response.underlying,
                )
            )

        # remove hidden messages from the list
        messages_copy = [m for m in messages_copy if not m.hidden]
        # only return the extra messages beyond the messages list that was passed in

        return (messages_copy, runtime_state)

