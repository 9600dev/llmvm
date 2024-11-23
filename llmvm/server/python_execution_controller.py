import ast
import asyncio
import base64
import copy
import dis
import inspect
import json
import marshal
import math
import random
import re
from importlib import resources
import traceback
import types
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, cast

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import (no_indent_debug, response_writer,
                                          role_debug, setup_logging)
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Answer, Assistant, AstNode, BrowserContent, Content,
                                  Controller, Executor, FileContent,
                                  FunctionCall, FunctionCallMeta, ImageContent,
                                  LLMCall, MarkdownContent, Message,
                                  PandasMeta, PdfContent, Statement, System, TextContent,
                                  TokenCompressionMethod, User, awaitable_none)
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class ExecutionController(Controller):
    def __init__(
        self,
        executor: Executor,
        tools: list[Callable],
        vector_search: VectorSearch,
        exception_limit: int = 3
    ):
        super().__init__()
        self.executor = executor
        self.tools = tools
        self.vector_search = vector_search
        self.exception_limit = exception_limit

    async def __llm_call(
        self,
        llm_call: LLMCall,
    ) -> Assistant:
        # execute the call to check to see if the Answer satisfies the original query
        messages: list[Message] = copy.deepcopy(llm_call.context_messages)

        # don't append the user message if it's empty
        if llm_call.user_message.get_str().strip() != '':
            messages.append(llm_call.user_message)

        try:
            assistant: Assistant = await llm_call.executor.aexecute(
                messages,
                max_output_tokens=llm_call.completion_tokens_len,
                temperature=llm_call.temperature,
                stream_handler=llm_call.stream_handler,
                model=llm_call.model,
                stop_tokens=llm_call.stop_tokens,
            )
            role_debug(logging, llm_call.prompt_name, 'User', llm_call.user_message.get_str())
            role_debug(logging, llm_call.prompt_name, 'Assistant', assistant.get_str())
        except Exception as ex:
            role_debug(logging, llm_call.prompt_name, 'User', llm_call.user_message.get_str())
            raise ex
        response_writer(llm_call.prompt_name, assistant)
        return assistant

    async def __llm_call_with_prompt(
        self,
        llm_call: LLMCall,
        template: Dict[str, Any],
    ) -> Assistant:
        prompt = Helpers.load_and_populate_prompt(
            prompt_name=llm_call.prompt_name,
            template=template,
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )
        llm_call.user_message = User(TextContent(prompt['user_message']))

        return await self.__llm_call(
            llm_call
        )

    def __parse_template(self, message: Message, template_args: Dict[str, Any]) -> Message:
        # {{templates}}
        # check to see if any of the content nodes have {{templates}} in them, and if so, replace
        # with template_args.
        for content in message.message:
            if isinstance(content, TextContent):
                message_text = content.get_str()
                for key, value in template_args.items():
                    key_replace = '{{' + key + '}}'
                    if key_replace in message_text:
                        content.sequence = message_text.replace(key_replace, value)
        return message

    def __parse_message_template(self, messages: list[Message], template_args: Dict[str, Any]) -> list[Message]:
        # check the [system_message] [user_message] pair, and parse those into messages
        # if they exist
        for i in range(len(messages)):
            for j in range(len(messages[i].message)):
                if (
                    (type(messages[i].message[j]) is TextContent or type(messages[i].message[j]) is FileContent)
                    and '[system_message]' in messages[i].message[j].get_str()
                    and '[user_message]' in messages[i].message[j].get_str()
                ):
                    system, user = Helpers.get_prompts(
                        messages[i].message[j].get_str(),
                        template_args,
                        self.get_executor().user_token(),
                        self.get_executor().assistant_token(),
                        self.get_executor().append_token(),
                    )
                    # inject system and user into messages
                    messages[i] = system
                    messages.insert(i + 1, user)
                    i += 1
        return messages

    def __parse_function(self, message: Message, tools: list[Callable]) -> Message:
        for content in [c for c in message.message if isinstance(c, TextContent) or isinstance(c, FileContent)]:
            content_text = content.get_str()
            if '{{functions}}' in content_text:
                new_content = content_text.replace(
                    '{{functions}}', '\n'.join([Helpers.get_function_description_flat(f) for f in tools])
                )
                if isinstance(content, TextContent):
                    content.sequence = new_content
                else:
                    content.sequence = new_content.encode('utf-8')
        return message

    async def __similarity(
        self,
        llm_call: LLMCall,
        query: str,
    ) -> Assistant:

        tokens_per_message = (
            math.floor((llm_call.max_prompt_len - await self.executor.count_tokens([llm_call.user_message])) / len(llm_call.context_messages))  # noqa E501
        )
        write_client_stream(TextContent(f'Performing context window compression type: similarity vector search with tokens per message {tokens_per_message}.\n'))  # noqa E501

        # for all messages, do a similarity search
        similarity_messages = []
        for i in range(len(llm_call.context_messages)):
            prev_message = llm_call.context_messages[i]

            similarity_chunks = await self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.count_tokens,
                content=prev_message.get_str(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=tokens_per_message - 16,
            )
            similarity_message: str = '\n\n'.join([content for content, _ in similarity_chunks])

            # check for the header of a statement_to_message. We probably need to keep this
            if 'Result:\n' in prev_message.get_str():
                similarity_message = 'Result:\n' + similarity_message

            similarity_messages.append(User(TextContent(similarity_message)))

        total_similarity_tokens = sum(
            [await self.executor.count_tokens(m.message.get_str()) for m in similarity_messages]
        )
        if total_similarity_tokens > llm_call.max_prompt_len:
            logging.error(f'__similarity() total_similarity_tokens: {total_similarity_tokens} is greater than max_prompt_len: {llm_call.max_prompt_len}, will perform map/reduce.')  # noqa E501

        assistant_result = await self.__llm_call(
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
        prompt_len = self.executor.count_tokens(llm_call.context_messages + [llm_call.user_message])
        write_client_stream(TextContent(f'Performing context window compression type: map/reduce with token length {prompt_len}.\n'))

        # collapse the context messages into single message
        context_message = User(TextContent('\n\n'.join([m.get_str() for m in llm_call.context_messages])))
        chunk_results = []

        # iterate over the data.
        map_reduce_prompt_tokens = await self.executor.count_tokens(
            [User(TextContent(Helpers.load_resources_prompt('map_reduce_map.prompt')['user_message']))]
        )

        chunk_size = llm_call.max_prompt_len - map_reduce_prompt_tokens - await self.executor.count_tokens([llm_call.user_message]) - 32  # noqa E501
        chunks = self.vector_search.chunk(
            content=context_message.get_str(),
            chunk_size=chunk_size,
            overlap=0
        )

        for chunk in chunks:
            chunk_assistant = await self.__llm_call_with_prompt(
                llm_call=LLMCall(
                    user_message=User(TextContent('')),
                    context_messages=[],
                    executor=llm_call.executor,
                    model=llm_call.model,
                    temperature=llm_call.temperature,
                    max_prompt_len=llm_call.max_prompt_len,
                    completion_tokens_len=llm_call.completion_tokens_len,
                    prompt_name='map_reduce_map.prompt',
                    stream_handler=llm_call.stream_handler,
                ),
                template={
                    'original_query': original_query,
                    'query': query,
                    'data': chunk,
                },
            )
            chunk_results.append(chunk_assistant.get_str())

        # perform the reduce
        map_results = '\n\n====\n\n' + '\n\n====\n\n'.join(chunk_results)

        assistant_result = await self.__llm_call_with_prompt(
            llm_call=LLMCall(
                user_message=User(TextContent('')),
                context_messages=[],
                executor=self.executor,
                model=llm_call.model,
                temperature=llm_call.temperature,
                max_prompt_len=llm_call.max_prompt_len,
                completion_tokens_len=llm_call.completion_tokens_len,
                prompt_name='map_reduce_reduce.prompt',
                stream_handler=llm_call.stream_handler,
            ),
            template={
                'original_query': original_query,
                'query': query,
                'map_results': map_results
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
            math.floor((llm_call.max_prompt_len - await self.executor.count_tokens([llm_call.user_message])) / len(llm_call.context_messages))  # noqa E501
        )
        tokens_per_message = tokens_per_message - header_text_token_len

        write_client_stream(TextContent(f'Performing context window compression type: summary map/reduce with tokens per message {tokens_per_message}.\n'))  # noqa E501

        llm_call_copy = llm_call.copy()

        if await llm_call.executor.count_tokens([llm_call.user_message]) > tokens_per_message:
            logging.debug('__summary_map_reduce() user message is longer than the summary window, will try to cut.')
            llm_call_copy.user_message = User(TextContent(llm_call.user_message.get_str()[0:tokens_per_message]))

        # for all messages, do a similarity search
        summary_messages = []
        for i in range(len(llm_call.context_messages)):
            current_message = llm_call.context_messages[i]

            summary_str = ''
            # we want to summarize the message into the appropriate 'tokens per message' size
            if isinstance(current_message.message, PdfContent):
                summary_str = f'This message contains a summary of the PDF document at: {current_message.message.url}.\n\n'
            elif isinstance(current_message.message, FileContent):
                summary_str = f'This message contains a summary of the file at: {current_message.message.url}.\n\n'
            elif isinstance(current_message.message, Content):
                summary_str = f'This message contains a summary of some arbitrary content. A future prompt will contain the full message.\n\n'  # noqa E501

            summary_str += f'{current_message.get_str()[0:tokens_per_message]}'
            summary_messages.append(User(TextContent(summary_str)))

        # we should have all the same messages, but summarized.
        llm_call_copy.context_messages = summary_messages

        assistant_result = await self.__llm_call(
            llm_call_copy,
        )
        return assistant_result

    async def __lifo(
        self,
        llm_call: LLMCall
    ) -> Assistant:
        write_client_stream(TextContent('Performing context window compression type: last-in-first-out.\n'))
        lifo_messages = copy.deepcopy(llm_call.context_messages)

        prompt_context_messages = [llm_call.user_message]
        current_tokens = await self.executor.count_tokens(
            llm_call.user_message.get_str()
        ) + llm_call.completion_tokens_len

        # reverse over the messages, last to first
        for i in range(len(lifo_messages) - 1, -1, -1):
            if (
                current_tokens + await self.executor.count_tokens(lifo_messages[i].get_str())
                < llm_call.max_prompt_len
            ):
                prompt_context_messages.append(lifo_messages[i])
                current_tokens += await self.executor.count_tokens(lifo_messages[i].get_str())
            else:
                break

        new_call = llm_call.copy()
        new_call.context_messages = prompt_context_messages[::-1]

        assistant_result = await self.__llm_call(
            llm_call=new_call
        )
        return assistant_result

    def statement_to_str(
        self,
        context: list[Statement] | Statement | str | list[Content] | Content,
    ):
        messages = self.statement_to_message(context)
        return '\n\n'.join([message.message.get_str() for message in messages if isinstance(message.message, Content)])

    def statement_to_message(
        self,
        context: list[Statement] | Statement | str | list[Content] | Content,
    ) -> list[Message]:
        from llmvm.server.base_library.function_bindable import \
            FunctionBindable

        statement_result_prompts = {
            'answer': 'answer_result.prompt',
            'assistant': 'assistant_result.prompt',
            'function_call': 'function_call_result.prompt',
            'function_meta': 'functionmeta_result.prompt',
            'llm_call': 'llm_call_result.prompt',
            'str': 'str_result.prompt',
            'foreach': 'foreach_result.prompt',
            'list': 'list_result.prompt',
        }

        if (
            isinstance(context, list)
            and len(context) > 0
            and (
                isinstance(context[0], Content)
                or isinstance(context[0], Message)
            )
        ):
            return Helpers.flatten([self.statement_to_message(c) for c in context])

        if isinstance(context, str):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts['str'],
                template={
                    'str_result': context,
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt['user_message']))]

        elif isinstance(context, FunctionCall):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts[context.token()],
                template={
                    'function_call': context.to_code_call(),
                    'function_signature': context.to_definition(),
                    'function_result': str(context.result()),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt['user_message']))]

        elif isinstance(context, FunctionCallMeta):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts['function_meta'],
                template={
                    'function_callsite': context.callsite,
                    'function_result': str(context.result()),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt['user_message']))]

        elif isinstance(context, Assistant):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_name=statement_result_prompts['assistant'],
                template={
                    'assistant_result': str(context.message),
                },
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                append_token=self.get_executor().append_token(),
            )
            return [User(TextContent(result_prompt['user_message']))]

        elif isinstance(context, PandasMeta):
            return [User(TextContent(context.df.to_csv()))]

        elif isinstance(context, MarkdownContent):
            return [User(cast(list[Content], ObjectTransformers.transform_markdown_to_content(context, self.executor)))]

        elif isinstance(context, PdfContent):
            return [User(cast(list[Content], ObjectTransformers.transform_pdf_to_content(context, self.executor)))]

        elif isinstance(context, BrowserContent):
            return [User(cast(list[Content], ObjectTransformers.transform_browser_to_content(context, self.executor)))]

        elif isinstance(context, FileContent):
            return [User(cast(list[Content], ObjectTransformers.transform_file_to_content(context, self.executor)))]

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
                    prompt_name=statement_result_prompts['list'],
                    template={
                        'list_result': '\n'.join([str(c) for c in context])
                    },
                    user_token=self.get_executor().user_token(),
                    assistant_token=self.get_executor().assistant_token(),
                    append_token=self.get_executor().append_token(),
                )
                return [User(TextContent(result_prompt['user_message']))]

        logging.debug(f'statement_to_message() unusual type {type(context)}, context is: {str(context)}')
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
        '''
        Internal function to execute an LLM call with prompt template and context messages.
        Deals with chunking, map/reduce, and other logic if the message/context messages
        are too long for the context window.
        '''
        model = llm_call.model if llm_call.model else self.executor.get_default_model()
        llm_call.model = model

        """
        Executes an LLM call on a prompt_message with a context of messages.
        Performs either a chunk_and_rank, or a map/reduce depending on the
        context relavence to the prompt_message.
        """

        # If we have a PdfContent here, we need to convert it into the appropriate format
        # before firing off the call.
        from llmvm.common.pdf import Pdf

        prompt_len = await self.executor.count_tokens(llm_call.context_messages + [llm_call.user_message])
        max_prompt_len = self.executor.max_input_tokens(output_token_len=llm_call.completion_tokens_len, model=model)

        # I have either a message, or a list of messages. They might need to be map/reduced.
        # todo: we usually have a prepended message of context to help the LLM figure out
        # what to do with the message at a later stage. This is getting removed right now.
        if prompt_len <= max_prompt_len:
            # straight call
            return await self.__llm_call(llm_call=llm_call)
        elif prompt_len > max_prompt_len and compression == TokenCompressionMethod.LIFO:
            return await self.__lifo(llm_call)
        elif prompt_len > max_prompt_len and compression == TokenCompressionMethod.SUMMARY:
            return await self.__summary_map_reduce(llm_call)
        elif prompt_len > max_prompt_len and compression == TokenCompressionMethod.MAP_REDUCE:
            return await self.__map_reduce(query, original_query, llm_call)
        elif prompt_len > max_prompt_len and compression == TokenCompressionMethod.SIMILARITY:
            return await self.__similarity(llm_call, query)
        else:
            # let's figure out what method to use
            write_client_stream(TextContent(f'The message prompt length: {prompt_len} is bigger than the max prompt length: {max_prompt_len} for executor {llm_call.executor.name()}\n'))  # noqa E501
            write_client_stream(TextContent(f'Context window compression strategy: {compression.name}.\n'))
            # check to see if we're simply lifo'ing the context messages (last in first out)
            context_message = User(TextContent('\n\n'.join([m.get_str() for m in llm_call.context_messages])))

            # see if we can do a similarity search or not.
            write_client_stream(TextContent(
                'Determining context window compression approach of either similarity vector search, or full map/reduce.\n'
            ))
            similarity_chunks = await self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.count_tokens,
                content=context_message.get_str(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=max_prompt_len - await self.executor.count_tokens([llm_call.user_message]) - 16,  # noqa E501
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.15)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: list[str] = []
            for chunk, _ in similarity_chunks[:5]:
                assistant_similarity = await self.__llm_call_with_prompt(
                    llm_call=LLMCall(
                        user_message=User(TextContent('')),
                        context_messages=[],
                        executor=llm_call.executor,
                        model=llm_call.model,
                        temperature=llm_call.temperature,
                        max_prompt_len=llm_call.max_prompt_len,
                        completion_tokens_len=llm_call.completion_tokens_len,
                        prompt_name='document_chunk.prompt',
                    ),
                    template={
                        'query': str(query),
                        'document_chunk': chunk,
                    },
                )

                decision_criteria.append(assistant_similarity.get_str())
                logging.debug('aexecute_llm_call() map_reduce_required, query_or_task: {}, response: {}'.format(
                    query,
                    assistant_similarity.message,
                ))
                if 'No' in assistant_similarity.get_str():
                    # we can break early, as the 'map_reduced_required' flag will not be set below
                    break

            map_reduce_required = all(['Yes' in d for d in decision_criteria])
            if map_reduce_required:
                return await self.__map_reduce(query, original_query, llm_call)
            else:
                return await self.__similarity(llm_call, query)

    def execute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> Assistant:
        llm_call.model = llm_call.model if llm_call.model else self.executor.get_default_model()

        return asyncio.run(self.aexecute_llm_call(
            llm_call=llm_call,
            query=query,
            original_query=original_query,
            compression=compression,
        ))

    async def aexecute(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        model: Optional[str] = None,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: Dict[str, Any] = {},
    ) -> list[Message]:
        # [system_message] [user_message] pairs - deal with them in text content or file content
        messages = self.__parse_message_template(messages, template_args)

        # {{templates}} check to see if the messages have {{templates}} in them, and if so, replace with template_args
        messages = [self.__parse_template(m, template_args) for m in messages]

        llm_call = LLMCall(
            user_message=messages[-1],
            context_messages=messages[0:-1],
            executor=self.executor,
            model=model if model else self.executor.get_default_model(),
            temperature=temperature,
            max_prompt_len=self.executor.max_input_tokens(),
            completion_tokens_len=self.executor.max_output_tokens(),
            prompt_name='',
            stop_tokens=[],
            stream_handler=stream_handler
        )

        response: Assistant = await self.aexecute_llm_call(
            llm_call,
            query=messages[-1].get_str(),
            original_query='',
            compression=compression
        )
        return [response]

    async def aexecute_continuation(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        model: Optional[str] = None,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: dict[str, Any] = {},
        tools: list[Callable] = [],
        cookies: list[Dict[str, Any]] = [],
        locals_dict: dict[str, Any] = {},
    ) -> Tuple[list[Message], dict[str, Any]]:
        def parse_code_block_result(result: AstNode | list[AstNode] | str | list[Answer]) -> list[AstNode]:
            if isinstance(result, str):
                return [TextContent(result)]
            elif isinstance(result, Content):
                return [result]
            elif isinstance(result, list) and len(result) == 1:
                return [result[0]]
            elif isinstance(result, list) and all([isinstance(a, Answer) for a in result]):
                logging.debug('parse_code_block_result() called with list of Answers')
                return [TextContent(str(a.result())) for a in result]  # type: ignore
            elif isinstance(result, list) and len(result) > 1:
                logging.debug('parse_code_block_result() called with list of AstNodes')
                return cast(list[AstNode], result)
            elif isinstance(result, Assistant):
                return cast(list[AstNode], result.message)
            else:
                raise ValueError(f'Unknown content type: {type(result)}')

        from llmvm.server.python_runtime import PythonRuntime
        python_runtime = PythonRuntime(
            self,
            tools=tools,
            vector_search=self.vector_search,
            locals_dict=locals_dict,
        )

        model = model if model else self.executor.get_default_model()

        response: Assistant = Assistant(TextContent(''))

        # [system_message] [user_message] pairs - deal with them in text content or file content
        messages = self.__parse_message_template(messages, template_args)

        # {{templates}} check to see if the messages have {{templates}} in them, and if so, replace with template_args
        messages = [self.__parse_template(m, template_args) for m in messages]

        # {{functions}} template
        messages = [self.__parse_function(m, tools) for m in messages]

        # bootstrap the continuation execution
        completed = False
        results: list[Statement] = []
        old_model = python_runtime.controller.get_executor().get_default_model()
        if model: python_runtime.controller.get_executor().set_default_model(model)

        # inject the python_continuation_execution.prompt prompt
        functions = [Helpers.get_function_description_flat(f) for f in tools]

        system_message, tools_message = Helpers.prompts(
            prompt_name='python_continuation_execution.prompt',
            template={
                'functions': '\n'.join(functions),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        # anthropic prompt caching through the first three messages
        assistant_reply = Assistant(TextContent('Yes, I am ready.'), hidden=True)
        assistant_reply.prompt_cached = True
        system_message.hidden = True
        tools_message.hidden = True

        # inject the tools_message into the messages, and make sure it's first.
        tools_message.pinned = 0  # pinned as the first message # todo: thread this through the system
        messages_copy = []
        messages_copy.append(system_message)
        messages_copy.append(tools_message)
        messages_copy.append(assistant_reply)
        # strip prompt caching from all messages so we can only have it on the last message
        for message in messages:
            message.prompt_cached = False

        messages_copy.extend(copy.deepcopy(messages))

        # todo: hack
        browser_content_start_token = '<helpers_result>BrowserContent('

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
                messages_copy[-1] = Assistant(TextContent(last_message_str))

            # undo the last prompt cached flag, because we're moving the cache checkpoint to the end
            prompt_cache_counter = 0
            for message in reversed(messages_copy):
                if message.prompt_cached and message != assistant_reply and message != system_message and message != tools_message:
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
                completion_tokens_len=self.executor.max_output_tokens(),
                prompt_name='',
                stop_tokens=['</helpers>', '</complete>'],
                stream_handler=stream_handler
            )

            response: Assistant = await self.aexecute_llm_call(
                llm_call,
                query=messages[-1].get_str(),
                original_query='',
                compression=compression
            )

            # openai doesn't return the stop token, so we should check the last message for the <helpers> token
            if (
                '<helpers>' in response.get_str()
                and response.stop_token == ''
                and (
                    self.get_executor().name() == 'openai'
                    or self.get_executor().name() == 'gemini'
                )
            ):
                response.stop_token = '</helpers>'

            if (
                response.stop_reason == 'stop'
                and response.stop_token != '</helpers>'
                and (
                    self.get_executor().name() == 'openai'
                    or self.get_executor().name() == 'gemini'
                )
            ):
                response.stop_token = '</complete>'

            # Two Assistant messages in a row: we don't want two assistant messages in a row (which is what happens if you're asking
            # for a continuation), so we remove the last Assistant message and replace it with the Assistant response
            # we just got, plus the previous Assistant response.
            if isinstance(messages_copy[-1], Assistant):
                previous_assistant = messages_copy.pop()
                response.message = [TextContent(previous_assistant.get_str() + ' ' + response.get_str())]

            # add the stop token to the response
            if response.stop_token and response.stop_token == '</helpers>':
                response.message = [TextContent(response.get_str() + '\n' + response.stop_token)]

            # extract any code blocks the Assistant wants to run
            assistant_response_str = response.get_str().replace('Assistant:', '').strip()
            code_blocks: list[str] = PythonRuntime.get_code_blocks(assistant_response_str)
            code_blocks_remove = []

            # filter code_blocks we've already seen
            for code_block in code_blocks:
                for code_block_executed in code_blocks_executed:
                    if Helpers.compare_code_blocks(code_block, code_block_executed):
                        code_blocks_remove.append(code_block)
            code_blocks = [code_block for code_block in code_blocks if code_block not in code_blocks_remove]

            # run code blocks we haven't run before
            if code_blocks and not response.stop_token == '</complete>':
                # emit some debugging
                code_block = '\n'.join(code_blocks)
                write_client_stream(TextContent('</helpers>\n'))
                write_client_stream(TextContent('Executing code block locally.\n'))

                no_indent_debug(logging, '')
                no_indent_debug(logging, '** [bold yellow]Python Abstract Syntax Tree:[/bold yellow] **')
                # debug out AST
                lines = Helpers.split_on_newline(Helpers.escape_newlines_in_strings(code_block))  # code_block.split('\n')
                line_counter = 1
                for line in lines:
                    line = line.replace('[', '\\[')
                    no_indent_debug(logging, f'{line_counter:02}  {line}')
                    line_counter+=1
                no_indent_debug(logging, '')

                # debug output
                response_writer('llm_call', code_block)

                # make sure the code block is valid and syntactically correct Python
                try:
                    _ = ast.parse(Helpers.escape_newlines_in_strings(code_block))
                except SyntaxError as ex:
                    logging.debug('aexecute() SyntaxError trying to parse code block: {}'.format(ex))
                    code_block = python_runtime.compile_error(
                        python_code=code_block,
                        error=str(ex),
                    )
                    if code_block.strip().startswith('```'):
                        code_block_extracted = Helpers.extract_code_blocks(code_block.strip())
                        code_block = code_block_extracted[0] if code_block_extracted else code_block

                if cookies: locals_dict['cookies'] = cookies

                code_execution_result: Optional[list[AstNode]] = []
                hidden = False

                try:
                    python_runtime.answers = []
                    # here we're using the original messages list because messages() wouldn't work without
                    # the original. we append the response to this messages list later in the code.
                    locals_dict = python_runtime.run(
                        python_code=code_block,
                        original_query=messages[-1].get_str(),
                        messages=messages,
                        locals_dict=locals_dict
                    )
                    # Python was executed without exceptions, reset the exception counter
                    # and add any answer() results to the results list
                    exception_counter = 0
                    code_blocks_executed.append(code_block)
                    # results.extend(python_runtime.answers)
                except Exception as ex:
                    # we call the assistant again, and the string will often contain the original code block
                    # even though we got an exception, we want to make sure we specify that we've executed it
                    code_blocks_executed.append(code_block)
                    hidden = True
                    # update the code_execution_result to expose the exception for the next iteration
                    logging.debug('ExecutionController.aexecute_continuation() Exception executing code block: {}'.format(ex))
                    # code_execution_result = Helpers.extract_stacktrace_until(traceback.format_exc(), type(python_runtime))
                    # the python_runtime __compile_and_execute() method will add line numbers to the original code and exception
                    code_execution_result = parse_code_block_result(f'\n{str(ex)}')

                    exception_counter += 1
                    if exception_counter == self.exception_limit:
                        EXCEPTION_PROMPT = """We have reached our exception limit.
                        Running any of the previous code again won't work.
                        You have one more shot, try something very different. Feel free to just emit a natural language message instead of code.\n"""
                        code_execution_result = parse_code_block_result(f'{code_execution_result}\n\n{EXCEPTION_PROMPT}')
                        logging.debug('aexecute() Exception limit reached)')

                # code_execution_result will be empty if the code block executed without exceptions
                # and will include the exception if it threw
                if python_runtime.answers and not code_execution_result:
                    # code block successfully executed, grab the result from any answer() blocks and stuff them
                    # in the code_execution_result var
                    code_execution_result = parse_code_block_result(python_runtime.answers)
                elif not python_runtime.answers and code_execution_result:
                    # exception!
                    code_execution_result = parse_code_block_result(f'{code_execution_result}')
                elif not python_runtime.answers and not code_execution_result:
                    # sometimes dove doesn't generate an answer() block, so we'll have to get the last assignment of the code and use that.
                    last_assignment = python_runtime.get_last_assignment(code_block, locals_dict)
                    if last_assignment:
                        # todo: this forces a string, but we can deal with more than that these days
                        code_execution_result = parse_code_block_result(last_assignment[1])

                if not code_execution_result:
                    # no answer() block, or last assignment was None
                    code_execution_result = parse_code_block_result(f'No answer() block found in code block, or last assignment was None.')

                assert(isinstance(code_execution_result, list) and len(code_execution_result) > 0)

                # todo: this should be AstNode or TextNode or ...
                # but for now we're just making it a str()
                code_execution_result_str = '\n'.join([str(c) for c in code_execution_result])

                # we have a <helpers_result></helpers_result> block, push it to the cli client
                if len(code_execution_result_str) > 300:
                    # grab the first and last 150 characters
                    write_client_stream(TextContent(f'<helpers_result>{code_execution_result_str[:150]}\n\n ...excluded for brevity...\n\n{code_execution_result_str[-150:]}</helpers_result>\n\n'))
                else:
                    write_client_stream(TextContent(f'<helpers_result>{code_execution_result_str}</helpers_result>\n\n'))

                # todo: we're using a string here to embed the answer in the helpers_result
                # but I think we can probably have all sorts of stuff in here, including images.
                messages_copy.append(response)
                content_messages = []
                content_messages.append(TextContent(f'<helpers_result>'))
                if isinstance(code_execution_result, Content):
                    content_messages.append(code_execution_result)
                else:
                    content_messages.append(TextContent(code_execution_result_str))
                content_messages.append(TextContent('</helpers_result>'))
                completed_code_user_message = User(content_messages, hidden=hidden)
                # required to complete the continuation
                messages_copy.append(completed_code_user_message)
                # required to make messages() work
                python_runtime.messages_list.append(completed_code_user_message)

            # we have a stop token and there are no code blocks. Assistant is finished, so we can just append the response
            elif response.stop_token and response.stop_token == '</complete>':
                # oh gemini, why do you do this? returning </complete> as a string not as a stop token. sigh.
                if response.get_str().strip() != '' and response.get_str().strip() != '</complete>':
                    results.append(Answer(
                        result=response.get_str()
                    ))
                # empty assistant, maybe we got a </helpers_result> that was correct. this should be in the code_execution_result
                elif code_execution_result:
                    results.extend([answer for answer in cast(list, code_execution_result) if isinstance(answer, Answer)])

                completed = True

            # code_blocks was filtered out, and we have a stop token, so it's trying to repeat the code again
            elif not code_blocks and response.stop_token == '</helpers>':
                assistant_response_str = response.get_str()
                assistant_response_str += '\n\n' + "I've repeated the same code block. I should try something different and not repeat the same code again."
                messages_copy.append(Assistant(TextContent(assistant_response_str), hidden=True))
            else:
                # if there are no code blocks, we're done
                if response.get_str().strip() != '':
                    results.append(Answer(
                        result=response.get_str()
                    ))
                completed = True
                # results.extend(python_runtime.answers)

        if model: python_runtime.controller.get_executor().set_default_model(old_model)
        dedupped: list[Statement] = list(reversed(Helpers.remove_duplicates(results, lambda a: a.result())))

        if messages_copy[-1].role() != 'assistant':
            messages_copy.append(Assistant(TextContent('\n'.join([str(statement) for statement in dedupped]))))

        # remove hidden messages from the list
        messages_copy = [m for m in messages_copy if not m.hidden]
        # only return the extra messages beyond the messages list that was passed in

        return (messages_copy, locals_dict)
        # return list(reverseddedupped)), locals_dict