import ast
import asyncio
import copy
import json
import math
import random
import re
from importlib import resources
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, cast

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import (no_indent_debug, response_writer,
                                          role_debug, setup_logging)
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Answer, Assistant, AstNode, Content,
                                  Controller, Executor, FileContent,
                                  FunctionCall, FunctionCallMeta, ImageContent,
                                  LLMCall, MarkdownContent, Message,
                                  PandasMeta, PdfContent, Statement, System,
                                  TokenCompressionMethod, User, awaitable_none)
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class ExecutionController(Controller):
    def __init__(
        self,
        executor: Executor,
        agents: List[Callable],
        vector_search: VectorSearch,
        edit_hook: Optional[Callable[[str], str]] = None,
        continuation_passing_style: bool = False,
        exception_limit: int = 3
    ):
        super().__init__()

        self.executor = executor
        self.agents = agents
        self.vector_search = vector_search
        self.edit_hook = edit_hook
        self.continuation_passing_style = continuation_passing_style
        self.exception_limit = exception_limit

    async def __llm_call(
        self,
        llm_call: LLMCall,
    ) -> Assistant:
        # execute the call to check to see if the Answer satisfies the original query
        messages: List[Message] = copy.deepcopy(llm_call.context_messages)

        # don't append the user message if it's empty
        if llm_call.user_message.message.get_str().strip() != '':
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
            role_debug(logging, llm_call.prompt_name, 'User', str(llm_call.user_message.message))
            role_debug(logging, llm_call.prompt_name, 'Assistant', str(assistant.message))
        except Exception as ex:
            role_debug(logging, llm_call.prompt_name, 'User', str(llm_call.user_message.message))
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
        llm_call.user_message = User(Content(prompt['user_message']))

        return await self.__llm_call(
            llm_call
        )

    async def __similarity(
        self,
        llm_call: LLMCall,
        query: str,
    ) -> Assistant:

        tokens_per_message = (
            math.floor((llm_call.max_prompt_len - await self.executor.count_tokens([llm_call.user_message], model=llm_call.model)) / len(llm_call.context_messages))  # noqa E501
        )
        write_client_stream(f'Performing context window compression type: similarity vector search with tokens per message {tokens_per_message}.\n')  # noqa E501

        # for all messages, do a similarity search
        similarity_messages = []
        for i in range(len(llm_call.context_messages)):
            prev_message = llm_call.context_messages[i]

            similarity_chunks = await self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.count_tokens,
                content=prev_message.message.get_str(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=tokens_per_message - 16,
            )
            similarity_message: str = '\n\n'.join([content for content, _ in similarity_chunks])

            # check for the header of a statement_to_message. We probably need to keep this
            if 'Result:\n' in prev_message.message.get_str():
                similarity_message = 'Result:\n' + similarity_message

            similarity_messages.append(User(Content(similarity_message)))

        total_similarity_tokens = sum(
            [await self.executor.count_tokens(m.message.get_str(), model=llm_call.model) for m in similarity_messages]
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
        prompt_len = self.executor.count_tokens(llm_call.context_messages + [llm_call.user_message], model=llm_call.model)
        write_client_stream(f'Performing context window compression type: map/reduce with token length {prompt_len}.\n')

        # collapse the context messages into single message
        context_message = User(Content('\n\n'.join([m.message.get_str() for m in llm_call.context_messages])))
        chunk_results = []

        # iterate over the data.
        map_reduce_prompt_tokens = await self.executor.count_tokens(
            [User(Content(Helpers.load_resources_prompt('map_reduce_map.prompt')['user_message']))],
            model=llm_call.model,
        )

        chunk_size = llm_call.max_prompt_len - map_reduce_prompt_tokens - await self.executor.count_tokens([llm_call.user_message], model=llm_call.model) - 32  # noqa E501
        chunks = self.vector_search.chunk(
            content=context_message.message.get_str(),
            chunk_size=chunk_size,
            overlap=0
        )

        for chunk in chunks:
            chunk_assistant = await self.__llm_call_with_prompt(
                llm_call=LLMCall(
                    user_message=User(Content()),
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
            chunk_results.append(chunk_assistant.message.get_str())

        # perform the reduce
        map_results = '\n\n====\n\n' + '\n\n====\n\n'.join(chunk_results)

        assistant_result = await self.__llm_call_with_prompt(
            llm_call=LLMCall(
                user_message=User(Content()),
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
            math.floor((llm_call.max_prompt_len - await self.executor.count_tokens([llm_call.user_message], model=llm_call.model)) / len(llm_call.context_messages))  # noqa E501
        )
        tokens_per_message = tokens_per_message - header_text_token_len

        write_client_stream(f'Performing context window compression type: summary map/reduce with tokens per message {tokens_per_message}.\n')  # noqa E501

        llm_call_copy = llm_call.copy()

        if await llm_call.executor.count_tokens([llm_call.user_message], llm_call.model) > tokens_per_message:
            logging.debug('__summary_map_reduce() user message is longer than the summary window, will try to cut.')
            llm_call_copy.user_message = User(Content(llm_call.user_message.message.get_str()[0:tokens_per_message]))

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

            summary_str += f'{current_message.message.get_str()[0:tokens_per_message]}'
            summary_messages.append(User(Content(summary_str)))

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
        write_client_stream('Performing context window compression type: last-in-first-out.\n')
        lifo_messages = copy.deepcopy(llm_call.context_messages)

        prompt_context_messages = [llm_call.user_message]
        current_tokens = await self.executor.count_tokens(
            llm_call.user_message.message.get_str(),
            model=llm_call.model
        ) + llm_call.completion_tokens_len

        # reverse over the messages, last to first
        for i in range(len(lifo_messages) - 1, -1, -1):
            if (
                current_tokens + await self.executor.count_tokens(lifo_messages[i].message.get_str(), model=llm_call.model)
                < llm_call.max_prompt_len
            ):
                prompt_context_messages.append(lifo_messages[i])
                current_tokens += await self.executor.count_tokens(lifo_messages[i].message.get_str(), model=llm_call.model)
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
        context: List[Statement] | Statement | str | List[Content] | Content,
    ):
        messages = self.statement_to_message(context)
        return '\n\n'.join([message.message.get_str() for message in messages if isinstance(message.message, Content)])

    def statement_to_message(
        self,
        context: List[Statement] | Statement | str | List[Content] | Content,
    ) -> List[Message]:
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
            return [User(Content(result_prompt['user_message']))]

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
            return [User(Content(result_prompt['user_message']))]

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
            return [User(Content(result_prompt['user_message']))]

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
            return [User(Content(result_prompt['user_message']))]

        elif isinstance(context, PandasMeta):
            return [User(Content(context.df.to_csv()))]  # type: ignore

        elif isinstance(context, MarkdownContent):
            return ObjectTransformers.transform_markdown_content(context, self.executor)

        elif isinstance(context, PdfContent):
            return ObjectTransformers.transform_pdf_content(context, self.executor)

        elif isinstance(context, FunctionBindable):
            return [User(Content(context._result.result()))]  # type: ignore

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
                return [User(Content(result_prompt['user_message']))]

        logging.debug(f'statement_to_message() unusual type {type(context)}, context is: {str(context)}')
        return [User(Content(str(context)))]

    def get_executor(self) -> Executor:
        return self.executor

    async def aclassify_tool_or_direct(
        self,
        message: User,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = awaitable_none,
        model: Optional[str] = None,
    ) -> Dict[str, float]:
        model = model if model else self.executor.get_default_model()

        def parse_result(result: str) -> Dict[str, float]:
            if ',' in result:
                if result.startswith('Assistant: '):
                    result = result[len('Assistant: '):].strip()

                first = result.split(',')[0].strip()
                second = result.split(',')[1].strip()

                match = re.search(r"[-+]?[0-9]*\.?[0-9]+", second)
                if match:
                    second = match.group(0)

                try:
                    if first.startswith('tool') or first.startswith('"tool"'):
                        return {'tool': float(second)}
                    elif first.startswith('direct') or first.startswith('"direct"'):
                        return {'direct': float(second)}
                    else:
                        return {'tool': 1.0}
                except ValueError as ex:
                    return {'tool': 1.0}
            return {'tool': 1.0}

        # assess the type of task
        function_list = [Helpers.get_function_description_flat(f) for f in self.agents]
        # todo rip out the probability from here
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_name='query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': message.message.get_str(),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        assistant: Assistant = await self.executor.aexecute(
            messages=[
                System(Content(query_understanding['system_message'])),
                User(Content(query_understanding['user_message']))
            ],
            temperature=0.0,
            stream_handler=stream_handler,
            model=model,
        )
        if assistant.error or not parse_result(assistant.message.get_str()):
            return {'tool': 1.0}
        return parse_result(assistant.message.get_str())

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

        prompt_len = await self.executor.count_tokens(llm_call.context_messages + [llm_call.user_message], model=llm_call.model)
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
            write_client_stream(f'The message prompt length: {prompt_len} is bigger than the max prompt length: {max_prompt_len} for executor {llm_call.executor.name()}\n')  # noqa E501
            write_client_stream(f'Context window compression strategy: {compression.name}.\n')
            # check to see if we're simply lifo'ing the context messages (last in first out)
            context_message = User(Content('\n\n'.join([m.message.get_str() for m in llm_call.context_messages])))

            # see if we can do a similarity search or not.
            write_client_stream(
                'Determining context window compression approach of either similarity vector search, or full map/reduce.\n'
            )
            similarity_chunks = await self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.count_tokens,
                content=context_message.message.get_str(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=max_prompt_len - await self.executor.count_tokens([llm_call.user_message], model=model) - 16,  # noqa E501
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.15)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: List[str] = []
            for chunk, _ in similarity_chunks[:5]:
                assistant_similarity = await self.__llm_call_with_prompt(
                    llm_call=LLMCall(
                        user_message=User(Content()),
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

                decision_criteria.append(assistant_similarity.message.get_str())
                logging.debug('aexecute_llm_call() map_reduce_required, query_or_task: {}, response: {}'.format(
                    query,
                    assistant_similarity.message,
                ))
                if 'No' in assistant_similarity.message.get_str():
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

    async def abuild_runnable_code_ast(
        self,
        llm_call: LLMCall,
        files: List[str],
    ) -> Assistant:
        messages = llm_call.context_messages + [llm_call.user_message]

        logging.debug(f'abuild_runnable_code_ast() user_message = {llm_call.user_message.message.get_str()[0:25]}')
        logging.debug(f'abuild_runnable_code_ast() model = {llm_call.model}, executor = {llm_call.executor.name()}')

        tools_message = Helpers.prompt_message(
            prompt_name='python_code_insights.prompt',
            template={
                'user_input': messages[-1].message.get_str(),
                'files': '\n'.join(files),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        llm_response = await self.aexecute_llm_call(
            llm_call=LLMCall(
                user_message=tools_message,
                context_messages=llm_call.context_messages + [llm_call.user_message],
                executor=llm_call.executor,
                model=llm_call.model,
                temperature=llm_call.temperature,
                max_prompt_len=llm_call.max_prompt_len,
                completion_tokens_len=llm_call.completion_tokens_len,
                prompt_name='python_code_insights.prompt',
                stream_handler=llm_call.stream_handler,
            ),
            query='',
            original_query='',
            compression=TokenCompressionMethod.MAP_REDUCE,
        )
        return llm_response

    async def abuild_runnable_tools_ast(
        self,
        llm_call: LLMCall,
        agents: List[Callable],
    ) -> Assistant:
        logging.debug(f'abuild_runnable_tools_ast() user_message = {llm_call.user_message.message.get_str()[0:25]}')
        logging.debug(f'abuild_runnable_tools_ast() model = {llm_call.model}, executor = {llm_call.executor.name()}')

        functions = [Helpers.get_function_description_flat(f) for f in agents]

        tools_message = Helpers.prompt_message(
            prompt_name='python_tool_execution.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': llm_call.user_message.message.get_str(),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        llm_response = await self.aexecute_llm_call(
            llm_call=LLMCall(
                user_message=tools_message,
                context_messages=llm_call.context_messages,  # + [llm_call.user_message],
                executor=llm_call.executor,
                model=llm_call.model,
                temperature=llm_call.temperature,
                max_prompt_len=llm_call.max_prompt_len,
                completion_tokens_len=llm_call.completion_tokens_len,
                prompt_name='python_tool_execution.prompt',
                stream_handler=llm_call.stream_handler,
            ),
            query='',
            original_query='',
            compression=TokenCompressionMethod.SUMMARY,
        )
        return llm_response

    async def aexecute(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        model: Optional[str] = None,
        mode: str = 'auto',
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: Optional[Dict[str, Any]] = None,
        cookies: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Statement]:

        from llmvm.server.python_runtime import PythonRuntime
        python_runtime = PythonRuntime(self, agents=self.agents, vector_search=self.vector_search)

        model = model if model else self.executor.get_default_model()

        def find_answers(d: Dict[Any, Any]) -> List[Statement]:
            current_results = []
            for _, value in d.items():
                if isinstance(value, Answer):
                    current_results.append(cast(Answer, value))
                if isinstance(value, dict):
                    current_results.extend(find_answers(value))
            return current_results

        results: List[Statement] = []

        # assess the type of task
        last_message = Helpers.last(lambda m: isinstance(m, User), messages)
        if not last_message: return []

        skip_generation = False
        response: Assistant = Assistant(Content(''))

        code_message = Helpers.first(lambda m: PythonRuntime.get_code_blocks(m.message.get_str()), messages)

        if code_message:
            skip_generation = True
            mode = 'tool'
            response.message = code_message.message
            messages.remove(code_message)

        # either classify, or we're going direct
        if mode == 'auto':
            classification = await self.aclassify_tool_or_direct(
                last_message,
                stream_handler=stream_handler,
                model=model,
            )
        elif mode == 'tool':
            classification = {'tool': 1.0}
        elif mode == 'code':
            classification = {'code': 1.0}
        else:
            classification = {'direct': 1.0}

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification or 'code' in classification:
            if 'tool' in classification and not skip_generation:
                response = await self.abuild_runnable_tools_ast(
                    llm_call=LLMCall(
                        user_message=messages[-1],
                        context_messages=messages[0:-1],
                        executor=self.executor,
                        model=model,
                        temperature=temperature,
                        max_prompt_len=self.executor.max_input_tokens(),
                        completion_tokens_len=self.executor.max_output_tokens(),
                        prompt_name='',
                        stream_handler=stream_handler
                    ),
                    agents=self.agents,
                )
            elif 'code' in classification and not skip_generation:
                files = template_args['files'] if template_args and 'files' in template_args else []
                if files:
                    python_runtime.globals_dict['source_project'].set_files(files)
                response = await self.abuild_runnable_code_ast(
                    llm_call=LLMCall(
                        user_message=messages[-1],
                        context_messages=messages[0:-1],
                        executor=self.executor,
                        model=model,
                        temperature=temperature,
                        max_prompt_len=self.executor.max_input_tokens(),
                        completion_tokens_len=self.executor.max_output_tokens(),
                        prompt_name='',
                        stream_handler=stream_handler
                    ),
                    files=files,
                )

            assistant_response_str = response.message.get_str().replace('Assistant:', '').strip()

            code_blocks = PythonRuntime.get_code_blocks(assistant_response_str)
            # todo for now, just join them
            assistant_response_str = '\n'.join(code_blocks)

            no_indent_debug(logging, '')
            no_indent_debug(logging, '** [bold yellow]Python Abstract Syntax Tree:[/bold yellow] **')
            # debug out AST
            lines = assistant_response_str.split('\n')
            line_counter = 1
            for line in lines:
                line = line.replace('[', '\\[')
                no_indent_debug(logging, f'{line_counter:02}  {line}')
                line_counter+=1
            no_indent_debug(logging, '')

            # debug output
            response_writer('llm_call', assistant_response_str)

            if self.edit_hook:
                assistant_response_str = self.edit_hook(assistant_response_str)

                # check to see if there is natural language in there or not
                try:
                    _ = ast.parse(assistant_response_str)
                except SyntaxError as ex:
                    logging.debug('aexecute() SyntaxError: {}'.format(ex))
                    assistant_response_str = python_runtime.compile_error(
                        python_code=assistant_response_str,
                        error=str(ex),
                    )

            if not self.continuation_passing_style:
                old_model = python_runtime.controller.get_executor().get_default_model()

                if model:
                    python_runtime.controller.get_executor().set_default_model(model)

                locals_dict = {'cookies': cookies} if cookies else {}
                _ = python_runtime.run(
                    python_code=assistant_response_str,
                    original_query=messages[-1].message.get_str(),
                    messages=messages,
                    locals_dict=locals_dict
                )
                results.extend(python_runtime.answers)

                if model:
                    python_runtime.controller.get_executor().set_default_model(old_model)

                return results
            else:
                old_model = python_runtime.controller.get_executor().get_default_model()

                if model:
                    python_runtime.controller.get_executor().set_default_model(model)

                locals_dict = {'cookies': cookies} if cookies else {}

                _ = python_runtime.run_continuation_passing(
                    python_code=assistant_response_str,
                    original_query=messages[-1].message.get_str(),
                    messages=messages,
                    locals_dict=locals_dict
                )
                results.extend(python_runtime.answers)
                return results
        else:
            # classified or specified as 'direct'
            assistant_reply: Assistant = await self.aexecute_llm_call(
                llm_call=LLMCall(
                    user_message=messages[-1],
                    context_messages=messages[0:-1],
                    executor=self.executor,
                    model=model,
                    temperature=temperature,
                    max_prompt_len=self.executor.max_input_tokens(),
                    completion_tokens_len=self.executor.max_output_tokens(),
                    prompt_name='',
                    stream_handler=stream_handler,
                ),
                query=messages[-1].message.get_str(),
                original_query='',
                compression=compression
            )

            results.append(Answer(
                conversation=[assistant_reply],
                result=assistant_reply.message
            ))

        return results

    def __serialize_locals_dict(self, locals_dict: Dict[str, Any]) -> Dict[str, Any]:
        temp_dict = {}
        for key, value in locals_dict.items():
            if isinstance(key, str) and key.startswith('__'):
                continue
            elif isinstance(value, dict):
                temp_dict[key] = self.__serialize_locals_dict(value)
            elif isinstance(value, list):
                temp_dict[key] = [str(v) for v in value]
            # all primitive types are fine
            elif (
                isinstance(value, str)
                or isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, bool)
            ):
                temp_dict[key] = value
            # all the types in objects.py are fine too
            elif (
                isinstance(value, Content)
                or isinstance(value, AstNode)
                or isinstance(value, Message)
                or isinstance(value, Statement)
            ):
                temp_dict[key] = value
            else:
                # check to see if serializable
                try:
                    json.dumps(value)
                    temp_dict[key] = value
                except Exception as ex:
                    pass
        return temp_dict

    async def aexecute_continuation(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        model: Optional[str] = None,
        compression: TokenCompressionMethod = TokenCompressionMethod.AUTO,
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: Dict[str, Any] = {},
        agents: List[Callable] = [],
        cookies: List[Dict[str, Any]] = [],
        locals_dict: Dict[str, Any] = {},
    ) -> Tuple[List[Statement], Dict[str, Any]]:

        from llmvm.server.python_runtime import PythonRuntime
        python_runtime = PythonRuntime(
            self,
            agents=agents,
            vector_search=self.vector_search,
            locals_dict=locals_dict,
        )

        model = model if model else self.executor.get_default_model()

        response: Assistant = Assistant(Content())

        # a single code block is supported as a special case which we execute immediately
        if (
            type(messages[-1]) is Content
            and PythonRuntime.only_code_block(messages[-1].message.get_str())
        ):
            response.message = messages[-1].message
            messages.remove(messages[-1])
            raise NotImplementedError('Code block execution is not yet supported.')

        # check the [system_message] [user_message] pair, and parse those into messages
        # if they exist
        i = 0
        while i < len(messages):
            if (
                (type(messages[i]) is Content or type(messages[i]) is FileContent)
                and '[system_message]' in messages[i].message.get_str()
                and '[user_message]' in messages[i].message.get_str()
            ):
                system, user = Helpers.get_prompts(
                    messages[i].message.get_str(),
                    template_args,
                    self.get_executor().user_token(),
                    self.get_executor().assistant_token(),
                    self.get_executor().append_token(),
                )
                # inject system and user into messages
                messages[i] = system
                messages.insert(i + 1, user)
                i += 1
            i += 1

        # {{templates}}
        # check to see if the messages have {{templates}} in them, and if so, replace
        # with template_args.
        for i in range(len(messages)):
            if type(messages[i]) is Content:
                message_text = messages[i].message.get_str()
                for key, value in template_args.items():
                    key_replace = '{{' + key + '}}'
                    if key_replace in message_text:
                        messages[i].message = Content(message_text.replace(key_replace, value))

        # {{functions}}
        # deal with the {{functions}} special case
        for i in range(len(messages)):
            if type(messages[i]) is Content:
                message_text = messages[i].message.get_str()
                if '{{functions}}' in message_text:
                    messages[i].message = Content(
                        message_text.replace(
                            '{{functions}}', '\n'.join([Helpers.get_function_description_flat(f) for f in agents])
                        )
                    )

        # bootstrap the continuation execution
        completed = False
        results: List[Statement] = []
        old_model = python_runtime.controller.get_executor().get_default_model()
        if model: python_runtime.controller.get_executor().set_default_model(model)

        # inject the python_continuation_execution.prompt prompt
        functions = [Helpers.get_function_description_flat(f) for f in agents]

        system_message, tools_message = Helpers.prompts(
            prompt_name='python_continuation_execution.prompt',
            template={
                'functions': '\n'.join(functions),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        # anthropic prompt caching
        assistant_reply = Assistant(Content('Yes, I am ready.'))
        system_message.prompt_cached = True
        tools_message.prompt_cached = True
        assistant_reply.prompt_cached = True

        # inject the tools_message into the messages, and make sure it's first.
        tools_message.pinned = 0  # pinned as the first message # todo: thread this through the system
        messages_copy = []
        messages_copy.append(system_message)
        messages_copy.append(tools_message)
        messages_copy.append(assistant_reply)
        messages_copy.extend(copy.deepcopy(messages[0:-1]))
        messages_copy.append(messages[-1])

        # execute the continuation loop
        exception_counter = 0
        code_blocks_executed: List[str] = []
        while not completed and exception_counter <= self.exception_limit:
            llm_call = LLMCall(
                user_message=messages_copy[-1],
                context_messages=messages_copy[0:-1],
                executor=self.executor,
                model=model,
                temperature=temperature,
                max_prompt_len=self.executor.max_input_tokens(),
                completion_tokens_len=self.executor.max_output_tokens(),
                prompt_name='',
                stop_tokens=['</code>', '</complete>'],
                stream_handler=stream_handler
            )

            response: Assistant = await self.aexecute_llm_call(
                llm_call,
                query=messages[-1].message.get_str(),
                original_query='',
                compression=compression
            )

            # openai doesn't return the stop token, so we should check the last message for the <code> token
            if (
                '<code>' in response.message.get_str()
                and response.stop_token == ''
                and self.get_executor().name() == 'openai'
            ):
                response.stop_token = '</code>'

            if (
                response.stop_reason == 'stop'
                and response.stop_token != '</code>'
                and self.get_executor().name() == 'openai'
            ):
                response.stop_token = '</complete>'

            # we don't want two assistant messages in a row (which is what happens if you're asking
            # for a continuation), so we remove the last Assistant message and replace it with the Assistant response
            # we just got, plus the previous Assistant response.
            if isinstance(messages_copy[-1], Assistant):
                previous_assistant = messages_copy.pop()
                response.message = Content(previous_assistant.message.get_str() + ' ' + response.message.get_str())

            # add the stop token to the response
            if response.stop_token and response.stop_token == '</code>':
                response.message = Content(response.message.get_str() + response.stop_token)

            # extract any code blocks the Assistant wants to run
            assistant_response_str = response.message.get_str().replace('Assistant:', '').strip()
            code_blocks: List[str] = PythonRuntime.get_code_blocks(assistant_response_str)
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
                write_client_stream('</code>\n')
                write_client_stream('Executing code block locally.\n')

                no_indent_debug(logging, '')
                no_indent_debug(logging, '** [bold yellow]Python Abstract Syntax Tree:[/bold yellow] **')
                # debug out AST
                lines = code_block.split('\n')
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
                    _ = ast.parse(code_block)
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
                code_execution_result = ''

                try:
                    # todo: made this change for t2
                    python_runtime.answers = []
                    locals_dict = python_runtime.run(
                        python_code=code_block,
                        original_query=messages[-1].message.get_str(),
                        messages=messages,
                        locals_dict=locals_dict
                    )
                    # Python was executed without exceptions, reset the exception counter
                    # and add any answer() results to the results list
                    exception_counter = 0
                    code_blocks_executed.append(code_block)
                    # results.extend(python_runtime.answers)
                except Exception as ex:
                    # update the code_execution_result to expose the exception for the next iteration
                    logging.debug('ExecutionController.aexecute_continuation() Exception executing code block: {}'.format(ex))
                    code_execution_result = Helpers.extract_stacktrace_until(traceback.format_exc(), type(python_runtime))
                    exception_counter += 1
                    if exception_counter == self.exception_limit:
                        EXCEPTION_PROMPT = """We have reached our exception limit.
                        Running any of the previous code again won't work.
                        You have one more shot, try something very different. Feel free to just emit a natural language message instead of code.\n"""
                        code_execution_result = f'{code_execution_result}\n\n{EXCEPTION_PROMPT}'
                        logging.debug('aexecute() Exception limit reached)')

                # code_execution_result will be empty if the code block executed without exceptions
                # and will include the exception if it threw
                if python_runtime.answers and not code_execution_result:
                    # code block successfully executed, grab the result from any answer() blocks and stuff them
                    # in the code_execution_result var
                    code_execution_result = '\n'.join([str(a.result()) for a in python_runtime.answers])
                elif not python_runtime.answers and code_execution_result:
                    # exception!
                    code_execution_result = f'{code_execution_result}'
                elif not python_runtime.answers and not code_execution_result:
                    # sometimes dove doesn't generate an answer() block, so we'll have to get the last assignment of the code and use that.
                    last_assignment = python_runtime.get_last_assignment(code_block, locals_dict)
                    if last_assignment:
                        code_execution_result = f'{str(last_assignment[1])}'

                assert(code_execution_result)

                # we have a <code_result></code_result> block, push it to the user
                if len(code_execution_result) > 300:
                    # grab the first and last 150 characters
                    write_client_stream(f'<code_result>{code_execution_result[:150]}\n\n ...excluded for brevity...\n\n{code_execution_result[-150:]}</code_result>\n\n')
                else:
                    write_client_stream(f'<code_result>{code_execution_result}</code_result>\n\n')

                # update the code_execution_result to include both the <code> and <code_result> blocks
                code_execution_result = f'<code>{code_block}</code>\n<code_result>{code_execution_result}</code_result>\n'

                # assistant_response_str will have the original code block <code></code> in it, so we need to replace it with the answers
                # use regex to replace the code block with the original code + answers
                try:
                    assistant_response_str = re.sub(r'<code>.*?</code>', code_execution_result, assistant_response_str, flags=re.DOTALL)
                except Exception as ex:
                    logging.debug(f'ExecutionController.aexecute_continuation() Error in regex replacing code block with code execution result: {ex}')
                    assistant_response_str = f'{assistant_response_str}\n\n{code_execution_result}'
                messages_copy.append(Assistant(Content(assistant_response_str)))

            # we have a stop token and there are no code blocks. Assistant is finished, so we can just append the response
            elif response.stop_token and response.stop_token == '</complete>':
                if response.message.get_str().strip() != '':
                    results.append(Answer(
                        conversation=[response],
                        result=response.message.get_str()
                    ))
                completed = True

            # code_blocks was filtered out, and we have a stop token, so it's trying to repeat the code again
            elif not code_blocks and response.stop_token == '</code>':
                assistant_response_str = response.message.get_str()
                assistant_response_str += '\n\n' + "I've repeated the same code block. I should try something different."
                messages_copy.append(Assistant(Content(assistant_response_str)))
            else:
                # if there are no code blocks, we're done
                if response.message.get_str().strip() != '':
                    results.append(Answer(
                        conversation=[response],
                        result=response.message.get_str()
                    ))
                completed = True
                # results.extend(python_runtime.answers)

        if model: python_runtime.controller.get_executor().set_default_model(old_model)
        dedupped = Helpers.remove_duplicates(results, lambda a: a.result())
        return list(reversed(dedupped)), self.__serialize_locals_dict(locals_dict)
