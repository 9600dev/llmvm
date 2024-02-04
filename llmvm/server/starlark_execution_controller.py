import ast
import asyncio
import copy
import math
import random
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import (no_indent_debug, response_writer,
                                          role_debug, setup_logging)
from llmvm.common.objects import (Answer, Assistant, AstNode, Content,
                                  Controller, Executor, FileContent, LLMCall,
                                  Message, PdfContent, Statement, System,
                                  TokenCompressionMethod, User, awaitable_none)
from llmvm.server.starlark_runtime import StarlarkRuntime
from llmvm.server.tools.pdf import PdfHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class StarlarkExecutionController(Controller):
    def __init__(
        self,
        executor: Executor,
        agents: List[Callable],
        vector_search: VectorSearch,
        edit_hook: Optional[Callable[[str], str]] = None,
        continuation_passing_style: bool = False,
    ):
        super().__init__()

        self.executor = executor
        self.agents = agents
        self.vector_search = vector_search
        self.edit_hook = edit_hook
        self.starlark_runtime = StarlarkRuntime(self, agents=self.agents, vector_search=self.vector_search)
        self.continuation_passing_style = continuation_passing_style

    async def __llm_call(
        self,
        llm_call: LLMCall,
    ) -> Assistant:
        # execute the call to check to see if the Answer satisfies the original query
        messages: List[Message] = copy.deepcopy(llm_call.context_messages)

        # don't append the user message if it's empty
        if llm_call.user_message.message.get_content().strip() != '':
            messages.append(llm_call.user_message)

        try:
            assistant: Assistant = await llm_call.executor.aexecute(
                messages,
                max_completion_tokens=llm_call.completion_tokens_len,
                temperature=llm_call.temperature,
                stream_handler=llm_call.stream_handler,
                model=llm_call.model,
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
        function_list = [Helpers.get_function_description_flat_extra(f) for f in self.agents]
        # todo rip out the probability from here
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_name='query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': message.message.get_content(),
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
        if assistant.error or not parse_result(assistant.message.get_content()):
            return {'tool': 1.0}
        return parse_result(assistant.message.get_content())

    async def __similarity(
        self,
        llm_call: LLMCall,
        query: str,
    ) -> Assistant:

        tokens_per_message = (
            math.floor((llm_call.max_prompt_len - self.executor.calculate_tokens([llm_call.user_message], model=llm_call.model)) / len(llm_call.context_messages))  # noqa E501
        )
        write_client_stream(f'Performing context window compression type: similarity vector search with tokens per message {tokens_per_message}.\n')  # noqa E501

        # for all messages, do a similarity search
        similarity_messages = []
        for i in range(len(llm_call.context_messages)):
            prev_message = llm_call.context_messages[i]

            similarity_chunks = self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.calculate_tokens,
                content=prev_message.message.get_content(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=tokens_per_message - 32,
            )
            similarity_message: str = '\n\n'.join([content for content, _ in similarity_chunks])

            # check for the header of a statement_to_message. We probably need to keep this
            # todo: hack
            if 'Result:\n' in prev_message.message.get_content():
                similarity_message = prev_message.message.get_content()[0:prev_message.message.get_content().index('Result:\n')] + similarity_message  # noqa E501

            similarity_messages.append(User(Content(similarity_message)))

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
        prompt_len = self.executor.calculate_tokens(llm_call.context_messages + [llm_call.user_message], model=llm_call.model)
        write_client_stream(f'Performing context window compression type: map/reduce with token length {prompt_len}.\n')

        # collapse the context messages into single message
        context_message = User(Content('\n\n'.join([m.message.get_content() for m in llm_call.context_messages])))
        chunk_results = []

        # iterate over the data.
        map_reduce_prompt_tokens = self.executor.calculate_tokens(
            [User(Content(open('map_reduce_map.prompt', 'r').read()))],
            model=llm_call.model,
        )

        chunk_size = llm_call.max_prompt_len - map_reduce_prompt_tokens - self.executor.calculate_tokens([llm_call.user_message], model=llm_call.model) - 32  # noqa E501
        chunks = self.vector_search.chunk(
            content=context_message.message.get_content(),
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
            chunk_results.append(chunk_assistant.message.get_content())

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
            math.floor((llm_call.max_prompt_len - self.executor.calculate_tokens([llm_call.user_message], model=llm_call.model)) / len(llm_call.context_messages))  # noqa E501
        )
        tokens_per_message = tokens_per_message - header_text_token_len

        write_client_stream(f'Performing context window compression type: summary map/reduce with tokens per message {tokens_per_message}.\n')  # noqa E501

        llm_call_copy = llm_call.copy()

        if llm_call.executor.calculate_tokens([llm_call.user_message], llm_call.model) > tokens_per_message:
            logging.debug('__summary_map_reduce() user message is longer than the summary window, will try to cut.')
            llm_call_copy.user_message = User(Content(llm_call.user_message.message.get_content()[0:tokens_per_message]))

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

            summary_str += f'{current_message.message.get_content()[0:tokens_per_message]}'
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
        current_tokens = self.executor.calculate_tokens(
            llm_call.user_message.message.get_content(),
            model=llm_call.model
        ) + llm_call.completion_tokens_len

        # reverse over the messages, last to first
        for i in range(len(lifo_messages) - 1, -1, -1):
            if (
                current_tokens + self.executor.calculate_tokens(lifo_messages[i].message.get_content(), model=llm_call.model)
                < llm_call.max_prompt_len
            ):
                prompt_context_messages.append(lifo_messages[i])
                current_tokens += self.executor.calculate_tokens(lifo_messages[i].message.get_content(), model=llm_call.model)
            else:
                break

        new_call = llm_call.copy()
        new_call.context_messages = prompt_context_messages[::-1]

        assistant_result = await self.__llm_call(
            llm_call=new_call
        )
        return assistant_result

    async def aexecute_llm_call(
        self,
        llm_call: LLMCall,
        query: str,
        original_query: str,
        token_compression_method: TokenCompressionMethod = TokenCompressionMethod.AUTO,
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
        # todo: this should probably use ContentDownloader.parse_pdf (so that context messages)
        # are included in the assessment of if the Pdf has been converted or not
        for c_message in llm_call.context_messages:
            if isinstance(c_message.message, PdfContent) and not c_message.message.is_text():
                text_result = PdfHelpers.parse_pdf(c_message.message.url)
                c_message.message.sequence = text_result

        prompt_len = self.executor.calculate_tokens(llm_call.context_messages + [llm_call.user_message], model=llm_call.model)
        max_prompt_len = self.executor.max_prompt_tokens(completion_token_len=llm_call.completion_tokens_len, model=model)

        # I have either a message, or a list of messages. They might need to be map/reduced.
        # todo: we usually have a prepended message of context to help the LLM figure out
        # what to do with the message at a later stage. This is getting removed right now.
        if prompt_len <= max_prompt_len:
            # straight call
            return await self.__llm_call(llm_call=llm_call)
        elif prompt_len > max_prompt_len and token_compression_method == TokenCompressionMethod.LIFO:
            return await self.__lifo(llm_call)
        elif prompt_len > max_prompt_len and token_compression_method == TokenCompressionMethod.SUMMARY:
            return await self.__summary_map_reduce(llm_call)
        elif prompt_len > max_prompt_len and token_compression_method == TokenCompressionMethod.MAP_REDUCE:
            return await self.__map_reduce(query, original_query, llm_call)
        elif prompt_len > max_prompt_len and token_compression_method == TokenCompressionMethod.SIMILARITY:
            return await self.__similarity(llm_call, query)
        else:
            # let's figure out what method to use
            write_client_stream(f'The prompt length: {prompt_len} is bigger than the max token count: {max_prompt_len} for executor {llm_call.executor.name()}.\n')  # noqa E501
            write_client_stream(f'Token Compression: {token_compression_method.name}.\n')
            # check to see if we're simply lifo'ing the context messages (last in first out)
            context_message = User(Content('\n\n'.join([m.message.get_content() for m in llm_call.context_messages])))

            # see if we can do a similarity search or not.
            write_client_stream('Determining map/reduce approach of either similarity vectorsearch or full map/reduce.\n')
            similarity_chunks = self.vector_search.chunk_and_rank(
                query=query,
                token_calculator=self.executor.calculate_tokens,
                content=context_message.message.get_content(),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=max_prompt_len - self.executor.calculate_tokens([llm_call.user_message], model=model) - 32,  # noqa E501
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

                decision_criteria.append(assistant_similarity.message.get_content())
                logging.debug('aexecute_llm_call() map_reduce_required, query_or_task: {}, response: {}'.format(
                    query,
                    assistant_similarity.message,
                ))
                if 'No' in assistant_similarity.message.get_content():
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
        token_compression_method: TokenCompressionMethod = TokenCompressionMethod.AUTO,
    ) -> Assistant:
        llm_call.model = llm_call.model if llm_call.model else self.executor.get_default_model()

        return asyncio.run(self.aexecute_llm_call(
            llm_call=llm_call,
            query=query,
            original_query=original_query,
            token_compression_method=token_compression_method,
        ))

    async def abuild_runnable_code_ast(
        self,
        llm_call: LLMCall,
        files: List[str],
    ) -> Assistant:
        messages = llm_call.context_messages + [llm_call.user_message]

        logging.debug(f'abuild_runnable_code_ast() user_message = {llm_call.user_message.message.get_content()[0:25]}')
        logging.debug(f'abuild_runnable_code_ast() model = {llm_call.model}, executor = {llm_call.executor.name()}')

        tools_message = Helpers.prompt_message(
            prompt_name='starlark_code_insights.prompt',
            template={
                'user_input': messages[-1].message.get_content(),
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
                prompt_name='starlark_code_insights.prompt',
                stream_handler=llm_call.stream_handler,
            ),
            query='',
            original_query='',
            token_compression_method=TokenCompressionMethod.MAP_REDUCE,
        )
        return llm_response

    async def abuild_runnable_tools_ast(
        self,
        llm_call: LLMCall,
        agents: List[Callable],
    ) -> Assistant:
        logging.debug(f'abuild_runnable_tools_ast() user_message = {llm_call.user_message.message.get_content()[0:25]}')
        logging.debug(f'abuild_runnable_tools_ast() model = {llm_call.model}, executor = {llm_call.executor.name()}')

        functions = [Helpers.get_function_description_flat_extra(f) for f in agents]

        tools_message = Helpers.prompt_message(
            prompt_name='starlark_tool_execution.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': llm_call.user_message.message.get_content(),
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
                prompt_name='starlark_tool_execution.prompt',
                stream_handler=llm_call.stream_handler,
            ),
            query='',
            original_query='',
            token_compression_method=TokenCompressionMethod.SUMMARY,
        )
        return llm_response

    async def aexecute(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        model: Optional[str] = None,
        mode: str = 'auto',
        stream_handler: Callable[[AstNode], Awaitable[None]] = awaitable_none,
        template_args: Optional[Dict[str, Any]] = None,
    ) -> List[Statement]:
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
            response: Assistant = Assistant(Content(''))

            if 'tool' in classification:
                response = await self.abuild_runnable_tools_ast(
                    llm_call=LLMCall(
                        user_message=messages[-1],
                        context_messages=messages[0:-1],
                        executor=self.executor,
                        model=model,
                        temperature=temperature,
                        max_prompt_len=self.executor.max_prompt_tokens(),
                        completion_tokens_len=self.executor.max_completion_tokens(),
                        prompt_name='',
                        stream_handler=stream_handler
                    ),
                    agents=self.agents,
                )
            elif 'code' in classification:
                files = template_args['files'] if template_args and 'files' in template_args else []
                if files:
                    self.starlark_runtime.globals_dict['source_project'].set_files(files)
                response = await self.abuild_runnable_code_ast(
                    llm_call=LLMCall(
                        user_message=messages[-1],
                        context_messages=messages[0:-1],
                        executor=self.executor,
                        model=model,
                        temperature=temperature,
                        max_prompt_len=self.executor.max_prompt_tokens(),
                        completion_tokens_len=self.executor.max_completion_tokens(),
                        prompt_name='',
                        stream_handler=stream_handler
                    ),
                    files=files,
                )
            assistant_response_str = response.message.get_content().replace('Assistant:', '').strip()

            # anthropic can often embed the code in ```python blocks
            if '```python' in assistant_response_str:
                match = re.search(r'```python\n(.*?)```', assistant_response_str, re.DOTALL)
                if match:
                    assistant_response_str = match.group(1)
            # openai can often embed the code in ```starlark blocks
            elif '```starlark' in assistant_response_str:
                match = re.search(r'```starlark\n(.*?)```', assistant_response_str, re.DOTALL)
                if match:
                    assistant_response_str = match.group(1)
            # mistral likes to embed the code in ``` blocks
            elif '```' in assistant_response_str:
                match = re.search(r'```(.*?)```', assistant_response_str, re.DOTALL)
                if match:
                    assistant_response_str = match.group(1)

            no_indent_debug(logging, '')
            no_indent_debug(logging, '** [bold yellow]Starlark Abstract Syntax Tree:[/bold yellow] **')
            # debug out AST
            lines = assistant_response_str.split('\n')
            for line in lines:
                no_indent_debug(logging, '  {}'.format(line.replace("[", "\\[")))
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
                    assistant_response_str = self.starlark_runtime.compile_error(
                        starlark_code=assistant_response_str,
                        error=str(ex),
                    )

            if not self.continuation_passing_style:
                old_model = self.starlark_runtime.controller.get_executor().get_default_model()

                if model:
                    self.starlark_runtime.controller.get_executor().set_default_model(model)

                _ = self.starlark_runtime.run(
                    starlark_code=assistant_response_str,
                    original_query=messages[-1].message.get_content(),
                    messages=messages,
                )
                results.extend(self.starlark_runtime.answers)

                if model:
                    self.starlark_runtime.controller.get_executor().set_default_model(old_model)

                return results
            else:
                _ = self.starlark_runtime.run_continuation_passing(
                    starlark_code=assistant_response_str,
                    original_query=messages[-1].message.get_content(),
                    messages=messages,
                )
                results.extend(self.starlark_runtime.answers)
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
                    max_prompt_len=self.executor.max_prompt_tokens(),
                    completion_tokens_len=self.executor.max_completion_tokens(),
                    prompt_name='',
                    stream_handler=stream_handler,
                ),
                query=messages[-1].message.get_content(),
                original_query='',
                token_compression_method=TokenCompressionMethod.AUTO,
            )

            results.append(Answer(
                conversation=[assistant_reply],
                result=assistant_reply.message
            ))

        return results
