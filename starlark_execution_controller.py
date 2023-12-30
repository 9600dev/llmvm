import ast
import asyncio
import copy
import math
import random
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

from helpers.helpers import Helpers
from helpers.logging_helpers import (no_indent_debug, response_writer,
                                     role_debug, setup_logging)
from objects import (Answer, Assistant, AstNode, Content, Controller, Executor,
                     Message, Statement, System, User, awaitable_none)
from starlark_runtime import StarlarkRuntime
from vector_search import VectorSearch

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
            prompt_filename='prompts/starlark/query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': str(message.message),
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
        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    async def aexecute_llm_call(
        self,
        message: Message,
        context_messages: List[Message],
        query: str,
        original_query: str,
        prompt_filename: Optional[str] = None,
        completion_tokens: int = 2048,
        temperature: float = 0.0,
        lifo: bool = False,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = awaitable_none,
        model: Optional[str] = None,
    ) -> Assistant:
        '''
        Internal function to execute an LLM call with prompt template and context messages.
        Deals with chunking, map/reduce, and other logic if the message/context messages
        are too long for the context window.
        '''
        model = model if model else self.executor.get_default_model()

        async def __llm_call(
            user_message: Message,
            context_messages: List[Message],
            executor: Executor,
            prompt_filename: Optional[str] = None,
        ) -> Assistant:
            if not prompt_filename:
                prompt_filename = ''
            # execute the call to check to see if the Answer satisfies the original query
            messages: List[Message] = copy.deepcopy(context_messages)

            # don't append the user message if it's empty
            if str(user_message.message).strip() != '':
                messages.append(user_message)

            try:
                assistant: Assistant = await executor.aexecute(
                    messages,
                    max_completion_tokens=completion_tokens,
                    temperature=temperature,
                    stream_handler=stream_handler,
                    model=model,
                )
                role_debug(logging, prompt_filename, 'User', str(user_message.message))
                role_debug(logging, prompt_filename, 'Assistant', str(assistant.message))
            except Exception as ex:
                role_debug(logging, prompt_filename, 'User', str(user_message.message))
                raise ex
            response_writer(prompt_filename, assistant)
            return assistant

        async def __llm_call_prompt(
            prompt_filename: str,
            context_messages: List[Message],
            executor: Executor,
            template: Dict[str, Any],
        ) -> Assistant:
            prompt = Helpers.load_and_populate_prompt(
                prompt_filename=prompt_filename,
                template=template,
                user_token=self.get_executor().user_token(),
                assistant_token=self.get_executor().assistant_token(),
                append_token=self.get_executor().append_token(),
            )
            return await __llm_call(
                User(Content(prompt['user_message'])),
                context_messages,
                executor,
                prompt_filename=prompt_filename,
            )

        """
        Executes an LLM call on a prompt_message with a context of messages.
        Performs either a chunk_and_rank, or a map/reduce depending on the
        context relavence to the prompt_message.
        """
        assistant_result: Assistant

        # I have either a message, or a list of messages. They might need to be map/reduced.
        # todo: we usually have a prepended message of context to help the LLM figure out
        # what to do with the message at a later stage. This is getting removed right now.
        if (
            self.executor.calculate_tokens(context_messages + [message], model=model)
            > self.executor.max_prompt_tokens(completion_token_count=completion_tokens, model=model)
        ):
            # check to see if we're simply lifo'ing the context messages (last in first out)
            if lifo:
                lifo_messages = copy.deepcopy(context_messages)
                prompt_context_messages = [message]
                current_tokens = self.executor.calculate_tokens(str(message.message), model=model) + completion_tokens

                # reverse over the messages, last to first
                for i in range(len(lifo_messages) - 1, -1, -1):
                    if (
                        current_tokens + self.executor.calculate_tokens(str(lifo_messages[i].message), model=model)
                        < self.executor.max_prompt_tokens(completion_token_count=completion_tokens, model=model)
                    ):
                        prompt_context_messages.append(lifo_messages[i])
                        current_tokens += self.executor.calculate_tokens(str(lifo_messages[i].message), model=model)
                    else:
                        break

                assistant_result = await __llm_call(
                    user_message=cast(User, message),
                    context_messages=prompt_context_messages[::-1],  # reversed, because of above
                    executor=self.executor,
                    prompt_filename=prompt_filename,
                )
                return assistant_result

            # not lifo
            context_message = User(Content('\n\n'.join([str(m.message) for m in context_messages])))

            # see if we can do a similarity search or not.
            similarity_chunks = self.vector_search.chunk_and_rank(
                query=query,
                content=str(context_message.message),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=self.executor.max_prompt_tokens(completion_token_count=completion_tokens, model=model) - self.executor.calculate_tokens([message], model=model) - 32,  # noqa E501
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.15)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: List[str] = []
            for chunk, _ in similarity_chunks[:5]:
                assistant_similarity = await __llm_call_prompt(
                    prompt_filename='prompts/document_chunk.prompt',
                    context_messages=[],
                    executor=self.executor,
                    template={
                        'query': str(query),
                        'document_chunk': chunk,
                    },
                )

                decision_criteria.append(str(assistant_similarity.message))
                logging.debug('aexecute_llm_call() map_reduce_required, query_or_task: {}, response: {}'.format(
                    query,
                    assistant_similarity.message,
                ))
                if 'No' in str(assistant_similarity.message):
                    # we can break early, as the 'map_reduced_required' flag will not be set below
                    break

            map_reduce_required = all(['Yes' in d for d in decision_criteria])

            # either similarity search, or map reduce required
            # here, we're not doing a map_reduce, we're simply populating the context window
            # with the highest ranked chunks from each message.
            if not map_reduce_required:
                tokens_per_message = (
                    math.floor((self.executor.max_prompt_tokens(completion_token_count=completion_tokens, model=model) - self.executor.calculate_tokens([message], model=model))  # noqa E501
                               / len(context_messages))
                )

                # for all messages, do a similarity search
                similarity_messages = []
                for i in range(len(context_messages)):
                    prev_message = context_messages[i]

                    similarity_chunks = self.vector_search.chunk_and_rank(
                        query=query,
                        content=str(prev_message),
                        chunk_token_count=256,
                        chunk_overlap=0,
                        max_tokens=tokens_per_message - 32,
                    )
                    similarity_message: str = '\n\n'.join([content for content, _ in similarity_chunks])

                    # check for the header of a statement_to_message. We probably need to keep this
                    # todo: hack
                    if 'Result:\n' in str(prev_message):
                        similarity_message = str(prev_message)[0:str(prev_message).index('Result:\n')] + similarity_message

                    similarity_messages.append(User(Content(similarity_message)))

                assistant_result = await __llm_call(
                    user_message=message,
                    context_messages=similarity_messages,
                    executor=self.executor,
                    prompt_filename=prompt_filename,
                )

            # do the map reduce instead of similarity
            else:
                # collapse the message
                context_message = User(Content('\n\n'.join([str(m.message) for m in context_messages])))
                chunk_results = []

                # iterate over the data.
                map_reduce_prompt_tokens = self.executor.calculate_tokens(
                    [User(Content(open('prompts/map_reduce_map.prompt', 'r').read()))],
                    model=model,
                )
                chunk_size = self.executor.max_prompt_tokens(completion_token_count=completion_tokens, model=model) - map_reduce_prompt_tokens - self.executor.calculate_tokens([message], model=model) - 32  # noqa E501

                chunks = self.vector_search.chunk(
                    content=str(context_message.message),
                    chunk_size=chunk_size,
                    overlap=0
                )

                for chunk in chunks:
                    chunk_assistant = await __llm_call_prompt(
                        prompt_filename='prompts/map_reduce_map.prompt',
                        context_messages=[],
                        executor=self.executor,
                        template={
                            'original_query': original_query,
                            'query': query,
                            'data': chunk,
                        },
                    )
                    chunk_results.append(str(chunk_assistant.message))

                # perform the reduce
                map_results = '\n\n====\n\n' + '\n\n====\n\n'.join(chunk_results)

                assistant_result = await __llm_call_prompt(
                    prompt_filename='prompts/map_reduce_reduce.prompt',
                    context_messages=[],
                    executor=self.executor,
                    template={
                        'original_query': original_query,
                        'query': query,
                        'map_results': map_results
                    },
                )
        else:
            assistant_result = await __llm_call(
                user_message=cast(User, message),
                context_messages=context_messages,
                executor=self.executor,
                prompt_filename=prompt_filename,
            )
        return assistant_result

    def execute_llm_call(
        self,
        message: Message,
        context_messages: List[Message],
        query: str,
        original_query: str,
        prompt_filename: Optional[str] = None,
        completion_tokens: int = 2048,
        temperature: float = 0.0,
        lifo: bool = False,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = awaitable_none,
        model: Optional[str] = None,
    ) -> Assistant:
        model = model if model else self.executor.get_default_model()

        return asyncio.run(self.aexecute_llm_call(
            message=message,
            context_messages=context_messages,
            query=query,
            original_query=original_query,
            prompt_filename=prompt_filename,
            completion_tokens=completion_tokens,
            temperature=temperature,
            model=model,
            lifo=lifo,
            stream_handler=stream_handler,
        ))

    async def abuild_runnable_code_ast(
        self,
        messages: List[Message],
        files: List[str],
        temperature: float = 0.0,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = awaitable_none,
        model: Optional[str] = None,
    ) -> Assistant:
        logging.debug('abuild_runnable_code_ast() messages[-1] = {}'.format(str(messages[-1])[0:25]))
        model = model if model else self.executor.get_default_model()
        logging.debug('abuild_runnable_code_ast() model = {}, executor = {}'.format(model, self.executor.name()))

        tools_message = Helpers.load_and_populate_message(
            prompt_filename='prompts/starlark/starlark_code_insights.prompt',
            template={
                'user_input': str(messages[-1].message),
                'files': '\n'.join(files),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        llm_response = await self.aexecute_llm_call(
            message=tools_message,
            context_messages=messages[0:-1],
            query='',
            original_query='',
            prompt_filename='prompts/starlark/starlark_code_insights.prompt',
            completion_tokens=4096,
            temperature=temperature,
            lifo=False,
            stream_handler=stream_handler,
            model=model,
        )
        return llm_response

    async def abuild_runnable_tools_ast(
        self,
        messages: List[Message],
        agents: List[Callable],
        temperature: float = 0.0,
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = awaitable_none,
        model: Optional[str] = None,
    ) -> Assistant:
        logging.debug('abuild_runnable_tools_ast() messages[-1] = {}'.format(str(messages[-1])[0:25]))
        model = model if model else self.executor.get_default_model()
        logging.debug('abuild_runnable_tools_ast() model = {}, executor = {}'.format(model, self.executor.name()))

        functions = [Helpers.get_function_description_flat_extra(f) for f in agents]

        tools_message = Helpers.load_and_populate_message(
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': str(messages[-1].message),
            },
            user_token=self.get_executor().user_token(),
            assistant_token=self.get_executor().assistant_token(),
            append_token=self.get_executor().append_token(),
        )

        llm_response = await self.aexecute_llm_call(
            message=tools_message,
            context_messages=messages[0:-1],
            query='',
            original_query='',
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
            completion_tokens=2048,
            temperature=temperature,
            lifo=False,
            stream_handler=stream_handler,
            model=model,
        )

        return llm_response

    async def aexecute(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        mode: str = 'auto',
        stream_handler: Optional[Callable[[AstNode], Awaitable[None]]] = None,
        model: Optional[str] = None,
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
                    messages=messages,
                    agents=self.agents,
                    temperature=temperature,
                    stream_handler=stream_handler,
                    model=model,
                )
            elif 'code' in classification:
                files = template_args['files'] if template_args and 'files' in template_args else []
                if files:
                    self.starlark_runtime.globals_dict['source_project'].set_files(files)
                response = await self.abuild_runnable_code_ast(
                    messages=messages,
                    files=files,
                    temperature=temperature,
                    stream_handler=stream_handler,
                    model=model,
                )

            assistant_response = str(response.message).replace('Assistant:', '').strip()

            # anthropic can often embed the code in ```python blocks
            if '```python' in assistant_response:
                match = re.search(r'```python\n(.*?)```', assistant_response, re.DOTALL)
                if match:
                    assistant_response = match.group(1)

            # openai can often embed the code in ```starlark blocks
            if '```starlark' in assistant_response:
                match = re.search(r'```starlark\n(.*?)```', assistant_response, re.DOTALL)
                if match:
                    assistant_response = match.group(1)

            no_indent_debug(logging, '')
            no_indent_debug(logging, '** [bold yellow]Starlark Abstract Syntax Tree:[/bold yellow] **')
            # debug out AST
            lines = str(assistant_response).split('\n')
            for line in lines:
                no_indent_debug(logging, '  {}'.format(str(line).replace("[", "\\[")))
            no_indent_debug(logging, '')

            # debug output
            response_writer('llm_call', assistant_response)

            if self.edit_hook:
                assistant_response = self.edit_hook(assistant_response)

                # check to see if there is natural language in there or not
                try:
                    _ = ast.parse(str(assistant_response))
                except SyntaxError as ex:
                    logging.debug('aexecute() SyntaxError: {}'.format(str(ex)))
                    assistant_response = self.starlark_runtime.compile_error(
                        starlark_code=str(assistant_response),
                        error=str(ex),
                    )

            if not self.continuation_passing_style:
                _ = self.starlark_runtime.run(
                    starlark_code=assistant_response,
                    original_query=str(messages[-1].message),
                    messages=messages,
                )
                results.extend(self.starlark_runtime.answers)
                return results
            else:
                _ = self.starlark_runtime.run_continuation_passing(
                    starlark_code=assistant_response,
                    original_query=str(messages[-1].message),
                    messages=messages,
                )
                results.extend(self.starlark_runtime.answers)
                return results
        else:
            assistant_reply: Assistant = await self.aexecute_llm_call(
                message=messages[-1],
                context_messages=messages[0:-1],
                query=str(messages[-1].message),
                temperature=temperature,
                original_query='',
                stream_handler=stream_handler,
                model=model,
            )

            results.append(Answer(
                conversation=[assistant_reply],
                result=assistant_reply.message
            ))

        return results
