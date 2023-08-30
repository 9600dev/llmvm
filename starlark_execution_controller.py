import ast
import copy
import math
import random
from typing import Any, Callable, Dict, List, Optional, cast

import rich
from openai import InvalidRequestError

from ast_parser import Parser
from container import Container
from helpers.helpers import Helpers, response_writer
from helpers.logging_helpers import console_debug, setup_logging
from objects import (Answer, Assistant, Content, Controller, Executor, Message,
                     Statement, System, User)
from openai_executor import OpenAIExecutor
from persistent_cache import PersistentCache
from starlark_runtime import StarlarkRuntime
from vector_store import VectorStore

logging = setup_logging()

class StarlarkExecutionController(Controller):
    def __init__(
        self,
        executor: Executor,
        agents: List[Callable],
        vector_store: VectorStore = VectorStore(),
        cache: PersistentCache = PersistentCache(),
        edit_hook: Optional[Callable[[str], str]] = None,
        stream_handler: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()

        self.executor = executor
        self.agents = agents
        self.vector_store = vector_store
        self.cache = cache
        self.edit_hook = edit_hook
        self.stream_handler = stream_handler
        self.tools_model = Container().get('openai_tools_model') \
            if Container().get('openai_tools_model') else 'gpt-3.5-turbo-16k-0613'
        self.model = Container().get('openai_model') \
            if Container().get('openai_model') else 'gpt-3.5-turbo-16k-0613'
        self.starlark_runtime = StarlarkRuntime(self, self.agents)

    def __classify_tool_or_direct(
        self,
        prompt: str,
    ) -> Dict[str, float]:
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
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': prompt,
            }
        )

        assistant: Assistant = self.executor.execute(
            messages=[
                System(Content(query_understanding['system_message'])),
                User(Content(query_understanding['user_message']))
            ],
            temperature=0.0,
            model=self.model
        )

        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    def execute_llm_call(
        self,
        message: Message,
        context_messages: List[Message],
        query: str,
        original_query: str,
        prompt_filename: Optional[str] = None,
        completion_tokens: int = 2048,
        temperature: float = 0.0,
        model: str = 'gpt-3.5-turbo-16k-0613',
        lifo: bool = False,
        stream_handler: Optional[Callable[[str], None]] = None,
    ) -> Assistant:
        # internal helper to wrap and execute LLM call to executor.
        def __llm_call(
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
                assistant: Assistant = executor.execute(
                    messages,
                    max_completion_tokens=completion_tokens,
                    temperature=temperature,
                    stream_handler=stream_handler,
                    model=model,
                )
                console_debug(prompt_filename, 'User', str(user_message.message))
                console_debug(prompt_filename, 'Assistant', str(assistant.message))
            except Exception as ex:
                console_debug(prompt_filename, 'User', str(user_message.message))
                raise ex
            response_writer(prompt_filename, assistant)
            return assistant

        def __llm_call_prompt(
            prompt_filename: str,
            context_messages: List[Message],
            executor: Executor,
            template: Dict[str, Any],
        ) -> Assistant:
            prompt = Helpers.load_and_populate_prompt(
                prompt_filename=prompt_filename,
                template=template,
            )
            return __llm_call(
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
            self.executor.calculate_tokens(context_messages + [message])
            > self.executor.max_prompt_tokens(completion_token_count=completion_tokens)
        ):
            # check to see if we're simply lifo'ing the context messages (last in first out)
            if lifo:
                lifo_messages = copy.deepcopy(context_messages)
                prompt_context_messages = [message]
                current_tokens = self.executor.calculate_tokens(str(message.message)) + completion_tokens

                # reverse over the messages, last to first
                for i in range(len(lifo_messages) - 1, -1, -1):
                    if (
                        current_tokens + self.executor.calculate_tokens(str(lifo_messages[i].message))
                        < self.executor.max_prompt_tokens(completion_token_count=completion_tokens)
                    ):
                        prompt_context_messages.append(lifo_messages[i])
                        current_tokens += self.executor.calculate_tokens(str(lifo_messages[i].message))
                    else:
                        break

                assistant_result = __llm_call(
                    user_message=cast(User, message),
                    context_messages=prompt_context_messages[::-1],  # reversed, because of above
                    executor=self.executor,
                    prompt_filename=prompt_filename,
                )
                return assistant_result

            # not lifo
            context_message = User(Content('\n\n'.join([str(m.message) for m in context_messages])))

            # see if we can do a similarity search or not.
            similarity_chunks = self.vector_store.chunk_and_rank(
                query=query,
                content=str(context_message.message),
                chunk_token_count=256,
                chunk_overlap=0,
                max_tokens=self.executor.max_prompt_tokens() - self.executor.calculate_tokens([message]) - 32
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.15)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: List[str] = []
            for chunk, _ in similarity_chunks[:5]:
                assistant_similarity = __llm_call_prompt(
                    prompt_filename='prompts/document_chunk.prompt',
                    context_messages=[],
                    executor=self.executor,
                    template={
                        'query': str(query),
                        'document_chunk': chunk,
                    },
                )

                decision_criteria.append(str(assistant_similarity.message))
                logging.debug('map_reduce_required, query_or_task: {}, response: {}'.format(
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
                    math.floor((self.executor.max_prompt_tokens() - self.executor.calculate_tokens([message]))
                               / len(context_messages))
                )

                # for all messages, do a similarity search
                similarity_messages = []
                for i in range(len(context_messages)):
                    prev_message = context_messages[i]

                    similarity_chunks = self.vector_store.chunk_and_rank(
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

                assistant_result = __llm_call(
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
                    [User(Content(open('prompts/map_reduce_map.prompt', 'r').read()))]
                )
                chunk_size = self.executor.max_prompt_tokens() - map_reduce_prompt_tokens - self.executor.calculate_tokens([message]) - 32  # noqa E501

                chunks = self.vector_store.chunk(
                    content=str(context_message.message),
                    chunk_size=chunk_size,
                    overlap=0
                )

                for chunk in chunks:
                    chunk_assistant = __llm_call_prompt(
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

                assistant_result = __llm_call_prompt(
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
            assistant_result = __llm_call(
                user_message=cast(User, message),
                context_messages=context_messages,
                executor=self.executor,
                prompt_filename=prompt_filename,
            )
        return assistant_result

    def execute_with_agents(
        self,
        messages: List[Message],
        agents: List[Callable],
        temperature: float = 0.0,
    ) -> Assistant:
        if self.cache and self.cache.has_key(messages):
            return cast(Assistant, self.cache.get(messages))

        logging.debug('StarlarkRuntime.execute_with_agents() messages[-1] = {}'.format(str(messages[-1])[0:25]))

        functions = [Helpers.get_function_description_flat_extra(f) for f in agents]

        tools_message = Helpers.load_and_populate_message(
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': str(messages[-1].message),
            }
        )

        with open('logs/tools_execution.log', 'w') as f:
            f.write(str(tools_message['system_message']['content']) + '\n')
            f.write(str(tools_message['user_message']['content']))

        llm_response = self.execute_llm_call(
            message=tools_message,
            context_messages=messages[0:-1],
            query='',
            original_query='',
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
            completion_tokens=1024,
            temperature=temperature,
            model=self.tools_model,
            lifo=False,
            stream_handler=self.stream_handler,
        )

        with open('logs/tools_execution.log', 'a') as f:
            f.write('\n\n')
            for message in messages:
                f.write(f'Message:\n{message.message}\n\n')
            f.write(f'\n\nResponse:\n{llm_response.message}\n\n')
            f.write('\n\n')

        if self.cache: self.cache.set(messages, llm_response)
        return llm_response

    def execute(
        self,
        messages: List[Message],
        temperature: float = 0.0,
    ) -> List[Statement]:
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
        classification = self.__classify_tool_or_direct(str(messages[-1].message))

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification:
            response = self.execute_with_agents(
                messages=messages,
                agents=self.agents,
                temperature=temperature,
            )

            assistant_response = str(response.message).replace('Assistant:', '').strip()

            rich.print()
            rich.print('[bold yellow]Abstract Syntax Tree:[/bold yellow]')
            # debug out AST
            lines = str(assistant_response).split('\n')
            for line in lines:
                rich.print('{}'.format(str(line).replace("[", "\\[")))
            rich.print()

            # debug output
            response_writer('llm_call', assistant_response)

            if self.edit_hook:
                assistant_response = self.edit_hook(assistant_response)

                # check to see if there is natural language in there or not
                try:
                    _ = ast.parse(str(assistant_response))
                except SyntaxError as ex:
                    logging.debug('StarlarkRuntime.execute() SyntaxError: {}'.format(str(ex)))
                    assistant_response = self.starlark_runtime.compile_error(
                        starlark_code=str(assistant_response),
                        error=str(ex),
                    )

            _ = self.starlark_runtime.run(
                starlark_code=assistant_response,
                original_query=str(messages[-1].message),
                messages=messages,
            )
            results.extend(self.starlark_runtime.answers)
            return results
        else:
            assistant_reply: Assistant = self.execute_llm_call(
                message=messages[-1],
                context_messages=messages[0:-1],
                query=str(messages[-1].message),
                temperature=temperature,
                original_query='',
                stream_handler=self.stream_handler,
            )

            results.append(Answer(
                conversation=[assistant_reply],
                result=assistant_reply.message
            ))

        return results
