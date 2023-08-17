import ast
import copy
import datetime as dt
import inspect
import math
import os
import random
import sys
import time
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)
from urllib.parse import urlparse

import astunparse
import pandas as pd
import pandas_gpt
from openai import InvalidRequestError

from ast_parser import Parser
from eightbitvicuna import VicunaEightBit
from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.firefox import FirefoxHelpers
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import console_debug, setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.search import SerpAPISearcher
from helpers.vector_store import VectorStore
from helpers.websearch import WebHelpers
from objects import (Answer, Assistant, AstNode, Content, DataFrame,
                     ExecutionFlow, Executor, ForEach, FunctionCall,
                     FunctionCallMeta, Get, LLMCall, Message, Order,
                     PandasMeta, Program, Set, StackNode, Statement, System,
                     UncertainOrError, User, tree_map)
from openai_executor import OpenAIExecutor

logging = setup_logging()
def response_writer(callee, message):
    with (open('logs/ast.log', 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')


class StarlarkExecutionController:
    def __init__(
        self,
        execution_contexts: List[Executor],
        agents: List[Callable] = [],
        vector_store: VectorStore = VectorStore(),
        cache: PersistentCache = PersistentCache(),
        edit_hook: Optional[Callable[[str], str]] = None,
    ):
        self.execution_contexts: List[Executor] = execution_contexts
        self.agents = agents
        self.parser = Parser()
        self.vector_store: VectorStore = vector_store
        self.cache = cache
        self.edit_hook = edit_hook
        self.starlark_runtime = StarlarkRuntime(self.execution_contexts[0], self.agents)

    def classify_tool_or_direct(
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

        executor = self.execution_contexts[0]

        # assess the type of task
        function_list = [Helpers.get_function_description_flat_extra(f) for f in self.agents]
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': prompt,
            }
        )

        assistant: Assistant = executor.execute(
            messages=[
                System(Content(query_understanding['system_message'])),
                User(Content(query_understanding['user_message']))
            ],
            temperature=0.0,
        )

        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    def execute_with_agents(
        self,
        executor: OpenAIExecutor,
        call: LLMCall,
        agents: List[Callable],
        temperature: float = 0.0,
    ) -> Assistant:
        if self.cache and self.cache.has_key((call.message, call.supporting_messages)):
            return cast(Assistant, self.cache.get((call.message, call.supporting_messages)))

        logging.debug('StarlarkRuntime.execute_with_agents()')

        user_message: User = cast(User, call.message)
        # todo: we should figure out if we should pass context messages in here or not
        messages = []
        message_results = []

        functions = [Helpers.get_function_description_flat_extra(f) for f in agents]

        prompt = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
            template={
                'functions': '\n'.join(functions),
                'user_input': str(user_message),
            }
        )

        messages.append({'role': 'system', 'content': prompt['system_message']})
        messages.append({'role': 'user', 'content': prompt['user_message']})

        chat_response = executor.execute_direct(
            model=executor.model,
            temperature=temperature,
            max_completion_tokens=1024,  # todo: calculate this properly
            messages=messages,
            chat_format=True,
        )

        chat_response = chat_response['choices'][0]['message']  # type: ignore
        message_results.append(chat_response)

        with open('logs/tools_execution.log', 'w') as f:
            f.write('\n\n')
            for message in messages:
                f.write(f'Message:\n{message["content"]}')
            f.write(f'\n\nResponse:\n{chat_response["content"]}')
            f.write('\n\n')

        if len(chat_response) == 0:
            return Assistant(Content('The model could not execute the query.'), error=True)
        else:
            assistant = Assistant(
                message=Content(chat_response['content']),
                error=False,
                messages_context=[Message.from_dict(m) for m in messages],
                system_context=prompt['system_message'],
                llm_call_context=call,
            )

            if self.cache: self.cache.set((call.message, call.supporting_messages), assistant)
            return assistant

    def execute_chat(
        self,
        messages: List[Message],
    ) -> Assistant:
        executor = self.execution_contexts[0]
        return executor.execute(messages, temperature=1.0, max_completion_tokens=1024)

    def execute(
        self,
        call: LLMCall
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
        executor = self.execution_contexts[0]

        # assess the type of task
        classification = self.classify_tool_or_direct(str(call.message))

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification:
            response = self.execute_with_agents(
                executor=cast(OpenAIExecutor, executor),
                call=call,
                agents=self.agents,
                temperature=0.0,
            )
            import rich

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
                    assistant_response = self.starlark_runtime.rewrite(
                        starlark_code=str(assistant_response),
                        error=str(ex),
                    )

            _ = self.starlark_runtime.run(
                starlark_code=assistant_response,
                original_query=str(call.message),
            )
            results.extend(self.starlark_runtime.answers)
            return results
        else:
            assistant_reply: Assistant = self.execute_chat(call.supporting_messages + [call.message])
            results.append(Answer(
                conversation=[Content(str(assistant_reply.message))],
                result=assistant_reply.message
            ))

        return results


class StarlarkRuntime:
    def __init__(
        self,
        executor: Executor,
        agents: List[Callable] = [],
        vector_store: VectorStore = VectorStore(),
    ):
        self.original_query = ''
        self.original_code = ''
        self.executor = executor
        self.agents = agents
        self.vector_store = vector_store
        self.locals_dict = {}
        self.globals_dict = {}
        self.answers: List[Answer] = []
        self.answer_error_correcting = False
        self.setup()

    def setup(self):
        class CallWrapper:
            def __init__(self, outer_self, wrapped_class):
                self.wrapped_class = wrapped_class
                self.outer_self = outer_self

            def __getattr__(self, name):
                # instantiate if it's not been instantiated yet
                if isinstance(self.wrapped_class, type):
                    try:
                        self.wrapped_class = self.wrapped_class()
                    except Exception:
                        logging.debug(f'Could not instantiate {self.wrapped_class}')

                if hasattr(self.wrapped_class, name):
                    attr = getattr(self.wrapped_class, name)
                    if callable(attr):
                        def wrapper(*args, **kwargs):
                            caller_frame = inspect.currentframe().f_back  # type: ignore
                            caller_frame_lineno = caller_frame.f_lineno  # type: ignore

                            # extract the line from the original starlark code
                            code_line = self.outer_self.original_code.split('\n')[caller_frame_lineno - 1]

                            # todo: we should probably do the marshaling here too
                            result = attr(*args, **kwargs)

                            meta = FunctionCallMeta(
                                callsite=code_line,
                                func=attr,
                                result=result,
                                lineno=caller_frame_lineno,
                            )
                            return meta
                        return wrapper
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        self.answers = []
        self.locals_dict = {}
        self.globals_dict = {}
        self.locals_dict['llm_bind'] = self.llm_bind
        self.locals_dict['llm_call'] = self.llm_call
        self.locals_dict['llm_loop_bind'] = self.llm_loop_bind
        self.locals_dict['search'] = self.search
        self.locals_dict['pandas_bind'] = self.pandas_bind
        for agent in self.agents:
            self.locals_dict[agent.__name__] = agent
        self.locals_dict['WebHelpers'] = CallWrapper(self, WebHelpers)
        self.locals_dict['PdfHelpers'] = CallWrapper(self, PdfHelpers)
        self.locals_dict['EdgarHelpers'] = CallWrapper(self, EdgarHelpers)
        self.locals_dict['EmailHelpers'] = CallWrapper(self, EmailHelpers)
        self.locals_dict['FirefoxHelpers'] = CallWrapper(self, FirefoxHelpers)
        self.locals_dict['MarketHelpers'] = CallWrapper(self, MarketHelpers)
        self.locals_dict['answer'] = self.answer

    def __find_variable_assignment(
        self,
        var_name,
        tree
    ):
        class AssignmentFinder(ast.NodeVisitor):
            def __init__(self, var_name):
                self.var_name = var_name
                self.assignment_node = None

            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == self.var_name:
                        self.assignment_node = node.value
                self.generic_visit(node)

        finder = AssignmentFinder(var_name)
        finder.visit(tree)
        return finder.assignment_node

    def __get_assignment(
        self,
        variable_name,
        code,
    ) -> Optional[Tuple[ast.expr, ast.expr]]:
        '''
        Get's the right hand side of a variable assignment
        by walking the abstract syntax tree
        '''
        tree = ast.parse(code)

        assignment_node = self.__find_variable_assignment(variable_name, tree)
        if assignment_node is None:
            return None

        var_node = ast.Name(variable_name, ctx=ast.Load())
        return var_node, assignment_node

    def statement_to_message(
        self,
        context: Statement | str,
    ) -> User:
        statement_result_prompts = {
            'answer': 'prompts/answer_result.prompt',
            'assistant': 'prompts/starlark/assistant_result.prompt',
            'function_call': 'prompts/function_call_result.prompt',
            'function_meta': 'prompts/starlark/functionmeta_result.prompt',
            'llm_call': 'prompts/llm_call_result.prompt',
            'str': 'prompts/starlark/str_result.prompt',
            'uncertain_or_error': 'prompts/uncertain_or_error_result.prompt',
            'foreach': 'prompts/foreach_result.prompt',
            'list': 'prompts/starlark/list_result.prompt',
        }

        if isinstance(context, FunctionCall):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[context.token()],
                template={
                    'function_call': context.to_code_call(),
                    'function_signature': context.to_definition(),
                    'function_result': str(context.result()),
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, FunctionCallMeta):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts['function_meta'],
                template={
                    'function_call': context.callsite,
                    'function_result': str(context.result()),
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, LLMCall) or isinstance(context, ForEach):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[context.token()],
                template={
                    f'{context.token()}_result': str(context.result()),
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, str):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts['str'],
                template={
                    'str_result': context,
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, Assistant):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts['assistant'],
                template={
                    'assistant_result': str(context.message),
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, list):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts['list'],
                template={
                    'list_result': '\n'.join([str(c) for c in context])
                }
            )
            return User(Content(result_prompt['user_message']))

        elif isinstance(context, PandasMeta):
            return User(Content(context.df.to_csv()))

        elif isinstance(context, FunctionBindable):
            # todo
            return User(Content(context._result.result()))  # type: ignore

        raise ValueError(f"{str(context)} not supported")

    def execute_llm_call(
        self,
        executor: Executor,
        message: Message,
        context_messages: List[Message],
        query: str,
        original_query: str,
        prompt_filename: Optional[str] = None,
        completion_tokens: int = 1024,
    ) -> Assistant:
        def __llm_call(
            user_message: Message,
            context_messages: List[Message],
            executor: Executor,
            prompt_filename: Optional[str] = None,
            completion_tokens: int = 1024,
        ) -> Assistant:
            if not prompt_filename:
                prompt_filename = ''
            # execute the call to check to see if the Answer satisfies the original query
            messages: List[Message] = copy.deepcopy(context_messages)
            messages.append(user_message)
            try:
                assistant: Assistant = executor.execute(messages, max_completion_tokens=completion_tokens)
                console_debug(prompt_filename, 'User', str(user_message.message))
                console_debug(prompt_filename, 'Assistant', str(assistant.message))
            except InvalidRequestError as ex:
                console_debug(prompt_filename, 'User', str(user_message.message))
                raise ex
            response_writer(prompt_filename, assistant)
            return assistant

        def __llm_call_prompt(
            prompt_filename: str,
            context_messages: List[Message],
            executor: Executor,
            template: Dict[str, Any],
            completion_tokens: int = 1024,
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
                completion_tokens=completion_tokens,

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
            executor.calculate_tokens(context_messages + [message])
            > executor.max_prompt_tokens(completion_token_count=completion_tokens)
        ):
            context_message = User(Content('\n\n'.join([str(m.message) for m in context_messages])))

            # see if we can do a similarity search or not.
            similarity_chunks = self.vector_store.chunk_and_rank(
                query=query,
                content=str(context_message.message),
                chunk_token_count=256,
                chunk_overlap=10,
                max_tokens=executor.max_prompt_tokens() - executor.calculate_tokens([message])
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.2)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: List[str] = []
            for chunk, _ in similarity_chunks:
                assistant_similarity = __llm_call_prompt(
                    prompt_filename='prompts/document_chunk.prompt',
                    context_messages=[],
                    executor=executor,
                    template={
                        'query': str(query),
                        'document_chunk': chunk,
                    },
                    completion_tokens=completion_tokens)

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
                    math.floor((executor.max_prompt_tokens() - executor.calculate_tokens([message])) / len(context_messages))
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
                    executor=executor,
                    prompt_filename=prompt_filename,
                    completion_tokens=completion_tokens,
                )

            # do the map reduce instead of similarity
            else:
                # collapse the message
                context_message = User(Content('\n\n'.join([str(m.message) for m in context_messages])))
                chunk_results = []

                # iterate over the data.
                map_reduce_prompt_tokens = executor.calculate_tokens(
                    [User(Content(open('prompts/map_reduce_map.prompt', 'r').read()))]
                )
                chunk_size = (executor.max_prompt_tokens() - map_reduce_prompt_tokens) - (
                    executor.calculate_tokens([message]) - 32
                )

                chunks = self.vector_store.chunk(
                    content=str(context_message.message),
                    chunk_size=chunk_size,
                    overlap=0
                )

                for chunk in chunks:
                    chunk_assistant = __llm_call_prompt(
                        prompt_filename='prompts/map_reduce_map.prompt',
                        context_messages=[],
                        executor=executor,
                        template={
                            'original_query': original_query,
                            'query': query,
                            'data': chunk,
                        },
                        completion_tokens=completion_tokens
                    )
                    chunk_results.append(str(chunk_assistant.message))

                # perform the reduce
                map_results = '\n\n====\n\n' + '\n\n====\n\n'.join(chunk_results)

                assistant_result = __llm_call_prompt(
                    prompt_filename='prompts/map_reduce_reduce.prompt',
                    context_messages=[],
                    executor=executor,
                    template={
                        'original_query': original_query,
                        'query': query,
                        'map_results': map_results
                    },
                    completion_tokens=completion_tokens
                )
        else:
            assistant_result = __llm_call(
                user_message=cast(User, message),
                context_messages=context_messages,
                executor=executor,
                prompt_filename=prompt_filename,
                completion_tokens=completion_tokens,
            )
        return assistant_result

    def rewrite_answer_error_correction(
        self,
        query: str,
        starlark_code: str,
        error: str,
        globals_dictionary: Dict[Any, Any],
    ) -> str:
        '''
        This handles the case where an answer() is not correct and we need to
        identify the single place in the code where this may have occurred, and
        see if we can manually patch it.
        '''
        dictionary = ''
        for key, value in globals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/answer_error_correction.prompt',
                template={
                    'task': query,
                    'code': starlark_code,
                    'error': error,
                    'dictionary': dictionary,
                }),
            context_messages=[],
            query=self.original_query,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/answer_error_correction.prompt',
        )

        line = str(assistant.message)

        if line not in self.original_code and ' = ' in line:
            var_binding = line.split(' = ')[0].strip()
            # example:
            # var3 = Assistant(To find the differences and similarities between the two papers, I would need the text or
            # content of the two papers. Please provide the text or relevant information from the papers, and I will be
            # able to analyze and compare them for you. False)
            rhs: Optional[Tuple[ast.expr, ast.expr]] = self.__get_assignment(var_binding, starlark_code)
            if (
                rhs
                and isinstance(rhs[1], ast.Call)
                and hasattr(cast(ast.Call, rhs[1]).func, 'value')
                and hasattr(cast(ast.Call, rhs[1]), 'attr')
                and cast(ast.Call, rhs[1]).func.value.id == 'WebHelpers'  # type: ignore
                and cast(ast.Call, rhs[1]).func.attr == 'get_url'  # type: ignore
            ):
                # we have a WebHelpers.get_url() call that looks like it failed,
                # let's try with Firefox
                starlark_code = starlark_code.replace('WebHelpers.get_url', 'WebHelpers.get_url_firefox_via_pdf')
                return starlark_code
            elif (
                rhs
                and isinstance(rhs[1], ast.Call)
                and hasattr(cast(ast.Call, rhs[1]).func, 'value')
                and hasattr(cast(ast.Call, rhs[1]), 'attr')
                and cast(ast.Call, rhs[1]).func.attr == 'llm_call'  # type: ignore
            ):
                starlark_code = line
                return starlark_code
            else:
                return ''
        else:
            return ''

    def rewrite_starlark_error_correction(
        self,
        query: str,
        starlark_code: str,
        error: str,
        globals_dictionary: Dict[Any, Any],
    ) -> str:
        dictionary = ''
        for key, value in globals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/starlark_error_correction.prompt',
                template={
                    'task': query,
                    'code': starlark_code,
                    'error': error,
                    'dictionary': dictionary,
                }),
            context_messages=[],
            query=self.original_query,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/starlark_error_correction.prompt',
        )

        # double shot try
        try:
            _ = ast.parse(str(assistant.message))
            return str(assistant.message)
        except SyntaxError as ex:
            logging.debug('SyntaxError: {}'.format(ex))
            try:
                _ = self.rewrite_starlark_error_correction(
                    query=query,
                    starlark_code=str(assistant.message),
                    error=str(ex),
                    globals_dictionary=globals_dictionary,
                )
                return str(assistant.message)
            except Exception as ex:
                logging.debug('Second exception rewriting starlark code: {}'.format(ex))
                return ''

    def uncertain_or_error(self):
        logging.debug('uncertain_or_error()')
        pass

    def pandas_bind(self, expr) -> PandasMeta:
        logging.debug('pandas_bind()')

        def bind_with_llm(expr_str: str) -> PandasMeta:
            assistant = self.execute_llm_call(
                executor=self.executor,
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/pandas_bind.prompt',
                    template={}),
                context_messages=[self.statement_to_message(expr)],  # type: ignore
                query=self.original_query,
                original_query=self.original_query,
                prompt_filename='prompts/starlark/pandas_bind.prompt',
            )
            return PandasMeta(expr_str=expr_str, pandas_df=pd.DataFrame(str(assistant.message)))

        if isinstance(expr, str) and '.csv' in expr:
            try:
                result = urlparse(expr)

                if result.scheme == '' or result.scheme == 'file' or result.scheme == 'https' or result.scheme == 'http':
                    df = pd.read_csv(expr)
                    return PandasMeta(expr_str=expr, pandas_df=df)
                else:
                    raise ValueError()
            except Exception:
                return bind_with_llm(expr)
        elif isinstance(expr, list) or isinstance(expr, dict):
            df = pd.DataFrame(expr)
            return PandasMeta(expr_str=str(expr), pandas_df=df)
        else:
            return bind_with_llm(expr)

    def llm_bind(self, expr, func: str) -> 'FunctionBindable':
        bindable = FunctionBindable(
            executor=self.executor,
            expr=expr,
            func=func,
            agents=self.agents,
            messages=[],
            lineno=inspect.currentframe().f_back.f_lineno,  # type: ignore
            expr_instantiation=inspect.currentframe().f_back.f_locals,  # type: ignore
            scope_dict=self.globals_dict,
            original_code=self.original_code,
            original_query=self.original_query,
            starlark_runtime=self,
        )
        bindable.bind(expr, func)
        return bindable

    def search(
        self,
        expr: str,
    ) -> str:
        logging.debug(f'search({expr})')
        searcher = Search(
            executor=self.executor,
            expr=expr,
            agents=self.agents,
            messages=[],
            starlark_runtime=self,
            original_code=self.original_code,
            original_query=self.original_query,
        )
        return searcher.search()

    def llm_call(self, expr_list: List[Any] | Any, llm_instruction: str) -> Assistant:
        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/llm_call.prompt',
                template={
                    'llm_call_message': llm_instruction,
                }),
            context_messages=[self.statement_to_message(expr) for expr in expr_list],
            query=llm_instruction,
            original_query=self.original_query,
            prompt_filename='prompts/llm_call.prompt',
        )
        return assistant

    def llm_loop_bind(self, expr, llm_instruction: str, count: int = sys.maxsize) -> List[Any]:
        assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/llm_loop_bind.prompt',
                template={
                    'goal': llm_instruction.replace('"', ''),
                    'context': str(expr),
                }),
            context_messages=[],
            query=llm_instruction,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/llm_loop_bind.prompt'
        )

        try:
            result = cast(list, eval(str(assistant.message)))
            if not isinstance(result, list):
                raise ValueError('llm_loop_bind result is not a list, or is malformed')
            return result[:count]
        except Exception as ex:
            logging.debug('llm_loop_bind error: {}'.format(ex))
            new_starlark_code = self.rewrite_starlark_error_correction(
                query=llm_instruction,
                starlark_code=str(assistant.message),
                error=str(ex),
                globals_dictionary=self.globals_dict,
            )
            logging.debug('llm_loop_bind new_starlark_code: {}'.format(new_starlark_code))
            result = cast(list, eval(new_starlark_code))
            if not isinstance(result, list):
                return []
            return result[:count]

    def answer(self, expr):
        messages: List[Message] = []

        if not self.answer_error_correcting:
            # todo: the 'rewriting' logic down below helps with the prettyness of
            # the output, and we're missing that here, but this is a nice shortcut.
            answer_assistant = self.execute_llm_call(
                executor=self.executor,
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/answer_nocontext.prompt',
                    template={
                        'original_query': self.original_query,
                    }),
                context_messages=[self.statement_to_message(expr)],
                query=self.original_query,
                original_query=self.original_query,
                prompt_filename='prompts/starlark/answer_nocontext.prompt',
                completion_tokens=2048,
            )

            if 'None' not in str(answer_assistant.message) and "[##]" not in str(answer_assistant.message):
                answer = Answer(
                    conversation=[],
                    result=str(answer_assistant.message)
                )
                self.answers.append(answer)
                return answer

            elif 'None' in str(answer_assistant.message) and "[##]" in str(answer_assistant.message):
                logging.debug("Found comment in answer_nocontext: {}".format(answer_assistant.message))

                self.answer_error_correcting = True
                error_correction = self.rewrite_answer_error_correction(
                    query=self.original_query,
                    starlark_code=self.original_code,
                    error=str(expr),
                    globals_dictionary=self.globals_dict,
                )

                if error_correction and 'None' not in error_correction:
                    # try and perform the error correction
                    self.setup()
                    # results_dict = self.compile_and_execute(error_correction)
                    results_dict = StarlarkRuntime(
                        self.executor,
                        self.agents,
                        self.vector_store
                    ).run(error_correction, self.original_query)
                    return

        # finally.
        context_messages: List[Message] = [
            self.statement_to_message(
                context=value,
            ) for key, value in self.globals_dict.items() if key.startswith('var')
        ]
        context_messages.append(self.statement_to_message(expr))

        answer_assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/answer.prompt',
                template={
                    'original_query': self.original_query,
                }),
            context_messages=context_messages,
            query=self.original_query,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/answer.prompt',
            completion_tokens=2048,
        )

        # check for comments
        if "[##]" in str(answer_assistant.message):
            answer_assistant.message = Content(str(answer_assistant.message).split("[##]")[0].strip())

        answer = Answer(
            conversation=[],
            result=str(answer_assistant.message)
        )
        self.answers.append(answer)
        return answer

    def rewrite(
        self,
        starlark_code: str,
        error: str,
    ):
        # SyntaxError, or other more global error. We should rewrite the entire code.
        function_list = [Helpers.get_function_description_flat_extra(f) for f in self.agents]
        code_prompt = \
            f'''The following code either didn't compile, or threw an exception while executing. Re-write the code.
            If there is natural language guidance in previous messages, follow it.

            Original User Query: {self.original_query}

            Error: {error}

            Original Code:

            {starlark_code}
            '''

        assistant = self.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
                template={
                    'functions': '\n'.join(function_list),
                    'user_input': code_prompt,
                }
            ),
            context_messages=[],
            query=self.original_query,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/starlark_tool_execution.prompt',
        )
        lines = str(assistant.message).split('\n')
        logging.debug(f'StarlarkRuntime.rewrite() Re-written Starlark code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def __compile_and_execute(
        self,
        starlark_code: str,
    ) -> Dict[Any, Any]:
        parsed_ast = ast.parse(starlark_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"), self.locals_dict, self.globals_dict)
        return self.globals_dict

    def run(
        self,
        starlark_code: str,
        original_query: str,
    ) -> Dict[Any, Any]:
        self.original_code = starlark_code
        self.original_query = original_query
        self.setup()

        counter = 0
        last_exception = ''
        while counter < 3:
            try:
                return self.__compile_and_execute(starlark_code)
            except Exception as ex:
                starlark_code = self.rewrite(starlark_code, str(ex))
                last_exception = str(ex)
            finally:
                counter += 1
        raise RuntimeError('StarlarkRuntime.run() failed to execute code. Last exception: {}'.format(last_exception))

    def execute_ast(
        self,
        ast_node: ast.AST,
    ):
        if isinstance(ast_node, ast.Module):
            for statement in ast_node.body:
                self.execute_statement(statement)

    def execute_expr(
        self,
        node: ast.expr | ast.Expression,
    ):
        if isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self.execute_expr(node.left)
            right = self.execute_expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
        elif isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return globals()[node.id]
        elif isinstance(node, ast.IfExp):
            test = self.execute_expr(node.test)
            return self.execute_expr(node.body) if test else self.execute_expr(node.orelse)
        else:
            raise NotImplementedError(f"Unhandled node type: {type(node).__name__}")

    def execute_statement(
        self,
        statement: ast.stmt,
    ):
        if isinstance(statement, ast.Expr):
            self.execute_expr(statement.value)
        elif isinstance(statement, ast.FunctionDef):
            # Handle function definition here (if needed)
            pass
        elif isinstance(statement, ast.If):
            # Handle if statement here (if needed)
            pass
        elif isinstance(statement, ast.For):
            # Handle for loop statement here (if needed)
            pass
        else:
            raise NotImplementedError(f"Unhandled statement type: {type(statement).__name__}")


class Search():
    def __init__(
        self,
        executor: Executor,
        expr,
        agents: List[Callable],
        messages: List[Message],
        starlark_runtime: StarlarkRuntime,
        original_code: str,
        original_query: str,
        total_links_to_return: int = 4,
    ):
        self.executor = executor
        self.query = expr
        self.messages: List[Message] = messages
        self.agents = agents
        self.original_code = original_code
        self.original_query = original_query
        self.starlark_runtime = starlark_runtime

        self.parser = WebHelpers.get_url
        self.ordered_snippets: List = []
        self.index = 0
        self._result = None
        self.total_links_to_return: int = total_links_to_return

    def search(
        self,
    ) -> str:
        # todo: we should probably return the Search instance, so we can futz with it later on.
        query_expander = self.starlark_runtime.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_expander.prompt',
                template={
                    'query': self.query,
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )

        queries = eval(str(query_expander.message))[:3]

        def yelp_to_text(reviews: Dict[Any, Any]) -> str:
            return_str = f"{reviews['title']} in {reviews['neighborhood']}."
            return_str += '\n\n'
            return_str += f"{reviews['reviews']}"
            return return_str

        engines = {
            'Google Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a general web search engine that is good at answering questions, finding knowledge and information, and has a complete scan of the Internet.'},  # noqa:E501
            'Google News': {'searcher': SerpAPISearcher().search_news, 'parser': WebHelpers.get_news_url, 'description': 'a news search engine. This engine is excellent at finding news about particular topics, people, companies and entities.'},  # noqa:E501
            'Google Product Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a product search engine that is excellent at finding the prices of products, finding products that match descriptions of products, and finding where to buy a particular product.'},  # noqa:E501
            'Google Patent Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a search engine that is exceptional at findind matching patents for a given query.'},  # noqa:E501
            'Yelp Search': {'searcher': SerpAPISearcher().search_yelp, 'parser': yelp_to_text, 'description': 'a search engine dedicated to finding geographically local establishments, restaurants, stores etc and extracing their user reviews.'},  # noqa:E501
            'Hacker News Search': {'searcher': SerpAPISearcher().search_internet, 'parser': WebHelpers.get_url, 'description': 'a search engine dedicated to technology, programming and science. This search engine finds and returns commentary from smart individuals about news, technology, programming and science articles.'},  # noqa:E501
        }  # noqa:E501

        # classify the search engine
        engine_rank = self.starlark_runtime.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_classifier.prompt',
                template={
                    'query': '\n'.join(queries),
                    'engines': '\n'.join([f'* {key}: {value["description"]}' for key, value in engines.items()]),
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )
        engine = str(engine_rank.message).split('\n')[0]
        searcher = SerpAPISearcher().search_internet

        for key, value in engines.items():
            if key in engine:
                self.parser = engines[key]['parser']
                searcher = engines[key]['searcher']

        # perform the search
        search_results = []

        # deal especially for yelp.
        if 'Yelp' in engine:
            # take the first query, and figure out the location
            location = self.starlark_runtime.execute_llm_call(
                executor=self.executor,
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/search_location.prompt',
                    template={
                        'query': queries[0],
                    }),
                context_messages=[],
                query=self.query,
                original_query=self.original_query,
                prompt_filename='prompts/starlark/search_location.prompt',
            )
            query_result, location = eval(str(location.message))
            yelp_result = SerpAPISearcher().search_yelp(query_result, location)
            return yelp_to_text(yelp_result)

        for query in queries:
            search_results.extend(list(searcher(query))[:10])

        import random

        snippets = {
            str(random.randint(0, 100000)): {'title': result['title'], 'snippet': result['snippet'], 'link': result['link']}
            for result in search_results
        }

        result_rank = self.starlark_runtime.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/search_ranker.prompt',
                template={
                    'queries': '\n'.join(queries),
                    'snippets': '\n'.join(
                        [f'* {str(key)}: {value["title"]} {value["snippet"]}' for key, value in snippets.items()]
                    ),
                }),
            context_messages=[],
            query=self.query,
            original_query=self.original_query,
        )

        ranked_results = eval(str(result_rank.message))
        self.ordered_snippets = [snippets[key] for key in ranked_results if key in snippets]
        return self.result()

    def result(self) -> str:
        return_results = []

        while len(return_results) < self.total_links_to_return and self.index < len(self.ordered_snippets):
            for result in self.ordered_snippets[self.index:]:
                self.index += 1
                try:
                    parser_result = self.parser(result['link']).strip()
                    if parser_result:
                        return_results.append(f"The following content is from: {result['link']} with the title: {result['title']} \n\n{parser_result}")  # noqa:E501
                    if len(return_results) >= 4:
                        break
                except Exception as e:
                    logging.error(e)
                    pass
        return '\n\n\n'.join(return_results)

class FunctionBindable():
    def __init__(
        self,
        executor: Executor,
        expr,
        func: str,
        agents: List[Callable],
        messages: List[Message],
        lineno: int,
        expr_instantiation,
        scope_dict: Dict[Any, Any],
        original_code: str,
        original_query: str,
        starlark_runtime: StarlarkRuntime,
    ):
        self.executor = executor
        self.expr = expr
        self.expr_instantiation = expr_instantiation
        self.messages: List[Message] = messages
        self.func = func.replace('"', '')
        self.agents = agents
        self.lineno = lineno
        self.scope_dict = scope_dict
        self.original_code = original_code
        self.original_query = original_query
        self.starlark_runtime = starlark_runtime
        self.bound_function: Optional[Callable] = None
        self._result = None

    def __call__(self, *args, **kwargs):
        if self._result:
            return self._result

    def __bind_helper(
        self,
        expr,
        func: str,
        context_messages: List[Message] = [],
    ) -> str:
        # if we have a list, we need to use a different prompt
        if isinstance(self.expr, list):
            raise ValueError('llm_bind() does not support lists. You should rewrite the code to use a for loop instead.')

        # get a function definition fuzzy binding
        function_str = Helpers.in_between(func, '', '(')
        function_callable = [f for f in self.agents if function_str in str(f)]
        if not function_callable:
            raise ValueError('could not find function: {}'.format(function_str))

        function_callable = function_callable[0]
        function_definition = Helpers.get_function_description_flat_extra(cast(Callable, function_callable))

        bindable = self.starlark_runtime.execute_llm_call(
            executor=self.executor,
            message=Helpers.load_and_populate_message(
                prompt_filename='prompts/starlark/llm_bind_global.prompt',
                template={
                    'function_definition': function_definition,
                }),
            context_messages=context_messages,
            query=self.original_query,
            original_query=self.original_query,
            prompt_filename='prompts/starlark/llm_bind_global.prompt',
        )
        return str(bindable.message)

    def binder(
        self,
        expr,
        func: str,
    ) -> Generator['FunctionBindable', None, None]:

        bound = False
        global_counter = 0
        messages: List[Message] = []
        extra: List[str] = []
        goal = ''
        bindable = ''
        function_call: Optional[FunctionCall] = None

        def find_string_instantiation(target_string, source_code):
            parsed_ast = ast.parse(source_code)

            for node in ast.walk(parsed_ast):
                # Check for direct assignment
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and isinstance(node.value, ast.Str):
                            if node.value.s == target_string:
                                return (node, None)
                        # Check for string instantiation in a list
                        elif isinstance(node.value, ast.List):
                            for element in node.value.elts:
                                if isinstance(element, ast.Str) and element.s == target_string:
                                    return (element, node)
            return None, None

        # the following code determines the progressive scope exposure to the LLM
        # to help it determine the binding
        expr_instantiation_message = User(Content())
        if isinstance(expr, str) and find_string_instantiation(expr, self.original_code):
            node, parent = find_string_instantiation(expr, self.original_code)
            if parent:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(parent)}"
                )
            elif node:
                expr_instantiation_message.message = Content(
                    f"The data in the next message was instantiated via this Starlark code: {astunparse.unparse(node.value)}"
                )

        # start with just the expression binding
        messages.append(self.starlark_runtime.statement_to_message(expr))
        # instantiation
        messages.append(expr_instantiation_message)
        # goal
        messages.append(User(Content(
            f"""The overall goal of the Starlark program is to: {self.original_query}."""
        )))
        messages.append(User(Content(
            f"""The Starlark code that is currently being executed is: {self.original_code}"""
        )))
        # program scope
        scope = '\n'.join(['{} = {}'.format(key, value) for key, value in self.scope_dict.items() if key.startswith('var')])
        messages.append(User(Content(
            f"""The Starlark programs current execution scope for all variables is:
            {scope}
            """
        )))

        counter = 1

        while global_counter < 3:
            # try and bind the callsite without executing
            while not bound and counter < 4:

                bindable = self.__bind_helper(
                    expr=expr,
                    func=func,
                    context_messages=messages[:counter][::-1],  # reversing the list using list slicing
                )

                # get function definition
                parser = Parser()
                parser.agents = self.agents
                function_call = parser.get_callsite(bindable)

                if 'None' in str(bindable):
                    # move forward a stage
                    counter += 1
                    if counter >= len(messages):
                        # we've run out of messages, so we'll just use the original code
                        break
                if 'None' not in str(bindable) and function_call:
                    break

            if not function_call:
                raise ValueError('couldn\'t bind function call for func: {}, expr: {}'.format(func, expr))

            # todo need to properly parse this.
            if ' = ' not in bindable:
                # no identifier, so we'll create one to capture the result
                identifier = 'result_{}'.format(str(time.time()).replace('.', ''))
                bindable = '{} = {}'.format(identifier, bindable)
            else:
                identifier = bindable.split(' = ')[0].strip()

            # execute the function
            # todo: figure out what to do when you get back None, or ''
            starlark_code = bindable
            globals_dict = self.scope_dict.copy()
            globals_result = {}

            try:
                global_counter += 1

                globals_result = StarlarkRuntime(
                    self.executor,
                    self.agents,
                    self.starlark_runtime.vector_store
                ).run(starlark_code, self.original_query)

                self._result = globals_result[identifier]
                yield self

                # if we're here, it's because we've been next'ed() and it was the wrong binding
                # reset the binding parameters and try again.
                counter = 0
                bound = False

            except Exception as ex:
                logging.debug('Error executing function call: {}'.format(ex))
                counter += 1
                starlark_code = self.starlark_runtime.rewrite_starlark_error_correction(
                    query=self.original_query,
                    starlark_code=starlark_code,
                    error=str(ex),
                    globals_dictionary=globals_dict,
                )

        raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))

    def bind(
        self,
        expr,
        func,
    ) -> 'FunctionBindable':
        for bindable in self.binder(expr, func):
            return bindable

        raise ValueError('could not bind and or execute the function: {} expr: {}'.format(func, expr))
