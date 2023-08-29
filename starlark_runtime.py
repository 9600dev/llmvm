import ast
import copy
import inspect
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, cast
from urllib.parse import urlparse

import pandas as pd
from openai import InvalidRequestError

from ast_parser import Parser
# from bcl import ContentDownloader, FunctionBindable, Searcher
from helpers.edgar import EdgarHelpers
from helpers.email_helpers import EmailHelpers
from helpers.firefox import FirefoxHelpers
from helpers.helpers import Helpers, response_writer
from helpers.logging_helpers import console_debug, setup_logging
from helpers.market import MarketHelpers
from helpers.pdf import PdfHelpers
from helpers.webhelpers import WebHelpers
from objects import (Answer, Assistant, AstNode, Content, Controller, Executor,
                     FunctionCall, FunctionCallMeta, Message, PandasMeta,
                     Statement, User)
from vector_store import VectorStore

logging = setup_logging()


class StarlarkRuntime:
    def __init__(
        self,
        executor: Controller,
        agents: List[Callable] = [],
        vector_store: VectorStore = VectorStore(),
    ):
        self.original_query = ''
        self.original_code = ''
        self.executor: Controller = executor
        self.agents = agents
        self.vector_store = vector_store
        self.locals_dict = {}
        self.globals_dict = {}
        self.answers: List[Answer] = []
        self.messages_list: List[Message] = []
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

        from bcl import BCL

        self.answers = []
        self.locals_dict = {}
        self.globals_dict = {}
        self.locals_dict['llm_bind'] = self.llm_bind
        self.locals_dict['llm_call'] = self.llm_call
        self.locals_dict['llm_loop_bind'] = self.llm_loop_bind
        self.locals_dict['messages'] = self.messages
        self.locals_dict['search'] = self.search
        self.locals_dict['download'] = self.download
        self.locals_dict['pandas_bind'] = self.pandas_bind
        for agent in self.agents:
            self.locals_dict[agent.__name__] = agent
        self.locals_dict['WebHelpers'] = CallWrapper(self, WebHelpers)
        self.locals_dict['PdfHelpers'] = CallWrapper(self, PdfHelpers)
        self.locals_dict['BCL'] = CallWrapper(self, BCL)
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
        from bcl import FunctionBindable

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
                    'function_callsite': context.callsite,
                    'function_result': str(context.result()),
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

        logging.debug('statement_to_message() unusual type, context is: {}'.format(context))
        return User(Content(str(context)))

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

        assistant = self.executor.execute_llm_call(
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
                # todo, now using firefox always
                raise ValueError('gotta fix this code')
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

        assistant = self.executor.execute_llm_call(
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
            assistant = self.executor.execute_llm_call(
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

    def messages(self):
        logging.debug('messages()')
        if len(self.messages_list) == 0:
            return []

        return [str(m.message) for m in self.messages_list[:-1] if m.role() != 'system']

    def llm_bind(self, expr, func: str):
        from bcl import FunctionBindable

        bindable = FunctionBindable(
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

    def download(
        self,
        expr: str,
    ) -> str:
        logging.debug(f'download({expr})')
        from bcl import ContentDownloader

        downloader = ContentDownloader(
            expr=expr,
            agents=self.agents,
            messages=[],
            starlark_runtime=self,
            original_code=self.original_code,
            original_query=self.original_query,
        )
        return downloader.get()

    def search(
        self,
        expr: str,
    ) -> str:
        logging.debug(f'search({expr})')
        from bcl import Searcher

        searcher = Searcher(
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

        assistant = self.executor.execute_llm_call(
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
        assistant = self.executor.execute_llm_call(
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

    def answer(self, expr) -> Answer:
        # if we have a list of answers, maybe just return them.
        if isinstance(expr, list) and all([isinstance(e, Assistant) for e in expr]):
            answer = Answer(
                conversation=self.messages_list,
                result='\n\n'.join([str(self.statement_to_message(assistant).message) for assistant in expr])
            )
            return answer

        # if we have a FunctionCallMeta object, it's likely we've called a helper function
        # and we're just keen to return that.
        # Handing this to the LLM means that call results that are larger than the context window
        # will end up being lost or transformed into the smaller output context window.
        # deal with base types
        if (
            isinstance(expr, FunctionCallMeta)
            or isinstance(expr, float)
            or isinstance(expr, int)
            or isinstance(expr, str)
            or isinstance(expr, bool)
            or expr is None
        ):
            answer_assistant = self.executor.execute_llm_call(
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/answer_primitive.prompt',
                    template={
                        'function_output': str(expr),
                        'original_query': self.original_query,
                    }),
                context_messages=[],  # type: ignore
                query=self.original_query,
                original_query=self.original_query,
                prompt_filename='prompts/starlark/answer_primitive.prompt',
                completion_tokens=512,
            )
            answer = Answer(
                conversation=self.messages_list,
                result=str(answer_assistant.message),
            )
            self.answers.append(answer)
            return answer

        if not self.answer_error_correcting:
            # todo: the 'rewriting' logic down below helps with the prettyness of
            # the output, and we're missing that here, but this is a nice shortcut.
            answer_assistant = self.executor.execute_llm_call(
                message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/starlark/answer_nocontext.prompt',
                    template={
                        'original_query': self.original_query,
                    }),
                context_messages=[self.statement_to_message(expr)],  # type: ignore
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

                # add the message to answers, just in case the user wants to see it.
                self.answers.append(Answer(
                    conversation=self.messages_list,
                    result=str(self.statement_to_message(expr).message)  # type: ignore
                ))

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

        # finally.
        context_messages: List[Message] = [
            self.statement_to_message(
                context=value,
            ) for key, value in self.globals_dict.items() if key.startswith('var')
        ]
        context_messages.append(self.statement_to_message(expr))  # type: ignore

        answer_assistant = self.executor.execute_llm_call(
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

    def compile_error(
        self,
        starlark_code: str,
        error: str,
    ):
        # SyntaxError, or other more global error. We should rewrite the entire code.
        # function_list = [Helpers.get_function_description_flat_extra(f) for f in self.agents]
        code_prompt = \
            f'''The following Starlark code (found under "Original Code") didn't parse or compile.
            Identify the error in the code below, and re-write the code and only that code.
            The error is found under "Error" and the code is found under "Code".

            Error: {error}

            Original Code:

            {starlark_code}
            '''

        assistant = self.executor.execute_llm_call(
            message=User(Content(code_prompt)),
            context_messages=[],
            query=self.original_query,
            original_query=self.original_query,
        )
        lines = str(assistant.message).split('\n')
        logging.debug('StarlarkRuntime.compile_error() Re-written Starlark code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def rewrite(
        self,
        starlark_code: str,
        error: str,
    ):
        # SyntaxError, or other more global error. We should rewrite the entire code.
        function_list = [Helpers.get_function_description_flat_extra(f) for f in self.agents]
        code_prompt = \
            f'''The following code (found under "Original Code") either didn't compile, or threw an exception while executing.
            Identify the error in the code below, and re-write the code and only that code.
            The error is found under "Error" and the code is found under "Original Code".

            If there is natural language guidance in previous messages, follow it to help re-write the original code.

            Original User Query: {self.original_query}

            Error: {error}

            Original Code:

            {starlark_code}
            '''

        assistant = self.executor.execute_llm_call(
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
        logging.debug('StarlarkRuntime.rewrite() Re-written Starlark code:')
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
        messages: List[Message] = [],
    ) -> Dict[Any, Any]:
        self.original_code = starlark_code
        self.original_query = original_query
        self.messages_list = messages
        self.setup()

        return self.__compile_and_execute(starlark_code)
