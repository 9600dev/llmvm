import ast
import asyncio
import datetime as dt
import inspect
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

import astunparse

from llmvm.common.helpers import Helpers, write_client_stream
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Answer, Assistant, Content, Controller,
                                  FunctionCall, FunctionCallMeta, LLMCall,
                                  MarkdownContent, Message, PandasMeta,
                                  Statement, User)
from llmvm.server.starlark_execution_controller import ExecutionController
from llmvm.server.tools.edgar import EdgarHelpers
from llmvm.server.tools.firefox import FirefoxHelpers
from llmvm.server.tools.market import MarketHelpers
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class StarlarkRuntime:
    def __init__(
        self,
        controller: ExecutionController,
        vector_search: VectorSearch,
        agents: List[Callable] = [],
    ):
        self.original_query = ''
        self.original_code = ''
        self.controller: ExecutionController = controller
        self.vector_search = vector_search
        self.agents = agents
        self.locals_dict = {}
        self.globals_dict = {}
        self.answers: List[Answer] = []
        self.messages_list: List[Message] = []
        self.answer_error_correcting = False
        self.setup()

    def statement_to_message(self, statement: Any) -> List[Message]:
        return self.controller.statement_to_message(statement)

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

        from llmvm.server.base_library.source_project import SourceProject
        from llmvm.server.bcl import BCL

        self.answers = []
        self.locals_dict = {}
        self.globals_dict = {}

        # todo: fix this hack
        self.globals_dict['llm_bind'] = self.llm_bind
        self.globals_dict['llm_call'] = self.llm_call
        self.globals_dict['llm_loop_bind'] = self.llm_loop_bind
        self.globals_dict['coerce'] = self.coerce
        self.globals_dict['messages'] = self.messages
        self.globals_dict['search'] = self.search
        self.globals_dict['download'] = self.download
        self.globals_dict['pandas_bind'] = self.pandas_bind
        for agent in self.agents:
            self.globals_dict[agent.__name__] = agent
        self.globals_dict['WebHelpers'] = CallWrapper(self, WebHelpers)
        self.globals_dict['BCL'] = CallWrapper(self, BCL)
        self.globals_dict['EdgarHelpers'] = CallWrapper(self, EdgarHelpers)
        self.globals_dict['FirefoxHelpers'] = CallWrapper(self, FirefoxHelpers)
        self.globals_dict['MarketHelpers'] = CallWrapper(self, MarketHelpers)
        self.globals_dict['answer'] = self.answer
        self.globals_dict['sys'] = sys
        self.globals_dict['os'] = os
        self.globals_dict['datetime'] = dt
        # code stuff
        source = SourceProject(self)
        self.globals_dict['source_project'] = source
        self.globals_dict['get_source_structure'] = source.get_source_structure
        self.globals_dict['get_source'] = source.get_source
        self.globals_dict['get_source_summary'] = source.get_source_summary
        self.globals_dict['get_classes'] = source.get_classes
        self.globals_dict['get_methods'] = source.get_methods
        self.globals_dict['get_references'] = source.get_references

    def pandas_bind(self, expr) -> PandasMeta:
        import pandas as pd

        logging.debug('pandas_bind()')

        def bind_with_llm(expr_str: str) -> PandasMeta:
            assistant = self.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_message(
                        prompt_name='pandas_bind.prompt',
                        template={},
                        user_token=self.controller.get_executor().user_token(),
                        assistant_token=self.controller.get_executor().assistant_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=self.statement_to_message(expr),  # type: ignore
                    executor=self.controller.get_executor(),
                    model=self.controller.get_executor().get_default_model(),
                    temperature=0.0,
                    max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                    completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                    prompt_name='pandas_bind.prompt',
                ),
                query=self.original_query,
                original_query=self.original_query,
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

    def messages(self) -> List[Message]:
        logging.debug('messages()')
        if len(self.messages_list) == 0:
            return []

        return [m for m in self.messages_list[:-1] if m.role() != 'system']

    def llm_bind(self, expr, func: str):
        # todo circular import if put at the top
        from llmvm.server.base_library.function_bindable import \
            FunctionBindable
        logging.debug(f'llm_bind({str(expr)[:20]}, {str(func)})')
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
            controller=self.controller,
            starlark_runtime=self,
        )
        bindable.bind(expr, func)
        return bindable

    def download(self, expr: str) -> Content:
        logging.debug(f'download({str(expr)})')

        from llmvm.server.base_library.content_downloader import \
            ContentDownloader
        cookies = self.locals_dict['cookies'] if 'cookies' in self.locals_dict else []

        downloader = ContentDownloader(
            expr=expr,
            cookies=cookies
        )
        return downloader.download()

    def search(self, expr: str) -> List[Content]:
        logging.debug(f'search({str(expr)})')
        from llmvm.server.base_library.searcher import Searcher

        searcher = Searcher(
            expr=expr,
            controller=self.controller,
            original_code=self.original_code,
            original_query=self.original_query,
            vector_search=self.vector_search,
        )
        return searcher.search()

    def coerce(self, expr, type_name: str) -> Any:
        logging.debug(f'coerce({str(expr)[:20]}, {str(type_name)})')
        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='coerce.prompt',
                    template={
                        'string': str(expr),
                        'type': type_name,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='coerce.prompt',
            ),
            query='',
            original_query=self.original_query,
        )
        return self.__eval_with_error_wrapper(str(assistant.message))

    def llm_call(self, expr_list: List[Any] | Any, llm_instruction: str) -> Assistant:
        logging.debug(f'llm_call({str(expr_list)[:20]}, {repr(llm_instruction)})')

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        # search returns a list of MarkdownContent objects, and the llm_call is typically
        # called with llm_call([var], ...), so we need to flatten
        expr_list = Helpers.flatten(expr_list)

        write_client_stream(Content(f'Calling LLM with instruction: {llm_instruction} ...\n'))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='llm_call.prompt',
                    template={
                        'llm_call_message': llm_instruction,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=self.statement_to_message(expr_list),
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='llm_call.prompt',
            ),
            query=llm_instruction,
            original_query=self.original_query,
        )
        write_client_stream(Content(f'LLM returned: {str(assistant.message)}\n'))
        return assistant

    def llm_loop_bind(self, expr, llm_instruction: str, count: int = sys.maxsize) -> List[Any]:
        logging.debug(f'llm_loop_bind({str(expr)[:20]}, {repr(llm_instruction)})')
        context = expr.message.get_str() if isinstance(expr, Message) else str(expr)

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='llm_loop_bind.prompt',
                    template={
                        'goal': llm_instruction.replace('"', ''),
                        'context': context,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='llm_loop_bind.prompt',
            ),
            query=llm_instruction,
            original_query=self.original_query,
        )

        list_result = str(assistant.message)

        # for anthropic
        if not list_result.startswith('['):
            pattern = r'\[\s*(?:(\d+|"[^"]*"|\'[^\']*\')\s*,\s*)*(\d+|"[^"]*"|\'[^\']*\')\s*\]'
            match = re.search(pattern, list_result)
            if match:
                list_result = match.group(0)

        try:
            result = cast(list, eval(list_result))
            if not isinstance(result, list):
                raise ValueError('llm_loop_bind result is not a list, or is malformed')
            return result[:count]
        except Exception as ex:
            logging.debug('llm_loop_bind error: {}'.format(ex))
            new_starlark_code = self.rewrite_starlark_error_correction(
                query=llm_instruction,
                starlark_code=str(assistant.message),
                error=str(ex),
                locals_dictionary=self.locals_dict,
            )
            logging.debug('llm_loop_bind new_starlark_code: {}'.format(new_starlark_code))
            result = cast(list, eval(new_starlark_code))
            if not isinstance(result, list):
                return []
            return result[:count]

    def answer(self, expr) -> Answer:
        logging.debug(f'answer({str(expr)[:20]}')
        # if we have a list of answers, maybe just return them.
        if isinstance(expr, list) and all([isinstance(e, Assistant) for e in expr]):
            # collapse the assistant answers and continue
            last = expr[-1]
            for e in expr[0:-1]:
                last.message = Content(f'{last.message}\n\n{e.message}')
            expr = last

        snippet = str(expr).replace('\n', ' ')[:150]
        write_client_stream(Content(f'I think I have an answer, but I am double checking it: answer("{snippet} ...")\n'))

        # if the original query is referring to an image, it's because we were in tool mode
        # so this is a todo: hack to fix answers() so that it works for images
        if "I've just pasted you an image." in self.original_query:
            answer = Answer(
                conversation=self.messages_list,
                result=str(expr)
            )
            self.answers.append(answer)
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
            or isinstance(expr, dict)
            or expr is None
        ):
            answer_assistant = self.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_message(
                        prompt_name='answer_primitive.prompt',
                        template={
                            'function_output': str(expr),
                            'original_query': self.original_query,
                        },
                        user_token=self.controller.get_executor().user_token(),
                        assistant_token=self.controller.get_executor().assistant_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=[],  # type: ignore
                    executor=self.controller.get_executor(),
                    model=self.controller.get_executor().get_default_model(),
                    temperature=0.0,
                    max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                    completion_tokens_len=512,
                    prompt_name='answer_primitive.prompt',
                ),
                query=self.original_query,
                original_query=self.original_query,
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
            answer_assistant = self.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_message(
                        prompt_name='answer_nocontext.prompt',
                        template={
                            'original_query': self.original_query,
                        },
                        user_token=self.controller.get_executor().user_token(),
                        assistant_token=self.controller.get_executor().assistant_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=self.statement_to_message(expr),  # type: ignore
                    executor=self.controller.get_executor(),
                    model=self.controller.get_executor().get_default_model(),
                    temperature=0.0,
                    max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                    completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                    prompt_name='answer_nocontext.prompt',
                ),
                query=self.original_query,
                original_query=self.original_query,
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
                    result=self.statement_to_message(expr)[-1].message  # type: ignore
                ))

                self.answer_error_correcting = True
                error_correction = self.rewrite_answer_error_correction(
                    query=self.original_query,
                    starlark_code=self.original_code,
                    error=str(expr),
                    locals_dictionary=self.locals_dict,
                )

                if error_correction and 'None' not in error_correction:
                    # try and perform the error correction
                    self.setup()
                    # results_dict = self.compile_and_execute(error_correction)

                    _ = StarlarkRuntime(
                        controller=self.controller,
                        agents=self.agents,
                        vector_search=self.vector_search,
                    ).run(error_correction, self.original_query)

        # finally.
        context_messages: List[Message] = Helpers.flatten([
            self.statement_to_message(
                statement=value,
            ) for key, value in self.locals_dict.items() if key.startswith('var')
        ])
        context_messages.extend(self.statement_to_message(expr))  # type: ignore

        answer_assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='answer.prompt',
                    template={
                        'original_query': self.original_query,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=context_messages,
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='answer.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
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

    def rewrite_answer_error_correction(
        self,
        query: str,
        starlark_code: str,
        error: str,
        locals_dictionary: Dict[Any, Any],
    ) -> str:
        '''
        This handles the case where an answer() is not correct and we need to
        identify the single place in the code where this may have occurred, and
        see if we can manually patch it.
        '''
        logging.debug('rewrite_answer_error_correction()')
        dictionary = ''
        for key, value in locals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='answer_error_correction.prompt',
                    template={
                        'task': query,
                        'code': starlark_code,
                        'error': error,
                        'functions': '\n'.join([Helpers.get_function_description_flat_extra(f) for f in self.agents]),
                        'dictionary': dictionary,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='answer_error_correction.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        return str(assistant.message)

    def rewrite_starlark_error_correction(
        self,
        query: str,
        starlark_code: str,
        error: str,
        locals_dictionary: Dict[Any, Any],
    ) -> str:
        logging.debug('rewrite_starlark_error_correction()')
        dictionary = ''
        for key, value in locals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='starlark_error_correction.prompt',
                    template={
                        'task': query,
                        'code': starlark_code,
                        'error': error,
                        'functions': '\n'.join([Helpers.get_function_description_flat_extra(f) for f in self.agents]),
                        'dictionary': dictionary,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='starlark_error_correction.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query
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
                    locals_dictionary=locals_dictionary,
                )
                return str(assistant.message)
            except Exception as ex:
                logging.debug('Second exception rewriting starlark code: {}'.format(ex))
                return ''

    def __eval_with_error_wrapper(
        self,
        starlark_code: str,
        retry_count: int = 2,
    ):
        counter = 0
        while counter < retry_count:
            try:
                return eval(starlark_code, self.globals_dict, self.locals_dict)
            except Exception as ex:
                starlark_code = self.rewrite(starlark_code, str(ex))
            counter += 1
        return None

    def __compile_and_execute(
        self,
        starlark_code: str,
    ) -> Dict[Any, Any]:
        logging.debug('__compile_and_execute()')
        parsed_ast = ast.parse(starlark_code)
        exec(compile(parsed_ast, filename="<ast>", mode="exec"), self.globals_dict, self.locals_dict)
        return self.locals_dict

    def __interpret(
        self,
        starlark_code: str,
    ) -> Dict[Any, Any]:
        parsed_ast = ast.parse(starlark_code)

        for node in parsed_ast.body:
            if isinstance(node, ast.Expr):
                logging.debug(f'[eval] {astunparse.unparse(node)}')
                result = eval(
                    compile(ast.Expression(node.value), '<string>', 'eval'),
                    {**self.globals_dict, **self.locals_dict}
                )
                logging.debug(f'[=>] {result}')
            else:
                logging.debug(f'[exec] {astunparse.unparse(node)}')
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), '<string>', 'exec'),
                    self.globals_dict,
                    self.locals_dict
                )
        return self.locals_dict

    def compile_error(
        self,
        starlark_code: str,
        error: str,
    ) -> str:
        logging.debug('compile_error()')
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

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=User(Content(code_prompt)),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        lines = str(assistant.message).split('\n')
        logging.debug('compile_error() Re-written Starlark code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def rewrite(
        self,
        starlark_code: str,
        error: str,
    ):
        logging.debug('rewrite()')
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

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='starlark_tool_execution.prompt',
                    template={
                        'functions': '\n'.join(function_list),
                        'user_input': code_prompt,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_prompt_tokens(),
                completion_tokens_len=self.controller.get_executor().max_completion_tokens(),
                prompt_name='starlark_tool_execution.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )
        lines = str(assistant.message).split('\n')
        logging.debug('rewrite() Re-written Starlark code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def run(
        self,
        starlark_code: str,
        original_query: str,
        messages: List[Message] = [],
        locals_dict: Dict = {}
    ) -> Dict[Any, Any]:
        self.original_code = starlark_code
        self.original_query = original_query
        self.messages_list = messages
        # todo: why are we running setup again here?
        # self.setup()
        self.locals_dict = locals_dict

        return self.__compile_and_execute(starlark_code)

    def run_continuation_passing(
        self,
        starlark_code: str,
        original_query: str,
        messages: List[Message] = [],
    ) -> Dict[Any, Any]:
        self.original_code = starlark_code
        self.original_query = original_query
        self.messages_list = messages
        self.setup()
        return self.__interpret(starlark_code)
