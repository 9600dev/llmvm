import ast
import builtins
import traceback
import numpy as np
import datetime as dt
import inspect
import pandas as pd
import os
import re
import sys
import scipy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, Any, Type
from urllib.parse import urlparse

import astunparse

from llmvm.common.helpers import Helpers, write_client_stream, get_stream_handler
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Answer, Assistant, Content, DownloadParams, FileContent,
                                  FunctionCallMeta, LLMCall,
                                  Message, PandasMeta, TextContent,
                                  User, coerce_to, awaitable_none)
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.tools.edgar import EdgarHelpers
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.server.tools.market import MarketHelpers
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class PythonRuntime:
    def __init__(
        self,
        controller: ExecutionController,
        vector_search: VectorSearch,
        tools: List[Callable] = [],
        answer_error_correcting: bool = False,
        locals_dict = {},
        globals_dict = {}
    ):
        self.original_query = ''
        self.original_code = ''
        self.controller: ExecutionController = controller
        self.vector_search = vector_search
        self.tools = tools
        self.locals_dict = locals_dict
        self.globals_dict = globals_dict
        self.answers: List[Answer] = []
        self.messages_list: List[Message] = []
        self.answer_error_correcting = answer_error_correcting
        self.setup()

    def statement_to_message(self, statement: Any) -> List[Message]:
        return self.controller.statement_to_message(statement)

    def setup(self):
        class InstantiationWrapper:
            def __init__(self, wrapped_class, python_runtime: PythonRuntime):
                self.wrapped_class = wrapped_class
                self.python_runtime = python_runtime

            def __call__(self, *args, **kwargs):
                init_method = self.wrapped_class.__init__
                params_call = {}

                for name, param in inspect.signature(init_method).parameters.items():
                    if name == 'self':
                        continue

                    if param.annotation and param.annotation is PythonRuntime:
                        params_call[name] = self.python_runtime
                    elif param.annotation and param.annotation is ExecutionController:
                        params_call[name] = self.python_runtime.controller
                    elif param.annotation and param.annotation is VectorSearch:
                        params_call[name] = self.python_runtime.vector_search
                    elif name == 'cookies':
                        cookies = self.python_runtime.locals_dict['cookies'] if 'cookies' in self.python_runtime.locals_dict else []
                        params_call[name] = cookies

                merged_kwargs = {**params_call, **kwargs}
                instance = self.wrapped_class(*args, **merged_kwargs)
                return instance

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

                            # extract the line from the original python code
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
                raise AttributeError(f"'{self.wrapped_class.__class__.__name__}' object has no attribute '{name}'")

        class float():
            def __new__(cls, value, *args, **kwargs):
                return builtins.float.__new__(builtins.float, coerce_to(value, builtins.float), *args, **kwargs)

            def __init__(self, value):
                self.value = coerce_to(value, builtins.float)

            @classmethod
            def __instancecheck__(cls, instance):
                return isinstance(instance, builtins.float)

            def __getattr__(self, name):
                return getattr(self.value, name)

            def __float__(self):
                return builtins.float(self.value)

            def __repr__(self):
                    return builtins.float.__repr__(self.value)

            def __str__(self):
                return builtins.float.__str__(self.value)

            @property
            def __class__(self):
                return builtins.float

        class int(builtins.int):
            def __new__(cls, value, *args, **kwargs):
                return builtins.int.__new__(builtins.int, coerce_to(value, builtins.int), *args, **kwargs)

            def __init__(self, value):
                self.value = coerce_to(value, builtins.int)

            @classmethod
            def __instancecheck__(cls, instance):
                return isinstance(instance, builtins.int)

            def __getattr__(self, name):
                return getattr(self.value, name)

            def __float__(self):
                return builtins.float(self.value)

            def __repr__(self):
                    return builtins.int.__repr__(self.value)

            def __str__(self):
                return builtins.int.__str__(self.value)

            @property
            def __class__(self):
                return builtins.int

        from llmvm.server.bcl import BCL

        self.answers = []
        self.locals_dict = {}
        self.globals_dict = {}

        # todo: fix this hack
        self.globals_dict['llm_bind'] = self.llm_bind
        self.globals_dict['llm_call'] = self.llm_call
        self.globals_dict['llm_list_bind'] = self.llm_list_bind
        self.globals_dict['coerce'] = self.coerce
        self.globals_dict['messages'] = self.messages
        self.globals_dict['search'] = self.search
        self.globals_dict['download'] = self.download
        self.globals_dict['pandas_bind'] = self.pandas_bind
        for tool in self.tools:
            # a tool is either a static method that is directly callable, or an instance method
            # which needs an instance of the class to be instantiated
            is_static, cls = Helpers.is_static_method(tool)
            if not is_static and cls and cls.__name__ not in self.globals_dict:
                self.globals_dict[cls.__name__] = InstantiationWrapper(cls, python_runtime=self)
            else:
                self.globals_dict[tool.__name__] = tool

        self.globals_dict['WebHelpers'] = CallWrapper(self, WebHelpers)
        self.globals_dict['BCL'] = CallWrapper(self, BCL)
        self.globals_dict['EdgarHelpers'] = CallWrapper(self, EdgarHelpers)
        self.globals_dict['MarketHelpers'] = CallWrapper(self, MarketHelpers)
        self.globals_dict['answer'] = self.answer
        self.globals_dict['sys'] = sys
        self.globals_dict['os'] = os
        self.globals_dict['datetime'] = dt
        self.globals_dict['numpy'] = np
        self.globals_dict['scipy'] = scipy
        self.globals_dict['np'] = np
        self.globals_dict['pd'] = pd
        self.globals_dict['float'] = float
        self.globals_dict['int'] = int


    @staticmethod
    def only_code_block(code: str) -> bool:
        code = code.strip()
        return (
            (code.startswith('```python') or code.startswith('<helpers>'))
            and (code.endswith('```') or code.endswith('</helpers>'))
        )

    @staticmethod
    def get_code_blocks(code: str) -> List[str]:
        def extract_code_blocks(text):
            # Pattern to match <helpers> blocks
            code_pattern = re.compile(r'<helpers>(.*?)</helpers>', re.DOTALL)
            code_blocks = code_pattern.findall(text)

            blocks = []
            for block in code_blocks:
                # Check for nested Markdown block
                markdown_pattern = re.compile(r'```(?:python)?\s*(.*?)\s*```', re.DOTALL)
                nested_markdown = markdown_pattern.search(block)
                if nested_markdown:
                    blocks.append(nested_markdown.group(1))
                else:
                    blocks.append(block)
            return blocks

        code = code.strip()
        ordered_blocks = []

        for block in extract_code_blocks(code):
            block = block.strip()
            if block:
                # Remove language identifier if present
                if block.startswith('python'):
                    block = block[6:].lstrip()
                try:
                    # ast.parse(block)
                    # syntax errors weren't allowing the model to try again and fix the error
                    ordered_blocks.append(block)
                except SyntaxError:
                    pass

        return ordered_blocks

    def pandas_bind(self, expr) -> PandasMeta:
        logging.debug(f'PythonRuntime.pandas_bind({expr})')

        def pandas_bind_with_llm(expr_str: str) -> PandasMeta:
            assistant: Assistant = self.controller.execute_llm_call(
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
                    max_prompt_len=self.controller.get_executor().max_input_tokens(),
                    completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                    prompt_name='pandas_bind.prompt',
                ),
                query=self.original_query,
                original_query=self.original_query,
            )
            return PandasMeta(expr_str=expr_str, pandas_df=pd.DataFrame(assistant.get_str()))

        if isinstance(expr, str) and 'FileContent' in expr:
            # sometimes the LLM generates code which is the "FileContent(...)" representation of the variable
            # rather than the actual FileContent variable
            try:
                file_content_url = re.search(r'FileContent\((.*)\)', expr).group(1)  # type: ignore
                df = pd.read_csv(file_content_url)
                return PandasMeta(expr_str=expr, pandas_df=df)
            except Exception:
                return pandas_bind_with_llm(expr)

        elif (
            isinstance(expr, str)
            and (
                expr.startswith('gsheet://') or expr.startswith('https://docs.google.com/spreadsheets/')
            )
        ):
            import gspread
            from gspread_dataframe import get_as_dataframe
            gp = gspread.oauth()  # type: ignore
            spreadsheet = gp.open_by_url(expr)
            ws = spreadsheet.get_worksheet(0)
            df = get_as_dataframe(ws, drop_empty_rows=True, drop_empty_columns=True)
            return PandasMeta(expr_str=expr, pandas_df=df)

        elif (
            isinstance(expr, str)
            and ('.csv' in expr or expr.startswith('http'))
        ):
            try:
                result = urlparse(expr)

                if result.scheme == '' or result.scheme == 'file' or result.scheme == 'https' or result.scheme == 'http':
                    df = pd.read_csv(expr)
                    return PandasMeta(expr_str=expr, pandas_df=df)
                else:
                    logging.error(f'PythonRuntime.pandas_bind({expr}) expr is an invalid URL')
                    raise Exception(f'PythonRuntime.pandas_bind({expr}) expr is an invalid URL')
            except FileNotFoundError as _:
                raise Exception(f'PythonRuntime.pandas_bind({expr}) file or url {expr} not found')
            except Exception as _:
                return pandas_bind_with_llm(expr)

        elif isinstance(expr, list) or isinstance(expr, dict):
            df = pd.DataFrame(expr)
            return PandasMeta(expr_str=str(expr), pandas_df=df)

        elif isinstance(expr, FileContent):
            df = pd.read_csv(expr.url)
            return PandasMeta(expr_str=expr.url, pandas_df=df)

        elif isinstance(expr, str) and os.path.exists(os.path.expanduser(expr)):
            df = pd.read_csv(os.path.expanduser(expr))
            return PandasMeta(expr_str=os.path.expanduser(expr), pandas_df=df)

        else:
            return pandas_bind_with_llm(expr)

    def messages(self) -> List[Message]:
        logging.debug('messages()')
        if len(self.messages_list) == 0:
            return []

        # return [m for m in self.messages_list[:-1] if m.role() != 'system']
        return [m for m in self.messages_list if m.role() != 'system']

    def llm_bind(self, expr, func: str):
        # todo circular import if put at the top
        from llmvm.server.base_library.function_bindable import \
            FunctionBindable
        logging.debug(f'llm_bind({str(expr)[:20]}, {str(func)})')
        bindable = FunctionBindable(
            expr=expr,
            func=func,
            tools=self.tools,
            messages=[],
            lineno=inspect.currentframe().f_back.f_lineno,  # type: ignore
            expr_instantiation=inspect.currentframe().f_back.f_locals,  # type: ignore
            scope_dict=self.locals_dict,
            original_code=self.original_code,
            original_query=self.original_query,
            controller=self.controller,
            python_runtime=self,
        )
        bindable.bind(expr, func)
        return bindable

    def download(self, expr: str) -> Content:
        logging.debug(f'download({str(expr)})')

        from llmvm.server.base_library.content_downloader import \
            WebAndContentDriver
        cookies = self.locals_dict['cookies'] if 'cookies' in self.locals_dict else []

        downloader = WebAndContentDriver(cookies=cookies)
        download_params: DownloadParams = {
            'url': expr,
            'goal': self.original_query,
            'search_term': ''
        }
        return downloader.download(download=download_params)

    def search(self, expr: str, total_links_to_return: int = 3, titles_seen: List[str] = [], preferred_search_engine: str = '') -> List[Content]:
        logging.debug(f'PythonRuntime.search({str(expr)})')
        from llmvm.server.base_library.searcher import Searcher

        if isinstance(expr, User):
            expr = ObjectTransformers.transform_content_to_string(expr.message, self.controller.get_executor(), xml_wrapper=False)

        searcher = Searcher(
            expr=expr,
            controller=self.controller,
            original_code=self.original_code,
            original_query=self.original_query,
            vector_search=self.vector_search,
            total_links_to_return=total_links_to_return,
            preferred_search_engine=preferred_search_engine
        )
        results = searcher.search(titles_seen=titles_seen)
        return results

    def coerce(self, expr, type_name: Union[str, Type]) -> Any:
        if isinstance(type_name, type):
            type_name = type_name.__name__

        logging.debug(f'coerce({str(expr)[:50]}, {str(type_name)}) length of expr: {len(str(expr))}')
        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='coerce.prompt',
                    template={
                        'string': str(expr),
                        'type': str(type_name),
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='coerce.prompt',
            ),
            query='',
            original_query=self.original_query,
        )
        logging.debug('Coercing {} to {} resulted in {}'.format(expr, type_name, assistant.get_str()))
        write_client_stream(TextContent(f'Coercing {expr} to {type_name} resulted in {assistant.get_str()}\n'))

        return self.__eval_with_error_wrapper(assistant.get_str())

    def llm_call(self, expr_list: List[Any] | Any, llm_instruction: str) -> Assistant:
        logging.debug(f'llm_call({str(expr_list)[:20]}, {repr(llm_instruction)})')

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        # search returns a list of MarkdownContent objects, and the llm_call is typically
        # called with llm_call([var], ...), so we need to flatten
        expr_list = Helpers.flatten(expr_list)

        write_client_stream(TextContent(f'llm_call() calling {self.controller.get_executor().get_default_model()} with instruction: "{llm_instruction}"\n'))

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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='llm_call.prompt',
                stream_handler=get_stream_handler() or awaitable_none,
            ),
            query=llm_instruction,
            original_query=self.original_query,
        )
        write_client_stream(TextContent(f'llm_call() finished...\n\n'))
        return assistant

    def llm_list_bind(self, expr, llm_instruction: str, count: int = sys.maxsize, list_type: Type[Any] = str) -> List[Any]:
        logging.debug(f'llm_list_bind({str(expr)[:20]}, {repr(llm_instruction)}, {count}, {list_type})')
        context = expr.get_str() if isinstance(expr, Message) else str(expr)

        assistant: Assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='llm_list_bind.prompt',
                    template={
                        'goal': llm_instruction.replace('"', ''),
                        'context': context,
                        'type': list_type.__name__,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='llm_list_bind.prompt',
            ),
            query=llm_instruction,
            original_query=self.original_query,
        )

        list_result = assistant.get_str()

        # for anthropic
        if not list_result.startswith('['):
            pattern = r'\[\s*((\d+(\.\d+)?|"[^"]*"|\'[^\']*\')\s*(,\s*(\d+(\.\d+)?|"[^"]*"|\'[^\']*\')\s*)*)?\]'
            match = re.search(pattern, list_result)
            if match:
                list_result = match.group(0)

        try:
            result = cast(list, eval(list_result))
            return result[:count]
        except Exception as ex:
            logging.debug('PythonRuntime.llm_list_bind error: {}'.format(ex))
            logging.debug('PythonRuntime.llm_list_bind list_result: {}, llm_instruction'.format(list_result, llm_instruction))

            new_python_code = self.rewrite_python_error_correction(
                query=llm_instruction,
                python_code=assistant.get_str(),
                error=str(ex),
                locals_dictionary=self.locals_dict,
            )
            logging.debug('PythonRuntime.llm_list_bind rewrote Python code: {}'.format(new_python_code))
            result = cast(list, eval(new_python_code))
            if not isinstance(result, list):
                logging.error('PythonRuntime.llm_list_bind result is not a list')
                return []
            return result[:count]

    def __rewrite_answer_error_correction(
        self,
        query: str,
        python_code: str,
        error: str,
        locals_dictionary: Dict[Any, Any],
    ) -> str:
        '''
        This handles the case where an answer() is not correct and we need to
        identify the single place in the code where this may have occurred, and
        see if we can manually patch it.
        '''
        logging.debug('__rewrite_answer_error_correction()')
        dictionary = ''
        for key, value in locals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='answer_error_correction.prompt',
                    template={
                        'task': query,
                        'code': python_code,
                        'error': error,
                        'functions': '\n'.join([Helpers.get_function_description_flat(f) for f in self.tools]),
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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='answer_error_correction.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        return assistant.get_str()

    def __generate_primitive_answer(self, expr) -> Answer:
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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=512,
                prompt_name='answer_primitive.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        answer = Answer(
            result=answer_assistant.get_str(),
        )
        self.answers.append(answer)
        return answer

    def __single_shot_answer(self, expr) -> Answer:
        # if we're here, we're in the error correction mode
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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='answer.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )
        # check for comments
        if "[##]" in answer_assistant.get_str():
            answer_assistant.message = [TextContent(answer_assistant.get_str().split("[##]")[0].strip())]

        answer = Answer(
            result=answer_assistant.get_str()
        )
        self.answers.append(answer)
        return answer

    def answer(self, expr, check_answer: bool = True) -> Answer:
        logging.debug(f'PythonRuntime.answer({str(expr)[:20]})')

        # if we have a list of answers, maybe just return them.
        if isinstance(expr, list) and all([isinstance(e, Assistant) for e in expr]):
            # collapse the assistant answers and continue
            expr = cast(list[Assistant], expr)
            last = expr[-1]
            for e in expr[0:-1]:
                last.message = [TextContent(f'{last.message}\n\n{e.message}')]
            expr = last

        # this typically won't be called, except when the user is passing in
        # code directly and doesn't want the answer to be checked.
        # (i.e. the last message is actually the input to the code)
        if not check_answer:
            answer = Answer(
                result=expr
            )
            self.answers.append(answer)
            return answer

        snippet = str(expr).replace('\n', ' ')[:150]

        # let's check the answer
        if not self.answer_error_correcting:
            write_client_stream(TextContent(f'I am double checking an answer: answer("{snippet} ...")\n'))
        else:
            write_client_stream(TextContent(f'I have a new answer, double checking it: answer("{snippet} ...")\n'))

        # if the original query is referring to an image, it's because we were in tool mode
        # so this is a todo: hack to fix answers() so that it works for images
        if "I've just pasted you an image." in self.original_query:
            answer = Answer(
                result=expr
            )
            self.answers.append(answer)
            return answer

        # todo: hack for continuations
        answer = Answer(result=expr)
        self.answers.append(answer)
        return answer

    def get_last_assignment(
        self,
        code: str,
        locals_dict: Dict[str, Any],
    ) -> Optional[Tuple[str, str]]:
        ast_parsed_code_block = ast.parse(Helpers.escape_newlines_in_strings(code))
        last_stmt = ast_parsed_code_block.body[-1]

        # Check if it's an assignment
        if isinstance(last_stmt, ast.Assign):
            # Get the target (left side of the assignment)
            target = last_stmt.targets[0]

            # Check if the target is a simple variable name
            if isinstance(target, ast.Name):
                var_name = target.id

                # Get the value from local_dict if it exists
                if var_name in locals_dict:
                    return (var_name, locals_dict[var_name])
                else:
                    return None

        # If it's not an assignment or doesn't have a simple variable name target
        return None

    def rewrite_python_error_correction(
        self,
        query: str,
        python_code: str,
        error: str,
        locals_dictionary: Dict[Any, Any],
    ) -> str:
        logging.debug('rewrite_python_error_correction()')
        dictionary = ''
        for key, value in locals_dictionary.items():
            dictionary += '{} = "{}"\n'.format(key, str(value)[:128].replace('\n', ' '))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='python_error_correction.prompt',
                    template={
                        'task': query,
                        'code': python_code,
                        'error': error,
                        'functions': '\n'.join([Helpers.get_function_description_flat(f) for f in self.tools]),
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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='python_error_correction.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query
        )

        # double shot try
        try:
            _ = ast.parse(Helpers.escape_newlines_in_strings(assistant.get_str()))
            return assistant.get_str()
        except SyntaxError as ex:
            logging.debug('SyntaxError: {}'.format(ex))
            try:
                _ = self.rewrite_python_error_correction(
                    query=query,
                    python_code=assistant.get_str(),
                    error=str(ex),
                    locals_dictionary=locals_dictionary,
                )
                return assistant.get_str()
            except Exception as ex:
                logging.debug('Second exception rewriting Python code: {}'.format(ex))
                return ''

    def __eval_with_error_wrapper(
        self,
        python_code: str,
        retry_count: int = 2,
    ):
        counter = 0
        while counter < retry_count:
            try:
                return eval(python_code, self.globals_dict, self.locals_dict)
            except Exception as ex:
                logging.debug(f'Error evaluating Python code, exception raised: {ex}')
                logging.debug(f'Python code: {python_code}')
                python_code = self.rewrite(python_code, str(ex))
            counter += 1
        return None

    def __compile_and_execute(
        self,
        python_code: str,
    ) -> Dict[Any, Any]:
        def build_exception_str(tb_string):
            def extract_relevant_traceback(tb_string):
                match = re.search(r'File "<ast>",.*', tb_string, re.DOTALL)
                if match:
                    str_result = match.group(0)
                    return str_result
                return tb_string

            lines = Helpers.split_on_newline(python_code)  # python_code.split('\n')
            python_code_with_line_numbers = '\n'.join([f'{(line_counter+1):02} {line}' for line_counter, line in enumerate(lines)])
            return f'An exception occurred while parsing or executing the following Python code in the <ast> module:\n\n{python_code_with_line_numbers}\n\nThe exception was: {extract_relevant_traceback(tb_string)}\n'

        logging.debug('PythonRuntime.__compile_and_execute()')
        python_code = Helpers.escape_newlines_in_strings(python_code)

        try:
            parsed_ast = ast.parse(python_code)
        except Exception as ex:
            logging.error(f'PythonRuntime.__compile_and_execute() threw an exception while parsing:\n{python_code}\n, exception: {ex}')
            raise Exception(build_exception_str(traceback.format_exc()))

        context = AutoGlobalDict(self.globals_dict, self.locals_dict)

        compilation_result = compile(parsed_ast, filename="<ast>", mode="exec")

        try:
            exec(compilation_result, context, context)
        except Exception as ex:
            logging.error(f'PythonRuntime.__compile_and_execute() threw an exception while executing:\n{python_code}\n')
            raise Exception(build_exception_str(traceback.format_exc()))

        self.locals_dict = context
        return self.locals_dict

    def __interpret(
        self,
        python_code: str,
    ) -> Dict[Any, Any]:
        parsed_ast = ast.parse(Helpers.escape_newlines_in_strings(python_code))

        # todo: this doesn't have the globals/local dictionary merging stuff like above
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
        python_code: str,
        error: str,
    ) -> str:
        logging.debug(f'PythonRuntime.compile_error(): error {error}')
        write_client_stream(TextContent(f'Python code failed to compile or run due to the following error or exception: {error}\n'))

        # SyntaxError, or other more global error. We should rewrite the entire code.
        # function_list = [Helpers.get_function_description_flat_extra(f) for f in self.tools]
        code_prompt = \
            f'''The following Python code:

            {python_code}

            Failed to compile or run due to the following error:

            {error}

            Analyze the error, and rewrite the entire code above to fix the error.
            Do not explain yourself, do not apologize. Just emit the re-written code and only the code.
            '''

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=User(TextContent(code_prompt)),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().get_default_model(),
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )

        lines = assistant.get_str().split('\n')
        write_client_stream(TextContent(f'Re-writing Python code\n'))
        logging.debug('PythonRuntime.compile_error() Re-written Python code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def rewrite(
        self,
        python_code: str,
        error: str,
    ):
        logging.debug('rewrite()')
        # SyntaxError, or other more global error. We should rewrite the entire code.
        function_list = [Helpers.get_function_description_flat(f) for f in self.tools]
        code_prompt = \
            f'''The following code (found under "Original Code") either didn't compile, or threw an exception while executing.
            Identify the error in the code below, and re-write the code and only that code.
            The error is found under "Error" and the code is found under "Original Code".

            If there is natural language guidance in previous messages, follow it to help re-write the original code.

            Original User Query: {self.original_query}

            Error: {error}

            Original Code:

            {python_code}
            '''

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='python_tool_execution.prompt',
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
                max_prompt_len=self.controller.get_executor().max_input_tokens(),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(),
                prompt_name='python_tool_execution.prompt',
            ),
            query=self.original_query,
            original_query=self.original_query,
        )
        lines = assistant.get_str().split('\n')
        logging.debug('rewrite() Re-written Python code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return assistant.get_str()

    def run(
        self,
        python_code: str,
        original_query: str,
        messages: List[Message] = [],
        locals_dict: Dict = {}
    ) -> Dict[Any, Any]:
        self.original_code = python_code
        self.original_query = original_query
        self.messages_list = messages
        # todo: why are we running setup again here?
        # self.setup()
        self.locals_dict = locals_dict

        return self.__compile_and_execute(python_code)

    def run_continuation_passing(
        self,
        python_code: str,
        original_query: str,
        messages: List[Message] = [],
        locals_dict: Dict = {}
    ) -> Dict[Any, Any]:
        self.original_code = python_code
        self.original_query = original_query
        self.messages_list = messages
        self.locals_dict = locals_dict

        return self.__interpret(python_code)
