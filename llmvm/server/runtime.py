import ast
import builtins
import numpy as np
import datetime as dt
import inspect
import json
import pandas as pd
import os
import re
import sys
import scipy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, Any, Type
from urllib.parse import urlparse

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream, get_stream_handler
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Answer, Assistant, Content, ContentEncoder, DownloadParams, FileContent,
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


class RuntimeError(Exception):
    def __init__(self, message: str, inner_exception: Optional[Exception] = None):
        self.message = message
        self.inner_exception = inner_exception
        super().__init__(message)


class InstantiationWrapper:
    def __init__(self, wrapped_class, python_runtime: 'Runtime'):
        self.wrapped_class = wrapped_class
        self.python_runtime = python_runtime

    def __call__(self, *args, **kwargs):
        init_method = self.wrapped_class.__init__
        params_call = {}

        for name, param in inspect.signature(init_method).parameters.items():
            if name == 'self':
                continue

            if param.annotation and param.annotation is Runtime:
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


class Runtime:
    def __init__(
        self,
        controller: ExecutionController,
        vector_search: VectorSearch,
        tools: list[Callable] = [],
        answer_error_correcting: bool = False,
        locals_dict = {},
        globals_dict = {},
        thread_id = 0
    ):
        self.locals_dict = locals_dict
        self.globals_dict = globals_dict
        self.thread_id = thread_id
        self.controller: ExecutionController = controller
        self.vector_search: VectorSearch = vector_search
        self.tools: list[Callable] = tools
        self.answers: List[Answer] = []
        self.messages_list: List[Message] = []
        self.answer_error_correcting = answer_error_correcting
        self.original_query = ''
        self.original_code = ''
        self.setup()

    def __statement_to_message(self, statement: Any) -> List[Message]:
        return self.controller.statement_to_message(statement)

    def setup(self):
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

        self.globals_dict['llm_bind'] = self.llm_bind
        self.globals_dict['llm_call'] = self.llm_call
        self.globals_dict['llm_list_bind'] = self.llm_list_bind
        self.globals_dict['coerce'] = self.coerce
        self.globals_dict['read_memory'] = self.read_memory
        self.globals_dict['write_memory'] = self.write_memory
        self.globals_dict['read_memory_keys'] = self.read_memory_keys
        self.globals_dict['write_file'] = self.write_file
        self.globals_dict['read_file'] = self.read_file
        self.globals_dict['last_assistant'] = self.last_assistant
        self.globals_dict['last_user'] = self.last_user
        self.globals_dict['search'] = self.search
        self.globals_dict['download'] = self.download
        self.globals_dict['pandas_bind'] = self.pandas_bind
        self.globals_dict['helpers'] = self.helpers
        self.globals_dict['locals'] = self.locals
        self.globals_dict['result'] = self.result
        self.globals_dict['print'] = self.print

        self.globals_dict['WebHelpers'] = CallWrapper(self, WebHelpers)
        self.globals_dict['BCL'] = CallWrapper(self, BCL)
        self.globals_dict['EdgarHelpers'] = CallWrapper(self, EdgarHelpers)
        self.globals_dict['MarketHelpers'] = CallWrapper(self, MarketHelpers)
        self.globals_dict['sys'] = sys
        self.globals_dict['os'] = os
        self.globals_dict['datetime'] = dt
        self.globals_dict['numpy'] = np
        self.globals_dict['scipy'] = scipy
        self.globals_dict['np'] = np
        self.globals_dict['pd'] = pd
        self.globals_dict['float'] = float
        self.globals_dict['int'] = int

        for tool in self.tools:
            self.install_helper(tool)

        return self.globals_dict

    def install_helper(self, helper: Callable):
        is_static, cls = Helpers.is_static_method(helper)
        if not is_static and cls and cls.__name__ not in self.globals_dict:
            self.globals_dict[cls.__name__] = InstantiationWrapper(cls, python_runtime=self)
        else:
            self.globals_dict[helper.__name__] = helper

        if helper not in self.tools:
            self.tools.append(helper)

    ########################
    ## llmvm special helpers
    ########################
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
                        scratchpad_token=self.controller.get_executor().scratchpad_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=self.__statement_to_message(expr),  # type: ignore
                    executor=self.controller.get_executor(),
                    model=self.controller.get_executor().default_model,
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
            return Helpers.get_google_sheet(expr)
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
                    raise RuntimeError(f'pandas_bind({expr}) expr is an invalid URL')
            except FileNotFoundError as _:
                raise RuntimeError(f'pandas_bind({expr}) file or url {expr} not found')
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

    def last_assistant(self) -> list[Content]:
        logging.debug('last_assistant()')
        if len(self.messages_list) == 0:
            return []

        result = Helpers.last(lambda x: x.role() == 'assistant', self.messages_list)
        if result is None:
            return []
        return result.content

    def last_user(self) -> list[Content]:
        logging.debug('last_user()')
        if len(self.messages_list) == 0:
            return []

        result = Helpers.last(lambda x: x.role() == 'user', self.messages_list)
        if result is None:
            return []
        return result.content

    def write_file(self, filename: str, content: list[Content] | str) -> bool:
        if os.path.basename(filename) != filename:
            logging.error(f'PythonRuntime.write_file() filename must be a basename, not a full path. filename: {filename}')
            raise RuntimeError(f'write_file() filename must be a basename, not a full path. filename: {filename}')

        memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')
        if not os.path.exists(os.path.expanduser(memory_dir)):
            os.makedirs(memory_dir)

        if not os.path.exists(os.path.expanduser(memory_dir) + f'/{self.thread_id}'):
            os.makedirs(os.path.expanduser(memory_dir) + f'/{self.thread_id}')

        with open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{filename}', 'w') as f:
            if isinstance(content, list):
                for c in content:
                    f.write(f'{c.get_str()}\n')
            elif isinstance(content, str):
                f.write(content)
        return True

    def read_file(self, full_path_filename: str) -> TextContent:
        memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')

        if not os.path.exists(full_path_filename) and os.path.exists(f'{memory_dir}/{self.thread_id}/{full_path_filename}'):
            with open(f'{memory_dir}/{self.thread_id}/{full_path_filename}', 'r') as f:
                return TextContent(f.read(), url=f'{memory_dir}/{self.thread_id}/{full_path_filename}')

        with open(full_path_filename, 'r') as f:
            return TextContent(f.read(), url=full_path_filename)

    def write_memory(self, key: str, summary: str, value: list[Content] | str) -> bool:
        try:
            memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')
            if not os.path.exists(os.path.expanduser(memory_dir)):
                os.makedirs(memory_dir)

            if not os.path.exists(os.path.expanduser(memory_dir) + f'/{self.thread_id}'):
                os.makedirs(os.path.expanduser(memory_dir) + f'/{self.thread_id}')

            if not isinstance(value, list):
                value = [value]  # type: ignore

            values = []
            for content in value:
                if isinstance(content, Assistant):
                    for c in content.message:
                        values.append(c)
                elif isinstance(content, str):
                    values.append(TextContent(content))
                else:
                    values.append(content)

            with open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{key}.meta', 'w') as f:
                f.write(summary)

            with open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{key}.json', 'w') as f:
                f.write(json.dumps(values, cls=ContentEncoder, indent=2))

            with open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{key}.txt', 'w') as f:
                for content in values:
                    if isinstance(content, Content):
                        f.write(content.get_str())
                    else:
                        f.write(f'{content}\n')
        except Exception as e:
            logging.error(f'PythonRuntime.write_memory() exception: {e}')
            return False
        return True

    def read_memory(self, key: str) -> list[Content]:
        memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')
        if not os.path.exists(os.path.expanduser(memory_dir)):
            os.makedirs(memory_dir)

        if not os.path.exists(os.path.expanduser(memory_dir) + f'/{self.thread_id}'):
            os.makedirs(os.path.expanduser(memory_dir) + f'/{self.thread_id}')

        if not os.path.exists(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{key}.json'):
            return []

        with open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{key}.json', 'r') as f:
            result = json.loads(f.read())
            object_list = [Content.from_json(content) for content in result]
            return object_list

    def read_memory_keys(self) -> list[dict[str, str]]:
        memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')
        if not os.path.exists(os.path.expanduser(memory_dir)):
            os.makedirs(memory_dir)

        if not os.path.exists(os.path.expanduser(memory_dir) + f'/{self.thread_id}'):
            os.makedirs(os.path.expanduser(memory_dir) + f'/{self.thread_id}')

        meta = [f for f in os.listdir(os.path.expanduser(memory_dir) + f'/{self.thread_id}') if f.endswith('.meta')]
        key_summaries = [{'key': f.replace('.meta', ''), 'summary': open(os.path.expanduser(memory_dir) + f'/{self.thread_id}/{f}', 'r').read()} for f in meta]
        return key_summaries

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

    def coerce(self, expr, type_name: Union[str, Type]) -> str:
        if isinstance(type_name, type):
            type_name = type_name.__name__

        logging.debug(f'coerce({str(expr)[:50]}, {str(type_name)}) length of expr: {len(str(expr))}')
        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='coerce.prompt',
                    template={
                        'string': Helpers.str_get_str(expr),
                        'type': str(type_name),
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    scratchpad_token=self.controller.get_executor().scratchpad_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().default_model,
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
        return assistant.get_str()

    def llm_call(self, expr_list: List[Any] | Any, llm_instruction: str) -> Assistant:
        logging.debug(f'llm_call({str(expr_list)[:20]}, {repr(llm_instruction)})')

        if not isinstance(expr_list, list):
            expr_list = [expr_list]

        # search returns a list of MarkdownContent objects, and the llm_call is typically
        # called with llm_call([var], ...), so we need to flatten
        expr_list = Helpers.flatten(expr_list)

        write_client_stream(TextContent(f'llm_call() calling {self.controller.get_executor().default_model} with instruction: "{llm_instruction}"\n'))

        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='llm_call.prompt',
                    template={
                        'llm_call_message': llm_instruction,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    scratchpad_token=self.controller.get_executor().scratchpad_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=self.__statement_to_message(expr_list),
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().default_model,
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
        context = Helpers.str_get_str(expr)

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
                    scratchpad_token=self.controller.get_executor().scratchpad_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=self.controller.get_executor(),
                model=self.controller.get_executor().default_model,
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
            logging.debug(f'PythonRuntime.llm_list_bind ex: {ex} list_result: {list_result}, llm_instruction {llm_instruction}')
            raise RuntimeError(f'PythonRuntime.llm_list_bind error: {ex} list_result: {list_result}, llm_instruction {llm_instruction}', ex)

    def print(self, *args, sep=' ', end='\n', file=None, flush=False):
        logging.debug(f'PythonRuntime.print({args})')

        args_as_strings = [str(arg) for arg in args]
        result = sep.join(args_as_strings) + end

        answer = Answer(
            result=result
        )
        self.answers.append(answer)
        write_client_stream(TextContent(result))

        return answer

    def helpers(self) -> str:
        tools = '\n'.join([Helpers.get_function_description_flat(f) for f in self.tools])
        return tools

    def locals(self) -> str:
        return '\n'.join([f'{key} = {Helpers.str_get_str(value)[:128]}' for key, value in self.locals_dict.items()])

    def result(self, expr, check_answer: bool = True) -> Content:
        def __result(expr):
            if isinstance(expr, Content):
                return expr
            else:
                return TextContent(Helpers.str_get_str(expr))

        logging.debug(f'PythonRuntime.result({Helpers.str_get_str(expr)[:20]})')

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
            return __result(expr)

        snippet = Helpers.str_get_str(expr).replace('\n', ' ')[:150]

        # let's check the answer
        if not self.answer_error_correcting:
            write_client_stream(TextContent(f'I am double checking a result: result("{snippet} ...")\n'))
        else:
            write_client_stream(TextContent(f'I have a new result, double checking it: result("{snippet} ...")\n'))

        # if the original query is referring to an image, it's because we were in tool mode
        # so this is a todo: hack to fix answers() so that it works for images
        if "I've just pasted you an image." in self.original_query:
            answer = Answer(
                result=expr
            )
            self.answers.append(answer)

        # todo: hack for continuations
        answer = Answer(result=expr)
        self.answers.append(answer)
        return __result(answer)


global _runtime
_runtime = None

def install(runtime: Runtime):
    global _runtime
    _runtime = runtime

def llm_bind(expr, func: str):
    global _runtime
    return cast(Runtime, _runtime).llm_bind(expr, func)

def llm_call(expr_list: List, instruction: str) -> Assistant:
    global _runtime
    return cast(Runtime, _runtime).llm_call(expr_list, instruction)

def llm_list_bind(expr, llm_instruction: str, count: int = sys.maxsize) -> list:
    global _runtime
    return cast(Runtime, _runtime).llm_list_bind(expr, llm_instruction, count)

def coerce(expr, type_name: Union[str, Type]) -> Any:
    global _runtime
    return cast(Runtime, _runtime).coerce(expr, type_name)

def read_memory(key: str) -> list[Content]:
    global _runtime
    return cast(Runtime, _runtime).read_memory(key)

def write_memory(key: str, summary: str, value: list[Content]) -> None:
    global _runtime
    cast(Runtime, _runtime).write_memory(key, summary, value)

def read_memory_keys() -> list[dict[str, str]]:
    global _runtime
    return cast(Runtime, _runtime).read_memory_keys()

def write_file(filename: str, content: list[Content]) -> bool:
    global _runtime
    return cast(Runtime, _runtime).write_file(filename, content)

def read_file(full_path_filename: str) -> TextContent:
    global _runtime
    return cast(Runtime, _runtime).read_file(full_path_filename)

def last_assistant() -> list[Content]:
    global _runtime
    return cast(Runtime, _runtime).last_assistant()

def last_user() -> list[Content]:
    global _runtime
    return cast(Runtime, _runtime).last_user()

def search(expr: str, total_links_to_return: int = 3, titles_seen: List[str] = []) -> list[Content]:
    global _runtime
    return cast(Runtime, _runtime).search(expr, total_links_to_return, titles_seen)

def download(expr: str) -> Content:
    global _runtime
    return cast(Runtime, _runtime).download(expr)

def pandas_bind(expr) -> PandasMeta:
    global _runtime
    return cast(Runtime, _runtime).pandas_bind(expr)

def helpers() -> str:
    global _runtime
    return cast(Runtime, _runtime).helpers()

def locals() -> str:
    global _runtime
    return cast(Runtime, _runtime).locals()

def result(expr, check_answer: bool = True) -> Content:
    global _runtime
    return cast(Runtime, _runtime).result(expr, check_answer)

def print(*args, sep=' ', end='\n', file=None, flush=False):
    global _runtime
    return cast(Runtime, _runtime).print(*args, sep, end, file, flush)