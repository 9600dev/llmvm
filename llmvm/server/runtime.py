import ast
import asyncio
import base64
import builtins
from dataclasses import asdict, dataclass
import linecache
import marshal
import math
import random
import textwrap
import types
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import inspect
import json
import pandas as pd
import numpy_financial as npf
import os
import re
import sys
import scipy
import bs4
from typing import Any, Callable, Optional, Tuple, Union, cast, Any, Type
from urllib.parse import urlparse

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream, get_stream_handler
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Answer, Assistant, AstNode, BrowserContent, Content, ContentEncoder, DownloadParams, FileContent,
                                  FunctionCallMeta, ImageContent, LLMCall, MarkdownContent, HTMLContent,
                                  Message, MessageModel, PandasMeta, PdfContent, SearchResult, Statement, TextContent, TokenCompressionMethod,
                                  User, coerce_to, awaitable_none, SessionThreadModel)
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.client.client import LLMVMClient, get_client


logging = setup_logging()


class RuntimeError(Exception):
    def __init__(self, message: str, inner_exception: Optional[Exception] = None):
        self.message = message
        self.inner_exception = inner_exception
        super().__init__(message)


class InstantiationWrapper:
    def __init__(self, wrapped_class, runtime: 'Runtime'):
        self.wrapped_class = wrapped_class
        self.runtime = runtime

    def __call__(self, *args, **kwargs):
        init_method = self.wrapped_class.__init__
        params_call = {}

        for name, param in inspect.signature(init_method).parameters.items():
            if name == 'self':
                continue

            if param.annotation and param.annotation is Runtime:
                params_call[name] = self.runtime
            elif param.annotation and param.annotation is ExecutionController:
                params_call[name] = self.runtime.controller
            elif name == 'cookies':
                cookies = self.runtime.runtime_state['cookies'] if 'cookies' in self.runtime.runtime_state else []
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
                    code_line = 'todo'

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


def m_isinstance(meta, type) -> bool:
    if isinstance(meta, FunctionCallMeta):
        return isinstance(meta.result(), type)

    return isinstance(meta, type)

@dataclass
class Todo(AstNode):
    id: int
    done: bool
    description: str
    expr_list: list[Any]

    def __str__(self):
        return f"[{'x' if self.done else ' '}] [{self.id}] {self.description}"

    def get_str(self):
        return str(self)

class Runtime:
    """
    The runtime class is responsible for:
     * Installing runtime functions and helper tools into a globals_dict for runtime code execution.
     * Providing the messages_list and thread_id for runtime use.
     * Capturing an answers into a list for all 'result()' calls within the runtime.
    """
    def __init__(
        self,
        controller: ExecutionController,
        helpers: list[Callable],
        messages_list: list[Message],
        runtime_state: AutoGlobalDict,
        answer_error_correcting: bool = False,
        thread_id = 0,
        thinking: int = 0,
    ):
        self.runtime_state: AutoGlobalDict = runtime_state
        self.controller: ExecutionController = controller
        # for read/write filesystem
        self.thread_id = thread_id
        self._helpers: list[Callable] = helpers
        self.answers: list[Answer] = []
        self.messages_list: list[Message] = messages_list
        self.answer_error_correcting = answer_error_correcting
        self.original_query = ''
        self.original_code = ''
        self.thinking = thinking

    def setup(self) -> 'Runtime':
        """
        Sets up the runtime by installing the global runtime functions and helper tools into globals_dict.
        """
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

        # libraries
        self.runtime_state['sys'] = sys
        self.runtime_state['os'] = os
        self.runtime_state['datetime'] = datetime
        self.runtime_state['numpy'] = np
        self.runtime_state['scipy'] = scipy
        self.runtime_state['scipy.stats'] = scipy.stats
        self.runtime_state['np'] = np
        self.runtime_state['pd'] = pd
        self.runtime_state['float'] = float
        self.runtime_state['int'] = int
        self.runtime_state['asyncio'] = asyncio
        self.runtime_state['npf'] = npf
        self.runtime_state['re'] = re
        self.runtime_state['math'] = math
        self.runtime_state['matplotlib'] = matplotlib
        self.runtime_state['plt'] = plt
        self.runtime_state['bs4'] = bs4
        self.runtime_state['json'] = json
        self.runtime_state['marshal'] = marshal
        self.runtime_state['base64'] = base64
        self.runtime_state['inspect'] = inspect
        self.runtime_state['types'] = types
        self.runtime_state['random'] = random

        # llmvm runtime
        self.runtime_state['add_thread'] = self.add_thread
        self.runtime_state['todo_list'] = []
        self.runtime_state['todos'] = self.todos
        self.runtime_state['create_todo'] = self.create_todo
        self.runtime_state['get_todo'] = self.get_todo
        self.runtime_state['done_todo'] = self.done_todo
        self.runtime_state['llmvm_call'] = self.llmvm_call
        self.runtime_state['delegate_task'] = self.delegate_task
        self.runtime_state['count_tokens'] = self.count_tokens
        self.runtime_state['guard'] = self.guard
        self.runtime_state['llm_bind'] = self.llm_bind
        self.runtime_state['llm_call'] = self.llm_call
        self.runtime_state['llm_list_bind'] = self.llm_list_bind
        self.runtime_state['llm_var_bind'] = self.llm_var_bind
        self.runtime_state['coerce'] = self.coerce
        self.runtime_state['read_memory'] = self.read_memory
        self.runtime_state['write_memory'] = self.write_memory
        self.runtime_state['read_memory_keys'] = self.read_memory_keys
        self.runtime_state['write_file'] = self.write_file
        self.runtime_state['read_file'] = self.read_file
        self.runtime_state['last_assistant'] = self.last_assistant
        self.runtime_state['last_user'] = self.last_user
        self.runtime_state['download'] = self.download
        self.runtime_state['pandas_bind'] = self.pandas_bind
        self.runtime_state['helpers'] = self.helpers
        self.runtime_state['locals'] = self.locals
        self.runtime_state['result'] = self.result
        self.runtime_state['print'] = self.print

        self.runtime_state['MarkdownContent'] = MarkdownContent
        self.runtime_state['HTMLContent'] = HTMLContent
        self.runtime_state['FileContent'] = FileContent
        self.runtime_state['ImageContent'] = ImageContent
        self.runtime_state['PdfContent'] = PdfContent
        self.runtime_state['BrowserContent'] = BrowserContent
        self.runtime_state['TextContent'] = TextContent

        for helper in self._helpers:
            self.install_helper(helper)

        return self

    def install_helper(self, helper: Callable):
        is_static, cls = Helpers.is_static_method(helper)
        if not is_static and cls and cls.__name__ not in self.runtime_state:
            self.runtime_state[cls.__name__] = InstantiationWrapper(cls, runtime=self)
        elif is_static:
            cls = Helpers.get_class_from_static_callable(helper)
            if cls:
                self.runtime_state[cls.__name__] = CallWrapper(self, cls)
            else:
                # a function without a self, probably defined by the runtime itself
                self.runtime_state[helper.__name__] = helper

        if helper not in self._helpers:
            self._helpers.append(helper)

    ########################
    ## llmvm runtime
    ########################
    def create_todo(self, todo_description: str, expr_list: list[Any] = []) -> Todo:
        todo = Todo(id=len(self.runtime_state['todo_list']), done=False, description=todo_description, expr_list=expr_list)
        self.runtime_state['todo_list'].append(todo)
        return todo

    def get_todo(self, id: int) -> Todo:
        if id < len(self.runtime_state['todo_list']):
            return self.runtime_state['todo_list'][id]
        else:
            raise ValueError(f"Todo with id {id} not found")

    def done_todo(self, id: int) -> None:
        if id < len(self.runtime_state['todo_list']):
            self.runtime_state['todo_list'][id].done = True
        else:
            raise ValueError(f"Todo with id {id} not found")

    def todos(self) -> str:
        return '\n'.join([str(todo) for todo in self.runtime_state['todo_list']])

    async def llmvm_call(self, context_or_content: list[Any], prompt: str) -> list[Content]:
        client: LLMVMClient = get_client(
            executor_name=self.controller.get_executor().name(),
            model_name=self.controller.get_executor().default_model
        )

        context_or_content = Helpers.flatten(context_or_content)
        messages = Helpers.flatten([self.controller.statement_to_message(context_or_content), User(TextContent(prompt))])

        default_model = self.controller.get_executor().default_model
        session_thread = await client.call(
            thread=SessionThreadModel(
                messages=[MessageModel.from_message(message) for message in messages],
                executor=self.controller.get_executor().name(),
                model=default_model,
                compression='auto',
                output_token_len=self.controller.get_executor().max_output_tokens(default_model),
                temperature=0.0,
                stop_tokens=[]
            ),
            stream_handler=get_stream_handler() or awaitable_none,
        )
        return Helpers.flatten([MessageModel.to_message(message).message for message in session_thread.messages])

    async def delegate_task(
        self,
        task_description: str,
        expr_list: list[Any],
        include_original_task: bool = True
    ) -> MarkdownContent:
        logging.debug(f'PythonRuntime.delegate({task_description}, {expr_list})')

        user_tasks = Helpers.compressed_user_messages(self.messages_list)
        user_tasks_message = """
        Here is a high level summary of the users previous task requests,
        including the current high level task you've been asked to help with.
        It may include extra instructions to help you with your sub-task below.\n\n
        """
        tasks_message = [User(TextContent(user_tasks_message + '\n\n'.join(user_tasks)))] if include_original_task else []

        conversation, _ = await self.controller.aexecute_continuation(
            messages=tasks_message + self.controller.statement_to_message(expr_list) + [User(TextContent(task_description))],
            temperature=0.0,
            model=self.controller.get_executor().default_model,
            runtime_state=self.runtime_state.copy(),
            stream_handler=get_stream_handler() or awaitable_none,
            thinking=self.thinking,
        )
        assistant = cast(Assistant, Helpers.last(lambda x: x.role() == 'assistant', conversation, default=Assistant(TextContent(''))))
        content = MarkdownContent(sequence=assistant.message)
        return content

    def pandas_bind(self, expr) -> PandasMeta:
        logging.debug(f'PythonRuntime.pandas_bind({expr})')

        def pandas_bind_with_llm(expr_str: str) -> PandasMeta:
            default_model = self.controller.get_executor().default_model
            assistant: Assistant = self.controller.execute_llm_call(
                llm_call=LLMCall(
                    user_message=Helpers.prompt_user(
                        prompt_name='pandas_bind.prompt',
                        template={},
                        user_token=self.controller.get_executor().user_token(),
                        assistant_token=self.controller.get_executor().assistant_token(),
                        scratchpad_token=self.controller.get_executor().scratchpad_token(),
                        append_token=self.controller.get_executor().append_token(),
                    ),
                    context_messages=self.controller.statement_to_message(expr),
                    executor=self.controller.get_executor(),
                    model=default_model,
                    temperature=0.0,
                    max_prompt_len=self.controller.get_executor().max_input_tokens(default_model),
                    completion_tokens_len=self.controller.get_executor().max_output_tokens(default_model),
                    prompt_name='pandas_bind.prompt',
                ),
                query=self.original_query,
                original_query=self.original_query,
            )
            return PandasMeta(expr_str=expr_str, pandas_df=pd.DataFrame(assistant.get_str()))

        if m_isinstance(expr, str) and 'FileContent' in expr:
            # sometimes the LLM generates code which is the "FileContent(...)" representation of the variable
            # rather than the actual FileContent variable
            try:
                file_content_url = re.search(r'FileContent\((.*)\)', expr).group(1)  # type: ignore
                df = pd.read_csv(file_content_url)
                return PandasMeta(expr_str=expr, pandas_df=df)
            except Exception:
                return pandas_bind_with_llm(expr)

        elif (
            m_isinstance(expr, str)
            and (
                expr.startswith('gsheet://') or expr.startswith('https://docs.google.com/spreadsheets/')
            )
        ):
            return Helpers.get_google_sheet(expr)
        elif (
            m_isinstance(expr, str)
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

        elif m_isinstance(expr, list) or m_isinstance(expr, dict):
            df = pd.DataFrame(expr)
            return PandasMeta(expr_str=str(expr), pandas_df=df)

        elif m_isinstance(expr, FileContent):
            df = pd.read_csv(expr.url)
            return PandasMeta(expr_str=expr.url, pandas_df=df)

        elif m_isinstance(expr, str) and os.path.exists(os.path.expanduser(expr)):
            df = pd.read_csv(os.path.expanduser(expr))
            return PandasMeta(expr_str=os.path.expanduser(expr), pandas_df=df)

        else:
            return pandas_bind_with_llm(expr)

    def count_tokens(self, content: list[Content] | Content | str) -> int:
        token_count = 0
        if isinstance(content, list) and Helpers.all(content, lambda x: isinstance(x, Content)):
            token_count = asyncio.run(self.controller.get_executor().count_tokens([User(content)]))
            return token_count
        elif isinstance(content, Content):
            token_count = asyncio.run(self.controller.get_executor().count_tokens([User(content)]))
            return token_count
        elif isinstance(content, str):
            token_count = asyncio.run(self.controller.get_executor().count_tokens([User([TextContent(content)])]))
            return token_count
        else:
            token_count = asyncio.run(self.controller.get_executor().count_tokens([User([TextContent(Helpers.str_get_str(content))])]))
            return token_count

    def last_assistant(self) -> list[Content]:
        logging.debug('last_assistant()')
        if len(self.messages_list) == 0:
            return []

        result: Optional[Assistant] = Helpers.last(lambda x: x.role() == 'assistant', self.messages_list)
        if result is None:
            return []
        return cast(Assistant, result).message

    def last_user(self) -> list[Content]:
        logging.debug('last_user()')
        if len(self.messages_list) == 0:
            return []

        result: Optional[User] = Helpers.last(lambda x: x.role() == 'user', self.messages_list)
        if result is None:
            return []
        return cast(User, result).message

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
            if m_isinstance(content, list):
                for c in content:
                    f.write(Helpers.str_get_str(c))
            elif isinstance(content, str):
                f.write(content)
        return True

    def read_file(self, full_path_filename: str) -> TextContent:
        memory_dir = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default='~/.local/share/llmvm/memory')

        full_path_filename_copy = full_path_filename
        full_path_filename = os.path.expanduser(full_path_filename)

        if not os.path.exists(full_path_filename) and os.path.exists(f'{memory_dir}/{self.thread_id}/{full_path_filename_copy}'):
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

            if not m_isinstance(value, list):
                value = [value]  # type: ignore

            values = []
            for content in value:
                if isinstance(content, Assistant):
                    for c in content.message:
                        values.append(c)
                elif isinstance(content, str):
                    values.append(TextContent(content))
                elif isinstance(content, FunctionCallMeta):
                    func_result = content.result()
                    if isinstance(func_result, Content):
                        values.append(func_result.get_str())
                    else:
                        values.append(TextContent(str(content.result())))
                elif isinstance(content, Content):
                    values.append(content)
                elif isinstance(content, Statement):
                    values.append(TextContent(str(content)))
                else:
                    values.append(TextContent(str(content)))

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
            helpers=self._helpers,
            messages=[],
            lineno=inspect.currentframe().f_back.f_lineno,  # type: ignore
            expr_instantiation=inspect.currentframe().f_back.f_locals,  # type: ignore
            runtime_state=self.runtime_state,
            original_code=self.original_code,
            original_query=self.original_query,
            controller=self.controller,
        )
        bindable.bind(expr, func)
        return bindable

    def add_thread(
        self,
        thread_id: Optional[int] = None,
        program_name: Optional[str] = None,
        last_message: bool = False,
    ) -> list[Content]:
        if thread_id is None and program_name is None:
            raise ValueError('add_thread() must be called with either thread_id or program_name')

        client: LLMVMClient = get_client(
            executor_name=self.controller.get_executor().name(),
            model_name=self.controller.get_executor().default_model
        )

        if thread_id:
            thread: SessionThreadModel = asyncio.run(client.get_thread(thread_id))
            if last_message:
                return MessageModel.to_message(thread.messages[-1]).message
            else:
                return Helpers.flatten([MessageModel.to_message(message).message for message in thread.messages])
        else:
            thread: SessionThreadModel = asyncio.run(client.get_program(thread_id, program_name))
            program_thread = '\n'.join([MessageModel.to_message(message).get_str() for message in thread.messages])
            program_code = Helpers.extract_program_code_block(program_thread)
            program_ast = Helpers.rewrite_late_binding(program_code)
            try:
                exec(program_ast, self.runtime_state)
            except Exception as ex:
                logging.error(f'PythonRuntime.add_thread() exception: {ex}')
                raise ex
            return [TextContent(f"{program_code}")]

    def guard(
        self,
        assertion: Callable[[], bool],
        error_message: str,
        bind_prompt: Optional[str] = None
    ) -> None:
        if assertion():
            return

        dumped_assertion = Helpers.dump_assertion(assertion)
        logging.error(f'PythonRuntime.guard() assertion failed: {dumped_assertion}')
        raise ValueError(f"A guard() test was executed and it returned False or threw an exception. The error message is: {error_message}. The assertion code and values is:\n{dumped_assertion}")

    def llm_var_bind(
        self,
        expr,
        type_name: str,
        description: str,
        default_value: Optional[object] = None
    ) -> Optional[Any]:
        if (
            expr in self.runtime_state
            and self.runtime_state[expr] is not None
            and isinstance(self.runtime_state[expr], Helpers.str_to_type(type_name))
        ):
            logging.debug(f'llm_var_bind({expr}) found in runtime_state, shortcircuiting')
            return self.runtime_state[expr]

        caller_frame   = inspect.currentframe().f_back  # type: ignore
        filename       = caller_frame.f_code.co_filename  # type: ignore
        call_lineno    = caller_frame.f_lineno  # type: ignore

        lines = [
            linecache.getline(filename, call_lineno + i)
            for i in range(1, 10 + 1)
        ]

        messages = self.messages_list
        caller_code = textwrap.dedent("".join(lines))

        PROMPT = f"""
        You are a helpful assistant that finds and returns the most likely value
        of a piece of data I'm using to bind to a local Python variable.

        I will give you the name of the varialbe, the type of the variable,
        a description of the data I'm looking for, a default value, and 10 lines of future code
        that may use the variable to give you a hint as to what kind of data I'm looking for.

        The previous messages contain the data you will use to find the value.

        Variable name: {expr}
        Type: {type_name}
        Description: {description}
        Default value: {default_value}
        Code: {caller_code}

        The previous messages contain the data you will use to find the value.

        Just return the value of the variable as something that can be eval'd directly.
        Do not return any other text. Return the "Default value" if you cannot find a value.
        """

        default_model = self.controller.get_executor().default_model
        assistant = asyncio.run(self.controller.aexecute_llm_call(
            llm_call=LLMCall(
                user_message=User(TextContent(PROMPT)),
                context_messages=messages,
                executor=self.controller.get_executor(),
                model=default_model,
                temperature=1.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(default_model),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(default_model),
                prompt_name='find_selector_or_mouse_x_y',
            ),
            query='',
            original_query='',
            compression=TokenCompressionMethod.AUTO,
        ))

        result = assistant.get_str()
        try:
            result = eval(result)
            return result
        except Exception as ex:
            logging.error(f'PythonRuntime.llm_bind_var() exception: {ex}')
            raise ex

    def download(self, expr: list[str] | list[SearchResult]) -> list[Content]:
        logging.debug(f'PythonRuntime.download({expr})')

        from llmvm.server.base_library.content_downloader import \
            WebAndContentDriver
        cookies = self.runtime_state['cookies'] if 'cookies' in self.runtime_state else []
        downloader = WebAndContentDriver(cookies=cookies)
        download_params: list[DownloadParams] = []

        if not m_isinstance(expr, list):
            expr = [expr]  # type: ignore

        expr = Helpers.flatten(expr)

        for url in expr:
            real_url = ''
            if isinstance(url, str):
                real_url = url
            elif isinstance(url, SearchResult):
                real_url = url.url
            elif 'url' in url:
                real_url = url['url']

            download_params.append({
                'url': real_url,
                'goal': self.original_query,
                'search_term': ''
            })

        if len(download_params) > 5:
            raise ValueError('Too many downloads requested, max is 5, got {}'.format(len(download_params)))

        result = asyncio.run(downloader.download_multiple_async(downloads=download_params))
        return [content for _, content in result]

    def coerce(self, expr, type_name: Union[str, Type]) -> str:
        if m_isinstance(type_name, type):
            type_name = type_name.__name__  # type: ignore

        default_model = self.controller.get_executor().default_model
        logging.debug(f'coerce({str(expr)[:50]}, {str(type_name)}) length of expr: {len(str(expr))}')
        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_user(
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
                model=default_model,
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(default_model),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(default_model),
                prompt_name='coerce.prompt',
            ),
            query='',
            original_query=self.original_query,
        )
        logging.debug('Coercing {} to {} resulted in {}'.format(expr, type_name, assistant.get_str()))
        write_client_stream(TextContent(f'Coercing {expr} to {type_name} resulted in {assistant.get_str()}. Trying to further coerce to {type_name}.\n'))

        try:
            coerce_result = eval(assistant.get_str())
            return coerce_result
        except Exception as ex:
            if isinstance(type_name, type):
                return coerce_to(assistant.get_str(), type_name)
            raise ex

    def llm_call(self, expr_list: list[Any] | Any, llm_instruction: str) -> Content:
        logging.debug(f'llm_call({str(expr_list)[:20]}, {repr(llm_instruction)})')

        if not m_isinstance(expr_list, list):
            expr_list = [expr_list]

        # search returns a list of MarkdownContent objects, and the llm_call is typically
        # called with llm_call([var], ...), so we need to flatten
        expr_list = Helpers.flatten(expr_list)

        write_client_stream(TextContent(f'llm_call() calling {self.controller.get_executor().default_model} with instruction: "{llm_instruction}"\n'))

        default_model = self.controller.get_executor().default_model
        assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_user(
                    prompt_name='llm_call.prompt',
                    template={
                        'llm_call_message': llm_instruction,
                    },
                    user_token=self.controller.get_executor().user_token(),
                    assistant_token=self.controller.get_executor().assistant_token(),
                    scratchpad_token=self.controller.get_executor().scratchpad_token(),
                    append_token=self.controller.get_executor().append_token(),
                ),
                context_messages=self.controller.statement_to_message(expr_list),
                executor=self.controller.get_executor(),
                model=default_model,
                temperature=1.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(default_model),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(default_model),
                prompt_name='llm_call.prompt',
                stream_handler=get_stream_handler() or awaitable_none,
            ),
            query=llm_instruction,
            original_query=self.original_query,
        )
        write_client_stream(TextContent(f'\nllm_call() finished...\n\n'))
        return TextContent(assistant.get_str())

    def llm_list_bind(self, expr, llm_instruction: str, count: int = sys.maxsize, list_type: Type[Any] = str) -> list[Any]:
        logging.debug(f'llm_list_bind({str(expr)[:20]}, {repr(llm_instruction)}, {count}, {list_type})')
        context = Helpers.str_get_str(expr)

        default_model = self.controller.get_executor().default_model
        assistant: Assistant = self.controller.execute_llm_call(
            llm_call=LLMCall(
                user_message=Helpers.prompt_user(
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
                model=default_model,
                temperature=0.0,
                max_prompt_len=self.controller.get_executor().max_input_tokens(default_model),
                completion_tokens_len=self.controller.get_executor().max_output_tokens(default_model),
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
        helpers = '\n'.join([Helpers.get_function_description_flat(f) for f in self._helpers])
        return helpers

    def locals(self) -> str:
        return '\n'.join([f'{key} = {Helpers.str_get_str(value)[:128]}' for key, value in self.runtime_state.items()])

    def result(self, expr, check_answer: bool = True) -> Content:
        def __result(expr) -> Content:
            if m_isinstance(expr, Content):
                return expr
            else:
                return TextContent(Helpers.str_get_str(expr))

        logging.debug(f'PythonRuntime.result({Helpers.str_get_str(expr)[:20]})')

        # if we have a list of answers, maybe just return them.
        if m_isinstance(expr, list) and all([m_isinstance(e, Assistant) for e in expr]):
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
            write_client_stream(TextContent(f'result("{snippet} ...")\n'))
        else:
            write_client_stream(TextContent(f'result("{snippet} ...")\n'))

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

def add_thread(thread_id: int, program_name: str, last_message: bool = False) -> list[Content]:
    global _runtime
    return cast(Runtime, _runtime).add_thread(thread_id, program_name, last_message)

def guard(assertion: Callable[[], bool], error_message: str, bind_prompt: Optional[str] = None) -> None:
    global _runtime
    return cast(Runtime, _runtime).guard(assertion, error_message, bind_prompt)

def llm_bind(expr, func: str):
    global _runtime
    return cast(Runtime, _runtime).llm_bind(expr, func)

def llm_call(expr_list: list, instruction: str) -> Content:
    global _runtime
    return cast(Runtime, _runtime).llm_call(expr_list, instruction)

def llm_var_bind(expr, default: object, type_name: str, description: str, default_value: Optional[object] = None) -> Any:
    global _runtime
    return cast(Runtime, _runtime).llm_var_bind(expr, type_name, description, default)

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

def download(expr: list[str] | list[SearchResult]) -> list[Content]:
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