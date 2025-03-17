import ast
import builtins
import traceback
import numpy as np
import datetime as dt
import inspect
import json
import pandas as pd
import os
import re
import sys
import scipy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, Any, Type, Tuple
from urllib.parse import urlparse

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers, write_client_stream, get_stream_handler
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Answer, Assistant, Content, ContentEncoder, DownloadParams, FileContent,
                                  FunctionCallMeta, LLMCall,
                                  Message, PandasMeta, TextContent,
                                  User, coerce_to, awaitable_none)
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.runtime import Runtime
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class PythonRuntimeBlockState:
    def __init__(
        self,
        python_code: str,
        runtime_state: AutoGlobalDict,
        helpers: list[Callable],
        answers: list[Answer],
    ):
        self.python_code = python_code
        self.runtime_state = runtime_state
        self.helpers = helpers
        self.answers = answers


class PythonRuntimeHost:
    def __init__(
        self,
        controller: ExecutionController,
        vector_search: VectorSearch,
        answer_error_correcting: bool = False,
        thread_id = 0
    ):
        self.controller: ExecutionController = controller
        self.vector_search = vector_search
        self.answer_error_correcting = answer_error_correcting
        self.thread_id = thread_id
        self.executed_code_blocks: list[PythonRuntimeBlockState] = []

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
                if block.startswith('python\n'):
                    block = block[7:].lstrip()
                try:
                    # ast.parse(block)
                    # syntax errors weren't allowing the model to try again and fix the error
                    ordered_blocks.append(block)
                except SyntaxError:
                    pass

        return ordered_blocks

    @staticmethod
    def get_last_assignment(
        code: str,
        runtime_state: AutoGlobalDict,
    ) -> Optional[Tuple[str, str]]:
        ast_parsed_code_block = ast.parse(Helpers.escape_newlines_in_strings(code))

        # Track only global-level variable assignments
        global_var_assignments = []

        # Process only global-level statements
        for stmt in ast_parsed_code_block.body:
            # Regular assignments
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        global_var_assignments.append(target.id)

            # With statements
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        global_var_assignments.append(item.optional_vars.id)

            # For loops
            elif isinstance(stmt, ast.For):
                if isinstance(stmt.target, ast.Name):
                    global_var_assignments.append(stmt.target.id)

            # Annotated assignments (x: int = 5)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                global_var_assignments.append(stmt.target.id)

            # Augmented assignments (x += 1)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                global_var_assignments.append(stmt.target.id)

            # Function definitions
            elif isinstance(stmt, ast.FunctionDef):
                global_var_assignments.append(stmt.name)

            # Class definitions
            elif isinstance(stmt, ast.ClassDef):
                global_var_assignments.append(stmt.name)

            # Try/except blocks with exception variables
            elif isinstance(stmt, ast.Try):
                for handler in stmt.handlers:
                    if handler.name:  # except Exception as e:
                        global_var_assignments.append(handler.name)

        # Check variables in reverse order (to get the last global assignment)
        for var_name in reversed(global_var_assignments):
            # Check locals first, then globals
            if var_name in runtime_state:
                return (var_name, runtime_state[var_name])
            elif var_name in runtime_state:
                return (var_name, runtime_state[var_name])

        return None

    @staticmethod
    def get_function_names(code_str):
        parsed = ast.parse(code_str)

        function_names = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)

        return function_names

    @staticmethod
    def fix_python_error(
        controller: ExecutionController,
        python_code: str,
        helpers: list[Callable],
        error: str,
        task_query: str,
        runtime_state: AutoGlobalDict,
    ) -> str:
        logging.debug(f'PythonRuntime.python_error_correction({python_code[0:100]}, {error[0:100]})')
        dictionary = ''

        for key, value in runtime_state.items():
            dictionary += '{} = "{}"\n'.format(key, Helpers.str_get_str(value)[:128].replace('\n', ' '))

        for key, value in runtime_state.items():
            dictionary += '{} = "{}"\n'.format(key, Helpers.str_get_str(value)[:128].replace('\n', ' '))

        assistant = controller.execute_llm_call_simple(
            llm_call=LLMCall(
                user_message=Helpers.prompt_message(
                    prompt_name='python_error_correction.prompt',
                    template={
                        'task': task_query,
                        'code': python_code,
                        'error': error,
                        'functions': '\n'.join([Helpers.get_function_description_flat(f) for f in helpers]),
                        'dictionary': dictionary,
                    },
                    user_token=controller.get_executor().user_token(),
                    assistant_token=controller.get_executor().assistant_token(),
                    scratchpad_token=controller.get_executor().scratchpad_token(),
                    append_token=controller.get_executor().append_token(),
                ),
                context_messages=[],
                executor=controller.get_executor(),
                model=controller.get_executor().default_model,
                temperature=0.0,
                max_prompt_len=controller.get_executor().max_input_tokens(),
                completion_tokens_len=controller.get_executor().max_output_tokens(),
                prompt_name='python_error_correction.prompt',
            ),
        )

        # double shot try
        try:
            _ = ast.parse(Helpers.escape_newlines_in_strings(assistant.get_str()))
            return assistant.get_str()
        except SyntaxError as ex:
            logging.debug(f'PythonRuntime.python_error_correction() SyntaxError: {ex}')
            try:
                _ = PythonRuntimeHost.fix_python_error(
                    controller=controller,
                    python_code=assistant.get_str(),
                    helpers=helpers,
                    error=str(ex),
                    task_query=task_query,
                    runtime_state=runtime_state,
                )
                return assistant.get_str()
            except Exception as ex:
                logging.debug(f'PythonRuntime.python_error_correction() Second exception rewriting Python code: {ex}')
                return ''

    @staticmethod
    def fix_python_parse_compile_error(
        controller: ExecutionController,
        python_code: str,
        error: str,
    ) -> str:
        logging.debug(f'PythonRuntime.python_compile_error(): error {error}')
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

        assistant = controller.execute_llm_call_simple(
            llm_call=LLMCall(
                user_message=User(TextContent(code_prompt)),
                context_messages=[],
                executor=controller.get_executor(),
                model=controller.get_executor().default_model,
                temperature=0.0,
                max_prompt_len=controller.get_executor().max_input_tokens(),
                completion_tokens_len=controller.get_executor().max_output_tokens(),
                prompt_name='',
            ),
        )

        lines = assistant.get_str().split('\n')
        write_client_stream(TextContent(f'PythonCompiler.python_compile_error() Re-writing Python code\n'))
        logging.debug('PythonRuntime.compile_error() Re-written Python code:')
        for line in lines:
            logging.debug(f'  {str(line)}')
        return str(assistant.message)

    def get_executed_code_blocks(self) -> list[PythonRuntimeBlockState]:
        return self.executed_code_blocks

    def compile_and_execute_code_block(
        self,
        python_code: str,
        messages_list: list[Message],
        helpers: list[Callable],
        runtime_state: AutoGlobalDict,
    ) -> list[Answer]:
        """
        Compiles and executes a code block.
        'helpers' will be updated with any new functions defined in the code block.
        'locals_dict' will be updated with any new variables defined in the code block.
        Returns a list of Answer objects defined with result() calls within the code block.
        """
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

        logging.debug('PythonRuntime.compile_and_execute()')
        python_code = Helpers.escape_newlines_in_strings(python_code)

        try:
            parsed_ast = ast.parse(python_code)
        except Exception as ex:
            logging.error(f'PythonRuntime.compile_and_execute() threw an exception while parsing:\n{python_code}\n, exception: {ex}')
            raise Exception(build_exception_str(traceback.format_exc()))

        runtime = Runtime(
            self.controller,
            self.vector_search,
            helpers,
            messages_list,
            answer_error_correcting=self.answer_error_correcting,
            runtime_state=runtime_state,
        ).setup()

        compilation_result = compile(parsed_ast, filename="<ast>", mode="exec")

        try:
            exec(compilation_result, runtime_state, runtime_state)
        except Exception as ex:
            logging.error(f'PythonRuntime.compile_and_execute() threw an exception while executing:\n{python_code}\n')
            raise Exception(build_exception_str(traceback.format_exc()))

        # add any helpers the llm defined to the passed in helpers list
        function_names = PythonRuntimeHost.get_function_names(python_code)
        callables = {name: runtime_state[name] for name in function_names if name in runtime_state and callable(runtime_state[name])}
        for helper in list(callables.values()):
            helpers.append(helper)

        # keep a list of executed code blocks and their final state
        self.executed_code_blocks.append(
            PythonRuntimeBlockState(
                python_code=python_code,
                runtime_state=runtime_state.copy(),
                helpers=helpers.copy(),
                answers=runtime.answers.copy(),
            )
        )

        return runtime.answers
