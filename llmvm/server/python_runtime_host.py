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
from llmvm.server.runtime import Runtime
from llmvm.server.tools.edgar import EdgarHelpers
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.server.tools.market import MarketHelpers
from llmvm.server.tools.webhelpers import WebHelpers
from llmvm.server.vector_search import VectorSearch

logging = setup_logging()


class PythonRuntimeHost:
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
        self.original_query = ''
        self.original_code = ''
        self.controller: ExecutionController = controller
        self.vector_search = vector_search
        self.tools: list[Callable] = tools
        self.locals_dict = locals_dict
        self.globals_dict = globals_dict
        self.answers: List[Answer] = []
        self.messages_list: List[Message] = []
        self.answer_error_correcting = answer_error_correcting
        self.thread_id = thread_id
        self.runtime = Runtime(
            controller,
            vector_search,
            tools,
            answer_error_correcting=answer_error_correcting,
            locals_dict=locals_dict,
            globals_dict=globals_dict
        )
        self.runtime.setup()

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
        locals_dict: Dict[Any, Any],
    ) -> str:
        logging.debug(f'PythonRuntime.python_error_correction({python_code[0:100]}, {error[0:100]})')
        dictionary = ''
        for key, value in locals_dict.items():
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
                    locals_dict=locals_dict,
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

    def compile_and_execute(
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

        logging.debug('PythonRuntime.compile_and_execute()')
        python_code = Helpers.escape_newlines_in_strings(python_code)

        try:
            parsed_ast = ast.parse(python_code)
        except Exception as ex:
            logging.error(f'PythonRuntime.compile_and_execute() threw an exception while parsing:\n{python_code}\n, exception: {ex}')
            raise Exception(build_exception_str(traceback.format_exc()))

        context = AutoGlobalDict(self.globals_dict, self.locals_dict)

        compilation_result = compile(parsed_ast, filename="<ast>", mode="exec")

        try:
            exec(compilation_result, context, context)
        except Exception as ex:
            logging.error(f'PythonRuntime.compile_and_execute() threw an exception while executing:\n{python_code}\n')
            raise Exception(build_exception_str(traceback.format_exc()))

        self.locals_dict = context
        # add any helpers the llm defined to the tools list
        function_names = PythonRuntimeHost.get_function_names(python_code)
        callables = {name: self.locals_dict[name] for name in function_names if name in self.locals_dict and callable(self.locals_dict[name])}
        self.tools = self.tools + list(callables.values())
        self.runtime.tools = self.runtime.tools + list(callables.values())
        return self.locals_dict

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
        self.locals_dict = locals_dict

        return self.compile_and_execute(python_code)