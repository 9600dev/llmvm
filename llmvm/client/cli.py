import asyncio
import csv
import io
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from importlib import resources
from threading import Event
import types
from typing import Optional, Sequence, cast

import click
import httpx
import jsonpickle
import nest_asyncio
import pyperclip
import rich
from click import MissingParameter
from click_default_group import DefaultGroup
from httpx import ConnectError
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer as PromptCompleter
from prompt_toolkit.completion import Completion as PromptCompletion
from prompt_toolkit.completion import WordCompleter, merge_completers
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.filters import Condition
from rich.markdown import Markdown
from rich.text import Text

from llmvm.client.client import LLMVMClient
from llmvm.client.custom_completer import CustomCompleter
from llmvm.client.markdown_renderer import markdown__rich_console__
from llmvm.client.parsing import (get_path_as_messages,
                                  get_string_thread_with_roles,
                                  parse_command_string, parse_path,
                                  read_from_pipe)
from llmvm.client.printing import (ConsolePrinter, HTMLPrinter, StreamPrinter,
                                   stream_response)
from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import serialize_messages, setup_logging
from llmvm.common.objects import (Assistant, DownloadItemModel, ImageContent,
                                  MarkdownContent, Message, MessageModel,
                                  PdfContent, SessionThreadModel, TextContent, HTMLContent, TokenCompressionMethod, TokenPriceCalculator,
                                  User)

invoke_context = None
logging = setup_logging()
nest_asyncio.apply()

global thread_id
global current_mode
global last_thread
global pipe_event
global escape
global console_printer


thread_id: int = 0
current_mode = 'auto'
escape = False
console = ConsolePrinter()


def setup_named_pipe(pid = os.getpid()):
    pid = os.getpid()
    FIFO = f'/tmp/llmvm_client_pipe_{pid}'
    if not os.path.exists(FIFO):
        os.mkfifo(FIFO)
    return FIFO



pipe_path = setup_named_pipe()
pipe_event = Event()


if os.environ.get('LLMVM_SERIALIZE', '') and os.path.exists(os.path.expanduser(os.environ.get('LLMVM_SERIALIZE', ''))):
    os.remove(os.path.expanduser(os.environ.get('LLMVM_SERIALIZE', '')))


def tear_down(ctx):
    global pipe_event

    pipe_event.set()

    try:
        loop = asyncio.get_running_loop()
        tasks = asyncio.all_tasks(loop)
        # get the current task and avoid that
        current_task = asyncio.current_task()
        other_tasks = [t for t in tasks if t is not current_task]

        for task in other_tasks:
            task.cancel()

        loop.run_until_complete(asyncio.gather(*other_tasks, return_exceptions=True))
    except RuntimeError:
        pass

    current_thread = threading.current_thread()
    threads = threading.enumerate()

    for thread in threads:
        if thread is not current_thread and isinstance(thread, threading.Thread):
            thread.join(timeout=0.3)

    # todo: force an exit, because for some reason
    # piping stuff to the cli hangs the process on exit only on Anthropic
    os._exit(0)


def invoke_context_wrapper(ctx):
    global invoke_context
    invoke_context = ctx


def apply_file_writes_and_diffs(message_str: str, prompt: bool = True) -> None:
    def extract_filename_and_match(text):
        # Check if text contains both opening and closing ```
        if not (text.strip().startswith('```') and '```' in text.strip()[3:]):
            return None, None

        # Get the first line (opening block)
        first_line = text.strip().split('\n')[0]

        # Remove leading ```
        if len(first_line) <= 3:
            return None, None

        # Get everything after the ```
        content = first_line[3:].strip()

        # Split by whitespace (in case there's a filename)
        parts = content.split()

        # If it's just a language with no filename
        if len(parts) == 1:
            return parts[0], None

        # If there's a language and filename
        elif len(parts) >= 2:
            return parts[0], parts[1]

        return None, None

    def extract_diff_info(markdown_block):
        # Pattern for the entire diff block
        # diff_block_pattern = r'```diff\s*([\s\S]*?)```'
        diff_block_pattern = r'```diff\s+(?:.*?)\n([\s\S]*?)```'

        # Pattern for Git-style diff
        git_pattern = r'(diff\s+--git\s+(\S+)\s+\S+)'

        # unified diff format, no patch, no git
        # @@ -1,3 +1,3 @@  # no patch, no git, just diff content
        unified_diff_pattern = r'^@@ -(\d+),(\d+) \+(\d+),(\d+) @@'

        # no patch, no git, just diff content
        filename_pattern = r'```diff\s+(\S+)'

        # Try to match the entire diff block
        block_match = re.search(diff_block_pattern, markdown_block, re.MULTILINE)
        if not block_match:
            # If no block found, treat the entire input as potential diff content
            diff_content = markdown_block.strip()
        else:
            diff_content = block_match.group(1).strip()

        # Try to match Git-style diff
        git_match = re.search(git_pattern, diff_content)
        if git_match:
            return {
                'command': git_match.group(1),
                'filename': git_match.group(2).split('/')[-1],
                'diff_content': diff_content
            }

        unified_match = re.search(unified_diff_pattern, diff_content)
        filename_match = re.search(filename_pattern, markdown_block, re.MULTILINE)
        if unified_match and filename_match:
            return {
                'command': 'apply_unified_diff',
                'filename': filename_match.group(1),
                'diff_content': diff_content
            }

        filename_match = re.search(filename_pattern, markdown_block, re.MULTILINE)
        if filename_match:
            return {
                'command': 'apply_context_free_diff',
                'filename': filename_match.group(1),
                'diff_content': diff_content
            }

        # If no specific format matched, but we found content
        return {
            'command': '',
            'filename': '',
            'diff_content': diff_content if diff_content else ''
        }

    while '```' in message_str and '```' in message_str[message_str.index('```') + 3:]:
        diff_info = extract_diff_info(message_str)
        # ['command'] ['filename'] ['diff_content']
        if diff_info['filename']:
            filename = diff_info['filename']
            if filename.startswith('~'):
                filename = os.path.expanduser(filename)
            filename = os.path.abspath(filename)
            if os.path.exists(filename) and prompt:
                rich.print(f'Found the filename: {filename} the diff wants to patch.')

            if prompt:
                if os.path.exists(filename):
                    rich.print(f'Apply diff to: {filename}? (y/n) ', end='')
                    answer = input()
                    if answer == 'n':
                        message_str = Helpers.after_end(message_str, '```diff', '```')
                        continue
                else:
                    rich.print(f'File {filename} does not exist. Filename to apply to? (enter to break) ', end='')
                    answer = input()
                    if not answer:
                        message_str = Helpers.after_end(message_str, '```diff', '```')
                        continue
                    else:
                        filename = os.path.expanduser(answer)
            command = diff_info['command']

            if diff_info['command'] == 'patch':
                with tempfile.NamedTemporaryFile(mode='w', delete=True) as temp_file:
                    temp_file.write(diff_info['diff_content'])
                    temp_file.flush()
                    command = f'patch -u {filename} {temp_file.name}'
                    rich.print(f'Applying diff to {filename} via command {command}')
                    subprocess.run(command, shell=True)
            elif diff_info['command'] == 'apply_context_free_diff':
                with open(filename, 'r') as f:
                    applied_diff = Helpers.apply_context_free_diff(f.read(), diff_info['diff_content'])
                with open(filename, 'w') as f:
                    rich.print(f'Applying diff to {filename} via command {command}')
                    f.write(applied_diff)
            elif diff_info['command'] == 'apply_unified_diff':
                with open(filename, 'r') as f:
                    applied_diff = Helpers.apply_unified_diff(f.read(), diff_info['diff_content'])
                with open(filename, 'w') as f:
                    rich.print(f'Applying diff to {filename} via command {command}')
                    f.write(applied_diff)

            message_str = Helpers.after_end(message_str, '```diff', '```')
            continue

        if extract_filename_and_match(message_str)[1]:
            language, filename = extract_filename_and_match(message_str)
            answer = 'n' if prompt else 'y'
            if prompt and filename and os.path.exists(filename):
                rich.print(f'File {filename} already exists. Overwrite (y/n)? ', end='')
                answer = input()
                if answer == 'n':
                    message_str = Helpers.after_end(message_str, '```', '```')
                    continue

            elif prompt and filename and not os.path.exists(filename):
                rich.print(f'File {filename} does not exist. Create (y/n)? ', end='')
                answer = input()
                if answer == 'n':
                    message_str = Helpers.after_end(message_str, '```', '```')
                    continue

            elif not prompt and filename and not os.path.exists(filename):
                answer = 'y'
                rich.print(f'File {filename} does not exist. Creating.')

            elif prompt and not filename:
                rich.print(f'No filename for diff specified by LLM. Filename? ', end='')
                filename = input()
                answer = 'y'

            if answer == 'y' and filename:
                with open(os.path.abspath(os.path.expanduser(filename)), 'w') as f:
                    f.write(message_str)
                    f.flush()
            else:
                rich.print(f'File {filename} not written. Skipping.')

            message_str = Helpers.after_end(message_str, '```', '```')
            continue

        message_str = Helpers.after_end(message_str, '```', '```')


def parse_optional_int(ctx, param, value):
    if value is None:
        return -1
    elif value == '':
        return 0
    try:
        return int(value)
    except ValueError:
        raise click.BadParameter('Must be an integer')


class ThinkingParamType(click.ParamType):
    name = 'thinking'

    def convert(self, value, param, ctx):
        # First, try converting to int
        try:
            return int(value)
        except ValueError:
            pass

        # Then check if it's a valid string option
        value = value.lower()
        if value in ('low', 'medium', 'high'):
            if value == 'low':
                return 0
            elif value == 'medium':
                return 1
            else:
                return 2
        self.fail(f'{value} is not a valid thinking value. Must be an integer (Anthropic) or one of "low", "medium", "high" (OpenAI).', param, ctx)


def execute_python_in_thread(python_str: str, id: int, endpoint: str) -> None:
    async def python_helper():
        data = ''
        async with httpx.AsyncClient(timeout=400.0) as client:
            response = await client.get(f'{endpoint}/python', params={'thread_id': id, 'python_str': python_str})
            response.raise_for_status()
            data = response.json()
        return data

    data = asyncio.run(python_helper())
    data = cast(dict, data)
    if data and 'var_name' in data and 'var_value' in data and data['var_name'] and data['var_value']:
        rich.print(f"{data['var_value']}")
    if data and 'results' in data and data['results']:
        for result in data['results']:
            # answer object
            if '_result' in result:
                rich.print(result["_result"])
    elif data and 'error' in data and data['error']:
        rich.print(f"{data['error']}")


THINKING = ThinkingParamType()
multiline_enabled = [False]
python_enabled = [False]

class Repl():
    def __init__(
        self,
    ):
        pass

    def open_editor(self, editor: str, initial_text: str) -> str:
        temp_file_name = ''
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.md') as temp_file:
            temp_file.write(initial_text)
            temp_file.flush()
            temp_file_name = temp_file.name

            if 'vim' in editor or 'nvim' in editor:
                cmd = f'{editor} -c "normal G" -c "normal A" +startinsert {temp_file.name}'
                proc = subprocess.Popen(cmd, shell=True, env=os.environ)
            else:
                proc = subprocess.Popen([editor, temp_file.name], env=os.environ)

            while proc.poll() is None:
                time.sleep(0.2)

        with open(temp_file_name, 'r') as temp_file:
            temp_file.seek(0)
            edited_text = temp_file.read()
        os.remove(temp_file_name)
        return edited_text

    def open_default_editor(self, initial_text: str) -> str:
        return self.open_editor(os.environ.get('EDITOR', 'vim'), initial_text)  # type: ignore

    def help(self):
        ctx = click.Context(cli)
        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).help or ''  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        rich.print()
        rich.print('[red]Commands:[/red]')
        for key, value in commands.items():
            if key == 'message':
                key = 'message [red](default)[/red]'
            rich.print(f' [green]{key.ljust(23)}[/green]  {value}')
            for argument in [param for param in ctx.command.get_command(ctx, key).params if isinstance(param, click.Argument)]:  # type: ignore  # NOQA: E501
                rich.print(f'  ({argument.name})')
            for option in [param for param in ctx.command.get_command(ctx, key).params if isinstance(param, click.Option)]:  # type: ignore  # NOQA: E501
                rich.print(f'  {str(", ".join(option.opts)).ljust(25)} {option.help if option.help else ""}')

        rich.print()
        rich.print(f'$LLMVM_EXECUTOR: {Container.get_config_variable("LLMVM_EXECUTOR", default="(not set)")}')
        rich.print(f'$LLMVM_MODEL: {Container.get_config_variable("LLMVM_MODEL", default="(not set)")}')
        rich.print(f'$LLMVM_FULL_PROCESSING: {str(Container.get_config_variable("LLMVM_FULL_PROCESSING", default="(not set)")).lower()}')
        rich.print(f'$LLMVM_PROFILING: {str(Container.get_config_variable("LLMVM_PROFILING", default="(not set)")).lower()}')
        rich.print()
        rich.print(f'Named pipe: {pipe_path}')
        rich.print('[bold]Keys:[/bold]')
        rich.print('[white](Ctrl-c or "exit" to exit, or cancel current request)[/white]')
        rich.print('[white](Ctrl-n to create a new thread)[/white]')
        rich.print('[white](Ctrl-e to open $EDITOR for multi-line User prompt)[/white]')
        rich.print('[white](Ctrl-g to open $EDITOR for full message thread editing)[/white]')
        rich.print('[white](Ctrl-r search prompt history)[/white]')
        rich.print('[white](Ctrl-i multi-line repl toggle)[/white]')
        rich.print('[white](Ctrl-u python in current thread repl toggle)[/white]')
        rich.print('[white](Ctrl-y+y yank the last message to the clipboard)[/white]')
        rich.print('[white](Ctrl-y+a yank entire message thread to clipboard)[/white]')
        rich.print('[white](Ctrl-y+c yank code blocks to clipboard)[/white]')
        rich.print('[white](Ctrl-y+p paste image from clipboard into message)[/white]')
        rich.print('[white](yy to yank the last message to the clipboard)[/white]')
        rich.print('[white](:w filename to save the current thread to a file)[/white]')
        rich.print('[white](:wh filename to save the current thread as HTML to a file)[/white]')
        rich.print('[white](:.) to open the LLMVM memory/sandbox directory in finder.')
        rich.print('[white](:ohc open ```html block in browser)[/white]')
        rich.print('[white](:omc open last message in browser)[/white]')
        rich.print('[white](:otc open message thread in browser)[/white]')
        rich.print('[white](:cb Show all code blocks)[/white]')
        rich.print('[white](:ycb0 Copy code block 0, 1, 2... ycb for all)[/white]')
        rich.print('[white](:vcb0 $EDITOR code block 0, 1, 2... vcb for all)[/white]')
        rich.print('[white](:sym show all functions that are defined in <helpers> by the LLM)[/white]')
        rich.print('[white](:csym symbol(arg1,arg2) call the function with arguments)[/white]')
        rich.print('[white](:py switch python mode to execute python code in the current thread[/white]')
        rich.print('[white]($ single line python code to execute in the current thread)[/white]')
        rich.print('[white]($(command) to execute a shell command and capture in query)[/white]')
        rich.print('[white]($$(command) to execute a shell command and display to screen)[/white]')
        rich.print('')
        rich.print('[white](If the LLMVM server.py is not running, messages are executed directly)[/white]')
        rich.print('[white]("message" is the default command, so you can omit it)[/white]')
        rich.print()
        rich.print('[bold]I am a helpful assistant that has access to tools. Use "mode" to switch tools on and off.[/bold]')
        rich.print()

    def __redraw(self, event):
        event.app.invalidate()
        event.app.renderer.reset()
        event.app._redraw()


    async def repl(
        self,
    ):
        global thread_id
        global current_mode
        global last_thread
        global console
        global python_enabled
        global multiline_enabled

        ctx = click.Context(cli)
        history = FileHistory(os.path.expanduser('~/.local/share/llmvm/.repl_history'))
        kb = KeyBindings()
        current_mode = 'tools'

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        @kb.add('escape', 'm')
        def _(event):
            multiline_enabled[0] = not multiline_enabled[0]
            rich.print(f"Multiline mode: {'enabled' if multiline_enabled[0] else 'disabled'}. Escape to exit.")
            self.__redraw(event)

        @kb.add('escape', 'p')
        def _(event):
            python_enabled[0] = not python_enabled[0]
            rich.print('Switching python mode ' + ('on' if python_enabled[0] else 'off'))
            self.__redraw(event)

        @kb.add('c-y', 'y')
        async def _(event):
            if 'last_thread' in globals():
                last_thread_t: SessionThreadModel = last_thread
                pyperclip.copy(MessageModel.to_message(last_thread_t.messages[-1]).get_str())
                rich.print('Last message copied to clipboard.')
                self.__redraw(event)
            else:
                rich.print('No message to copy to clipboard.')
                self.__redraw(event)

        @kb.add('c-y', 'a')
        def _(event):
            if 'last_thread' in globals():
                last_thread_t: SessionThreadModel = last_thread
                whole_thread = get_string_thread_with_roles(last_thread_t)
                pyperclip.copy(str(whole_thread))
                rich.print('Thread copied to clipboard.')
                self.__redraw(event)
            else:
                rich.print("No thread to copy to clipboard.")
                self.__redraw(event)

        @kb.add('c-y', 'c')
        def _(event):
            if 'last_thread' in globals():
                code_blocks = get_code_blocks()

                if code_blocks:
                    code = '\n\n'.join(code_blocks)
                    pyperclip.copy(code)
                    rich.print('Code blocks copied to clipboard.')
                    self.__redraw(event)
                else:
                    rich.print('No code block found.')
                    self.__redraw(event)
            else:
                rich.print('No code block found.')
                self.__redraw(event)

        @kb.add('c-y', 'p')
        def _(event):
            global thread_id
            global last_thread

            from PIL import ImageGrab  # type: ignore
            try:
                im = ImageGrab.grabclipboard()
            except Exception as ex:
                im = None

            if im is not None:
                with io.BytesIO() as output:
                    im.save(output, format='PNG')  # type: ignore
                    output.seek(0)
                    raw_data = Helpers.load_resize_save(output.read(), 'PNG')

                    asyncio.run(StreamPrinter().display_image(raw_data))

                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.png', delete=False) as temp_file:
                        temp_file.write(raw_data)
                        temp_file.flush()

                        # check to see if there is text already present in the query>>
                        current_text = event.app.current_buffer.text
                        event.app.invalidate()
                        if len(current_text) <= 0:
                            event.app.current_buffer.text = f'[ImageContent({temp_file.name})] '
                            event.app.current_buffer.cursor_position = len(event.app.current_buffer.text)
                            event.app.layout.focus(event.app.current_buffer)
                        else:
                            event.app.current_buffer.text = current_text + f' [ImageContent({temp_file.name})] '
                            event.app.current_buffer.cursor_position = len(event.app.current_buffer.text)
                            event.app.layout.focus(event.app.current_buffer)
                        self.__redraw(event)
            else:
                rich.print('No image found in clipboard.')
                self.__redraw(event)

        @kb.add('c-n')
        def _(event):
            global thread_id
            llmvm_client = LLMVMClient(
                api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
                default_executor_name='openai',
                default_model_name='',
                api_key='',
            )

            thread = asyncio.run(llmvm_client.get_thread(0))
            thread_id = thread.id
            rich.print('New thread created.')
            event.app.current_buffer.text = ''
            event.app.current_buffer.cursor_position = 0
            self.__redraw(event)

        @kb.add('c-g')
        def _(event):
            global last_thread
            global thread_id
            llmvm_client = LLMVMClient(
                api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
                default_executor_name='openai',
                default_model_name='',
                api_key='',
            )

            editor = os.environ.get('EDITOR', 'vim')

            if 'thread_id' in globals() and thread_id > 0:
                try:
                    last_thread = asyncio.run(
                        llmvm_client.get_thread(thread_id)
                    )
                except Exception as ex:
                    pass

            thread_text = get_string_thread_with_roles(last_thread)

            current_text = event.app.current_buffer.text
            if len(current_text) > 0:
                thread_text = thread_text + 'User:  ' + current_text
            else:
                thread_text = thread_text + 'User:  '

            text = self.open_editor(editor, thread_text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        command_completer = WordCompleter(list(commands.keys()), ignore_case=True, display_dict=commands)
        custom_completer = CustomCompleter()
        combined_completer = merge_completers([custom_completer, command_completer])

        session = PromptSession(
            completer=combined_completer,
            auto_suggest=AutoSuggestFromHistory(),
            history=history,
            enable_history_search=True,
            vi_mode=True,
            key_bindings=kb,
            complete_while_typing=True,
            multiline=Condition(lambda: multiline_enabled[0]),
        )

        self.help()

        async def process_pipe_messages(self):
            async for message in read_from_pipe(pipe_path, pipe_event):
                session.app.current_buffer.text = message
                session.app.current_buffer.cursor_position = len(message) - 1
                # session.app.text insert_text(message)
                session.default_buffer.validate_and_handle()  # type: ignore

        def has_option(option_str, command):
            if isinstance(command, click.core.Command):
                return any(option_str in param.opts for param in command.params if isinstance(param, click.Option))
            elif isinstance(command, list):
                return any(option_str in token for token in command)
            return False

        def repl_default_option(command, tokens):
            if (
                thread_id > 0
                and (has_option('--id', command) or has_option('-i', command))
                and ('--id' not in tokens or '-i' not in tokens)
            ):
                tokens = ['--id', str(thread_id)] + tokens
            if (
                (has_option('--direct', command))
                and (
                    ('--direct' not in tokens) or ('-d' not in tokens)
                )
            ):
                if current_mode == 'direct':
                    tokens = ['--direct'] + tokens
            return tokens

        def get_blocks(block_type: str) -> list[str]:
            if 'last_thread' in globals():
                last_thread_t: SessionThreadModel = last_thread
                last_message = last_thread_t.messages[-1].to_message().get_str()
                blocks = Helpers.extract_blocks(last_message, block_type)
                return blocks
            else: return []

        def get_code_blocks() -> list[str]:
            if 'last_thread' in globals():
                last_thread_t: SessionThreadModel = last_thread
                last_message = last_thread_t.messages[-1].to_message().get_str()
                code_blocks = Helpers.extract_code_blocks(last_message)
                return code_blocks
            else:
                return []

        def get_total_tokens(session_thread: SessionThreadModel) -> int:
            if session_thread and session_thread.messages and session_thread.messages[-1].role == 'assistant':
                return session_thread.messages[-1].total_tokens
            else:
                return 0

        command_executing = False

        while True:
            try:
                global last_thread

                ctx = click.Context(cli)
                pipe_task = asyncio.create_task(process_pipe_messages(self))

                token_count = get_total_tokens(last_thread) if 'last_thread' in globals() else 0
                repl_mode = "query" if not python_enabled[0] else "python"

                repl_stats = f'[id: {thread_id}] {repl_mode}>> '
                if token_count > 0:
                    repl_stats = f'[id: {thread_id} n_toks: {token_count}] {repl_mode}>> '

                query = await session.prompt_async(
                    repl_stats,
                    complete_while_typing=True,
                    style=Style.from_dict({
                        'prompt': Container.get_config_variable('client_repl_color', default='')
                    })
                )

                pipe_task.cancel()

                if query == ':py':
                    python_enabled[0] = not python_enabled[0]
                    rich.print('Switching python mode ' + ('on' if python_enabled[0] else 'off'))
                    continue

                # deal with python single liner
                if query.startswith('$ ') and len(query) > 2:
                    rest = query.split('$ ')[1]
                    # execute the python command
                    ctx.invoke(python,
                        python_str=rest,
                        id=thread_id,
                        endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011')
                    )
                    continue

                # deal with $(...) command substitution
                if query.startswith('$$(') and query.endswith(')'):
                    command_substitution_result = Helpers.command_substitution(query)
                    console.print(Text.from_ansi(command_substitution_result))
                    continue

                # deal with $(...) command substitution sent to the LLM
                if (
                    isinstance(query, str)
                    and '$(' in query
                    and ')' in query
                ):
                    # command substitution
                    query = Helpers.command_substitution(query)

                if query.strip() == '':
                    continue

                if query == ':sym':
                    last_thread_t: SessionThreadModel = last_thread
                    result = Helpers.deserialize_locals_dict(last_thread_t.locals_dict)
                    for key, value in result.items():
                        if isinstance(value, types.FunctionType) and value.__code__.co_filename == '<ast>':
                            args = ', '.join(value.__code__.co_varnames)
                            function_def_str = f'def {value.__name__}({args})'
                            rich.print(function_def_str)
                    continue

                if query.startswith(':csym ') and len(query) > 6 and '(' in query and ')' in query:
                    last_thread_t: SessionThreadModel = last_thread
                    symbol_name = query[6:].split('(')[0]
                    call_str = query[6:]
                    print(call_str)
                    # call the symbol
                    result = Helpers.deserialize_locals_dict(last_thread_t.locals_dict)
                    for key, value in result.copy().items():
                        if (
                            isinstance(value, types.FunctionType)
                            and value.__code__.co_filename == '<ast>'
                            and value.__name__ == symbol_name
                        ):
                            eval_result = eval(call_str, globals(), result)
                            rich.print(eval_result)
                    continue

                if query.startswith(':q'):
                    # quit the assistant
                    tear_down(ctx)
                    break

                if query.startswith(':.'):
                    thread_id = last_thread.id
                    directory = Container().get_config_variable('memory_directory', 'LLMVM_MEMORY_DIRECTORY', default=f'~/.local/share/llmvm/memory/')
                    # macos only?
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    command = f'open {directory}/{thread_id}'
                    subprocess.run(command, shell=True)
                    continue

                # there are a few special commands that aren't 'clickified'
                if query == 'yy':
                    # copy the last assistant message to the clipboard
                    last_thread_t: SessionThreadModel = last_thread
                    pyperclip.copy(MessageModel.to_message(last_thread_t.messages[-1]).get_str())
                    rich.print('Last message copied to clipboard.')
                    continue

                if query == ':cb':
                    code_blocks = get_code_blocks()
                    for i in range(len(code_blocks)):
                        rich.print(f"[i] block {i}:")
                        markdown_snippet = f'```python\n{code_blocks[i][0:300]}\n```'
                        Markdown.__rich_console__ = markdown__rich_console__
                        console.print(Markdown(markdown_snippet))
                    continue

                if query == ':ycb' or (
                    query.startswith(':ycb')
                    and query[4].isdigit()
                ):
                    # find the code block to copy
                    if query == ':ycb':
                        pyperclip.copy('\n\n'.join(get_code_blocks()))
                    else:
                        pyperclip.copy(get_code_blocks()[int(query[4])])
                    continue

                if query == ':vcb' or (
                    query.startswith(':vcb')
                    and query[4].isdigit()
                ):
                    # find the code block to copy
                    if query == ':vcb':
                        result = '\n\n'.join(get_code_blocks())
                        self.open_default_editor(result)
                    else:
                        self.open_default_editor(get_code_blocks()[int(query[4])])
                    continue

                # save a thread
                if query.startswith(':w ') and len(query) > 3:
                    last_thread_t: SessionThreadModel = last_thread
                    # save the current thread to a file
                    filename = query[3:]
                    thread_text = get_string_thread_with_roles(last_thread_t)
                    with open(filename, 'w') as f:
                        f.write(thread_text)
                        rich.print(f'Thread saved to {filename}')
                    continue

                if query.startswith(':wh ') and len(query) > 4:
                    last_thread_t: SessionThreadModel = last_thread
                    # save current thread to a file with HTML formatting
                    filename = query[4:]
                    html_printer = HTMLPrinter(filename)
                    html_printer.print_messages([MessageModel.to_message(message) for message in last_thread_t.messages])
                    rich.print(f'HTML thread saved to {filename}')
                    Helpers.find_and_run_chrome(filename)
                    continue

                if query.startswith(':ohc'):
                    last_thread_t: SessionThreadModel = last_thread
                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.html', delete=False) as temp_file:
                        blocks = get_blocks('html')
                        if not blocks:
                            rich.print('No HTML blocks found.')
                            continue
                        block = blocks[-1]
                        html_printer = HTMLPrinter(temp_file.name)
                        html_printer.print(block)
                        Helpers.find_and_run_chrome(temp_file.name)
                    continue

                if query.startswith(':title '):
                    llmvm_client = LLMVMClient(
                        api_endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
                        default_executor_name='',
                        default_model_name='',
                        api_key='',
                    )
                    last_thread_t: SessionThreadModel = last_thread
                    title = query[7:]
                    asyncio.run(llmvm_client.set_thread_title(last_thread_t.id, title))
                    continue

                if query.startswith(':omc'):
                    last_thread_t: SessionThreadModel = last_thread
                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.html', delete=False) as temp_file:
                        last_message = last_thread_t.messages[-1]
                        html_printer = HTMLPrinter(temp_file.name)
                        html_printer.print(html_printer.get_str([last_message.to_message()]))
                        Helpers.find_and_run_chrome(temp_file.name)
                    continue

                if query.startswith(':otc'):
                    last_thread_t: SessionThreadModel = last_thread
                    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.html', delete=False) as temp_file:
                        html_printer = HTMLPrinter(temp_file.name)
                        html_printer.print_messages([MessageModel.to_message(message) for message in last_thread_t.messages])
                        Helpers.find_and_run_chrome(temp_file.name)
                    continue

                if python_enabled[0]:
                    # execute the python command
                    ctx.invoke(python,
                        python_str=query,
                        id=thread_id,
                        endpoint=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011')
                    )
                    continue

                # see if the first argument is a command
                args = query.split(' ')

                if args[0] in commands:
                    command = ctx.command.get_command(ctx, args[0])  # type: ignore
                    tokens = parse_command_string(query, command)
                    tokens = repl_default_option(command, tokens)

                    command_executing = True
                    thread = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))  # args[1:]))
                    })
                    command_executing = False
                    if thread and isinstance(thread, SessionThreadModel):
                        thread_id = thread.id
                else:
                    # default message command
                    command = ctx.command.get_command(ctx, 'message')  # type: ignore
                    tokens = parse_command_string(query, command)
                    tokens = repl_default_option(command, tokens)

                    command_executing = True
                    thread = command.invoke(ctx, **{
                        param.name: value
                        for param, value in zip(command.params, command.parse_args(ctx, tokens))
                    })
                    command_executing = False
                    if thread and isinstance(thread, SessionThreadModel):
                        thread_id = thread.id

            except KeyboardInterrupt:
                rich.print("\nKeyboardInterrupt")
                if command_executing:
                    command_executing = False
                    continue
                pipe_event.set()
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
                break
            except Exception:
                console.print_exception(max_frames=10)
                pipe_event.set()
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)


@click.group(
    cls=DefaultGroup,
    default='message',
    default_if_no_args=True,
)
@click.pass_context
def cli(ctx):
    global invoke_context
    global renderer

    if ctx.invoked_subcommand is None:
        ctx.invoke(invoke_context)
    elif not invoke_context:
        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        # there's a weird assumption I've made here around parsing, which I'm going to hackily fix
        for i in range(len(sys.argv)):
            if sys.argv[i] == '-t' or sys.argv[i] == '--context' and not sys.argv[i+1].startswith('"'):
                sys.argv[i+1] = f'"{sys.argv[i+1]}"'

        query = ' '.join(sys.argv[1:])
        args = query.split(' ')

        # see if the first argument is a command
        if args[0] in commands:
            command = ctx.command.get_command(ctx, args[0])  # type: ignore
            tokens = parse_command_string(query, command)

            command.invoke(ctx, **{
                param.name: value
                for param, value in zip(command.params, command.parse_args(ctx, tokens))  # args[1:]))
            })
            tear_down(ctx)
        else:
            # default message command
            command = ctx.command.get_command(ctx, 'message')  # type: ignore
            tokens = parse_command_string(query, command)

            command.invoke(ctx, **{
                param.name: value
                for param, value in zip(command.params, command.parse_args(ctx, tokens))
            })
            tear_down(ctx)


@cli.command('status')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def status(
    endpoint: str,
):
    # don't need an executor or model name set for status
    llmvm_client = LLMVMClient(
        api_endpoint=endpoint, default_executor_name='openai', default_model_name='', api_key='',
    )
    rich.print(asyncio.run(llmvm_client.status()))


@cli.command('mode', help='Switch between "tools" and "direct" mode. Direct avoids using LLMVM tools.')
@click.argument('mode', type=str, required=False, default='')
def mode(
    mode: str,
):
    global current_mode
    old_mode = current_mode

    if mode.startswith('"') and mode.endswith('"') or mode.startswith("'") and mode.endswith("'"):
        mode = mode[1:-1]

    if not mode:
        if current_mode == 'tools': current_mode = 'direct'
        elif current_mode == 'direct': current_mode = 'tools'
    elif mode == 'tools' or mode == 'direct':
        current_mode = mode
    else:
        rich.print(f'Invalid mode: {mode}')
        return

    rich.print(f'Switching mode from {old_mode} to {current_mode}')


@cli.command('python', help='Execute python code in the current thread.')
@click.argument('python_str', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0)
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'))
def python(
    python_str: str,
    id: int,
    endpoint: str,
):
    global last_thread
    global thread_id

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='anthropic',
        default_model_name='',
        api_key='',
    )
    last_thread = asyncio.run(llmvm_client.get_thread(id))
    thread_id = last_thread.id
    id = thread_id

    if id <= 0:
        raise ValueError('Thread id must be greater than 0')

    execute_python_in_thread(python_str, id, endpoint)

    last_thread = asyncio.run(llmvm_client.get_thread(id))
    thread_id = last_thread.id


@cli.command('exit', hidden=True)
def exit():
    tear_down(None)


@cli.command('clear', hidden=True)
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


@cli.command('help', hidden=True)
def help():
    Repl().help()


@cli.command('ls', hidden=True)
@click.argument('args', nargs=-1)
def ls(args):
    os.system('ls -la --color ' + ' '.join(args).strip('"\''))


@cli.command('cd', hidden=True)
@click.argument('args', type=str, required=False, default='')
def cd(args):
    current_dir = os.getcwd()
    args = args.strip()

    if not args:
        rich.print(current_dir)
        return

    if args[0] == '"' and args[-1] == '"':
        args = args[1:-1]
    if args[0] == '~':
        args = os.path.expanduser(args)
    full_path = os.path.join(current_dir, args)
    rich.print(f'Changing directory from {current_dir} to {full_path}')
    os.chdir(full_path)


@cli.command('todo', help='List all todo items in the specified todo file.')
@click.option('--filename', '-f', type=str, required=False,
              default=Container.get_config_variable('todo_file', 'LLMVM_TODO_FILE', default='~/.local/share/llmvm/todo.md'),
              help='location of todo markdown file.')
@click.option('--add', '-a', type=str, required=False, default='')
@click.option('--edit', '-e', is_flag=True, required=False, default=False)
@click.option('--insert', '-i', is_flag=True, required=False, default=False,
              help='insert the todo list into the current thread.')
@click.option('--extract_thread', '-t', is_flag=True, default=False,
              help='extract todo items from the current thread.')
@click.option('--extract', '-x', type=str, required=False, default='',
              help='extract todo items from supplied text')
def todo(
    filename: str,
    add: str,
    edit: bool,
    insert: bool,
    extract_thread: bool,
    extract: str = '',

):
    global thread_id
    global last_thread
    role_color = Container().get_config_variable('client_role_color', default='cyan')

    if not os.path.exists(filename):
        rich.print(f'[{role_color}]Todo file {filename} does not exist, creating.[/{role_color}]')
        with open(filename, 'w') as f:
            f.write('')

    llmvm_client = LLMVMClient(
        api_endpoint=Container().get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
        default_executor_name=Container().get_config_variable('LLMVM_EXECUTOR', default='anthropic'),
        default_model_name=Container().get_config_variable('LLMVM_MODEL', default='claude-3-7-sonnet-latest'),
        api_key='',
        throw_if_server_down=False,
    )

    if edit:
        editor = os.environ.get('EDITOR', 'vim')
        if 'vim' in editor or 'nvim' in editor:
            cmd = f'{editor} -c "normal G" -c "normal A" +startinsert {filename}'
            proc = subprocess.Popen(cmd, shell=True, env=os.environ)
        else:
            proc = subprocess.Popen([editor, filename], env=os.environ)

    elif add:
        # check to see if there is a \n at the end of the file
        with open(filename, 'r') as f:
            last_line = f.readlines()[-1]
            if not last_line.endswith('\n'):
                with open(filename, 'a') as f:
                    f.write('\n')

        with open(filename, 'a') as f:
            f.write(f'- [ ] {add}')
            rich.print(f'[{role_color}]Added todo item:[/{role_color}] {add}')

    elif insert:
        todo_list = ''
        with open(filename, 'r') as f:
            todo_list = f.read()

        if thread_id <= 0 and 'last_thread' in globals() and last_thread:
            thread = last_thread
        elif thread_id <= 0 and 'last_thread' in globals():
            thread = SessionThreadModel(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
            )
        elif thread_id <= 0:
            thread = SessionThreadModel(id=-1)
        else:
            thread = asyncio.run(llmvm_client.get_thread(thread_id))

        rich.print(f'[{role_color}]Inserting todo list into thread {thread_id}[/{role_color}]')
        thread.messages.append(MessageModel.from_message(User(TextContent(todo_list))))
        last_thread = thread
        thread_id = last_thread.id

    if extract_thread or extract:
        todo_list = open(filename, 'r').read()

        PROMPT = f"""
        You're a personal assistant that helps users with their tasks. Your job is to look at the previous messages in this
        thread and extract out all the relavent todo's and tasks, compare them to the current todo list, and then generate
        a list of new tasks/todos. If there are no new tasks or todos, just emit "NONE". If it's not obvious that there
        is really a new task or todo, just emit "NONE".

        You should emit the new items as a list in markdown TODO list format, where "-" represents a task or todo
        and "[ ]" represents that the task is not yet complete. If you're adding tasks/todos then you should always
        start them as unfinished by using "- [ ] ... task sentence". Examples:

        - [ ] this is an unfinished todo item 1 and its description
        - [ ] todo item 2 which is unfinished
        - [ ] another task or todo

        The current todo list is:
        {todo_list}

        Only emit the markdown list of new todo and task items. Do not emit anything else.
        If there are no new tasks or todos, just emit "NONE"
        """
        messages: list[Message] = []
        if extract_thread:
            # if we have a last_thread but the thread_id is 0 or -1, then we don't
            # have a connection to the server, so we'll just use the last thread
            if thread_id <= 0 and 'last_thread' in globals() and last_thread:
                thread = last_thread
            elif thread_id <= 0 and 'last_thread' in globals():
                thread = SessionThreadModel(
                    id=-1,
                    executor=last_thread.executor,
                    model=last_thread.model,
                    current_mode=last_thread.current_mode,
                    cookies=last_thread.cookies,
                )
            elif thread_id <= 0:
                thread = SessionThreadModel(id=-1)
            else:
                thread = asyncio.run(llmvm_client.get_thread(thread_id))
            messages = [MessageModel.to_message(m) for m in thread.messages]
        else:
            messages = [User(TextContent(extract))]

        assistant: Assistant = asyncio.run(llmvm_client.call_direct(
            messages=messages + [User(TextContent(PROMPT))],
        ))

        if assistant.get_str().strip() == 'NONE':
            rich.print(f'[{role_color}]No new tasks or todos found.[/{role_color}]')
            return
        else:
            with open(filename, 'a') as f:
                for line in assistant.get_str().strip().split('\n'):
                    if line.startswith('-'):
                        rich.print(f"[{role_color}]Adding new task:[/{role_color}] {line}")
                        f.write(line + '\n')

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            rich.print(line.strip())


@cli.command('cookies', help='Set cookies for a message thread so that the tooling is able to access authenticated content.')
@click.option('--file_location', '-l', type=str, required=False,
              help='location of cookies txt file in Netscape format.')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread id to attach cookies to')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def cookies(
    file_location: str,
    id: int,
    endpoint: str,
):
    global thread_id

    if id <= 0:
        id = thread_id

    cookies = []

    if file_location:
        with open(file_location, 'r') as f:
            cookies = Helpers.read_netscape_cookies(f.read())

    async def cookies_helper():
        async with httpx.AsyncClient(timeout=400.0) as client:
            response = await client.post(f'{endpoint}/v1/chat/cookies', json={'id': id, 'cookies': cookies})  # type: ignore
            session_thread = SessionThreadModel.model_validate(response.json())
        return session_thread

    thread = asyncio.run(cookies_helper())
    rich.print(f'Set cookies for thread [{thread.id}]')
    return thread


@cli.command('act', help='Use a prompt from awesome_prompts to set a character, actor or persona.')
@click.argument('actor', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to retrieve.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),  # type: ignore
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def act(
    actor: str,
    id: int,
    endpoint: str,
):
    global thread_id
    global last_thread
    global console
    thread: SessionThreadModel

    prompt_file = resources.files('llmvm.client') / 'awesome_prompts.csv'
    rows = []
    with open(prompt_file, 'r') as f:  # type: ignore
        reader = csv.reader(f)
        rows = list(reader)

    column_names = rows[0]
    if actor.startswith('"') and actor.endswith('"') or actor.startswith("'") and actor.endswith("'"):
        actor = actor[1:-1]

    if not actor:
        from rich.table import Table
        table = Table(show_header=True, header_style="bold magenta")
        for column in column_names:
            table.add_column(column)
        for row in rows[1:]:  # type: ignore
            table.add_row(*row)

        console.print(table)
    else:
        if isinstance(id, str) and id.startswith('"') and id.endswith('"'):
            int_id = int(id[1:-1])
        else:
            int_id = int(id)

        try:
            llmvm_client = LLMVMClient(
                api_endpoint=endpoint,
                default_executor_name='anthropic',
                default_model_name='',
                api_key='',
            )
        except Exception as ex:
            if int_id >=0:
                raise ex
            pass

        # if we have a last_thread but the thread_id is 0 or -1, then we don't
        # have a connection to the server, so we'll just use the last thread
        if id <= 0 and 'last_thread' in globals() and last_thread:
            thread = last_thread
        elif id <= 0 and 'last_thread' in globals():
            thread = SessionThreadModel(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
            )
        elif id <= 0:
            thread = SessionThreadModel(id=-1)
        else:
            thread = asyncio.run(llmvm_client.get_thread(int_id))

        prompt_result = Helpers.tfidf_similarity(actor, [row[0] + ' ' + row[1] for row in rows[1:]])

        rich.print()
        rich.print('[bold green]Setting actor mode by adding the actor to the current message prompt.[/bold green]')
        rich.print()
        rich.print('Prompt: {}'.format(prompt_result))
        rich.print()

        thread.messages.append(MessageModel.from_message(User(TextContent(prompt_result))))
        last_thread = thread
        thread_id = last_thread.id
        try:
            asyncio.run(llmvm_client.set_thread(last_thread))
        except Exception as ex:
            pass
        return last_thread


@cli.command('url', help='download a url and insert the content into the message thread.')
@click.argument('url', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to download and push the content to. Default is last thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def url(
    url: str,
    id: int,
    endpoint: str,
):
    item = DownloadItemModel(url=url, id=id)
    global thread_id
    global last_thread

    if url.startswith('"') and url.endswith('"') or url.startswith("'") and url.endswith("'"):
        url = url[1:-1]

    async def download_helper():
        global last_thread

        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f'{endpoint}/health')
                response.raise_for_status()

            async with httpx.AsyncClient(timeout=400.0) as client:
                async with client.stream('POST', f'{endpoint}/download', json=item.model_dump()) as response:
                    objs = await stream_response(response, StreamPrinter().write)
            await response.aclose()

            session_thread = SessionThreadModel.model_validate(objs[-1])
            return session_thread

        except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
            if 'last_thread' not in globals():

                with click.Context(new) as ctx:
                    ctx.ensure_object(dict)
                    ctx.params['endpoint'] = endpoint
                    new.invoke(ctx)

            message = get_path_as_messages([url])[0]
            rich.print(f'Downloaded content from {url}.')

            new_thread = SessionThreadModel(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
                messages=last_thread.messages + [MessageModel.from_message(message)],
            )
            last_thread = new_thread
            return new_thread

    thread: SessionThreadModel = asyncio.run(download_helper())
    thread_id = thread.id
    return thread

@cli.command('compile', help='Compile a thread into a program.')
@click.option('--id', '-i', type=str, required=False, help='thread id to compile. Default is last thread.')
@click.option('--program_name', '-p', type=str, required=False, default='',
              help='name of the program to compile. Default is the thread id.')
@click.option('--compile_instructions', '-s', type=str, required=False, default='',
              help='instructions for the compiler to use. Default is empty.')
@click.option('--executor', '-x', type=str, required=False, default=Container.get_config_variable('LLMVM_EXECUTOR', default=''),
              help='model to use. Default is $LLMVM_EXECUTOR or LLMVM server default.')
@click.option('--model', '-m', type=str, required=False, default=Container.get_config_variable('LLMVM_MODEL', default=''),
              help='model to use. Default is $LLMVM_MODEL or LLMVM server default.')
@click.option('--compression', '-c', type=click.Choice(['auto', 'lifo', 'similarity', 'mapreduce', 'summary']), required=False,
              default='lifo', help='context window compression method if the message is too large. Default is "lifo" last in first out.')
@click.option('--thinking', '-z', type=THINKING, required=False, default=0, help='enable thinking mode. Token count int for Anthropic, "low", "medium", or "high" for OpenAI.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def compile(
    id: str,
    program_name: str,
    compile_instructions: str,
    executor: str,
    model: str,
    compression: str,
    thinking: int,
    endpoint: str,
):
    global thread_id
    global last_thread

    if not id:
        id = str(thread_id)

    if id.startswith('"') and id.endswith('"'):
        int_id = int(id[1:-1])
    else:
        int_id = int(id)

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name=executor,
        default_model_name=model,
        api_key='',
    )
    session_thread: SessionThreadModel = asyncio.run(llmvm_client.compile(
        thread=int_id,
        program_name=program_name,
        compile_instructions=compile_instructions,
        executor_name=executor,
        model_name=model,
        compression=TokenCompressionMethod.from_str(compression),
        thinking=thinking,
    ))
    rich.print(f'Compiled thread {int_id} into a program {program_name}.')
    rich.print(session_thread.messages[-1].to_message().get_str())


@cli.command('programs', help='List all the compiled threads that can be used as helpers.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def programs(
    endpoint: str,
):
    global thread_id
    global last_thread

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='anthropic',
        default_model_name='',
        api_key='',
    )

    threads = asyncio.run(llmvm_client.get_threads())

    for thread in [t for t in threads if t.current_mode == 'program']:
        if len(thread.messages) > 0:
            message_content = thread.messages[-1].to_message().get_str().replace('\n', ' ')[0:75]
            title = thread.title if thread.title else ''
            rich.print(f'[{thread.id}]: {title} - {message_content}')


@cli.command('threads', help='List all message threads and set to last.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def threads(
    endpoint: str,
):
    global thread_id
    global last_thread

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='anthropic',
        default_model_name='',
        api_key='',
    )

    threads = asyncio.run(llmvm_client.get_threads())

    for thread in threads:
        if len(thread.messages) > 0:
            message_content = thread.messages[-1].to_message().get_str().replace('\n', ' ')[0:75]
            title = thread.title if thread.title else ''
            rich.print(f'[{thread.id}]: {title} - {message_content}')

    active_threads = [t for t in threads if len(t.messages) > 0]

    if thread_id == 0 and len(active_threads) > 0:
        thread_id = active_threads[-1].id
        last_thread = active_threads[-1]
        return active_threads[-1]


@cli.command('thread', help='List all message threads.')
@click.argument('id', type=str, required=True)
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def thread(
    id: str,
    endpoint: str,
):
    global thread_id
    global last_thread
    global console

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='',
        default_model_name='',
        api_key='',
    )

    if id.startswith('"') and id.endswith('"'):
        int_id = int(id[1:-1])
    else:
        int_id = int(id)

    thread = asyncio.run(llmvm_client.get_thread(int_id))
    console.print_thread(thread=thread)
    last_thread = thread
    thread_id = thread.id
    return thread


@cli.command('messages', help='List all messages in a message thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
@click.option('--escape', '-s', type=bool, is_flag=True, required=False, default=False)
def messages(
    endpoint: str,
    escape: bool,
):
    global thread_id
    global last_thread
    global console

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='anthropic',
        default_model_name='',
        api_key='',
    )

    try:
        thread = asyncio.run(llmvm_client.get_thread(thread_id))
        console.print_thread(thread=thread, escape=escape)
    except Exception:
        if 'last_thread' in globals():
            rich.print('LLMVM server not available. Showing local thread:')
            console.print_thread(thread=last_thread, escape=escape)
        else:
            rich.print('LLMVM server not available.')


@cli.command('new', help='Create a new message thread.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def new(
    endpoint: str,
):
    global thread_id
    global last_thread
    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='anthropic',
        default_model_name='',
        api_key='',
    )

    try:
        thread = asyncio.run(llmvm_client.get_thread(0))
        thread_id = thread.id
        last_thread = thread
    except ConnectError:
        rich.print('LLMVM server not available. Creating new local thread.')
        if 'last_thread' in globals():
            new_thread = SessionThreadModel(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
            )
            last_thread = new_thread
        else:
           last_thread = SessionThreadModel(id=-1)


@cli.command('message')
@click.argument('message', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to send message to. The default is create new thread (or use last thread if in repl mode).')
@click.option('--path', '-p', callback=parse_path, required=False, multiple=True,
              help='Adds a single file, multiple files, directory of files, glob, or url to the User message stack. If using a glob, put inside "" quotes.')
@click.option('--path_names', '-pn', callback=parse_path, required=False, multiple=True,
              help='Adds a single file name, multiple file names, directory of file names, glob, or url to add to User message stack.')
@click.option('--context', '-t', required=False, multiple=True,
              help='a string to add as a context message to the User message stack. Use quotes \"\' .. \'\" for multi-word strings.')
@click.option('--direct', '-d', is_flag=True, required=False, default=False,
              help='avoid using LLMVM tools and talk directly to the LLM provider. Default is false.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is $LLMVM_ENDPOINT or http://127.0.0.1:8011')
@click.option('--cookies', '-k', type=str, required=False, default=Container.get_config_variable('LLMVM_COOKIES', default=''),
              help='cookies.txt file (Netscape) for the request. Default is $LLMVM_COOKIES or empty.')
@click.option('--executor', '-x', type=str, required=False, default=Container.get_config_variable('LLMVM_EXECUTOR', default=''),
              help='model to use. Default is $LLMVM_EXECUTOR or LLMVM server default.')
@click.option('--model', '-m', type=str, required=False, default=Container.get_config_variable('LLMVM_MODEL', default=''),
              help='model to use. Default is $LLMVM_MODEL or LLMVM server default.')
@click.option('--compression', '-c', type=click.Choice(['auto', 'lifo', 'similarity', 'mapreduce', 'summary']), required=False,
              default='lifo', help='context window compression method if the message is too large. Default is "lifo" last in first out.')
@click.option('--output_token_len', '-o', type=int, required=False, default=0,
              help='maximum output tokens for the call.')
@click.option('--file-writes', type=bool, required=False, default=False, is_flag=True, help='automatically apply file writes and diffs')
@click.option('--temperature', type=float, required=False, default=0.2, help='temperature for the call.')
@click.option('--stop_tokens', type=str, required=False, multiple=True, help='stop tokens for the call.')
@click.option('--escape', type=bool, is_flag=True, required=False, help='escape the message content.')
@click.option('--throw', type=bool, is_flag=True, required=False, default=False, help='throw an exception if the LLMVM server is down. Default is false.')
@click.option('--thinking', '-z', type=THINKING, required=False, default=0, help='enable thinking mode. Token count int for Anthropic, "low", "medium", or "high" for OpenAI.')
@click.option('--context_messages', required=False, multiple=True, hidden=True)
def message(
    message: Optional[str | bytes | Message],
    id: int,
    path: list[str],
    path_names: list[str],
    context: list[str],
    direct: bool,
    endpoint: str,
    cookies: str,
    executor: str,
    model: str,
    output_token_len: int,
    compression: str,
    file_writes: bool,
    temperature: float,
    stop_tokens: list[str],
    escape: bool,
    throw: bool,
    thinking,
    context_messages: Sequence[Message] = [],
):
    global thread_id
    global last_thread
    context_messages = list(context_messages)

    console = ConsolePrinter()

    if not stop_tokens:
        stop_tokens = []

    if not escape and not sys.stdin.isatty():
        escape = True

    if model:
        if (model.startswith('"') and model.endswith('"')) or (model.startswith("'") and model.endswith("'")):
            model = model[1:-1]

    if executor:
        if (executor.startswith('"') and executor.endswith('"')) or (executor.startswith("'") and executor.endswith("'")):
            executor = executor[1:-1]

    if compression:
        if (compression.startswith('"') and compression.endswith('"')) or (compression.startswith("'") and compression.endswith("'")):  # NOQA
            compression = compression[1:-1]

    if thinking == 0 and Container().get_config_variable('LLMVM_THINKING', default=0):
        thinking = int(Container().get_config_variable('LLMVM_THINKING', default=0))

    if path_names:
        file_metadata = []
        for path_name in path_names:
            if os.path.exists(path_name):
                file_metadata.append(f"{os.path.abspath(path_name)}, filesize: {os.path.getsize(path_name)} bytes, number of tokens: {float(os.path.getsize(path_name)) * 0.75}")
            else:
                raise FileNotFoundError(f'File {path_name} not found')
        files_text = '\n'.join(file_metadata)
        if files_text:
            context_messages.append(User(TextContent(f"You have access to the following files: \n\n {files_text}")))
            logging.debug(f'path_names: {[os.path.abspath(path_name) for path_name in path_names]}')

    if path:
        context_messages.extend(get_path_as_messages(path, []))
        logging.debug(f'path: {path}')

    if context:
        for c in reversed(context):
            if (c.startswith('"') and c.endswith('"')) or (c.startswith("'") and c.endswith("'")):
                c = c[1:-1]
            context_messages.insert(0, User(TextContent(c)))

    # if we have files, but no message, grab the last file and use it as the message
    if not message and context_messages:
        message = context_messages[-1]
        context_messages = context_messages[:-1]

    # deal with output token length and thinking tokens
    if output_token_len == 0:
        output_token_len = TokenPriceCalculator().max_output_tokens(model, executor=executor, default=8192)

    if thinking > output_token_len:
        raise ValueError(f'--thinking value {thinking} is greater than --output_token_len {output_token_len}. Thinking must be less than or equal to output_token_len.')

    # input is coming from a pipe, could be binary or text
    if not sys.stdin.isatty():
        import select
        ready, _, _ = select.select([sys.stdin], [], [], 1)

        if ready:
            lines = []
            while True:
                line = sys.stdin.buffer.readline()
                if not line:
                    break
                lines.append(line)
            # file_content = sys.stdin.buffer.read()

            file_content = b''.join(lines)
            with io.BytesIO(file_content) as bytes_buffer:
                if Helpers.is_image(bytes_buffer):
                    image_bytes = Helpers.load_resize_save(bytes_buffer.getvalue(), 'PNG')

                    tty_message = User(ImageContent(image_bytes, url='cli'))
                    if message:
                        context_messages.insert(0, tty_message)
                    else:
                        message = tty_message
                elif Helpers.is_pdf(bytes_buffer):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(bytes_buffer.read())
                        temp_file.flush()
                        tty_message = User(PdfContent(bytes_buffer.read(), url=temp_file.name))
                        if message:
                            context_messages.insert(0, tty_message)
                        else:
                            message = tty_message
                elif Helpers.is_markdown(bytes_buffer):
                    tty_message = User(MarkdownContent([TextContent(bytes_buffer.read().decode('utf-8', errors='ignore'))]))
                    if message:
                        context_messages.insert(0, tty_message)
                    else:
                        message = tty_message
                else:
                    bytes_buffer_content = bytes_buffer.read().decode('utf-8', errors='ignore')
                    if bytes_buffer_content:
                        tty_message = User(TextContent(bytes_buffer_content))
                        if message:
                            context_messages.insert(0, tty_message)
                        else:
                            message = tty_message
                    else:
                        logging.debug('we are in a tty, but there is no content to read.')

    # if we don't have a message here, something went wrong.
    if not message:
        raise MissingParameter('message')

    if isinstance(message, str) and (
        (message.startswith('"') and message.endswith('"'))
        or (message.startswith("'") and message.endswith("'"))
    ):
        message = message[1:-1]

    if id <= 0:
        id = thread_id  # type: ignore

    # if we have a last_thread but the thread_id is 0 or -1, then we don't
    # have a connection to the server, so we'll just use the last thread
    if thread_id <= 0 and 'last_thread' in globals() and last_thread:
        # unless we have a full parsable thread
        # hacky as anything todo: lift this logic somehwere else as llm() does the same thing
        role_strings = ['Assistant: ', 'System: ', 'User: ']
        if isinstance(message, str) and any(role_string in message for role_string in role_strings):
            pass
        else:
            context_messages = [MessageModel.to_message(m) for m in last_thread.messages] + list(context_messages)

    cookies_list = []
    if cookies:
        with open(cookies, 'r') as f:
            cookies_list = Helpers.read_netscape_cookies(f.read())

    # we used to do context_messages + [User(Content(message))] but this is wrong
    # because we might swap the context message to be the message and it's already a Message object
    thread_messages: list[Message] = []
    if isinstance(message, Message):
        thread_messages = context_messages + [message]
    elif isinstance(message, list) and all(isinstance(m, Message) for m in message):
        thread_messages = context_messages + message  # type: ignore
    elif isinstance(message, str):
        thread_messages = cast(list[Message], context_messages + [User(TextContent(message))])
    else:
        raise ValueError('not supported')

    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name=executor,
        default_model_name=model,
        api_key='',
        throw_if_server_down=throw,
        default_stream_handler=StreamPrinter().write
    )

    thread: SessionThreadModel = SessionThreadModel()

    stream_printer = StreamPrinter()
    thread = asyncio.run(llmvm_client.call(
        thread=id,
        messages=thread_messages,
        executor_name=executor,
        model_name=model,
        temperature=temperature,
        output_token_len=output_token_len,
        stop_tokens=stop_tokens,
        mode='direct' if direct else 'tools',
        compression=TokenCompressionMethod.from_str(compression),
        cookies=cookies_list,
        thinking=thinking,
        stream_handler=stream_printer.write,
    ))

    # Finalize the stream if inline markdown is enabled
    stream_printer.finalize_stream()

    if not thread.messages:
        rich.print(f'No messages were returned from either the LLMVM server, or the LLM model {model}.')
        return

    # Skip final markdown rendering if inline rendering was used
    if not stream_printer.inline_markdown_render or not Helpers.is_callee('repl'):
        console.print_messages([MessageModel.to_message(thread.messages[-1])], escape)

    serialize_messages([MessageModel.to_message(m) for m in thread.messages])

    last_thread = thread
    thread_id = thread.id

    # apply file writes with or without prompting
    if file_writes or Helpers.is_callee('repl'):
        apply_file_writes_and_diffs(thread.messages[-1].to_message().get_str(), not file_writes)

    return thread


if __name__ == '__main__':
    # special case the hijacking of --help
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        Repl().help()
        if os.path.exists(pipe_path): os.unlink(pipe_path)
        sys.exit(0)

    if len(sys.argv) <= 1 and sys.stdin.isatty():
        repl_inst = Repl()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(repl_inst.repl())
    else:
        cli()