import asyncio
import csv
import glob
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
from typing import List, Optional, Sequence

import click
import httpx
import nest_asyncio
import pyperclip
import rich
from click import MissingParameter
from click_default_group import DefaultGroup
from httpx import ConnectError
from prompt_toolkit import PromptSession
from prompt_toolkit.keys import Keys
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer as PromptCompleter
from prompt_toolkit.completion import Completion as PromptCompletion
from prompt_toolkit.completion import WordCompleter, merge_completers
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import CodeBlock, Markdown
from rich.text import Text
from threading import Event

from llmvm.client.printing import StreamPrinter, markdown__rich_console__, print_response, print_thread, stream_response
from llmvm.client.client import LLMVMClient
from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Content, DownloadItem,
                                  ImageContent, MarkdownContent, Message,
                                  MessageModel, PdfContent, SessionThread,
                                  User)
from llmvm.client.parsing import get_string_thread_with_roles, parse_command_string, read_from_pipe, get_path_as_messages, parse_path


invoke_context = None
logging = setup_logging()
nest_asyncio.apply()

global thread_id
global current_mode
global last_thread
global pipe_event
global escape


thread_id: int = 0
current_mode = 'auto'
escape = False


def setup_named_pipe(pid = os.getpid()):
    pid = os.getpid()
    FIFO = f'/tmp/llmvm_client_pipe_{pid}'
    if not os.path.exists(FIFO):
        os.mkfifo(FIFO)
    return FIFO



pipe_path = setup_named_pipe()
pipe_event = Event()


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


def call_click_message(
    user_message: Optional[str | bytes | Message],
    id: int,
    path: List[str],
    context: List[str],
    mode: str,
    endpoint: str,
    cookies: str,
    executor: str,
    model: str,
    upload: bool = False,
    compression: str = 'auto',
    escape: bool = False,
    context_messages: list[str] = []
):
    params = {
        'message': user_message,
        'id': id,
        'path': path,
        'context': context,
        'upload': upload,
        'mode': mode,
        'endpoint': endpoint,
        'cookies': cookies,
        'executor': executor,
        'model': model,
        'compression': compression,
        'escape': escape,
        'context_messages': context_messages
    }

    with click.Context(message) as ctx:
        ctx.ensure_object(dict)
        ctx.params = {k: v for k, v in params.items() if v is not None}
        return message.invoke(ctx)


def apply_file_writes_and_diffs(message_str: str, prompt: bool = True) -> None:
    def extract_filename_and_match(markdown_block):
        # Updated pattern to handle ```diff case correctly
        pattern = r'```(\w+)?(?:\s+(?:Filename:\s*)?([^\s\n]+\.[^\s\n]+))?'
        match = re.search(pattern, markdown_block)
        if match:
            language = match.group(1) or ''
            filename = match.group(2) or ''
            full_match = match.group(0)  # The entire matched string

            # Don't return a filename for ```diff with no specified filename
            if language.lower() == 'diff' and not filename:
                return '', full_match

            return filename, full_match
        return '', None

    def extract_diff_info(markdown_block):
        # Pattern for the entire diff block
        # diff_block_pattern = r'```diff\s*([\s\S]*?)```'
        diff_block_pattern = r'```diff\s+(?:.*?)\n([\s\S]*?)```'

        # Pattern for Git-style diff
        git_pattern = r'(diff\s+--git\s+(\S+)\s+\S+)'

        # Updated pattern for standard patch format
        patch_pattern = r'---\s+(?:a/)?(\S+)'

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

        # If not Git-style, try to match standard patch format
        patch_match = re.search(patch_pattern, diff_content)
        if patch_match:
            return {
                'command': 'patch',
                'filename': patch_match.group(1),
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
                rich.print(f'File {filename} already exists')

            if prompt:
                rich.print(f'Apply diff to: {filename}? (y/n) ', end='')
                answer = input()
                if answer == 'n':
                    message_str = Helpers.after_end(message_str, '```diff', '```')
                    continue

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
                    f.write(applied_diff)
            elif diff_info['command'] == 'apply_unified_diff':
                with open(filename, 'r') as f:
                    applied_diff = Helpers.apply_unified_diff(f.read(), diff_info['diff_content'])
                with open(filename, 'w') as f:
                    f.write(applied_diff)

            message_str = Helpers.after_end(message_str, '```diff', '```')
            continue

        if extract_filename_and_match(message_str)[0]:
            filename, _ = extract_filename_and_match(message_str)
            answer = 'n' if prompt else 'y'
            if os.path.exists(filename) and prompt:
                rich.print(f'File {filename} already exists. Overwrite (y/n)? ', end='')
                answer = input()
                if answer == 'n':
                    message_str = Helpers.after_end(message_str, '```', '```')
                    continue

            if prompt and filename and not os.path.exists(filename):
                rich.print(f'File {filename} does not exist. Create (y/n)? ', end='')
                answer = input()
                if answer == 'n':
                    message_str = Helpers.after_end(message_str, '```', '```')
                    continue
            elif not prompt and filename and not os.path.exists(filename):
                answer = 'y'
                rich.print(f'File {filename} does not exist. Creating.')

            if not filename and prompt:
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


class CustomCompleter(PromptCompleter):
    def get_completions(self, document, complete_event):
        # Your logic to compute completions
        word = document.get_word_before_cursor()
        current_dir = os.getcwd()
        # get the files and directories recursively from the current directory
        filter_out = ['.git', '.venv', '.vscode', '.pytest_cache', '__pycache__']
        files_and_dirs = [os.path.relpath(f, current_dir)
                          for f in glob.glob(f'{current_dir}/**', recursive=True)
                          if f not in filter_out]

        filtered_files_and_dirs = [f for f in files_and_dirs if f.startswith(word)]

        for completion in filtered_files_and_dirs:
            yield PromptCompletion(completion, start_position=-len(word))


class Repl():
    def __init__(
        self,
    ):
        pass

    def open_editor(self, editor: str, initial_text: str) -> str:
        temp_file_name = ''
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
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
        rich.print()
        rich.print(f'Named pipe: {pipe_path}')
        rich.print('[bold]Keys:[/bold]')
        rich.print('[white](Ctrl-c or "exit" to exit, or cancel current request)[/white]')
        rich.print('[white](Ctrl-n to create a new thread)[/white]')
        rich.print('[white](Ctrl-e to open $EDITOR for multi-line User prompt)[/white]')
        rich.print('[white](Ctrl-g to open $EDITOR for full message thread editing)[/white]')
        rich.print('[white](Ctrl-r search prompt history)[/white]')
        rich.print('[white](Ctrl-y+y yank the last message to the clipboard)[/white]')
        rich.print('[white](Ctrl-y+a yank entire message thread to clipboard)[/white]')
        rich.print('[white](Ctrl-y+c yank code blocks to clipboard)[/white]')
        rich.print('[white](Ctrl-y+p paste image from clipboard into message)[/white]')
        rich.print('[white](:w filename to save the current thread to a file)[/white]')
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

        ctx = click.Context(cli)
        console = Console()
        history = FileHistory(os.path.expanduser('~/.local/share/llmvm/.repl_history'))
        kb = KeyBindings()
        current_mode = 'auto'

        @kb.add('c-e')
        def _(event):
            editor = os.environ.get('EDITOR', 'vim')
            text = self.open_editor(editor, event.app.current_buffer.text)
            event.app.current_buffer.text = text
            event.app.current_buffer.cursor_position = len(text) - 1

        @kb.add('c-y', 'y')
        async def _(event):
            if 'last_thread' in globals():
                last_thread_t: SessionThread = last_thread
                pyperclip.copy(str(last_thread_t.messages[-1].content))
                rich.print('Last message copied to clipboard.')
                self.__redraw(event)
            else:
                rich.print('No message to copy to clipboard.')
                self.__redraw(event)

        @kb.add('c-y', 'a')
        def _(event):
            if 'last_thread' in globals():
                last_thread_t: SessionThread = last_thread
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
                last_thread_t: SessionThread = last_thread
                last_message = str(last_thread_t.messages[-1].content)

                code_blocks = Helpers.extract_code_blocks(last_message)
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

            from PIL import ImageGrab # type: ignore
            try:
                im = ImageGrab.grabclipboard()
            except Exception as ex:
                im = None

            if im is not None:
                with io.BytesIO() as output:
                    im.save(output, format='PNG')  # type: ignore
                    output.seek(0)
                    raw_data = Helpers.load_resize_save(output.read(), 'PNG')

                    asyncio.run(StreamPrinter('user').display_image(raw_data))

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

        custom_style = Style.from_dict({
            'suggestion': 'bg:#888888 #444444'
        })

        commands = {
            cmd_name: ctx.command.get_command(ctx, cmd_name).get_short_help_str()  # type: ignore
            for cmd_name in ctx.command.list_commands(ctx)  # type: ignore
        }

        command_completer = WordCompleter(list(commands.keys()), ignore_case=True, display_dict=commands)
        # path_completer = PathCompleter()
        # combined_completer = merge_completers([command_completer, path_completer, custom_completer])
        custom_completer = CustomCompleter()
        combined_completer = merge_completers([custom_completer, command_completer])

        session = PromptSession(
            completer=combined_completer,
            auto_suggest=AutoSuggestFromHistory(),
            style=custom_style,
            history=history,
            enable_history_search=True,
            vi_mode=True,
            key_bindings=kb,
            complete_while_typing=True,
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
                (has_option('--mode', command))
                and (
                    ('--mode' not in tokens) or ('-o' not in tokens)
                )
            ):
                tokens = ['--mode', current_mode] + tokens
            return tokens

        command_executing = False

        while True:
            try:
                ctx = click.Context(cli)
                pipe_task = asyncio.create_task(process_pipe_messages(self))

                query = await session.prompt_async(
                    f'[{thread_id}] query>> ',
                    complete_while_typing=True,
                )

                pipe_task.cancel()

                # deal with $(...) command substitution
                if query.startswith('$$(') and query.endswith(')'):
                    command_substitution_result = Helpers.command_substitution(query)
                    console.print(Text.from_ansi(command_substitution_result))
                    continue

                if (
                    isinstance(query, str)
                    and '$(' in query
                    and ')' in query
                ):
                    # command substitution
                    query = Helpers.command_substitution(query)

                # there are a few special commands that aren't 'clickified'
                if query == 'yy':
                    # copy the last assistant message to the clipboard
                    last_thread_t: SessionThread = last_thread
                    pyperclip.copy(str(last_thread_t.messages[-1].content))
                    rich.print('Last message copied to clipboard.')
                    continue

                # save a thread
                if query.startswith(':w ') and len(query) > 3:
                    # save the current thread to a file
                    filename = query[3:]
                    last_thread_t: SessionThread = last_thread
                    thread_text = get_string_thread_with_roles(last_thread_t)
                    with open(filename, 'w') as f:
                        f.write(thread_text)
                        rich.print(f'Thread saved to {filename}')
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
                    if thread and isinstance(thread, SessionThread):
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
                    if thread and isinstance(thread, SessionThread):
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


@cli.command('mode', help='Switch between "auto", "tool", "direct" and "code" mode.')
@click.argument('mode', type=str, required=False, default='')
def mode(
    mode: str,
):
    global current_mode
    old_mode = current_mode

    if mode.startswith('"') and mode.endswith('"') or mode.startswith("'") and mode.endswith("'"):
        mode = mode[1:-1]

    if not mode:
        if current_mode == 'auto': current_mode = 'tool'
        elif current_mode == 'tool': current_mode = 'direct'
        elif current_mode == 'direct': current_mode = 'code'
        elif current_mode == 'code': current_mode = 'auto'
    elif mode == 'auto' or mode == 'tool' or mode == 'direct' or mode == 'code':
        current_mode = mode
    else:
        rich.print(f'Invalid mode: {mode}')
        return

    rich.print(f'Switching mode from {old_mode} to {current_mode}')


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
@click.argument('args', type=str, required=False, default='')
def ls(args):
    os.system(f'ls -la --color {args}')


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
            session_thread = SessionThread.model_validate(response.json())
        return session_thread

    thread = asyncio.run(cookies_helper())
    rich.print(f'Set cookies for thread [{thread.id}]')
    return thread


@cli.command('act', help='Use a prompt from awesome_prompts to set a character, actor or persona.')
@click.argument('actor', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to retrieve.')
@click.option('--mode', '-o', type=click.Choice(['direct', 'auto', 'tool'], case_sensitive=False), required=False, default='direct',
              help='mode to use "auto", "tool" or "direct". Default is "direct".')
@click.option('--executor', '-x', type=str, required=False, default=Container.get_config_variable('LLMVM_EXECUTOR', default=''),
              help='executor to use. Default is $LLMVM_EXECUTOR or LLMVM server default.')
@click.option('--model', '-m', type=str, required=False, default=Container.get_config_variable('LLMVM_MODEL', default=''),
              help='model to use. Default is $LLMVM_MODEL or LLMVM server default.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),  # type: ignore
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
@click.option('--escape', '-s', type=bool, is_flag=True, required=False, default=False)
def act(
    actor: str,
    id: int,
    mode: str,
    executor: str,
    model: str,
    endpoint: str,
    escape: bool,
):
    prompt_file = resources.files('llmvm.client') / 'awesome_prompts.csv'
    rows = []
    with open(prompt_file, 'r') as f:  # type: ignore
        reader = csv.reader(f)
        rows = list(reader)

    column_names = rows[0]
    if actor.startswith('"') and actor.endswith('"') or actor.startswith("'") and actor.endswith("'"):
        actor = actor[1:-1]

    if mode.startswith('"') and mode.endswith('"') or mode.startswith("'") and mode.endswith("'"):
        mode = mode[1:-1]

    if not actor:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(show_header=True, header_style="bold magenta")
        for column in column_names:
            table.add_column(column)
        for row in rows[1:]:  # type: ignore
            table.add_row(*row)

        console.print(table)
    else:
        prompt_result = Helpers.tfidf_similarity(actor, [row[0] + ' ' + row[1] for row in rows[1:]])

        rich.print()
        rich.print('[bold green]Setting actor mode.[/bold green]')
        rich.print()
        rich.print('Prompt: {}'.format(prompt_result))
        rich.print()

        call_click_message(prompt_result, id, [], [], mode, endpoint, '', executor, model, escape)


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
    item = DownloadItem(url=url, id=id)
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
                    objs = await stream_response(response, StreamPrinter('').write)
            await response.aclose()

            session_thread = SessionThread.model_validate(objs[-1])
            return session_thread

        except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
            if 'last_thread' not in globals():

                with click.Context(new) as ctx:
                    ctx.ensure_object(dict)
                    ctx.params['endpoint'] = endpoint
                    new.invoke(ctx)

            message = get_path_as_messages([url])[0]
            rich.print(f'Downloaded content from {url}.')

            new_thread = SessionThread(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
                messages=last_thread.messages + [MessageModel.from_message(message)],
            )
            last_thread = new_thread
            return new_thread

    thread: SessionThread = asyncio.run(download_helper())
    thread_id = thread.id
    return thread


@cli.command('search', help='perform a search on ingested content using the LLMVM search engine.')
@click.argument('query', type=str, required=False, default='')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def search(
    query: str,
    endpoint: str,
):
    results = LLMVMClient(
        api_endpoint=endpoint, default_executor_name='openai', default_model_name='', api_key='',
    ).search(query)

    console = Console()

    if len(results) == 0:
        console.print(f'No results found for query: {query}')
        return

    for result in results:
        title = result['title']
        snippet = result['snippet']
        link = result['link']
        score = result['score']

        if not link.startswith('http'):
            link = f'file:///{link}'

        console.print(f'[bold]Title: {title[0:80]}[/bold]')
        console.print(f'    [link={link}]{link}[/link]')
        if link.endswith('.py'):
            markdown_snippet = f'```python\n{snippet}\n```'
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(Markdown(markdown_snippet))
        else:
            wrapped_text = textwrap.wrap(snippet, width=console.width - 6)
            wrapped_lines = '\n'.join('    ' + line for line in wrapped_text[:4])
            console.print(f'{wrapped_lines}')
        console.print(f'    [italic]Score: {score}[/italic]')
        console.print('\n\n')


@cli.command('ingest', help='Ingest a file into the LLMVM search engine.')
@click.option('--path', '-p', callback=parse_path, required=True, multiple=True,
              help='Path to a single file, glob, or url to add to LLMVM server.')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def ingest(
    path: List[str],
    endpoint: str,
):
    files = path
    rich.print(f'Uploading {len(files)} files to {endpoint}/ingest')

    async def upload_helper():
        async with httpx.AsyncClient(timeout=400.0) as client:
            responses = []
            for filename_str in files:
                with open(filename_str, 'rb') as f:
                    file = {
                        'file': f,
                    }
                    response = await client.post(f'{endpoint}/ingest', files=file)
                    responses.append(response.text)
                    rich.print(response.text)
            return responses

    return asyncio.run(upload_helper())


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
        default_executor_name='openai',
        default_model_name='',
        api_key='',
    )

    threads = asyncio.run(llmvm_client.get_threads())

    for thread in threads:
        if len(thread.messages) > 0:
            message_content = str(thread.messages[-1].content).replace('\n', ' ')[0:75]
            rich.print(f'[{thread.id}]: {message_content}')

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
    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='openai',
        default_model_name='',
        api_key='',
    )

    if id.startswith('"') and id.endswith('"'):
        int_id = int(id[1:-1])
    else:
        int_id = int(id)

    thread = asyncio.run(llmvm_client.get_thread(int_id))
    print_thread(thread=thread)
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
    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='openai',
        default_model_name='',
        api_key='',
    )

    try:
        thread = asyncio.run(llmvm_client.get_thread(thread_id))
        print_thread(thread=thread, escape=escape)
    except Exception:
        if 'last_thread' in globals():
            rich.print('LLMVM server not available. Showing local thread:')
            print_thread(thread=last_thread, escape=escape)
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
    global last_thread
    llmvm_client = LLMVMClient(
        api_endpoint=endpoint,
        default_executor_name='openai',
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
            new_thread = SessionThread(
                id=-1,
                executor=last_thread.executor,
                model=last_thread.model,
                current_mode=last_thread.current_mode,
                cookies=last_thread.cookies,
            )
            last_thread = new_thread
        else:
           last_thread = SessionThread(id=-1)


@cli.command('message')
@click.argument('message', type=str, required=False, default='')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread ID to send message to. The default is create new thread (or use last thread if in repl mode).')
@click.option('--path', '-p', callback=parse_path, required=False, multiple=True,
              help='path to a single file, multiple files, directory of files, glob, or url to add to User message stack.')
@click.option('--context', '-t', required=False, multiple=True,
              help='a string to add as a context message to the User message stack. Use quotes \"\' .. \'\" for multi-word strings.')
@click.option('--upload', '-u', is_flag=True, required=True, default=False,
              help='upload the files to the LLMVM server. If false, LLMVM server must be run locally. Default is false.')
@click.option('--mode', '-o', type=click.Choice(['auto', 'direct', 'tool', 'code'], case_sensitive=False),
              required=False, default='auto',
              help='mode to use "auto", "tool", "code", or "direct". Default is "auto".')
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
              default='auto', help='context window compression method if the message is too large. Default is "auto".')
@click.option('--file-writes', type=bool, required=False, default=False, is_flag=True, help='automatically apply file writes and diffs')
@click.option('--temperature', type=float, required=False, default=0.0, help='temperature for the call.')
@click.option('--output_token_len', type=int, required=False, default=4096, help='maximum output tokens for the call.')
@click.option('--stop_tokens', type=str, required=False, multiple=True, help='stop tokens for the call.')
@click.option('--escape', type=bool, is_flag=True, required=False, help='escape the message content.')
@click.option('--context_messages', required=False, multiple=True, hidden=True)
def message(
    message: Optional[str | bytes | Message],
    id: int,
    path: List[str],
    context: List[str],
    upload: bool,
    mode: str,
    endpoint: str,
    cookies: str,
    executor: str,
    model: str,
    compression: str,
    file_writes: bool,
    temperature: float,
    output_token_len: int,
    stop_tokens: List[str],
    escape: bool,
    context_messages: Sequence[Message] = [],
):
    global thread_id
    global last_thread
    context_messages = list(context_messages)

    if mode == 'code' and not path:
        raise MissingParameter('path')

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

    if path:
        allowed_extensions = ['.py', '.md', 'Dockerfile', '.sh', '.txt'] if mode == 'code' else []
        context_messages = get_path_as_messages(path, upload, allowed_extensions)
        logging.debug(f'path: {path}')

    if context:
        for c in reversed(context):
            if (c.startswith('"') and c.endswith('"')) or (c.startswith("'") and c.endswith("'")):
                c = c[1:-1]
            context_messages.insert(0, User(Content(c)))

    # if we have files, but no message, grab the last file and use it as the message
    if not message and context_messages:
        message = context_messages[-1]
        context_messages = context_messages[:-1]

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
                    # asyncio.run(StreamPrinter('user').display_image(image_bytes))
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
                    tty_message = User(MarkdownContent(bytes_buffer.read().decode('utf-8', errors='ignore')))
                    if message:
                        context_messages.insert(0, tty_message)
                    else:
                        message = tty_message
                else:
                    tty_message = User(Content(bytes_buffer.read().decode('utf-8', errors='ignore')))
                    if message:
                        context_messages.insert(0, tty_message)
                    else:
                        message = tty_message

    if message:
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

        throw_if_server_down = mode == 'tool' or mode == 'code'

        # we used to do context_messages + [User(Content(message))] but this is wrong
        # because we might swap the context message to be the message and it's already a Message object
        thread_messages: List[Message] = []
        if isinstance(message, Message):
            thread_messages = context_messages + [message]
        elif isinstance(message, list) and all(isinstance(m, Message) for m in message):
            thread_messages = context_messages + message  # type: ignore
        else:
            thread_messages = context_messages + [User(Content(message))]

        llmvm_client = LLMVMClient(
            api_endpoint=endpoint,
            default_executor_name=executor,
            default_model_name=model,
            api_key='',
            throw_if_server_down=throw_if_server_down,
            default_stream_handler=StreamPrinter('').write
        )

        thread = asyncio.run(llmvm_client.call(
            thread=id,
            messages=thread_messages,
            executor_name=executor,
            model_name=model,
            temperature=temperature,
            output_token_len=output_token_len,
            stop_tokens=stop_tokens,
            mode=mode,
            compression=compression,
            cookies=cookies_list,
            stream_handler=StreamPrinter('').write,
        ))

        if not thread.messages:
            rich.print(f'No messages were returned from either the LLMVM server, or the LLM model {model}.')
            return

        if not escape: asyncio.run(StreamPrinter('').write_string('\n'))
        print_response([MessageModel.to_message(thread.messages[-1])], escape)
        if not escape: asyncio.run(StreamPrinter('').write_string('\n'))
        last_thread = thread
        thread_id = thread.id

        # apply file writes with or without prompting
        apply_file_writes_and_diffs(thread.messages[-1].to_message().message.get_str(), not file_writes)

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