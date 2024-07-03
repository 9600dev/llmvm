import asyncio
import base64
import csv
import glob
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from importlib import resources
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
from urllib.parse import urlparse

import async_timeout
import click
import httpx
import jsonpickle
import nest_asyncio
import pyperclip
import requests
import rich
from click import MissingParameter
from click_default_group import DefaultGroup
from httpx import ConnectError
from PIL import Image
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer as PromptCompleter
from prompt_toolkit.completion import Completion as PromptCompletion
from prompt_toolkit.completion import WordCompleter, merge_completers
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from pydantic.type_adapter import TypeAdapter
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax

from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.container import Container
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content, DownloadItem,
                                  Executor, FileContent, ImageContent, Message,
                                  MessageModel, PdfContent, SessionThread,
                                  StreamNode, System, TokenStopNode, User)
from llmvm.common.openai_executor import OpenAIExecutor

nest_asyncio.apply()

invoke_context = None
logging = setup_logging()

global thread_id
global current_mode
global last_thread
global escape


thread_id: int = 0
current_mode = 'auto'
escape = False


def parse_action(token) -> Content:
    """
    For a given [Action(...)] token, parse it into either
    ImageContent, PdfContent, or FileContent
    """
    action_type = token[1:].split('(')[0]
    action_data = Helpers.in_between(token, '(', ')')

    if action_type == 'ImageContent':
        with open(action_data, 'rb') as file_content:
            return ImageContent(file_content.read(), url=action_data)
    elif action_type == 'PdfContent':
        with open(action_data, 'rb') as file_content:
            return PdfContent(file_content.read(), url=action_data)
    elif action_type == 'FileContent':
        with open(action_data, 'r') as file_content:
            return FileContent(file_content.read().encode('utf-8'), url=action_data)
    else:
        raise ValueError('Unknown action type')

def parse_message_actions(role_type: type, message: str) -> list[Message]:
    actions = ['[ImageContent(', '[PdfContent(', '[FileContent(']
    accumulated_tokens = []
    messages = []
    # go through the message, create User() nodes for normal text content
    # and for actions, create the appropriate action node
    tokens = message.split(' ')
    for token in tokens:
        if any(token.startswith(action) for action in actions):
            if accumulated_tokens:
                messages.append(role_type(Content(' '.join(accumulated_tokens))))
            messages.append(role_type(parse_action(token)))
            accumulated_tokens = []
        elif token:
            accumulated_tokens.append(token)
    if accumulated_tokens:
        messages.append(role_type(Content(' '.join(accumulated_tokens))))
    return messages

def parse_message_thread(message: str):
    def create_message(type) -> Message:
        MessageClass = globals()[type]
        return MessageClass('')

    messages = []
    roles = ['Assistant: ', 'System: ', 'User: ']

    while any(message.startswith(role) for role in roles):
        role = next(role for role in roles if message.startswith(role))
        parsed_message = create_message(role.replace(': ', ''))
        content = Helpers.in_between_ends(message, role, roles)
        sub_messages = parse_message_actions(type(parsed_message), content)
        for sub_message in sub_messages:
            messages.append(sub_message)
        message = message[len(role) + len(content):]
    return messages


def parse_path(ctx, param, value) -> List[str]:
    def parse_helper(value, exclusions):
        files = []
        # deal with ~
        item = os.path.expanduser(value)
        # shell escaping
        item = item.replace('\\', '')

        if os.path.isdir(item):
            # If it's a directory, add all files within
            for dirpath, dirnames, filenames in os.walk(item):
                for filename in filenames:
                    files.append(os.path.join(dirpath, filename))

                if not Helpers.is_glob_recursive(item):
                    dirnames.clear()
        elif os.path.isfile(item):
            # If it's a file, add it to the list
            files.append(item)
        # check for glob
        elif Helpers.is_glob_pattern(item):
            # check for !
            if item.startswith('!'):
                for filepath in glob.glob(item[1:], recursive=Helpers.is_glob_recursive(item)):
                    exclusions.append(filepath)
            elif any('{' in item and '}' in item for item in value):
                brace_items = Helpers.flatten([Helpers.glob_brace(item) for item in value if '{' in item and '}' in item])
                files = files + brace_items
            else:
                for filepath in glob.glob(item, recursive=Helpers.is_glob_recursive(item)):
                    files.append(filepath)
        elif item.startswith('http'):
            files.append(item)
        return files

    if not value:
        return []

    if (
        (isinstance(value, str))
        and (value.startswith('"') or value.startswith("'"))
        and (value.endswith("'") or value.endswith('"'))
    ):
        value = value[1:-1]

    files = []

    if not value:
        return files

    if isinstance(value, str):
        value = [value]

    if isinstance(value, tuple):
        value = list(value)

    # see if there are any brace glob patterns, and if so, expand them
    # and include them in the value array
    if any('{' in item and '}' in item for item in value):
        brace_globs = [Helpers.glob_brace(item) for item in value if '{' in item and '}' in item]
        brace_globs = Helpers.flatten(brace_globs)
        value += list(brace_globs)

    # split by ' '
    # value = Helpers.flatten([item.split(' ') for item in value])

    exclusions = []

    for item in value:
        files.extend(parse_helper(item, exclusions))

    # deal with exclusions
    files = [file for file in files if file not in exclusions]
    return files


def parse_command_string(s, command):
    def parse_option(part):
        return next((param for param in command.params if part in param.opts), None)

    try:
        parts = shlex.split(s)
    except ValueError:
        parts = s.split(' ')

    tokens = []
    skip_n = 0

    for i, part in enumerate(parts):
        if skip_n > 0:
            skip_n -= 1
            continue

        if i == 0 and part == command.name:
            continue

        # Check if this part is an option in the given click.Command
        option = parse_option(part)

        # If the option is found and it's not a flag, consume the next value.
        if option and not option.is_flag:
            tokens.append(part)
            path_part = ''
            # special case path because of strange shell behavior with " "
            if part == '-p' or part == '--path':
                z = i
                while (
                    (z + 1 < len(parts))
                    and (
                        parse_path(None, None, parts[z + 1])
                        or Helpers.glob_brace(parts[z + 1])
                        or parts[z + 1].startswith('!')
                    )
                ):
                    path_part += parts[z + 1] + ' '
                    # tokens.append(parts[z + 1])
                    skip_n += 1
                    z += 1
                if path_part:
                    path_part = path_part.strip()
                    tokens.append(f'{path_part}')
                else:
                    # couldn't find any path, so display that to the user
                    logging.debug(f'Glob pattern not found for path: {parts[z + 1]}')
                    z += 1
                    skip_n += 1
                    tokens.append('')
            else:
                if i + 1 < len(parts):
                    tokens.append(parts[i + 1])
                    skip_n += 1
        elif option and option.is_flag:
            tokens.append(part)
        else:
            message = '"' + ' '.join(parts[i:]) + '"'
            tokens.append(message)
            break
    return tokens


def get_path_as_messages(
    path: List[str],
    upload: bool = False,
    allowed_file_types: List[str] = []
) -> List[User]:

    files: Sequence[User] = []
    for file_path in path:
        # check for url
        result = urlparse(file_path)

        if (result.scheme == 'http' or result.scheme == 'https'):
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}  # noqa
            download_request = requests.get(result.geturl(), headers=headers)
            if download_request.status_code == 200:
                file_name = os.path.basename(result.path)
                _, file_extension = os.path.splitext(file_name)
                file_extension = '.html' if file_extension == '' else file_extension.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_file.write(download_request.content)
                    temp_file.flush()
                    file_path = temp_file.name
                    result = urlparse(file_path)
            else:
                logging.debug(f'Unable to download {result.geturl()} as we got status code {download_request.status_code}')

        if allowed_file_types and not any(file_path.endswith(parsable_file_type) for parsable_file_type in allowed_file_types):
            continue

        if result.scheme == '' or result.scheme == 'file':
            if '.pdf' in result.path:
                if upload:
                    with open(file_path, 'rb') as f:
                        files.append(User(PdfContent(f.read(), url=os.path.abspath(file_path))))
                else:
                    files.append(User(PdfContent(b'', url=os.path.abspath(file_path))))
            elif '.htm' in result.path or '.html' in result.path:
                try:
                    with open(file_path, 'r') as f:
                        file_content = f.read().encode('utf-8')
                        # try to parse as markdown
                        markdown = Helpers.late_bind(
                            'helpers.webhelpers',
                            'WebHelpers',
                            'convert_html_to_markdown',
                            file_content,
                        )
                        file_content = markdown if markdown else file_content
                        file_content = f.read().encode('utf-8')
                        files.append(User(FileContent(file_content, url=os.path.abspath(file_path))))
                except UnicodeDecodeError:
                    raise ValueError(f'File {file_path} is not a valid text file, pdf or image.')
            elif '.pdf' in result.path:
                if upload:
                    with open(file_path, 'rb') as f:
                        files.append(User(PdfContent(f.read(), url=os.path.abspath(file_path))))
                else:
                    files.append(User(PdfContent(b'', url=os.path.abspath(file_path))))
            elif Helpers.classify_image(open(file_path, 'rb').read()) in ['image/jpeg', 'image/png', 'image/webp']:
                if upload:
                    with open(file_path, 'rb') as f:
                        files.append(User(ImageContent(f.read(), url=os.path.abspath(file_path))))
                else:
                    raw_image_data = open(file_path, 'rb').read()
                    files.append(User(ImageContent(Helpers.load_resize_save(raw_image_data), url=os.path.abspath(file_path))))
            else:
                try:
                    with open(file_path, 'r') as f:
                        if upload:
                            file_content = f.read().encode('utf-8')
                        else:
                            file_content = b''
                        files.append(User(FileContent(file_content, url=os.path.abspath(file_path))))
                except UnicodeDecodeError:
                    raise ValueError(f'File {file_path} is not a valid text file, pdf or image.')
    return files


async def stream_response(response, print_lambda: Callable) -> List[AstNode]:
    def strip_string(str):
        if str.startswith('"'):
            str = str[1:]
        if str.endswith('"'):
            str = str[:-1]
        return str

    def decode(content) -> bool:
        try:
            data = jsonpickle.decode(content)

            # tokens
            if isinstance(data, Content):
                print_lambda(str(cast(Content, data)))
            elif isinstance(data, TokenStopNode):
                print_lambda(str(cast(TokenStopNode, data)))
            elif isinstance(data, StreamNode):
                print_lambda(cast(StreamNode, data))
            elif isinstance(data, AstNode):
                response_objects.append(data)
            elif isinstance(data, (dict, list)):
                response_objects.append(data)
            else:
                print_lambda(strip_string(data))
            return True
        except Exception as ex:
            return False

    response_objects = []
    async with async_timeout.timeout(400):
        try:
            buffer = ''
            async for raw_bytes in response.aiter_raw():
                content = raw_bytes.decode('utf-8')
                content = content.replace('data: ', '').strip()

                if content == '[DONE]':
                    pass
                elif content == '':
                    pass
                else:
                    result = decode(content)
                    if not result:
                        buffer += content
                        result = decode(buffer)
                        if result:
                            buffer = ''
                    else:
                        buffer = ''

        except asyncio.TimeoutError as ex:
            logging.exception(ex)
            # await response.aclose()
            raise ex
        except KeyboardInterrupt as ex:
            await response.aclose()
            raise ex
    return response_objects


async def get_thread(
    api_endpoint: str,
    id: int,
):
    params = {
        'id': id,
    }
    response: httpx.Response = httpx.get(f'{api_endpoint}/v1/chat/get_thread', params=params)
    thread = SessionThread.model_validate(response.json())
    return thread


async def set_thread(
    api_endpoint: str,
    thread: SessionThread,
) -> SessionThread:
    async with httpx.AsyncClient(timeout=400.0) as client:
        response = await client.post(
            f'{api_endpoint}/v1/chat/set_thread',
            json=thread.model_dump()
        )
        session_thread = SessionThread.model_validate(response.json())
        return session_thread


async def get_threads(
    api_endpoint: str,
):
    response: httpx.Response = httpx.get(f'{api_endpoint}/v1/chat/get_threads')
    thread = cast(List[SessionThread], TypeAdapter(List[SessionThread]).validate_python(response.json()))
    return thread


async def execute_llm_call_direct(
    message: Message,
    api_key: str,
    executor_name: str,
    model_name: str,
    context_messages: Sequence[Message] = [],
    api_endpoint: str = ''
) -> Assistant:
    printer = StreamPrinter('')

    async def __stream_handler(node: AstNode):
        printer.write(node)  # type: ignore

    executor: Optional[Executor] = None

    if executor_name == 'openai':
        executor = OpenAIExecutor(
            api_key=api_key,
            default_model=model_name,
            api_endpoint=api_endpoint or Container.get_config_variable('LLMVM_API_BASE', default='https://api.openai.com/v1')
        )
    elif executor_name == 'anthropic':
        executor = AnthropicExecutor(
            api_key=api_key,
            default_model=model_name,
            api_endpoint=api_endpoint or Container.get_config_variable('LLMVM_API_BASE', default='https://api.anthropic.com')
        )
    elif executor_name == 'gemini':
        executor = GeminiExecutor(
            api_key=api_key,
            default_model=model_name,
        )
    else:
        raise ValueError('No executor specified.')

    messages = list(context_messages) + [message]
    assistant = await executor.aexecute(
        messages=messages,
        stream_handler=__stream_handler,
    )
    return assistant

    # messages.append(assistant)

    # response_messages = list([MessageModel.from_message(m) for m in messages])
    # result = SessionThread(id=-1, messages=response_messages)
    # return result


async def execute_llm_call(
    api_endpoint: str,
    id: int,
    message: Message,
    executor: str,
    model: str,
    mode: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
    compression: str = 'auto',
    clear_thread: bool = False,
) -> SessionThread:

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f'{api_endpoint}/health')
            response.raise_for_status()

        thread = await get_thread(api_endpoint, id)

        if clear_thread:
            thread.messages = []

        for context_message in context_messages:
            thread.messages.append(MessageModel.from_message(message=context_message))

        thread.messages.append(MessageModel.from_message(message=message))
        thread.current_mode = mode
        thread.cookies = cookies
        thread.executor = executor
        thread.model = model
        thread.compression = compression

        if mode == 'direct' or mode == 'tool' or mode == 'auto':
            endpoint = 'tools_completions_continuation'
        else:
            endpoint = 'code_completions'

        async with httpx.AsyncClient(timeout=400.0) as client:
            async with client.stream(
                'POST',
                f'{api_endpoint}/v1/chat/{endpoint}',
                json=thread.model_dump(),
            ) as response:
                objs = await stream_response(response, StreamPrinter('').write)

        await response.aclose()

        if objs:
            session_thread = SessionThread.model_validate(objs[-1])
            return session_thread
        return thread

    except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
        if mode == 'tool' or mode == 'code':
            logging.debug('LLMVM server is down, but we are in tool mode. Cannot execute directly')
            raise ex

    # server is down, go direct. this means that executor and model can't be nothing
    if not executor and not model:
        if Container.get_config_variable('LLMVM_EXECUTOR', 'executor'):
            executor = Container.get_config_variable('LLMVM_EXECUTOR')

        if Container.get_config_variable('LLMVM_MODEL', 'model'):
            model = Container.get_config_variable('LLMVM_MODEL')

    if executor and model:
        if executor == 'openai' and Container.get_config_variable('OPENAI_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('OPENAI_API_KEY'),
                'openai',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif executor == 'anthropic' and Container.get_config_variable('ANTHROPIC_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('ANTHROPIC_API_KEY'),
                'anthropic',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        elif executor == 'gemini' and Container.get_config_variable('GOOGLE_API_KEY'):
            assistant = await execute_llm_call_direct(
                message,
                Container.get_config_variable('GOOGLE_API_KEY'),
                'gemini',
                model,
                context_messages
            )
            return SessionThread(
                id=-1,
                messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
            )
        else:
            raise ValueError(f'Executor {executor} and model {model} are set, but no API key is set.')
    elif Container.get_config_variable('OPENAI_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('OPENAI_API_KEY'),
            'openai',
            'gpt-4o',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    elif os.environ.get('ANTHROPIC_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('ANTHROPIC_API_KEY'),
            'anthropic',
            'claude-2.1',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    elif os.environ.get('GOOGLE_API_KEY'):
        assistant = await execute_llm_call_direct(
            message,
            Container.get_config_variable('GOOGLE_API_KEY'),
            'gemini',
            'gemini-pro',
            context_messages
        )
        return SessionThread(
            id=-1,
            messages=[MessageModel.from_message(message) for message in list(context_messages) + [message, assistant]]
        )
    else:
        logging.warning('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa
        raise ValueError('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY is set. Unable to execute direct call to LLM.')  # noqa


def llm(
    message: Optional[str | bytes | Message],
    id: int,
    mode: str,
    endpoint: str,
    executor: str,
    model: str,
    context_messages: Sequence[Message] = [],
    cookies: List[Dict[str, Any]] = [],
    compression: str = 'auto',
) -> SessionThread:
    user_message = User(Content(''))
    if isinstance(message, str):
        user_message = User(Content(message))
    elif isinstance(message, bytes):
        user_message = User(Content(message.decode('utf-8')))
    elif isinstance(message, Message):
        user_message = message

    context_messages_list = list(context_messages)

    clear_thread = False
    # if the incoming message is a thread, parse it and send it through
    role_strings = ['Assistant: ', 'System: ', 'User: ']
    action_strings = ['ImageContent(', 'PdfContent(', 'FileContent(']

    if isinstance(message, str) and any(role_string in message for role_string in role_strings):
        all_messages = parse_message_thread(message)
        user_message = all_messages[-1]
        context_messages_list += all_messages[:-1]
        clear_thread = True
    # if the incoming message has actions [ImageContent(...), PdfContent(...), FileContent(...)] etc
    # parse those actions
    elif isinstance(message, str) and any(action_string in message for action_string in action_strings):
        all_messages = parse_message_actions(User, message)
        user_message = all_messages[-1]
        context_messages_list += all_messages[:-1]

    return asyncio.run(
        execute_llm_call(
            endpoint,
            id,
            user_message,
            executor,
            model,
            mode,
            context_messages_list,
            cookies,
            compression,
            clear_thread,
        )
    )


class StreamPrinter():
    def __init__(self, role: str):
        self.buffer = ''
        self.console = Console(file=sys.stderr)
        self.markdown_mode = False
        self.role = role
        self.started = False

    def display_image(self, image_bytes):
        if len(image_bytes) < 10:
            return
        try:
            # Create a temporary file to store the output from kitty icat
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                if (
                    Helpers.is_emulator('kitty')
                    and (
                        shutil.which('kitty')
                        or os.path.exists('/Applications/kitty.app/Contents/MacOS/kitty')
                    )
                    or (
                        Helpers.is_emulator('tmux')
                        and (
                            shutil.which('kitty')
                            or os.path.exists('/Applications/kitty.app/Contents/MacOS/kitty')
                        )
                        and Helpers.is_running('kitty')
                    )
                ):
                    # kitty + tmux is a feature of later versions of kitty, this may not work
                    # Use kitty icat to save its output to the temporary file
                    cmd_path = shutil.which('kitty') or '/Applications/kitty.app/Contents/MacOS/kitty'
                    process = subprocess.Popen(
                        [cmd_path, 'icat', '--transfer-mode', 'file'],
                        stdin=subprocess.PIPE,
                        stdout=temp_file
                    )
                    process.communicate(input=image_bytes)
                    process.wait()
                    # Now cat the temporary file to stderr
                    subprocess.run(['cat', temp_file.name], stdout=sys.stderr)
                elif (
                    shutil.which('viu')
                ):
                    temp_file.write(image_bytes)
                    temp_file.flush()
                    subprocess.run(['viu', temp_file.name], stdout=sys.stderr)
        except Exception as e:
            pass

    def write_string(self, string: str):
        if logging.level <= 20:  # INFO
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')

    def write(self, node: AstNode):
        if logging.level <= 20:  # INFO
            if not self.started and self.role:
                self.console.print(f'[bold green]{self.role}[/bold green]: ', end='')
                self.started = True

            string = ''

            if isinstance(node, Content):
                string = str(node)
            elif isinstance(node, TokenStopNode):
                string = '\n'
            elif isinstance(node, StreamNode):
                if isinstance(node.obj, bytes):
                    self.display_image(node.obj)
                    return
            else:
                string = str(node)

            self.buffer += string
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')


def markdown__rich_console__(
    self,
    console: Console,
    options: ConsoleOptions,
) -> RenderResult:
    code = str(self.text).rstrip()
    syntax = Syntax(
        code, self.lexer_name, theme=self.theme, word_wrap=True, padding=0
    )
    yield syntax


def print_response(messages: List[Message], escape: bool = False):
    # if we're being piped or redirected, we probably want to keep the
    # original output of the LLM, rather than render markdown.
    def contains_token(s, tokens):
        return any(token in s for token in tokens)

    def escape_string(input_str):
        # input_str = re.sub(r'\r?\n', '\\n', input_str)
        # input_str = re.sub(r'\t', '\\t', input_str)
        if escape:
            input_str = re.sub(r'"', r'\"', input_str)
            return input_str
        else:
            return input_str

    def pprint(prepend: str, content: Content):
        markdown_tokens = ['###', '* ', '](', '```', '## ']
        console = Console()

        if isinstance(content, ImageContent):
            console.print(f'{prepend}\n', end='')
            StreamPrinter('user').display_image(content.sequence)
        elif isinstance(content, PdfContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(f'[PdfContent({content.url})]'))
        elif isinstance(content, FileContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(f'[FileContent({content.url})]'))
        elif contains_token(str(content), markdown_tokens) and sys.stdout.isatty():
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(str(content)))
        else:
            console.print(escape_string(f'{prepend}{content}'))

    def fire_helper(string: str):
        if 'digraph' and 'edge' and 'node' in string:
            # fire up graphvis.
            graphvis_code = 'digraph' + Helpers.in_between(string, 'digraph', '}') + '}\n\n'
            temp_file = tempfile.NamedTemporaryFile(mode='w+')
            temp_file.write(graphvis_code)
            temp_file.flush()
            cmd = 'dot -Tx11 {}'.format(temp_file.name)
            subprocess.run(cmd, text=True, shell=True, env=os.environ)

    for message in messages:
        if message.role() == 'assistant':
            if not escape:
                pprint('[bold cyan]Assistant[/bold cyan]: ', message.message)
            else:
                pprint('', message.message)
            fire_helper(str(message))
        elif message.role() == 'system':
            if not escape:
                pprint('[bold red]System[/bold red]: ', message.message)
            else:
                pprint('', message.message)
        elif message.role() == 'user':
            if not escape:
                pprint('[bold cyan]User[/bold cyan]: ', message.message)
            else:
                pprint('', message.message)


def print_thread(thread: SessionThread, escape: bool = False):
    print_response([MessageModel.to_message(message) for message in thread.messages], escape)


def get_string_thread_with_roles(thread: SessionThread):
    string_result = ''
    for message in [MessageModel.to_message(message) for message in thread.messages]:
        if message.role() == 'assistant':
            string_result += 'Assistant: '
        elif message.role() == 'system':
            string_result += 'System: '
        elif message.role() == 'user':
            string_result += 'User: '

        if isinstance(message.message, ImageContent):
            if message.message.url.startswith('data:image'):
                decoded_base64 = base64.b64decode(message.message.url.split(',')[1])
                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpg', delete=False) as temp_file:
                    temp_file.write(decoded_base64)
                    temp_file.flush()
                    string_result += f'[ImageContent({temp_file.name})]\n\n'
            else:
                string_result += f'[ImageContent({message.message.url})]\n\n'
        elif isinstance(message.message, FileContent):
            string_result += f'[FileContent({message.message.url})]\n\n'
        elif isinstance(message.message, PdfContent):
            string_result += f'[PdfContent({message.message.url})]\n\n'
        else:
            string_result += str(message) + '\n\n'
    return string_result


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

        for completion in files_and_dirs:
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
                rich.print(f'  ({argument.name}')
            for option in [param for param in ctx.command.get_command(ctx, key).params if isinstance(param, click.Option)]:  # type: ignore  # NOQA: E501
                rich.print(f'  {str(", ".join(option.opts)).ljust(25)} {option.help if option.help else ""}')

        rich.print()
        rich.print(f'$LLMVM_EXECUTOR: {Container.get_config_variable("LLMVM_EXECUTOR", default="(not set)")}')
        rich.print(f'$LLMVM_MODEL: {Container.get_config_variable("LLMVM_MODEL", default="(not set)")}')
        rich.print()
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

            from PIL import ImageGrab
            try:
                im = ImageGrab.grabclipboard()
            except Exception as ex:
                im = None

            if im is not None:
                with io.BytesIO() as output:
                    im.save(output, format='PNG')
                    output.seek(0)
                    raw_data = Helpers.load_resize_save(output.read(), 'PNG')

                    StreamPrinter('user').display_image(raw_data)

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

            thread = asyncio.run(get_thread(Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'), 0))
            thread_id = thread.id
            rich.print('New thread created.')
            event.app.current_buffer.text = ''
            event.app.current_buffer.cursor_position = 0
            self.__redraw(event)

        @kb.add('c-g')
        def _(event):
            global last_thread
            global thread_id

            editor = os.environ.get('EDITOR', 'vim')

            if 'thread_id' in globals() and thread_id > 0:
                try:
                    last_thread = asyncio.run(
                        get_thread(Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'), thread_id)
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

                query = await session.prompt_async(
                    f'[{thread_id}] query>> ',
                    complete_while_typing=True,
                )

                # there are a few special commands that aren't 'clickified'
                if query == 'yy':
                    # copy the last assistant message to the clipboard
                    last_thread_t: SessionThread = last_thread
                    pyperclip.copy(str(last_thread_t.messages[-1].content))
                    rich.print('Last message copied to clipboard.')
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
                break

            except Exception:
                console.print_exception(max_frames=10)


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
            ctx.exit(0)
        else:
            # default message command
            command = ctx.command.get_command(ctx, 'message')  # type: ignore
            tokens = parse_command_string(query, command)

            command.invoke(ctx, **{
                param.name: value
                for param, value in zip(command.params, command.parse_args(ctx, tokens))
            })
            ctx.exit(0)


@cli.command('status')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def status(
    endpoint: str,
):
    async def status_helper():
        async with httpx.AsyncClient(timeout=2.0) as client:
            try:
                response = await client.get(f'{endpoint}/health')
                return response.json()
            except (httpx.HTTPError, httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError, httpx.ConnectTimeout) as ex:
                return {'status': f'LLMVM server not available at {endpoint}. Set endpoint using $LLMVM_ENDPOINT.'}

    rich.print(asyncio.run(status_helper()))


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
    os._exit(os.EX_OK)


@cli.command('clear', hidden=True)
def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


@cli.command('help', hidden=True)
def help():
    Repl().help()


@cli.command('ls', hidden=True)
@click.argument('args', type=str, required=False, default='')
def ls(args):
    os.system(f'ls --color {args}')

@cli.command('cookies', help='Set cookies for a message thread so that the tooling is able to access authenticated content.')
@click.option('--sqlite', '-s', type=str, required=False,
              help='location of Firefox/Chrome cookies sqlite file.')
@click.option('--file_location', '-l', type=str, required=False,
              help='location of Firefox/Chrome cookies txt file in Netscape format.')
@click.option('--id', '-i', type=int, required=False, default=0,
              help='thread id to attach cookies to')
@click.option('--endpoint', '-e', type=str, required=False,
              default=Container.get_config_variable('LLMVM_ENDPOINT', default='http://127.0.0.1:8011'),
              help='llmvm endpoint to use. Default is http://127.0.0.1:8011')
def cookies(
    sqlite: str,
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
    else:
        if not sqlite:
            # find either firefox or chrome cookies
            start_locations = []

            if os.path.exists(os.path.expanduser('~/.mozilla/firefox')):
                # find any cookies.sqlite file and print the directory
                for root, _, files in os.walk(os.path.expanduser('~/.mozilla/firefox')):
                    for file in files:
                        if file == 'cookies.sqlite':
                            start_locations.append(os.path.join(root, file))

            if os.path.exists(os.path.expanduser('~/.config/google-chrome')):
                # find any cookies.sqlite file and print the directory
                for root, dirs, files in os.walk(os.path.expanduser('~/.config/google-chrome')):
                    for file in files:
                        if file == 'Cookies':
                            start_locations.append(os.path.join(root, file))

            if len(start_locations) == 0:
                rich.print('No cookies files found.')
                return

            # print the list of cookies files and ask the user to pick one
            rich.print('Select a cookies file:')
            for i, location in enumerate(start_locations):
                rich.print(f'[{i}]: {location}')

            selection = int(input('Selection: '))
            if selection < 0 or selection >= len(start_locations):
                rich.print('Invalid selection.')
                return

            sqlite = start_locations[selection]

        # we have a location, now extract and upload
        # run the scripts/extract_cookies.py script and capture the text output
        browser = 'firefox' if 'sqlite' in sqlite else 'chrome'
        cmd = f'scripts/extract_{browser}_cookies.sh {sqlite}'
        result = subprocess.run(cmd, text=True, shell=True, env=os.environ, capture_output=True)
        if result.returncode != 0:
            rich.print(result.stderr)
            return
        cookies = Helpers.read_netscape_cookies(result.stdout)

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
    console = Console()

    response: httpx.Response = httpx.get(f'{endpoint}/search/{query}', timeout=400.0)
    if response.status_code == 200:
        results = response.json()

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

    threads = asyncio.run(get_threads(endpoint))

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

    if id.startswith('"') and id.endswith('"'):
        int_id = int(id[1:-1])
    else:
        int_id = int(id)

    thread = asyncio.run(get_thread(endpoint, int_id))
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

    try:
        thread = asyncio.run(get_thread(endpoint, thread_id))
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

    try:
        thread = asyncio.run(get_thread(endpoint, 0))
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
@click.option('--escape', '-s', type=bool, is_flag=True, required=False)
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
    escape: bool,
    context_messages: Sequence[Message] = [],
):
    global thread_id
    global last_thread
    context_messages = list(context_messages)

    if mode == 'code' and not path:
        raise MissingParameter('path')

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
                    StreamPrinter('user').display_image(image_bytes)
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

        thread = llm(
            message=message,
            id=id,
            mode=mode,
            endpoint=endpoint,
            executor=executor,
            model=model,
            context_messages=context_messages,
            cookies=cookies_list,
            compression=compression,
        )

        if not thread.messages:
            rich.print(f'No messages were returned from either the LLMVM server, or the LLM model {model}.')
            return

        if not escape: StreamPrinter('').write_string('\n')
        print_response([MessageModel.to_message(thread.messages[-1])], escape)
        if not escape: StreamPrinter('').write_string('\n')
        last_thread = thread
        thread_id = thread.id
        return thread
    else:
        rich.print('No message to send.')


if __name__ == '__main__':
    # special case the hijacking of --help
    if len(sys.argv) == 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        Repl().help()
        sys.exit(0)

    if len(sys.argv) <= 1 and sys.stdin.isatty():
        repl_inst = Repl()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(repl_inst.repl())
    else:
        cli()
