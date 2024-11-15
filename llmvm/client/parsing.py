import base64
import os
from threading import Event
import time
import asyncio
import requests
import tempfile
import urllib.parse
from llmvm.common.objects import Content, ImageContent, MarkdownContent, MessageModel, PdfContent, FileContent, SessionThreadModel, TextContent, User, Message, Assistant, System
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from typing import List, Sequence
from urllib.parse import urlparse
from asyncio import CancelledError

import shlex
import glob


logging = setup_logging()


def get_string_thread_with_roles(thread: SessionThreadModel):
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


async def read_from_pipe(pipe_path, pipe_event: Event, timeout=0.3):
    pipe_fd = None
    try:
        pipe_fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(pipe_fd, 'r') as pipe:
            buffer = ""
            last_read_time = time.time()
            while not pipe_event.is_set():
                try:
                    chunk = pipe.read(4096)  # Read in larger chunks
                    if chunk:
                        buffer += chunk
                        last_read_time = time.time()
                    else:
                        # No new data, check if we've been idle for a while
                        if time.time() - last_read_time > timeout and buffer:
                            return_buffer = buffer
                            buffer = ''
                            yield return_buffer # Return the entire collected message

                    await asyncio.sleep(0.1)
                except IOError:
                    await asyncio.sleep(0.1)

                current_task = asyncio.current_task()
                if current_task and current_task.cancelled():
                    raise CancelledError
    except Exception as _:
        pass
    finally:
        if pipe_fd is not None:
            try:
                os.close(pipe_fd)
            except Exception as _:
                pass
        if os.path.exists(pipe_path):
            try:
                os.unlink(pipe_path)
                os.remove(pipe_path)
            except Exception as _:
                return



def parse_action(token) -> Content:
    """
    For a given [Action(...)] token, parse it into either
    [ImageContent()], [PdfContent()], or [FileContent()]
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


def parse_message_actions(role_type: type, message: str, actions: list[str]) -> list[Message]:
    accumulated_tokens = []
    messages = []
    # go through the message, create User() nodes for normal text content
    # and for actions, create the appropriate action node
    tokens = message.split(' ')
    for token in tokens:
        if any(token.startswith(action) for action in actions):
            if accumulated_tokens:
                messages.append(role_type(TextContent(' '.join(accumulated_tokens))))
            messages.append(role_type(parse_action(token)))
            accumulated_tokens = []
        elif token:
            accumulated_tokens.append(token)
    if accumulated_tokens:
        messages.append(role_type(TextContent(' '.join(accumulated_tokens))))
    return messages


def parse_message_thread(message: str, actions: list[str]):
    def create_message(type) -> Message:
        MessageClass = globals()[type]
        return MessageClass('')

    messages = []
    roles = ['Assistant: ', 'System: ', 'User: ']

    while any(message.startswith(role) for role in roles):
        role = next(role for role in roles if message.startswith(role))
        parsed_message = create_message(role.replace(': ', ''))
        content = Helpers.in_between_ends(message, role, roles)
        sub_messages = parse_message_actions(type(parsed_message), content, actions=actions)
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
            if os.path.isdir(file_path):
                continue
            elif result.path.endswith('.pdf'):
                if upload:
                    with open(file_path, 'rb') as f:
                        files.append(User(PdfContent(f.read(), url=os.path.abspath(file_path))))
                else:
                    files.append(User(PdfContent(b'', url=os.path.abspath(file_path))))
            elif result.path.endswith('.md'):
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    files.append(User(MarkdownContent([TextContent(file_content)], url=os.path.abspath(file_path))))
            elif result.path.endswith('.htm') or result.path.endswith('.html'):
                try:
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                        # try to parse as markdown
                        markdown = Helpers.late_bind(
                            'helpers.webhelpers',
                            'WebHelpers',
                            'convert_html_to_markdown',
                            file_content,
                        )
                        file_content = markdown if markdown else file_content
                        files.append(User(MarkdownContent([TextContent(file_content)], url=os.path.abspath(file_path))))
                except UnicodeDecodeError:
                    raise ValueError(f'File {file_path} is not a valid text file, pdf or image.')
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