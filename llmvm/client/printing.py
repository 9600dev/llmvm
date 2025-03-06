import base64
import sys
import tempfile
import os
import shutil
import subprocess
import re
import jsonpickle
import async_timeout
import asyncio
from typing import Any, Awaitable, Callable, cast

from rich.console import Console
from rich.markdown import Markdown

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.client.markdown_renderer import markdown__rich_console__
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import BrowserContent, MarkdownContent, Message, Content, ImageContent, PdfContent, FileContent, AstNode, StreamingStopNode, TextContent, TokenNode, TokenStopNode, StreamNode, SessionThreadModel, MessageModel, TokenThinkingNode


logging = setup_logging()


async def stream_response(response, print_lambda: Callable[[Any], Awaitable]) -> list:
    # this was changed mar 6, to support idle timoue so there's streaming issues, this will be the root cause
    def strip_string(s: str) -> str:
        if s.startswith('"'):
            s = s[1:]
        if s.endswith('"'):
            s = s[:-1]
        return s

    async def decode(content: str) -> bool:
        try:
            # Only attempt to decode if content looks like a JSON object.
            if not content.startswith('{') or not content.endswith('}'):
                return False

            data = jsonpickle.decode(content)

            # tokens
            if isinstance(data, TokenNode):
                await print_lambda(data.token)
            elif isinstance(data, TextContent):
                await print_lambda(data.get_str())
            elif isinstance(data, (TokenStopNode, StreamingStopNode)):
                await print_lambda(str(data))
            elif isinstance(data, StreamNode):
                await print_lambda(cast(StreamNode, data))
            elif isinstance(data, AstNode):
                response_objects.append(data)
            elif isinstance(data, (dict, list)):
                response_objects.append(data)
            # todo: this shouldn't happen - they all need to be objects
            elif isinstance(data, str) and data.startswith('"') and data.endswith('"'):
                await print_lambda(strip_string(data))
            else:
                return False
            return True
        except Exception:
            return False

    response_objects = []
    buffer = ''
    response_iterator = response.aiter_raw()

    while True:
        try:
            raw_bytes = await asyncio.wait_for(response_iterator.__anext__(), timeout=60)
        except asyncio.TimeoutError as ex:
            logging.exception(ex)
            raise ex
        except StopAsyncIteration:
            # End of stream
            break
        except KeyboardInterrupt as ex:
            await response.aclose()
            raise ex

        content = raw_bytes.decode('utf-8')
        content = content.replace('data: ', '').strip()

        if content in ('[DONE]', ''):
            continue
        else:
            result = await decode(content)
            if not result:
                # If the chunk couldn't be decoded alone, accumulate and try decoding again.
                buffer += content
                result = await decode(buffer)
                if result:
                    buffer = ''
            else:
                buffer = ''

    return response_objects


class StreamPrinter():
    def __init__(self, file=sys.stderr):
        self.buffer = ''
        self.console = Console(file=file)
        self.markdown_mode = False
        self.token_color = Container.get_config_variable('client_stream_token_color', default='bright_black')
        self.thinking_token_color = Container.get_config_variable('client_stream_thinking_token_color', default='cyan')

    async def display_image(self, image_bytes):
        if len(image_bytes) < 10:
            return
        try:
            # Create a temporary file to store the output from kitty icat
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
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
                    Helpers.is_emulator('wezterm')
                    and (
                        shutil.which('wezterm')
                    )
                    or (
                        Helpers.is_emulator('tmux')
                        and (
                            shutil.which('wezterm')
                        )
                        and Helpers.is_running('wezterm')
                    )
                ):
                    cmd_path = shutil.which('wezterm')
                    if not cmd_path:
                        logging.debug('wezterm not found')
                        return

                    # check to see if it's a webp image, because wezterm doesn't support webp
                    if Helpers.is_webp(image_bytes):
                        image_bytes = Helpers.convert_image_to_png(image_bytes)

                    self.console.file.flush()

                    if Helpers.is_image(image_bytes):
                        process = subprocess.Popen(
                            [cmd_path, 'imgcat'],
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

    async def write(self, node: AstNode):
        if logging.level <= 20:  # INFO
            token_color = self.token_color

            if isinstance(node, StreamNode):
                if isinstance(node.obj, bytes):
                    await self.display_image(node.obj)
                    return
                raise ValueError(f'StreamNode.obj must be bytes, not {type(node.obj)}')
            elif isinstance(node, TextContent):
                string = node.get_str()
            elif isinstance(node, TokenThinkingNode):
                string = node.token
                token_color = self.thinking_token_color
            elif isinstance(node, TokenNode):
                string = node.token
            elif isinstance(node, TokenStopNode) or isinstance(node, StreamingStopNode):
                string = node.print_str
            else:
                string = str(node)

            if string:
                self.buffer += string
                self.console.print(string, end='', style=f"{token_color}", highlight=False)


class ConsolePrinter:
    def __init__(self, file=sys.stdout):
        self.console = Console(file=file)

    def print(self, any: Any, end: str = ''):
        self.console.print(any, end)

    def print_exception(self, max_frames: int = 10):
        self.console.print_exception(max_frames=max_frames)

    def width(self) -> int:
        return self.console.width

    def height(self) -> int:
        return self.console.height

    def pprint(self, prepend: str, content_list: list[Content], escape: bool = False):
        def escape_string(input_str):
            return re.sub(r'"', r'\"', input_str) if escape else input_str

        if prepend:
            self.console.print(f'{prepend}\n', end='')

        helpers_open = False
        helpers_result_open = False

        def compress(content: Content, compress: bool = False) -> Content:
            if isinstance(content, TextContent) and compress:
                if len(content.get_str()) > 10000:
                    return TextContent(content.get_str()[:300] + '\n\n ... \n\n' + content.get_str()[-300:])
            return content

        inline_markdown = Helpers.flatten([ObjectTransformers.transform_inline_markdown_to_image_content_list(content) for content in content_list])

        for content in inline_markdown:
            if isinstance(content, TextContent) and '<helpers_result>' in content.get_str() or '</helpers_result>' in content.get_str():
                helpers_result_open = '<helpers_result>' in content.get_str()
                self.console.print(Markdown(content.get_str()))

            elif isinstance(content, TextContent) and '<helpers>' in content.get_str() or '</helpers>' in content.get_str():
                helpers_open = '<helpers>' in content.get_str()
                self.console.print(Markdown(content.get_str()))

            elif isinstance(content, ImageContent):
                asyncio.run(StreamPrinter().display_image(content.sequence))

            elif isinstance(content, PdfContent):
                self.console.print(Markdown(f'[PdfContent({content.url})]'))

            elif isinstance(content, FileContent):
                self.console.print(Markdown(f'[FileContent({content.url})]'))

            elif isinstance(content, MarkdownContent):
                self.console.print(Markdown(f'[MarkdownContent({content.url})]'))

            elif isinstance(content, BrowserContent):
                self.console.print(Markdown(f'[BrowserContent({content.url})]'))

            elif isinstance(content, TextContent) and Helpers.is_markdown_simple(content.get_str()) and sys.stdout.isatty():
                self.console.print(Markdown(compress(content, helpers_open or helpers_result_open).get_str()))

            else:
                self.console.print(Markdown(escape_string(f'{compress(content, helpers_open or helpers_result_open).get_str()}')))

    def print_messages(self, messages: list[Message], escape: bool = False, role_new_line: bool = True):
        # make both ```markdown and markdown'ish responses look like a CodeBlock
        Markdown.__rich_console__ = markdown__rich_console__

        def fire_helpers(s: str):
            if '```digraph' in s:
                # fire up graphvis.
                graphvis_code = Helpers.in_between(s, '```digraph', '```')
                temp_file = tempfile.NamedTemporaryFile(mode='w+')
                temp_file.write(graphvis_code)
                temp_file.flush()
                # check for linux
                if sys.platform.startswith('linux'):
                    cmd = 'dot -Tx11 {}'.format(temp_file.name)
                elif sys.platform.startswith('darwin'):
                    cmd = 'dot -Tpdf {} | open -f -a Preview'.format(temp_file.name)
                subprocess.run(cmd, text=True, shell=True, env=os.environ)

        role_color = Container.get_config_variable('client_role_color', default='bold cyan')

        for message in messages:
            if escape:
                self.pprint('', message.message, escape)
            else:
                self.pprint(f'[{role_color}]{message.role().capitalize()}[/{role_color}]: ', message.message, escape)
            fire_helpers(message.get_str())
            if not escape and role_new_line: self.console.print('\n', end='')

    def print_thread(self, thread: SessionThreadModel, escape: bool = False, role_new_line: bool = True):
        self.print_messages([MessageModel.to_message(message) for message in thread.messages], escape=escape, role_new_line=role_new_line)

