import sys
import tempfile
import os
import shutil
import subprocess
import re
import jsonpickle
import async_timeout
import asyncio
from typing import Any, Awaitable, List, Callable, cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import MarkdownContent, Message, Content, ImageContent, PdfContent, FileContent, AstNode, TokenStopNode, StreamNode, SessionThread, MessageModel


logging = setup_logging()


async def stream_response(response, print_lambda: Callable[[Any], Awaitable]) -> List[AstNode]:
    def strip_string(str) -> str:
        if str.startswith('"'):
            str = str[1:]
        if str.endswith('"'):
            str = str[:-1]
        return str

    async def decode(content) -> bool:
        try:
            data = jsonpickle.decode(content)

            # tokens
            if isinstance(data, Content):
                await print_lambda(str(cast(Content, data)))
            elif isinstance(data, TokenStopNode):
                await print_lambda(str(cast(TokenStopNode, data)))
            elif isinstance(data, StreamNode):
                await print_lambda(cast(StreamNode, data))
            elif isinstance(data, AstNode):
                response_objects.append(data)
            elif isinstance(data, (dict, list)):
                response_objects.append(data)
            else:
                await print_lambda(strip_string(data))
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
                    result = await decode(content)
                    if not result:
                        buffer += content
                        result = await decode(buffer)
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


class StreamPrinter():
    def __init__(self, role: str):
        self.buffer = ''
        self.console = Console(file=sys.stderr)
        self.markdown_mode = False
        self.role = role
        self.started = False

    async def display_image(self, image_bytes):
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

    async def write_string(self, string: str):
        if logging.level <= 20:  # INFO
            self.console.print(f'[bright_black]{string}[/bright_black]', end='')

    async def write(self, node: AstNode):
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
                    await self.display_image(node.obj)
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
        console = Console()

        if isinstance(content, ImageContent):
            console.print(f'{prepend}\n', end='')
            asyncio.run(StreamPrinter('user').display_image(content.sequence))
        elif isinstance(content, PdfContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(f'[PdfContent({content.url})]'))
        elif isinstance(content, FileContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(f'[FileContent({content.url})]'))
        elif isinstance(content, MarkdownContent):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(f'[MarkdownContent({content.url})]'))
        elif isinstance(content, Markdown):
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(content.get_str())
        elif isinstance(content, Content) and Helpers.is_markdown_simple(content.get_str()) and sys.stdout.isatty():
            CodeBlock.__rich_console__ = markdown__rich_console__
            console.print(f'{prepend}', end='')
            console.print(Markdown(content.get_str()))
        else:
            console.print(escape_string(f'{prepend}{content.get_str()}'))

    def fire_helper(s: str):
        if 'digraph' in s and 'edge' in s and 'node' in s:
            # fire up graphvis.
            graphvis_code = 'digraph' + Helpers.in_between(s, 'digraph', '}') + '}\n\n'
            temp_file = tempfile.NamedTemporaryFile(mode='w+')
            temp_file.write(graphvis_code)
            temp_file.flush()
            cmd = 'dot -Tx11 {}'.format(temp_file.name)
            subprocess.run(cmd, text=True, shell=True, env=os.environ)

    for message in messages:
        if message.role() == 'assistant':
            temp_content = message.message
            if type(temp_content) is Content:
                # sometimes openai will do a funny thing where it:
                # var1 = search("....")
                # answer(var1)
                # and the answer will be a tonne of markdown in the var1 string
                # remove everything in between <helpers_result> and </helpers_result> including the code_result tag
                # todo: I dunno about this
                # temp_content.sequence = Helpers.outside_of(temp_content.get_str(), '<helpers_result>', '</helpers_result>')
                # temp_content.sequence = temp_content.get_str().replace('<helpers_result>', '').replace('</helpers_result>', '')
                code_result = Helpers.in_between(temp_content.get_str(), '<helpers_result>', '</helpers_result>')
                if len(code_result) > 10000:
                    # using regex, replace the stuff inside of <helpers_result></helpers_result> with a 20 character summary string
                    code_result_str = '<helpers_result>\n' + code_result[:300] + ' ... ' + code_result[-300:] + '\n</helpers_result>'
                    temp_content.sequence = re.sub(r'<helpers_result>.*?</helpers_result>', code_result_str, temp_content.get_str(), flags=re.DOTALL)
                # embed ```python around the code_result
                if '<helpers>' in temp_content.get_str() and '</helpers>' in temp_content.get_str():
                    temp_content.sequence = temp_content.get_str().replace('<helpers>', '```python\n<helpers>\n').replace('</helpers>', '\n</helpers>\n```')
                if '<helpers_result>' in temp_content.get_str() and '</helpers_result>' in temp_content.get_str():
                    temp_content.sequence = temp_content.get_str().replace('<helpers_result>', '```\n<helpers_result>\n').replace('</helpers_result>', '\n</helpers_result>\n```')
            if not escape:
                pprint('[bold cyan]Assistant[/bold cyan]: ', temp_content)
            else:
                pprint('', temp_content)
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

