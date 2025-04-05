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
from llmvm.common.objects import BrowserContent, HTMLContent, MarkdownContent, Message, Content, ImageContent, PdfContent, FileContent, AstNode, StreamingStopNode, TextContent, TokenNode, TokenStopNode, StreamNode, SessionThreadModel, MessageModel, TokenThinkingNode


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

            elif isinstance(content, HTMLContent):
                Helpers.find_and_run_chrome_with_html(content.get_str())

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

            if not escape:
                fire_helpers(message.get_str())

            if not escape and role_new_line: self.console.print('\n', end='')

    def print_thread(self, thread: SessionThreadModel, escape: bool = False, role_new_line: bool = True):
        self.print_messages([MessageModel.to_message(message) for message in thread.messages], escape=escape, role_new_line=role_new_line)


class HTMLPrinter:
    def __init__(self, filename: str):
        self.filename = filename

    def header(self):
        HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Message Conversation</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-okaidia.min.css" rel="stylesheet" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
        }

        header {
            background-color: #fff;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            text-align: center;
        }

        h1 {
            color: #2c3e50;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        .message {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }

        .message-header {
            padding: 0.8rem 1.5rem;
            border-bottom: 1px solid #eee;
        }

        .user .message-header {
            background-color: #e3f2fd;
            color: #0d47a1;
        }

        .assistant .message-header {
            background-color: #e8f5e9;
            color: #1b5e20;
        }

        .message-body {
            padding: 1.5rem;
        }

        h2 {
            font-weight: 500;
            margin: 0;
            font-size: 1.2rem;
        }

        p {
            margin-bottom: 1rem;
            font-size: 1rem;
            color: #555;
        }

        /* Code formatting */
        pre[class*="language-"] {
            margin: 1.5rem 0;
            border-radius: 8px;
            max-height: 500px;
            overflow: auto;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        code {
            font-family: 'JetBrains Mono', 'Fira Code', Consolas, Monaco, monospace;
            font-size: 0.9em;
            font-feature-settings: "liga" 0; /* Disable ligatures */
        }

        code:not([class*="language-"]) {
            background-color: #f0f0f0;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
            color: #e53935;
        }

        /* Better Prism styling overrides */
        .token.comment,
        .token.prolog,
        .token.doctype,
        .token.cdata {
            color: #8292a2;
            font-style: italic;
        }

        .token.punctuation {
            color: #f8f8f2;
        }

        .token.property,
        .token.tag,
        .token.constant,
        .token.symbol,
        .token.deleted {
            color: #f92672;
        }

        .token.boolean,
        .token.number {
            color: #ae81ff;
        }

        .token.selector,
        .token.attr-name,
        .token.string,
        .token.char,
        .token.builtin,
        .token.inserted {
            color: #a6e22e;
        }

        .token.operator,
        .token.entity,
        .token.url,
        .language-css .token.string,
        .style .token.string,
        .token.variable {
            color: #f8f8f2;
        }

        .token.atrule,
        .token.attr-value,
        .token.function,
        .token.class-name {
            color: #e6db74;
        }

        .image-container img {
            max-width: 100%;
            border-radius: 4px;
            margin: 0.5rem 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        table, th, td {
            border: 1px solid #ddd;
        }

        th, td {
            padding: 10px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }

            .message-body {
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Message Conversation</h1>
    </header>
    <div class="container">
"""
        return HEADER

    def footer(self):
        FOOTER = """
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-css.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markup.min.js"></script>
    <script>
        // Initialize Prism.js highlighting
        document.addEventListener('DOMContentLoaded', (event) => {
            // Force re-highlighting to ensure all code blocks are properly highlighted
            if (typeof Prism !== 'undefined') {
                Prism.highlightAll();
            }
        });
    </script>
</body>
</html>"""
        return FOOTER

    def _detect_language(self, code_text):
        """Try to detect the programming language from the code content."""
        # More comprehensive language detection
        code_text = code_text.strip()

        # Python detection
        if re.search(r'^\s*def\s+\w+\s*\(.*\)\s*:', code_text, re.MULTILINE) or \
           re.search(r'^\s*class\s+\w+(\s*\(.*\))?\s*:', code_text, re.MULTILINE) or \
           re.search(r'^\s*import\s+\w+', code_text, re.MULTILINE) or \
           re.search(r'^\s*from\s+[\w\.]+\s+import', code_text, re.MULTILINE) or \
           re.search(r'^\s*@\w+', code_text, re.MULTILINE):
            return "python"

        # JavaScript/TypeScript detection
        elif re.search(r'(const|let|var)\s+\w+\s*=', code_text) or \
             re.search(r'function\s+\w+\s*\(.*\)\s*\{', code_text) or \
             re.search(r'=>\s*\{', code_text) or \
             re.search(r'import\s+.*\s+from\s+[\'"]', code_text) or \
             re.search(r'export\s+(default\s+)?(class|function|const)', code_text):
            if re.search(r':\s*(string|number|boolean|any|void|React)', code_text):
                return "typescript"
            return "javascript"

        # HTML detection
        elif re.search(r'<!DOCTYPE\s+html>', code_text, re.IGNORECASE) or \
             re.search(r'<html\b', code_text, re.IGNORECASE) or \
             (re.search(r'<\w+>', code_text) and re.search(r'</\w+>', code_text)):
            return "markup"

        # CSS detection
        elif re.search(r'[\.\#]?[\w-]+\s*\{[^}]*\}', code_text) and \
             re.search(r':\s*[\w\-\'"\s\d#]+;', code_text):
            return "css"

        # JSON detection
        elif (code_text.startswith('{') and code_text.endswith('}')) or \
             (code_text.startswith('[') and code_text.endswith(']')):
            if re.search(r'"\s*:\s*[{\[\"\w]', code_text):
                return "json"

        # Shell/Bash detection
        elif re.search(r'^#!.*sh', code_text) or \
             re.search(r'^\s*\$\s+', code_text, re.MULTILINE) or \
             re.search(r'^\s*echo\s+', code_text, re.MULTILINE) or \
             re.search(r'^\s*if\s+\[\[', code_text, re.MULTILINE):
            return "bash"

        # SQL detection
        elif re.search(r'SELECT\s+.+\s+FROM', code_text, re.IGNORECASE) or \
             re.search(r'CREATE\s+TABLE', code_text, re.IGNORECASE) or \
             re.search(r'INSERT\s+INTO', code_text, re.IGNORECASE):
            return "sql"

        # C/C++ detection
        elif re.search(r'#include\s+[<"][\w\.]+[>"]', code_text) or \
             re.search(r'int\s+main\s*\(', code_text) or \
             re.search(r'(void|int|char|float|double)\s+\w+\s*\(', code_text):
            if re.search(r'std::', code_text) or re.search(r'class\s+\w+\s*\{', code_text):
                return "cpp"
            return "c"

        # Default if no clear match
        return "plaintext"

    def get_str(self, messages: list[Message]):
        import markdown2
        import re
        import html
        from io import StringIO
        s = StringIO()

        s.write(self.header())

        for message in messages:
            role = message.role().lower()
            s.write(f'<div class="message {role}">\n')
            s.write(f'    <div class="message-header">\n')
            s.write(f'        <h2>{message.role().capitalize()}</h2>\n')
            s.write(f'    </div>\n')
            s.write(f'    <div class="message-body">\n')

            for content in message.message:
                if isinstance(content, TextContent) or isinstance(content, MarkdownContent):
                    content_str = content.get_str()

                    # Advanced code block processing for better syntax highlighting
                    def code_replace(match):
                        code = match.group(2)
                        lang = match.group(1).strip() if match.group(1) else self._detect_language(code)

                        # Escape HTML in code to prevent rendering issues
                        escaped_code = html.escape(code)

                        # For Python code, preserve indentation which is semantically important
                        if lang == "python":
                            escaped_code = escaped_code.replace("    ", "&#160;&#160;&#160;&#160;")
                            escaped_code = escaped_code.replace("\t", "&#160;&#160;&#160;&#160;")

                        return f'<pre><code class="language-{lang}">{escaped_code}</code></pre>'

                    # Replace markdown code blocks with HTML that Prism can highlight
                    # More robust pattern to handle various code block formats
                    content_str = re.sub(r'```([\w-]*)\n(.*?)```', code_replace, content_str, flags=re.DOTALL)

                    # Handle inline code with backticks
                    def inline_code_replace(match):
                        code = html.escape(match.group(1))
                        return f'<code>{code}</code>'

                    content_str = re.sub(r'`([^`]+)`', inline_code_replace, content_str)

                    # Convert markdown to HTML with extended options for better rendering
                    result = markdown2.markdown(
                        content_str,
                        extras=[
                            'tables',
                            'fenced-code-blocks',
                            'code-friendly',
                            'break-on-newline',
                            'smarty-pants',  # For better typography
                            'cuddled-lists'  # Better list formatting
                        ]
                    )

                    # Apply final formatting
                    s.write(f'{result}\n')

                elif isinstance(content, ImageContent):
                    s.write(f'    <div class="image-container">\n')
                    s.write(f'        <img src="data:{content.image_type};base64, {content.sequence}" alt="{content.url}" />\n')
                    s.write(f'    </div>\n')
                elif isinstance(content, BrowserContent):
                    s.write(f'    <div class="browser-content">\n')
                    s.write(f'        <h3>Browser Content: {content.url}</h3>\n')
                    s.write(f'    </div>\n')
                elif isinstance(content, FileContent):
                    s.write(f'    <div class="file-content">\n')
                    s.write(f'        <h3>File Content: {content.url}</h3>\n')
                    s.write(f'    </div>\n')
                elif isinstance(content, HTMLContent):
                    result = markdown2.markdown(content.get_str(), extras=['tables', 'fenced-code-blocks'])
                    s.write(f'    <div class="html-content">\n')
                    s.write(f'        {result}\n')
                    s.write(f'    </div>\n')
                elif isinstance(content, PdfContent):
                    s.write(f'    <div class="pdf-content">\n')
                    s.write(f'        <h3>PDF Content: {content.url}</h3>\n')
                    s.write(f'    </div>\n')

            s.write(f'    </div>\n')
            s.write(f'</div>\n')

        s.write(self.footer())
        val = s.getvalue()
        s.close()  # close the file
        return val

    def print(self, s: str):
        with open(self.filename, 'w') as f:
            f.write(s)

    def print_messages(
        self,
        messages: list[Message],
        escape: bool = False,
        role_new_line: bool = True
    ):
        with open(self.filename, 'w') as f:
            f.write(self.get_str(messages))

