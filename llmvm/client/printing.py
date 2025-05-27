import base64
import html
from io import StringIO
import sys
import tempfile
import os
import shutil
import subprocess
import re
import jsonpickle
import async_timeout
import asyncio
import markdown2
from typing import Any, Awaitable, Callable, cast

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme
from rich.control import Control
from rich.text import Text

from llmvm.common.container import Container
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.client.markdown_renderer import markdown__rich_console__
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import BrowserContent, HTMLContent, MarkdownContent, Message, Content, ImageContent, \
    PdfContent, FileContent, AstNode, StreamingStopNode, TextContent, TokenNode, TokenStopNode, SessionThreadModel, \
    MessageModel, TokenThinkingNode, StreamNode


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
            if isinstance(data, TokenThinkingNode):
                await print_lambda(data)
            elif isinstance(data, TokenNode):
                await print_lambda(data)
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
        self.inline_markdown_render = Container.get_config_variable('client_markdown_inline', default=False)
        self.current_line = ''
        self.line_count = 0
        self.rendered_lines = []
        self.printed_on_current_line = False
        self.in_helpers_block = False
        self.in_code_block = False
        self.in_diff_block = False
        self.printed_length = 0  # Track actual printed characters for accurate wrapping

    async def display_image(self, image_bytes):
        if len(image_bytes) < 10:
            return
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Check for Kitty
                kitty_path = Helpers.find_kitty()
                if (
                    Helpers.is_emulator('kitty') and kitty_path
                    or (
                        Helpers.is_emulator('tmux')
                        and kitty_path
                        and Helpers.is_running('kitty')
                    )
                ):
                    process = subprocess.Popen(
                        [kitty_path, 'icat', '--transfer-mode', 'file'],
                        stdin=subprocess.PIPE,
                        stdout=temp_file
                    )
                    process.communicate(input=image_bytes)
                    process.wait()

                    subprocess.run(['cat', temp_file.name], stdout=sys.stderr)
                    return  # Done displaying with kitty

                # Check for WezTerm
                wezterm_path = Helpers.find_wezterm()
                if (
                    Helpers.is_emulator('wezterm') and wezterm_path
                    or (
                        Helpers.is_emulator('tmux')
                        and wezterm_path
                        and Helpers.is_running('wezterm')
                    )
                    or (
                        Helpers.is_emulator('wezterm-gui')
                        and wezterm_path
                        and Helpers.is_running('wezterm-gui')
                    )
                ):
                    if not wezterm_path:
                        logging.debug('wezterm not found')
                        return

                    if Helpers.is_webp(image_bytes):
                        image_bytes = Helpers.convert_image_to_png(image_bytes)

                    self.console.file.flush()

                    if Helpers.is_image(image_bytes):
                        process = subprocess.Popen(
                            [wezterm_path, 'imgcat'],
                            stdin=subprocess.PIPE,
                            stdout=temp_file
                        )
                        process.communicate(input=image_bytes)
                        process.wait()

                        subprocess.run(['cat', temp_file.name], stdout=sys.stderr)
                    return  # Done displaying with wezterm

                # Fallback to viu if available
                viu_path = shutil.which('viu')
                if viu_path:
                    temp_file.write(image_bytes)
                    temp_file.flush()
                    subprocess.run([viu_path, temp_file.name], stdout=sys.stderr)
        except Exception as e:
            return

    def _clear_current_line(self):
        """Clear the current line in the terminal"""
        # Calculate how many lines the current line takes up
        if self.printed_length > 0 and self.printed_on_current_line:
            # Get terminal width
            terminal_width = self.console.width
            # Use the tracked printed length for accurate calculation
            lines_used = max(1, (self.printed_length + terminal_width - 1) // terminal_width)  # Ceiling division

            # Clear all the lines that were used
            for i in range(lines_used):
                if i > 0:
                    # Move up one line
                    self.console.file.write('\033[1A')
                # Move to beginning of line and clear it
                self.console.file.write('\r\033[K')

            # After clearing, cursor is at the beginning of the first line
        else:
            # Simple case - just clear current line
            self.console.file.write('\r\033[K')

        self.console.file.flush()
        self.printed_on_current_line = False
        self.printed_length = 0

    def _get_visible_length(self, text: str) -> int:
        """Get the visible length of text, excluding ANSI escape sequences"""
        import re
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        visible_text = ansi_escape.sub('', text)
        return len(visible_text)

    def _render_line_as_markdown(
        self, line: str,
        should_highlight_as_python: bool = False,
        should_highlight_as_diff: bool = False
    ):
        """Render a line as markdown using rich"""
        if not line.strip():
            return

        # Clear the current line first
        self._clear_current_line()

        # Check if this line should be highlighted as Python
        if should_highlight_as_diff:
            text = Text()
            if line.startswith('+'):
                text.append(line, style='green')
            elif line.startswith('-'):
                text.append(line, style='red')
            else:
                text.append(line, style='default')
            self.console.print(text)
        elif should_highlight_as_python:
            # Render as Python code with syntax highlighting
            from rich.syntax import Syntax
            syntax = Syntax(line, "python", theme="monokai", background_color="default", word_wrap=True, padding=0)
            self.console.print(syntax)
        else:
            # Apply the custom markdown renderer
            Markdown.__rich_console__ = markdown__rich_console__

            # Create markdown object and render it
            md = Markdown(line)
            self.console.print(md)  # Remove end='' to ensure proper line ending

        # Store that we've rendered this line
        self.rendered_lines.append(self.line_count)


    def finalize_stream(self):
        """Finalize the stream by rendering any remaining partial line"""
        if self.inline_markdown_render and self.current_line.strip():
            # Render the final partial line
            self._render_line_as_markdown(self.current_line)

    def _check_in_code_block(self, line: str) -> bool:
        if self.in_helpers_block:
            return True

        line = line.strip()

        if line == '```' and self.in_code_block:
            self.in_code_block = False
            return False

        if (
            line.startswith('```python') or
            line.startswith('```javascript') or
            line.startswith('```html') or
            line.startswith('```json') or
            line.startswith('```css') or
            line.startswith('```digraph') or
            line.startswith('```mermaid') or
            line.startswith('```haskell') or
            line.startswith('```rust') or
            line.startswith('```java') or
            line.startswith('```c') or
            line.startswith('```c++') or
            line.startswith('```c#') or
            line.startswith('```ruby') or
            line.startswith('```php') or
            line.startswith('```bash')
        ):
            self.in_code_block = True
            return self.in_code_block
        return self.in_code_block

    def _check_in_diff_block(self, line: str) -> bool:
        line = line.strip()

        if line.startswith('```diff'):
            self.in_diff_block = True
            return self.in_diff_block

        elif line == '```' and self.in_diff_block:
            self.in_diff_block = False
            return self.in_diff_block

        elif self.rendered_lines and self.rendered_lines[-1] == '```':
            self.in_diff_block = False
            return self.in_diff_block

        return self.in_diff_block

    def _check_helpers_tags(self, line: str) -> bool:
        """Check if line contains opening or closing helpers tags. Returns True if this line should be syntax highlighted."""
        # Check if we're currently in a helpers block or if this line contains helpers tags
        should_highlight = (
            self.in_helpers_block or
            '<helpers>' in line or
            '<helpers_result>' in line or
            '</helpers>' in line or
            '</helpers_result>' in line
        )

        # Update state for next line
        if '<helpers>' in line or '<helpers_result>' in line:
            self.in_helpers_block = True
        elif '</helpers>' in line or '</helpers_result>' in line:
            self.in_helpers_block = False

        return should_highlight

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
                # Finalize the stream when we hit a stop node
                if self.inline_markdown_render:
                    self.finalize_stream()
                    return
            else:
                string = str(node)

            if string:
                self.buffer += string

                if self.inline_markdown_render:
                    # Check if we've hit a newline
                    if '\n' in string:
                        # Split on newlines
                        parts = string.split('\n')

                        # Complete the current line with the first part
                        self.current_line += parts[0]

                        # Only print tokens if we haven't already printed on this line
                        if parts[0]:
                            self.console.print(parts[0], end='', style=f"{token_color}", highlight=False)
                            self.printed_on_current_line = True
                            self.printed_length += len(parts[0])

                        if self.current_line.strip() and self.printed_on_current_line:
                            self._clear_current_line()

                        if self.current_line.strip():
                            # Check for helpers tags before rendering
                            should_highlight = self._check_helpers_tags(self.current_line) or self._check_in_code_block(self.current_line)
                            should_diff_highlight = self._check_in_diff_block(self.current_line)
                            self._render_line_as_markdown(self.current_line, should_highlight, should_diff_highlight)
                        else:
                            self._clear_current_line()
                            self.console.print()

                        # Process any additional complete lines in the middle
                        for i in range(1, len(parts) - 1):
                            self.line_count += 1
                            if parts[i].strip():
                                # First print the tokens
                                self.console.print(parts[i], end='', style=f"{token_color}", highlight=False)
                                self.printed_on_current_line = True
                                self.printed_length = len(parts[i])
                                # Clear before rendering
                                self._clear_current_line()
                                # Check for helpers tags before rendering
                                should_highlight = self._check_helpers_tags(parts[i]) or self._check_in_code_block(parts[i])
                                should_diff_highlight = self._check_in_diff_block(self.current_line)
                                # Then render as markdown
                                self._render_line_as_markdown(parts[i], should_highlight, should_diff_highlight)
                            else:
                                # Empty line
                                self.console.print()

                        # Start new current line
                        self.line_count += 1
                        self.current_line = parts[-1]
                        self.printed_length = 0  # Reset for new line

                        # Print the start of the new line if not empty
                        if self.current_line:
                            self.console.print(self.current_line, end='', style=f"{token_color}", highlight=False)
                            self.printed_on_current_line = True
                            self.printed_length = len(self.current_line)
                    else:
                        # No newline - accumulate in current line and print token
                        self.current_line += string
                        self.console.print(string, end='', style=f"{token_color}", highlight=False)
                        self.printed_on_current_line = True
                        self.printed_length += len(string)
                else:
                    # Normal mode - just print tokens
                    self.console.print(string, end='', style=f"{token_color}", highlight=False)


class ConsolePrinter:
    def __init__(self, file=sys.stdout):
        client_assistant_color = Container.get_config_variable('client_assistant_color', default='white')
        custom_theme = Theme({
            "default": client_assistant_color
        })
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
        assistant_color = Container.get_config_variable('client_assistant_color', default='white')

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


class HTMLMessageRenderer:
    """Render Message objects to HTML with automatic code detection/high-lighting.

    In addition to fenced blocks (```), this renderer now detects:
    • Bare code blocks that *start* on a line matching language patterns.
    • Custom <helpers> … </helpers> and <helpers_result> … </helpers_result> sections
      (always Python) used by the toolchain.
    """

    # ---------------------------------------------------------------------
    # Language-detection helpers
    # ---------------------------------------------------------------------

    _HELPERS_RE  = re.compile(
        r"<helpers\b[^>]*>(.*?)</helpers\s*>",           # note \b and [^>]* then \s*
        re.DOTALL | re.IGNORECASE,
    )
    _HELPERS_RES = re.compile(
        r"<helpers_result\b[^>]*>(.*?)</helpers_result\s*>",
        re.DOTALL | re.IGNORECASE,
    )

    def _render_helpers(self, match: re.Match, lang_hint: str | None = "python") -> str:
        """Return a <pre><code …> block that *includes* the wrapper tags."""
        full_block = match.group(0)                 # <helpers> … </helpers>
        inner      = match.group(1)
        lang = self._detect_language(inner) if lang_hint is None else lang_hint
        if lang is None:
            lang = "plaintext"

        escaped = (
            html.escape(full_block)
                .replace("    ", "&#160;&#160;&#160;&#160;")
                .replace("\t",  "&#160;&#160;&#160;&#160;")
        )
        return f'\n<pre><code class="language-{lang}">{escaped}</code></pre>\n'

    _LANG_START_PATTERNS: list[tuple[re.Pattern, str]] = [
        # Bash / shell first because shebang must win
        (re.compile(r"^#!.*\bsh\b"), "bash"),

        # -------------------- Python --------------------
        (re.compile(r"^\s*def\s+\w+\s*\(.*\)\s*:"), "python"),
        (re.compile(r"^\s*async\s+def\s+\w+\s*\(.*\)\s*:"), "python"),
        (re.compile(r"^\s*\w+\s*=.+"), "python"),
        (re.compile(r"^\s*class\s+\w+(\s*\(.*\))?\s*:"), "python"),
        (re.compile(r"^\s*(import|from)\s+[\w\.]+"), "python"),
        (re.compile(r"^\s*@\w+"), "python"),

        # ---------------- JavaScript / TypeScript ----------------
        (re.compile(r"^\s*(?:const|let|var)\s+\w+\s*="), "javascript"),
        (re.compile(r"^\s*function\s+\w+\s*\(.*\)\s*\{"), "javascript"),
        (re.compile(r"^\s*export\s+(?:default\s+)?(?:class|function|const)"), "javascript"),
        (re.compile(r"^\s*import\s+.+\s+from\s+['\"]"), "javascript"),

        # ----------------------- HTML -----------------------
        (re.compile(r"^\s*<!DOCTYPE\s+html>", re.IGNORECASE), "markup"),
        (re.compile(r"^\s*<html\b", re.IGNORECASE), "markup"),
        (re.compile(r"^\s*<\w+[^>]*>"), "markup"),

        # ----------------------- JSON -----------------------
        (re.compile(r"^\s*\{[\s\S]*:\s*.+"), "json"),  # rough but effective
        (re.compile(r"^\s*\[[\s\S]*\]$"), "json"),

        # ----------------------- CSS ------------------------
        (re.compile(r"^\s*[.#]?[\w-]+\s*\{[^}]*\}"), "css"),
    ]

    # ---------------------------------------------------------------------
    # Full-block language detection
    # ---------------------------------------------------------------------
    @staticmethod
    def _detect_language(code_text: str) -> str:
        """Attempt to detect the programming language looking at the **whole** block."""
        text = code_text.strip()

        # Python
        if re.search(r"^\s*def\s+\w+\s*\(.*\)\s*:", text, re.MULTILINE) or \
           re.search(r"^\s*async\s+def\s+\w+\s*\(.*\)\s*:", text, re.MULTILINE) or \
           re.search(r"^\s*class\s+\w+(\s*\(.*\))?\s*:", text, re.MULTILINE) or \
           re.search(r"^\s*(import|from)\s+\w+", text, re.MULTILINE) or \
           re.search(r"^\s*@\w+", text, re.MULTILINE):
            return "python"

        # JavaScript / TypeScript
        if re.search(r"(?:const|let|var)\s+\w+\s*=", text) or \
           re.search(r"function\s+\w+\s*\(.*\)\s*\{", text) or \
           re.search(r"=>\s*\{", text) or \
           re.search(r"import\s+.*\s+from\s+['\"]", text) or \
           re.search(r"export\s+(?:default\s+)?(?:class|function|const)", text):
            return "javascript"

        # HTML (Prism registers HTML/XML as "markup")
        if re.search(r"<!DOCTYPE\s+html>", text, re.IGNORECASE) or \
           re.search(r"<html\b", text, re.IGNORECASE):
            return "markup"

        # JSON
        if (text.startswith("{") and text.endswith("}")) or \
           (text.startswith("[") and text.endswith("]")):
            return "json"

        # Bash
        if re.search(r"^#!.*\bsh\b", text) or \
           re.search(r"^\s*echo\s+", text, re.MULTILINE):
            return "bash"

        # CSS
        if re.search(r"[.#]?[\w-]+\s*\{[^}]*\}", text):
            return "css"

        return "plaintext"

    # ---------------------------------------------------------------------
    # Public rendering API
    # ---------------------------------------------------------------------

    def render_messages(self, messages: list["Message"]) -> str:
        html_out = StringIO()

        for msg in messages:
            html_out.write(self._render_single_message(msg))

        rendered = html_out.getvalue()
        html_out.close()
        return rendered

    # ------------------------------------------------------------------
    # Message-level rendering
    # ------------------------------------------------------------------

    def _render_single_message(self, message: "Message") -> str:
        parts: list[str] = []
        role = message.role().lower()
        parts.append(f'<div class="message {role}">')
        parts.append("    <div class=\"message-header\">")
        parts.append(f"        <h2>{message.role().capitalize()}</h2>")
        parts.append("    </div>")
        parts.append("    <div class=\"message-body\">")

        # ── coalesce adjacent TextContent so helper-blocks stay intact ──
        buf: list[str] = []
        def _flush():
            if buf:
                joined = "\n".join(buf)
                parts.append(self._handle_markdown_and_code(joined))
                buf.clear()

        for content in message.message:
            if isinstance(content, TextContent):
                buf.append(content.get_str())
            else:
                _flush()
                if isinstance(content, MarkdownContent):
                    md = content.get_str()
                    if len(md) > 10000:
                        md = md[:300] + "\n\n … \n\n" + md[-300:]
                    parts.append(markdown2.markdown(md, extras=["tables", "fenced-code-blocks", "break-on-newline"]))
                elif isinstance(content, ImageContent):
                    # NB: base64 import omitted for brevity; assume in scope
                    parts.append(
                        f'<div class="image-container">\n'
                        f'<img src="data:{content.image_type};base64,'
                        f'{base64.b64encode(content.sequence).decode()}" alt="{content.url}" />\n'
                        f'</div>'
                    )
                # TODO: other content types (BrowserContent, FileContent, …) as required
        _flush()

        parts.append("    </div>\n</div>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Markdown / code processing pipeline
    # ------------------------------------------------------------------
    def _handle_markdown_and_code(self, text: str) -> str:
        """
        Pipeline:
        0.   Turn <helpers>… and <helpers_result>… into <pre><code> blocks.
        1.   Inline fenced ``` blocks.
        2.   Auto-wrap bare (unfenced) code blocks.
        3.   Inline single-back-tick code.
        4.   Run markdown2.
        """
        # ------------------------------------------------------------------
        # 0) custom helper blocks  -----------------------------------------
        text = self._HELPERS_RE.sub(lambda m: self._render_helpers(m, "python"), text)
        text = self._HELPERS_RES.sub(lambda m: self._render_helpers(m, None), text)

        # ------------------------------------------------------------------
        # 1) fenced  ```lang\n … ```  --------------------------------------
        def fenced_repl(match: re.Match) -> str:
            lang_hint = match.group(1).strip()
            code      = match.group(2)
            lang      = lang_hint or self._detect_language(code)

            escaped = html.escape(code)
            if lang == "python":
                escaped = escaped.replace("    ", "&#160;&#160;&#160;&#160;").replace("\t", "&#160;&#160;&#160;&#160;")
            return f'<pre><code class="language-{lang}">{escaped}</code></pre>'

        text = re.sub(r"```([\w-]*)\n([\s\S]*?)```", fenced_repl, text)

        # ------------------------------------------------------------------
        # 2) bare-block auto-wrapper  --------------------------------------
        text = self._auto_wrap_bare_code(text)

        # ------------------------------------------------------------------
        # 3) inline   `code`   ---------------------------------------------
        text = re.sub(r'`([^`]+)`', lambda m: f'<code>{html.escape(m.group(1))}</code>', text)

        # ------------------------------------------------------------------
        # 4) markdown conversion  ------------------------------------------
        return markdown2.markdown(
            text,
            extras=[
                "tables",
                "fenced-code-blocks",
                "code-friendly",
                "break-on-newline",
                "smarty-pants",
                "cuddled-lists",
            ],
        )

    # ------------------------------------------------------------------
    # Bare code-block auto-wrapper
    # ------------------------------------------------------------------

    def _auto_wrap_bare_code(self, text: str) -> str:
        """
        Scan the incoming plain-text message line-by-line.  If we see a line that
        matches one of the registered “start-of-code” patterns *and* we are not
        already inside a raw <pre>...</pre> HTML block, we gather the contiguous
        code lines and wrap them in a <pre><code class="language-…"> … </code></pre>.

        A “code line” is either:
        • non-empty, or
        • starts with at least one space or tab
        so we capture typical indented blocks cleanly.
        """
        import html

        lines = text.splitlines()
        n = len(lines)
        out: list[str] = []        # <--- here’s the missing variable
        i = 0
        in_pre = False

        while i < n:
            line = lines[i]

            if line.lstrip().startswith(("<helpers", "</helpers")):
                out.append(line)
                i += 1
                continue

            # ── 1. Pass raw <pre> blocks straight through ─────────────────────────
            if "<pre" in line:
                in_pre = True
                out.append(line)
                i += 1
                continue

            if in_pre:
                out.append(line)
                if "</pre>" in line:
                    in_pre = False
                i += 1
                continue

            # ── 2. Does this line look like the *start* of code? ──────────────────
            detected_lang = None
            for pat, lang in self._LANG_START_PATTERNS:
                if pat.search(line):
                    detected_lang = lang
                    break

            if detected_lang:
                # ── 2a.  Slurp up the whole contiguous block ────────────────────
                block: list[str] = [line]

                i += 1
                while i < n and (
                    lines[i].strip() == ""                       # blank line
                    or lines[i].startswith((" ", "\t"))          # indented
                    or lines[i].lstrip().startswith(("#", "//")) # comment
                    or any(pat.search(lines[i])                  # OR *another*
                        for pat, _ in self._LANG_START_PATTERNS)  # code line
                ):
                    block.append(lines[i])
                    i += 1

                code_block = "\n".join(block)
                escaped = (
                    html.escape(code_block)
                    .replace("    ", "&#160;&#160;&#160;&#160;")
                    .replace("\t", "&#160;&#160;&#160;&#160;")
                )
                out.append(
                    f'<pre><code class="language-{detected_lang}">{escaped}</code></pre>'
                )
            else:
                # ── 3. Regular text line ────────────────────────────────────────
                out.append(line)
                i += 1

        return "\n".join(out)


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
            font-family: ui-sans-serif, -apple-system, system-ui, "Segoe UI", Helvetica, "Apple Color Emoji", Arial, sans-serif, "Segoe UI Emoji", "Segoe UI Symbol";
            font-size: 16px;
            line-height: 1.3;
            color: rgb(13, 13, 13);
            background-color: #f8f9fa;
            padding: 20px;
            max-width: 1200px;
            font-feature-settings: normal;
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
            font-weight: 700;
        }

        .message {
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            overflow: hidden;
        }

        .message-header {
            padding: 0.6rem 1.5rem;
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
            padding: 1.0rem;
        }

        .message-body h2 {
            margin-bottom: 1rem;
        }

        h2 {
            font-weight: 600;
            margin: 0;
            font-size: 1rem;
        }

        p {
            margin-bottom: 1rem;
            font-size: 1rem;
            color: #0d0d0d;
        }

        pre[class*="language-"] {
            padding: 1.5rem 1.5rem 1.5rem;
            margin: 1.0rem 0;
            border-radius: 8px;
            max-height: 500px;
            overflow: auto;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-x: hidden !important;
            font-family: 'JetBrains Mono', 'Fira Code', Consolas, Monaco, monospace !important;
            font-size: 0.7em;
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

        ul {
            margin-left: 1.5rem;
            margin-bottom: 1rem;
        }

        ol {
            margin-left: 1.5rem;
            margin-bottom: 1rem;
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

    def get_str(self, messages: list[Message]) -> str:
        s = StringIO()
        s.write(self.header())

        s.write(HTMLMessageRenderer().render_messages(messages))

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

