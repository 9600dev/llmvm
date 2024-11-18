from rich.markdown import Markdown, CodeBlock, Heading
from rich.console import Console, ConsoleOptions, RenderResult
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel
from typing import Iterable, Iterator, Optional
import re


def markdown__rich_console__(
    self: Markdown,
    console: Console,
    options: ConsoleOptions,
) -> RenderResult:
    CodeBlock.__rich_console__ = code_block__rich_console__
    syntax = Syntax(
        self.markup,
        'markdown',
        theme="monokai",
        background_color="default",
        word_wrap=True,
        padding=0
    )
    yield syntax

def code_block__rich_console__(
    self: CodeBlock,
    console: Console,
    options: ConsoleOptions,
) -> RenderResult:
    code = str(self.text).rstrip()

    # Check for markdown links pattern [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    # If there are links in the text
    if re.search(link_pattern, code):
        # Split the text and handle links specially
        parts = []
        last_end = 0
        for match in re.finditer(link_pattern, code):
            # Add any text before the link
            if match.start() > last_end:
                parts.append(Text(code[last_end:match.start()]))

            # Create a clickable link
            text, url = match.groups()
            parts.append(Text(text, style="link " + url))

            last_end = match.end()

        # Add any remaining text
        if last_end < len(code):
            parts.append(Text(code[last_end:]))

        text = Text.join("", parts)
        yield text
    else:
        # If no links, use regular syntax highlighting
        syntax = Syntax(
            code,
            self.lexer_name,
            theme="monokai",
            background_color="default",
            word_wrap=True,
            padding=0
        )
        yield syntax