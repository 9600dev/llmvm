import asyncio
import base64
import io
import os
import random
import re
import sys
from typing import Optional, cast
from urllib.parse import urlparse

import click
import rich

sys.path.append('..')

from llmvm.client.client import get_executor, llm
from llmvm.common.helpers import Helpers
from llmvm.common.object_transformers import ObjectTransformers
from llmvm.common.objects import (Assistant, AstNode, Content, Executor,
                                  MarkdownContent, Message, TextContent, User)
from llmvm.common.pdf import Pdf
from llmvm.server.tools.chrome import ChromeHelpers
from llmvm.server.tools.webhelpers import WebHelpers


async def stream_handler(node: AstNode):
    pass


def pdf_to_markdown(pdf_url: str, executor: Executor) -> MarkdownContent:
    pdf = Pdf(executor=executor)
    content: list[Content] = pdf.get_pdf(pdf_url)
    supported_content = Helpers.flatten([ObjectTransformers.transform_to_supported_content(c, executor) for c in content])

    PROMPT = f"""
    The previous message contains text from a PDF file extracted from the following path: {pdf_url}.
    I want you to convert this text into Markdown. Do a thorough job. Don't miss any text.
    There are image references in the PDF text that are already in Markdown format (e.g. ![text](image_path) format).
    These image references have been placed at the bottom of each converted PDF page.
    If you think you can move these image references to a more appropriate location within the text of the page, feel free to do so.
    Do not emit any preamble or metadata, just the converted Markdown content.
    """

    context_messages: list[Message] = [User(c) for c in supported_content]
    prompt_message = User(TextContent(PROMPT))

    assistant: Assistant = llm(
        messages=cast(list[Message], context_messages + [prompt_message]),
        executor=executor,
        stream_handler=stream_handler
    )

    continuation_messages: list[Message] = []

    while assistant.stop_reason == 'max_tokens':
        continuation_messages.append(assistant)
        continuation_messages.append(User(TextContent("please continue from your last message.")))
        assistant: Assistant = llm(
            messages=cast(list[Message], context_messages + [prompt_message] + continuation_messages),
            executor=executor,
            stream_handler=stream_handler
        )

    full_stack = context_messages + [prompt_message] + continuation_messages + [assistant]
    assistant_content: list[Content] = Helpers.flatten([message.message for message in full_stack if isinstance(message, Message)])
    return MarkdownContent(assistant_content, url=pdf_url)


def url_to_markdown(url: str, inline_images: bool = False, browser: bool = False) -> MarkdownContent:
    def get_image_urls(text):
        return re.findall(r"!\[.*?\]\((.*?)\)", text)

    def inline_image(text, image_url, url):
        if image_url.startswith('/') and url.startswith('http'):
            image_url = url + image_url

        if image_url.startswith('http'):
            image_bytes = asyncio.run(Helpers.download_bytes(image_url, throw=False))
            if Helpers.is_image(image_bytes):
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = Helpers.classify_image(image_bytes)
                markdown_pattern = rf'!\[(.*?)\]\({re.escape(image_url)}(?:\s+"[^"]*")?\)'

                match = re.search(markdown_pattern, text)
                if match:
                    alt_text = match.group(1) or f"image_{str(random.randint(10, 100000))}"
                else:
                    alt_text = f"image_{str(random.randint(10, 100000))}"

                inline_image = f'![{alt_text}](data:{mime_type};base64,{base64_image})'
                text = re.sub(markdown_pattern, inline_image, text)
        elif os.path.exists(image_url):
            try:
                image_bytes = open(image_url, 'rb').read()
                if Helpers.is_image(image_bytes):
                    text = text.replace(url, f"data:image/{Helpers.classify_image(image_bytes)};base64,{base64.b64encode(image_bytes).decode('utf-8')}")
            except Exception as ex:
                pass
        return text

    if os.path.exists(url):
        markdown = WebHelpers.convert_html_to_markdown(open(url, 'r').read(), url=url).get_str()
    else:
        html = ''
        if browser:
            chrome = ChromeHelpers()
            asyncio.run(chrome.goto(url=url))
            asyncio.run(chrome.wait(1500))
            html = asyncio.run(chrome.get_html())
        else:
            html = asyncio.run(Helpers.download(url))

        markdown = WebHelpers.convert_html_to_markdown(html, url=url).get_str()

    if inline_images:
        for image_url in get_image_urls(markdown):
            markdown = inline_image(markdown, image_url, url)

    return MarkdownContent([TextContent(markdown)], url=url)


@click.command()
@click.argument('url', type=str, required=False, default='')
@click.option('--executor_name', '-e', default='anthropic', required=True)
@click.option('--model', '-m', default='', required=False)
@click.option('--output', '-o', default='', required=False)
@click.option('--inline_images', '-i', is_flag=True, default=False, required=False)
@click.option('--browser', '-b', is_flag=True, default=False, required=False)
def main(
    url: str,
    executor_name: str,
    model: str,
    output: str,
    inline_images: bool,
    browser: bool,
):
    if not url:
        rich.print('[red]Please provide a url[/red]')
        sys.exit(1)

    executor = get_executor(executor_name, model, '')
    result = urlparse(os.path.expanduser(url))
    rich.print(f'[green]Converting {result.geturl()} to markdown[/green]', file=sys.stderr)
    markdown_result: Optional[MarkdownContent] = None

    if '.pdf' in result.geturl():
        markdown_result = pdf_to_markdown(result.geturl(), executor)
    elif 'http' in result.scheme:
        markdown_result = url_to_markdown(result.geturl(), inline_images=inline_images, browser=browser)

    if markdown_result:
        if output:
            Helpers.write_markdown(markdown_result, open(output, 'w'))
            rich.print(f'[green]Markdown content saved to {output}[/green]', file=sys.stderr)
        else:
            str_io = io.StringIO()
            Helpers.write_markdown(markdown_result, str_io)
            str_io.seek(0)
            rich.print(str_io.read())
    else:
        rich.print(f'[red]No content found for url {url}[/red]')


if __name__ == '__main__':
    main()

