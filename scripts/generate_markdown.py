import os
import sys
from typing import List
import rich

import click

sys.path.append('..')

from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.objects import Content, ImageContent, Message, User, Executor
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.common.pdf import Pdf
from llmvm.client.client import llm, llmvm
from llmvm.common.container import Container
from llmvm.common.helpers import Helpers


@click.command()
@click.argument('pdf_path', type=str, required=False, default='')
@click.option('--executor', '-e', default='anthropic', required=True)
@click.option('--model', '-m', default='', required=False)
@click.option('--output_path', '-o', default='./', required=True)
@click.option('--title_as_dir', '-t', default=False, is_flag=True, required=False)
def main(
    pdf_path: str,
    executor: str,
    model: str,
    output_path: str,
    title_as_dir: bool,
):
    if not pdf_path:
        rich.print('[red]Please provide a PDF file path[/red]')
        sys.exit(1)

    if not os.path.exists(pdf_path):
        rich.print(f'[red]File not found: {pdf_path}[/red]')
        sys.exit(1)

    output_path = os.path.expanduser(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.abspath(output_path)

    if model:
        e = AnthropicExecutor(default_model=model) if executor == 'anthropic' else OpenAIExecutor(default_model=model)
    else:
        e = AnthropicExecutor() if executor == 'anthropic' else OpenAIExecutor()

    pdf = Pdf(e)
    messages: List[Content] = pdf.get_pdf(pdf_path)

    pdf_content = ''

    TITLE_PROMPT = """
    The previous message contains the first page of a PDF file.
    I want you to extract the best possible title of the PDF file you can and
    convert that title into a directory path that will work on macos, linux and windows.
    For example, if you extract a title like "Greatest Hits of the 80s", the directory path should be "greatest_hits_of_the_80s".
    Only return the directory name, nothing else. Do not include the full path, just the directory name.
    """

    generated_title_path = False

    for i, message in enumerate(messages):
        if type(message) is Content and title_as_dir and not generated_title_path:
            result = message.get_str()
            title_response = llm(
                messages=[User(Content(result)), User(Content(TITLE_PROMPT))],
                executor=e,
            )
            output_path += os.path.sep + title_response.get_str().strip()
            rich.print(f'[green]Title of the PDF extracted and converted to directory path {output_path}[/green]')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            generated_title_path = True

        if type(message) is ImageContent:
            image_path = os.path.join(output_path, f'image_{i}.png')
            image_name = f'image_{i}.png'
            with open(image_path, 'wb') as f:
                f.write(message.sequence)
            image_ref = f'![image_{i}]({image_name})'
            pdf_content += f'{image_ref}\n'
            rich.print(f'[green]Image saved to {image_path}[/green]')
        if type(message) is Content:
            pdf_content += message.get_str() + '\n'

    PROMPT = f"""
    The previous message contains text from a PDF file extracted from the following path: {pdf_path}.
    I want you to convert this text into Markdown. Do a thorough job. Don't miss any text.
    There are image references in the PDF text that are already in Markdown format (e.g. ![text](image_path) format).
    These image references have been placed at the bottom of each converted PDF page.
    If you think you can move these image references to a more appropriate location within the text of the page, feel free to do so.
    Do not emit any preamble or metadata, just the converted Markdown content.
    """

    pdf_content = User(Content(pdf_content))
    prompt = User(Content(PROMPT))

    response = llm(
        messages=[pdf_content, prompt],
        executor=e,
    )

    responses = [response]
    while response.stop_reason == 'max_tokens':
        rich.print(f'[yellow]Max tokens reached. Asking for continuation.[/yellow]')
        response = llm(
            messages=Helpers.flatten([pdf_content, prompt, responses]),
            executor=e,
        )

    rich.print('[green]Finished.[/green]')

    markdown_content = response.message.get_str()
    with open(os.path.join(output_path, 'index.md'), 'w') as f:
        f.write(markdown_content)

    rich.print(f'[green]Markdown content saved to {output_path}/index.md[/green]')

if __name__ == '__main__':
    main()

