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
@click.option('--output_path', '-o', default='output', required=True)
def main(
    pdf_path: str,
    executor: str,
    model: str,
    output_path: str,
):
    if not pdf_path:
        rich.print('[red]Please provide a PDF file path[/red]')
        sys.exit(1)

    if not os.path.exists(pdf_path):
        rich.print(f'[red]File not found: {pdf_path}[/red]')
        sys.exit(1)

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

    for i, message in enumerate(messages):
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
    There are image references in the PDF text that are already in Markdown format (e.g. ![text](image_path) format). Keep them.
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
