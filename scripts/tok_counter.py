import asyncio
import os
import sys
import click

sys.path.append('..')

from llmvm.common.openai_executor import OpenAIExecutor

def count_tokens(text: str) -> int:
    executor = OpenAIExecutor()
    return asyncio.run(executor.count_tokens(text))


@click.command()
@click.argument('filename')
def main(filename: str):
    with open(filename, 'r') as file:
        text = file.read()
        click.echo(count_tokens(text))


if __name__ == '__main__':
    main()
