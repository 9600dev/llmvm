import os
import sys

import click

sys.path.append('..')

from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.objects import Content, User
from llmvm.common.openai_executor import OpenAIExecutor


def get_tokens(num_tokens: int, executor_name: str = 'openai'):
    # start with roughly 1.3x the number of tokens as words
    num_words = num_tokens * 1.3
    words = []

    with open('../docs/war_and_peace.txt', 'r') as f:
        word_str = f.read().split(' ')
        word_str[:int(num_words)]

    executor = OpenAIExecutor('', 'gpt-4')
    if executor_name == 'openai':
        executor = OpenAIExecutor('', 'gpt-4')
    elif executor_name == 'anthropic':
        executor = AnthropicExecutor('', 'claude-2.1')

    total_tokens = 0
    min_words = 0
    max_words = len(word_str)

    while True:
        total_tokens = executor.count_tokens([User(Content(' '.join(words)))])

        if num_tokens * 0.96 <= total_tokens <= num_tokens * 1.04:
            break  # Desired range reached

        if total_tokens < num_tokens * 0.96:
            # Add words
            additional_words = 4
            words.extend(word_str[:additional_words])
            word_str = word_str[additional_words:]
        else:
            # Remove words
            words_to_remove = 4
            words = words[:-words_to_remove]

    return words


    # # while total_tokens is not within 4% of num_tokens, keep adding or subtracting words
    # while total_tokens < num_tokens * 0.96 or total_tokens > num_tokens * 1.04:
    #     total_tokens = executor.count_tokens([User(Content(' '.join(words)))])

    #     if total_tokens < num_tokens * 0.96:
    #         words.append(word_str.pop(0))
    #     else:
    #         words.pop()

    # return words


@click.command()
@click.option('--tokens', '-t', default=0, required=True,
              help='number of tokens to extract from war and peace')
@click.option('--output', '-o', default='output.txt', required=False)
@click.option('--executor', '-e', default='openai', required=False)
def main(
    tokens: int,
    output: str,
    executor: str,
):
    print(f'generating {tokens} war and peace tokens')
    print(f'and outputting to {output}')

    words = get_tokens(tokens)
    with open(output, 'w') as f:
        f.write(' '.join(words))
        f.write('\n\n')

if __name__ == '__main__':
    main()
