import os
import sys

import click

PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from typing import Dict, List, Optional

from helpers.logging_helpers import setup_logging, suppress_logging
from helpers.vector_store import VectorStore
from objects import Content, LLMCall, User
from openai_executor import OpenAIExecutor
from repl import agents
from runtime import ExecutionController

suppress_logging()

def run_ast_tests(
    openai_key: str,
    test_filename: str
):

    executor = OpenAIExecutor(openai_key)
    store = VectorStore()
    controller = ExecutionController([executor], agents, store)

    tests: List[str] = []
    with open(test_filename, 'r') as f:
        tests = f.readlines()
        tests = [test for test in tests if len(test) > 1]

    with open('tests/test_results.txt', 'w') as f:
        for test in tests:
            assistant = executor.execute_with_agents(
                LLMCall(messages=[User(Content(test))]),
                agents,
                0.5,
            )
            print('Test: ' + test)
            print()
            print(assistant.message)
            print()
            print()

            f.write('Test: ' + test + '\n')
            f.write(str(assistant.message) + '\n')
            f.write('\n')
            f.flush()


@click.command()
def main(
):
    if not os.environ.get('OPENAI_API_KEY'):
        raise Exception('OPENAI_API_KEY environment variable not set')

    run_ast_tests(os.environ.get('OPENAI_API_KEY'), 'tests/test_examples.txt')  # type: ignore


if __name__ == '__main__':
    main()
