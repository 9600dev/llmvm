#!/usr/bin/env python3
import glob
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import subprocess
import shlex
import re
import datetime as dt
import click

from typing import Tuple, Any, Optional, Union, List, Dict
from pydantic import BaseModel
from llmvm.client.client import llm
from llmvm.common.helpers import Helpers
from llmvm.common.objects import Message, TextContent, User

from pathlib import Path


class CommentaryPair(BaseModel, arbitrary_types_allowed=True):
    a: Message
    b: Message
    commentary: str

    class Config:
        json_encoders = {
            Message: lambda u: u.to_json()
        }


class Comparison(BaseModel, arbitrary_types_allowed=True):
    date_time: str = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    test_name: str
    commentaries: list[CommentaryPair]
    model_a: str
    model_b: str
    model_a_command: Optional[str]
    model_b_command: Optional[str]


def line_up_messages(
        model_a_messages: list[Message],
        model_b_messages: list[Message]
) -> Tuple[list[Message], list[Message]]:
    def debug():
        click.echo(f'debug(): length of a_msg: {len(a_msg)}, length of b_msg: {len(b_msg)}')
        for i in range(len(a)):
            click.echo(f'debug():  {i+1} .. a[{i}]: {a[i].role()} len(a): {len(a[i].get_str())}, b[{i}]: {b[i].role()} len(b): {len(b[i].get_str())}')
        click.echo()

    a_msg = model_a_messages.copy()
    b_msg = model_b_messages.copy()

    def extract_to_assistant(msgs: list[Message], index: int) -> list[Message]:
        for i in range(index, len(msgs)):
            if msgs[i].role() == 'assistant':
                return msgs[index:i+1]
        return msgs[index:]

    a_index = 0
    b_index = 0
    max_outputs = max(len(model_a_messages), len(model_b_messages))

    a = []
    b = []

    i = 0
    while i < max_outputs:
        a_block = extract_to_assistant(a_msg, a_index)
        b_block = extract_to_assistant(b_msg, b_index)

        a_index += len(a_block)
        b_index += len(b_block)

        for j in range(max(len(a_block) - 1, len(b_block) - 1)):
            if j < len(a_block) and j < len(b_block):
                a.append(a_block[j])
                b.append(b_block[j])
            elif j >= len(a_block) and j < len(b_block):
                a.append(User(TextContent('')))
                b.append(b_block[j])
            elif j < len(a_block) and j >= len(b_block):
                a.append(a_block[j])
                b.append(User(TextContent('')))

        # add the assistant messages
        if len(a_block) > 0:
            a.append(a_block[-1])
        else:
            a.append(User(TextContent('')))

        if len(b_block) > 0:
            b.append(b_block[-1])
        else:
            b.append(User(TextContent('')))

        i = max(a_index, b_index) + 1

    debug()

    if not a[-1].role() == 'assistant' or not b[-1].role() == 'assistant':
        a[-1] = a_msg[-1]
        b[-1] = b_msg[-1]

        for i in range(len(a) - 2, 0, -1):
            if a[i] == a[-1]:
                a[i] = User(TextContent(''))

        for i in range(len(b) - 2, 0, -1):
            if b[i] == b[-1]:
                b[i] = User(TextContent(''))

    debug()

    return a, b


def llm_line_up(model_a_messages: list[Message], model_b_messages: list[Message]) -> Tuple[list[Message], list[Message]]:
    PROMPT = f"""
    Your job is to take two lists of User/Assistant exchanges from two different LLMs and align the messages so that they can be
    efficiently compared.

    The last message of both lists is always an Assistant message that should be an answer the first User message. These should
    always be aligned.

    The first message of both lists is always a User message with a task or query. These should always be aligned.

    The lists can and are usually different sizes, but in general, there will be pairs of messages that look similar. You should align these.

    The lists should be equal length at the end. Alignment means that a[2] should be aligned to a similar message in b[2].

    The resulting lists should be the same size as the max size of the two lists. No exception.
    For this task, the max size is {max(len(model_a_messages), len(model_b_messages))}.

    You can't reorder the messages. You can only align them. You can remove messages with no text, which is a way
    you may reorder things.

    If a message contains no text, then you are free to remove that message from the list or leave it as is. You can use
    these spots in the list to help you align the messages and move the order around.

    I'm going to give you list of model (a) first. Then the list of model (b). You'll get the role (User or Assistant)
    and a brief summary of the content - usually the first 500 characters of the message. You also get the message unique id.

    You should return two Python lists of messages of the same length. They will have the id's of the messages you want
    to order. If you need to create a new empty message that represents that there is no message alignment, just use "None".

    Only return two Python lists. Don't return anything else.

    Example returned output:

    ["a_1", "a_2", None, None, "a_3"]
    ["b_1", "b_2", "b_3", "b_4", "b_5"]

    The two lists will be in the next two messages as json strings. (a) will be first, (b) will be second.
    """

    a = []
    b = []

    for i, message in enumerate(model_a_messages):
        a.append({
            'id': f'a_{i}',
            'role': message.role(),
            'content': message.get_str()[:500],
        })
        b.append({
            'id': f'b_{i}',
            'role': message.role(),
            'content': message.get_str()[:500],
        })

    click.echo(f'before reorder (a): {[a["id"] for a in a]}')
    click.echo(f'before reorder (b): {[b["id"] for b in b]}')

    assistant = llm([User(TextContent(PROMPT)), User(TextContent(json.dumps(a))), User(TextContent(json.dumps(b)))])

    click.echo()
    click.echo(assistant.get_str())

    lists = Helpers.parse_lists_from_string(assistant.get_str())
    list_a = lists[0]
    list_b = lists[1]

    new_a = []
    new_b = []

    for i in range(len(list_a)):
        if list_a[i] != None:
            new_a.append(model_a_messages[int(list_a[i].replace("a_", ""))])
        else:
            new_a.append(User(TextContent('')))

    for i in range(len(list_b)):
        if list_b[i] != None:
            new_b.append(model_b_messages[int(list_b[i].replace("b_", ""))])
        else:
            new_b.append(User(TextContent('')))

    for i in range(len(new_a)):
        click.echo(f"  {i+1} .. {list_b[i]} == {list_a[i]}")

    return new_a, new_b


def llm_message_commentary(
    model_a_messages: list[Message],
    model_b_messages: list[Message],
    a_msg: Message,
    b_msg: Message
) -> str:
    PROMPT = """
    Your job is to to take two messages from two different LLMs and figure out what the differences are so that a
    human can understand the 'vibes' differences between the two messages. You should be opinionated about what
    the differences are, and express a preference for one or the other.

    The previous two messages contain the full message lists from both LLMs. First message is the full conversation of
    model (a), then the second message is the full conversation of model (b).

    You're going to use these previous messages as context only to figure out the major interesting differences
    between the pair of single messages from (a) and (b) that I'm going to give you now in the next two messages.

    Look at the pair. Compare them. Use the previous messages for context. Give me an opinionated answer on
    what you think the differences are, what could be done better and what could be done differently.
    Be very succinct as there isn't a lot of room for a detailed explanation.

    If the pair is the same, then you can just say 'same' and move on.

    If you see code blocks in the messages, you can generally ignore them unless you think there's something
    interesting going on that is relevant to the overall difference between the two sets of messages.
    (i.e. there was some broken code or something which meant that the future messages of that broken code were bad or
    took a different direction compared to the other set of messages from the other llm).
    <helper_results> blocks are the output of tools, so you can completely ignore them.
    <helpers> blocks are code generated from the LLM, so you can pay attention to them.

    Model (a) full message will be first. Then model (b) full message.
    """
    a = []
    b = []
    for i, message in enumerate(model_a_messages):
        a.append({
            'id': f'a_{i}',
            'role': message.role(),
            'content': message.get_str()[:500],
        })

    for i, message in enumerate(model_b_messages):
        b.append({
            'id': f'b_{i}',
            'role': message.role(),
            'content': message.get_str()[:500],
        })

    a_msg_json = {
        'role': a_msg.role(),
        'content': a_msg.get_str()
    }
    b_msg_json = {
        'role': b_msg.role(),
        'content': b_msg.get_str()
    }

    assistant = llm([
        User(TextContent(json.dumps(a))),
        User(TextContent(json.dumps(b))),
        User(TextContent(PROMPT)),
        User(TextContent(json.dumps(a_msg_json))),
        User(TextContent(json.dumps(b_msg_json))),
    ])

    assistant_result = assistant.get_str().replace("$", "\\$")
    click.echo(f'\nCommentary:\n\n{assistant_result}')
    return assistant_result


def get_model_alias_command(model_name: str) -> str:
    """
    Get the full command for a model alias by:
      1. Reading ~/.zshrc to find the alias definition.
      2. Extracting the alias command without sourcing the entire zshrc,
         so that your current conda environment remains active.
      3. Using zsh's eval to expand any environment variables using the
         current os.environ.

    Parameters:
        model_name (str): The alias name to look up.

    Returns:
        str: The fully expanded command string.

    Raises:
        ValueError: If the alias is not found or expansion fails.
    """
    # Expand the path to ~/.zshrc
    zshrc_path = os.path.expanduser("~/.zshrc")
    try:
        with open(zshrc_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Failed to open {zshrc_path}: {e}")

    # Look for a line that defines the alias.
    # This regex assumes the alias is defined on a single line like:
    #   alias camel='ANTHROPIC_API_KEY=$ANT_KEY ... llm'
    pattern = re.compile(rf"^\s*alias\s+{re.escape(model_name)}=")
    alias_line = None
    for line in lines:
        if pattern.match(line):
            alias_line = line.strip()
            break

    if not alias_line:
        raise ValueError(f"Model alias '{model_name}' not found in {zshrc_path}")

    # Extract the command definition from the alias line.
    # Remove everything up to and including the equals sign.
    cmd_def = re.sub(r'^[^=]*=', '', alias_line).strip()

    # Remove surrounding quotes (single or double) if present.
    if (cmd_def.startswith("'") and cmd_def.endswith("'")) or \
       (cmd_def.startswith('"') and cmd_def.endswith('"')):
        cmd_def = cmd_def[1:-1]

    # Optionally remove a trailing ' llm' if that's known to be appended.
    cmd_def = re.sub(r'\s+llm$', '', cmd_def)

    # Use zsh to expand environment variables in the command.
    # We run zsh without sourcing ~/.zshrc so that the current environment (e.g. conda) is preserved.
    eval_cmd = f"eval echo {shlex.quote(cmd_def)}"
    result = subprocess.run(
        ["zsh", "-c", eval_cmd],
        capture_output=True,
        text=True,
        env=os.environ.copy()  # use current environment
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(f"Failed to expand environment variables in alias for '{model_name}'")

    expanded_cmd = result.stdout.strip()
    return expanded_cmd


def run_single_test(
    test_item: Dict[str, Any],
    model: str,
    output_file: str,
    base_dir: Path
) -> None:
    """Run a single test for a specific model and save the output to a file."""
    # Get the model's full command from alias
    try:
        model_cmd = get_model_alias_command(model)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Build the command with all arguments
    args = test_item.get('args', [])

    # Make paths relative to the base directory
    fixed_args = []
    for arg in args:
        if arg.startswith('-p'):
            fixed_args.append(arg)
        else:
            # Convert to absolute path if it's a file path
            arg_path = base_dir / arg
            if arg_path.exists():
                fixed_args.append(str(arg_path))
            else:
                fixed_args.append(arg)

    # get the test_runner.py file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Build the full command
    full_cmd = (
        f"{model_cmd} "
        f"LLMVM_SERIALIZE={output_file} "
        f"PYTHONPATH={parent_dir} "
        f"python -m llmvm.client {' '.join(fixed_args)}"
    )

    click.echo(f"Running command:\n{full_cmd}\n")

    # Run the command
    result = subprocess.run(["zsh", "-c", full_cmd], cwd=str(base_dir))

    if result.returncode != 0:
        click.echo(f"Warning: Test '{test_item['name']}' for model '{model}' failed with exit code {result.returncode}", err=True)


def find_test_by_name(json_file: Path, test_name: str) -> Optional[Dict[str, Any]]:
    """
    Find a specific test in the JSON file by name.

    Args:
        json_file: Path to the JSON file containing test specifications
        test_name: Name of the test to find

    Returns:
        The test specification if found, None otherwise
    """
    with open(json_file, 'r') as f:
        test_specs = json.load(f)

    for test_item in test_specs:
        if test_item.get('name') == test_name:
            return test_item

    return None


def run_single_test_by_name(
    json_file: Union[str, Path],
    test_name: str,
    model_a: str,
    model_b: str,
    output_dir: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Run a single specific test for two different models.

    Args:
        json_file: Path to the JSON file containing test specifications
        test_name: Name of the test to run
        model_a: Name of the first model to test
        model_b: Name of the second model to test
        output_dir: Directory to save test results
        base_dir: Base directory for test files (defaults to json_file's directory)
    """
    # Convert all paths to Path objects
    json_file = Path(json_file).resolve()
    output_dir = Path(output_dir).resolve()

    # Set base_dir to json_file's directory if not specified
    if base_dir is None:
        base_dir = json_file.parent
    else:
        base_dir = Path(base_dir).resolve()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the specified test
    test_item = find_test_by_name(json_file, test_name)
    if not test_item:
        click.echo(f"Error: Test '{test_name}' not found in {json_file}", err=True)
        sys.exit(1)

    click.echo(f"Running test: {test_name}")

    # Output files for model results
    output_file1 = output_dir / f"{test_name}_{model_a}.json"
    output_file2 = output_dir / f"{test_name}_{model_b}.json"

    # Run test for first model
    click.echo(f"Running test for model '{model_a}'")
    run_single_test(test_item, model_a, str(output_file1), base_dir)

    # Run test for second model
    click.echo(f"Running test for model '{model_b}'")
    run_single_test(test_item, model_b, str(output_file2), base_dir)

    click.echo(f"Completed test: {test_name}")
    click.echo(f"\nTest completed. Results saved to {output_dir}")


def run_tests_from_json(
    json_file: Union[str, Path],
    model_a: str,
    model_b: str,
    output_dir: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Run tests specified in a JSON file for two different models.

    Args:
        json_file: Path to the JSON file containing test specifications
        model_a: Name of the first model to test
        model_b: Name of the second model to test
        output_dir: Directory to save test results
        base_dir: Base directory for test files (defaults to json_file's directory)
    """
    # Convert all paths to Path objects
    json_file = Path(json_file).resolve()
    output_dir = Path(output_dir).resolve()

    # Set base_dir to json_file's directory if not specified
    if base_dir is None:
        base_dir = json_file.parent
    else:
        base_dir = Path(base_dir).resolve()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test specifications from JSON
    with open(json_file, 'r') as f:
        test_specs = json.load(f)

    click.echo(f"Running {len(test_specs)} tests from {json_file}")

    # Run each test for both models
    for i, test_item in enumerate(test_specs, 1):
        test_name = test_item['name']
        click.echo(f"\nRunning test {i}/{len(test_specs)}: {test_name}")

        # Output files for model results
        output_file1 = output_dir / f"{test_name}_{model_a}.json"
        output_file2 = output_dir / f"{test_name}_{model_b}.json"

        # Run test for first model
        click.echo(f"Running test for model '{model_a}'")
        run_single_test(test_item, model_a, str(output_file1), base_dir)

        # Run test for second model
        click.echo(f"Running test for model '{model_b}'")
        run_single_test(test_item, model_b, str(output_file2), base_dir)

        click.echo(f"Completed test: {test_name}")

    click.echo(f"\nAll tests completed. Results saved to {output_dir}")


def make_test_comparison(
    a_json: str,
    b_json: str,
    test_name: str,
    model_a: str,
    model_b: str,
    commentary: bool = True
) -> Comparison:
    if not (os.path.exists(a_json) and os.path.exists(b_json)):
        raise ValueError(f"JSON files not found: {a_json} and {b_json}")

    model_a_messages = Helpers.deserialize_messages(a_json)
    model_b_messages = Helpers.deserialize_messages(b_json)

    click.echo(f"Lining up messages for {a_json} and {b_json}")
    # Convert Message objects to a ComparisonSet
    model_a_messages, model_b_messages = line_up_messages(model_a_messages, model_b_messages)

    # we have what we think is a basic line up. let's see if the llm can do better.
    model_a_messages, model_b_messages = llm_line_up(model_a_messages, model_b_messages)

    # Initialize commentary list
    commentary_list = [''] * len(model_a_messages)

    if commentary:
        click.echo(f"Generating commentary for {a_json} and {b_json}")
        for i in range(1, len(model_a_messages)):
            if len(model_a_messages[i].get_str()) > 10 and len(model_b_messages[i].get_str()) > 10:
                commentary_text = llm_message_commentary(model_a_messages, model_b_messages, model_a_messages[i], model_b_messages[i])
                commentary_list[i] = commentary_text

    commentaries = []
    for i in range(len(model_a_messages)):
        commentaries.append(CommentaryPair(a=model_a_messages[i], b=model_b_messages[i], commentary=commentary_list[i]))

    return Comparison(
        commentaries=commentaries,
        model_a=model_a,
        model_b=model_b,
        test_name=test_name,
        model_a_command='',
        model_b_command='',
    )


def make_test_comparisons(
    file_path_1: Union[str, list[str]],
    file_path_2: Union[str, list[str]],
    test_name: str,
    model_a: str,
    model_b: str,
    commentary: bool = True,
) -> list[Comparison]:
    # deals with globs and other stuff
    test_pairs: list[Tuple[str, str]] = []

    # deal with the glob
    if isinstance(file_path_1, str) and isinstance(file_path_2, str) and '*' in file_path_1 and '*' in file_path_2:
        import glob
        file_path_1 = sorted(glob.glob(file_path_1))
        file_path_2 = sorted(glob.glob(file_path_2))
    elif not isinstance(file_path_1, list) and not isinstance(file_path_2, list):
        file_path_1 = [file_path_1]
        file_path_2 = [file_path_2]

    if len(file_path_1) != len(file_path_2):
        raise ValueError(f"Number of files in file_path_1 {len(file_path_1)} does not match number of files in file_path_2 {len(file_path_2)}")

    for file_1, file_2 in zip(file_path_1, file_path_2):
       test_pairs.append((file_1, file_2))

    click.echo(f"Found {len(test_pairs)} test pairs.")

    comparisons = []
    counter = 0
    for file_1, file_2 in test_pairs:
        if len(file_path_1) > 1:
            test_name = os.path.basename(file_1).split('_')[0]

        click.echo()
        click.echo(f"{counter+1}/{len(test_pairs)} - Comparing {file_1} and {file_2} for test {test_name}")

        comparison: Comparison = make_test_comparison(file_1, file_2, test_name, model_a, model_b, commentary)
        comparisons.append(comparison)
        counter+=1
    return comparisons


@click.group()
def cli():
    """LLM Test Runner - Run and compare tests for different LLM models."""
    pass


@cli.command('run-tests')
@click.argument('json_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('model_a', type=str)
@click.argument('model_b', type=str)
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.option('--base-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              help="Base directory for test files (defaults to json_file's directory)")
def run_tests(json_file, model_a, model_b, output_dir, base_dir):
    """Run tests and produce comparisons.

    JSON_FILE: Path to the JSON file containing test specifications
    MODEL_A: First model alias to test
    MODEL_B: Second model alias to test
    OUTPUT_DIR: Directory to save test results
    """
    run_tests_from_json(
        json_file=json_file,
        model_a=model_a,
        model_b=model_b,
        output_dir=output_dir,
        base_dir=base_dir
    )


@cli.command('run-tests-single')
@click.argument('test_name', type=str)
@click.argument('json_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument('model_a', type=str)
@click.argument('model_b', type=str)
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.option('--base-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
              help="Base directory for test files (defaults to json_file's directory)")
def run_tests_single(test_name, json_file, model_a, model_b, output_dir, base_dir):
    """Run a single test and produce comparison.

    TEST_NAME: Name of the specific test to run
    JSON_FILE: Path to the JSON file containing test specifications
    MODEL_A: First model alias to test
    MODEL_B: Second model alias to test
    OUTPUT_DIR: Directory to save test results
    """
    run_single_test_by_name(
        json_file=json_file,
        test_name=test_name,
        model_a=model_a,
        model_b=model_b,
        output_dir=output_dir,
        base_dir=base_dir
    )


@cli.command('make-comparisons')
@click.argument('file_or_glob_a', type=str)
@click.argument('file_or_glob_b', type=str)
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.option('--model_a', type=str, default="", help="Name for model 1")
@click.option('--model_b', type=str, default="", help="Name for model 2")
@click.option('--test-name', type=str, default="", help="Name for the test (if not glob)")
def make_comparisons(file_or_glob_a, file_or_glob_b, output_dir, model_a, model_b, test_name):
    """Generate comparison(s) between two models for a given pair of json model traces"""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if (
        '*' not in file_or_glob_a
        and '*' not in file_or_glob_b
        and model_a == ''
        and model_b == ''
        and '_' in file_or_glob_a
        and '_' in file_or_glob_b
    ):
        model_a = file_or_glob_a.split('_')[-1].replace('.json', '')
        model_b = file_or_glob_b.split('_')[-1].replace('.json', '')

        if test_name == '':
            if '/' in file_or_glob_a:
                test_name = file_or_glob_a.split('/')[-1].split('_')[0]
            else:
                test_name = file_or_glob_a.split('_')[0]

    # globs need to work over lots of files, so make sure the file
    # name contains the test name.
    if '*' in file_or_glob_a:
        test_name = ''

    comparisons: list[Comparison] = make_test_comparisons(
        file_or_glob_a,
        file_or_glob_b,
        test_name,
        model_a,
        model_b,
        True
    )

    for comparison in comparisons:
        output_file = os.path.join(output_dir, f"{comparison.test_name}_{comparison.model_a}_{comparison.model_b}.json")
        click.echo(f"Writing out test comparison to {output_file}")
        with open(output_file, 'w') as f:
            f.write(comparison.model_dump_json(indent=2))


@cli.command('show-comparisons')
@click.argument('file_or_glob', type=str)
def show_comparisons(file_or_glob):
    """Run Streamlit viewer of comparisons.
    FILE_OR_GLOB: Path to the JSON file or glob containing comparison results
    """
    # get the test_runner.py file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Prepare the Streamlit command
    streamlit_base_cmd = (
        f"PYTHONPATH={parent_dir} "
        f"python {parent_dir}/scripts/comparison/main.py"
    )

    full_cmd = f"{streamlit_base_cmd} \"{file_or_glob}\""
    click.echo(f"Starting Streamlit comparison:\n{full_cmd}\n")
    subprocess.run(["zsh", "-c", full_cmd])


@cli.command('test-compare-and-show')
@click.argument('json_file', type=str)
@click.argument('model_a', type=str)
@click.argument('model_b', type=str)
@click.argument('test_output_dir', type=str)
@click.argument('comparison_output_dir', type=str)
@click.option('--clean', is_flag=True, default=False, help="Clean the output directories before running")
@click.pass_context
def test_compare_and_show(
    ctx,
    json_file,
    model_a,
    model_b,
    test_output_dir,
    comparison_output_dir,
    clean
):
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    if not os.path.exists(comparison_output_dir):
        os.makedirs(comparison_output_dir)

    click.echo(f'Cleaning output directories: {test_output_dir} and {comparison_output_dir}')
    if clean:
        for file in glob.glob(f"{test_output_dir}/*"):
            os.remove(file)
        for file in glob.glob(f"{comparison_output_dir}/*"):
            os.remove(file)

    """Generate test comparisons then show the results"""
    # def run_tests(json_file, model_a, model_b, output_dir, base_dir):
    ctx.invoke(
        run_tests,
        json_file=json_file,
        model_a=model_a,
        model_b=model_b,
        output_dir=test_output_dir
    )

    # def make_comparisons(file_or_glob_a, file_or_glob_b, output_dir, model_a, model_b, test_name):
    ctx.invoke(
        make_comparisons,
        file_or_glob_a=f"{test_output_dir}/*_{model_a}.json",
        file_or_glob_b=f"{test_output_dir}/*_{model_b}.json",
        output_dir=comparison_output_dir,
        model_a=model_a,
        model_b=model_b,
        test_name=''
    )

    ctx.invoke(
        show_comparisons,
        file_or_glob=f"{comparison_output_dir}/*_{model_a}_{model_b}.json"
    )


if __name__ == "__main__":
    cli()
