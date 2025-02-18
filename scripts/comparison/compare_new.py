import json
import os
import uuid
import sys

sys.path.append("../..")

import streamlit as st
from typing import Tuple
from models import ComparisonSet, Comparison, ComparisonPair, ModelOutput
from comparison_ui import ComparisonUI, ComparisonUIContainer
from message_converter import create_comparison_from_message_lists
from llmvm.common.helpers import Helpers
from llmvm.common.objects import Message, User, TextContent
from llmvm.client.client import llm


def line_up_messages(model_a_messages: list[Message], model_b_messages: list[Message]) -> Tuple[list[Message], list[Message]]:
    def debug():
        print(f'length of a_msg: {len(a_msg)}, length of b_msg: {len(b_msg)}')
        for i in range(len(a)):
            print(f'{i+1} .. a[{i}]: {a[i].role()} len(a): {len(a[i].get_str())}, b[{i}]: {b[i].role()} len(b): {len(b[i].get_str())}')
        print()

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

    print(f'before reorder (a): {[a["id"] for a in a]}')
    print(f'before reorder (b): {[b["id"] for b in b]}')

    assistant = llm([User(TextContent(PROMPT)), User(TextContent(json.dumps(a))), User(TextContent(json.dumps(b)))])

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
        print(f"{i+1} .. {list_b[i]} == {list_a[i]}")

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
    print(f'\nCommentary:\n\n{assistant_result}')
    return assistant_result

@st.cache_data
def build_comparison(a_file: str, b_file: str, commentary: bool = True):
    model_a = Helpers.deserialize_messages(a_file)
    model_b = Helpers.deserialize_messages(b_file)

    # Convert Message objects to a ComparisonSet
    model_a_messages, model_b_messages = line_up_messages(model_a, model_b)

    # we have what we think is a basic line up. let's see if the llm can do better.
    model_a_messages, model_b_messages = llm_line_up(model_a_messages, model_b_messages)

    # Initialize commentary list
    commentary_list = [''] * len(model_a_messages)

    # Generate commentary if enabled
    if commentary:
        for i in range(1, len(model_a_messages)):
            if len(model_a_messages[i].get_str()) > 10 and len(model_b_messages[i].get_str()) > 10:
                commentary_text = llm_message_commentary(model_a_messages, model_b_messages, model_a_messages[i], model_b_messages[i])
                commentary_list[i] = commentary_text

    comparison_set = create_comparison_from_message_lists(model_a_messages, model_b_messages, commentary_list=commentary_list)
    return comparison_set


def run_with_files(a_file: str, b_file: str, a_name: str, b_name: str, commentary: bool = True, index: int = 0):
    comparison_set = build_comparison(a_file=a_file, b_file=b_file, commentary=commentary)

    # Initialize the UI with our comparison set
    ui = ComparisonUI(comparison_set, a_name, b_name, index)

    # Render the UI
    ui.render()


def render(
    file_or_glob: str,
):
    files = []

    if '*' in file_or_glob:
        import glob
        files = sorted(glob.glob(file_or_glob))
    else:
        files = [file_or_glob]

    # load the comparisons
    # comparisons have a list of commentary pairs
    comparison_set: list[Comparison] = []

    for file in files:
        with open(file, 'r') as f:
            json_data = json.loads(f.read())
            comparison_set.append(Comparison.model_validate(json_data))

    print(f'Loaded {len(comparison_set)} comparisons')

    comparison_ui_container = ComparisonUIContainer()

    for comparison in comparison_set:
        ui_output_comparison_pairs: list[ComparisonPair] = []
        for commentary_pair in comparison.commentaries:
            a_model_output = ModelOutput(
                content=commentary_pair.a.get_str(),
                role=commentary_pair.a.role(),
                model_id=comparison.model_a,
                output_id=str(uuid.uuid4())
            )
            b_model_output = ModelOutput(
                content=commentary_pair.b.get_str(),
                role=commentary_pair.b.role(),
                model_id=comparison.model_b,
                output_id=str(uuid.uuid4())
            )
            ui_output_comparison_pairs.append(
                ComparisonPair(
                    output_a=a_model_output,
                    output_b=b_model_output,
                    commentary=commentary_pair.commentary
                )
            )

        ui_comparison_set = ComparisonSet(model_a=comparison.model_a, model_b=comparison.model_b)
        for comparison_pair in ui_output_comparison_pairs:
            ui_comparison_set.pairs.append(comparison_pair)

        comparison_ui_container.add_comparison_ui(ui_comparison_set, comparison.model_a, comparison.model_b)

    comparison_ui_container.render()

def streamlit_ui():
    """Run the Streamlit UI version"""
    st.title("Model Output Comparison")

    # Default setup for single comparison
    use_multiple = st.checkbox("Use multiple comparison sets", value=False)

    if use_multiple:
        col1, col2 = st.columns(2)
        with col1:
            a_pattern = st.text_input("Model A glob pattern", value="*.a.json")
            a_name = st.text_input("Model A Name", value="Model A")
        with col2:
            b_pattern = st.text_input("Model B glob pattern", value="*.b.json")
            b_name = st.text_input("Model B Name", value="Model B")

        commentary = st.checkbox("Generate Commentary", value=True)

        if st.button("Compare Outputs"):
            from glob import glob
            a_files = sorted(glob(a_pattern))
            b_files = sorted(glob(b_pattern))

            if len(a_files) != len(b_files):
                st.error(f"Number of A files ({len(a_files)}) does not match B files ({len(b_files)})")
                return

            comparison_sets = []
            for a_file, b_file in zip(a_files, b_files):
                comparison_sets.append({
                    "a_path": os.path.abspath(a_file),
                    "b_path": os.path.abspath(b_file),
                    "a_name": f"{a_name} ({os.path.basename(a_file)})",
                    "b_name": f"{b_name} ({os.path.basename(b_file)})"
                })

            counter = 0
            for comparison in comparison_sets:
                st.subheader(f"Comparing {comparison['a_name']} with {comparison['b_name']}")
                run_with_files(comparison['a_path'], comparison['b_path'],
                             comparison['a_name'], comparison['b_name'],
                             commentary=commentary, index=counter)
                counter+=1
    else:
        col1, col2 = st.columns(2)
        with col1:
            a_file = st.text_input("Model A JSON file path", value="a.json")
            a_name = st.text_input("Model A Name", value="Model A")
        with col2:
            b_file = st.text_input("Model B JSON file path", value="b.json")
            b_name = st.text_input("Model B Name", value="Model B")

        commentary = st.checkbox("Generate Commentary", value=True)

        if st.button("Compare Outputs"):
            run_with_files(a_file, b_file, a_name, b_name, commentary=commentary, index=0)

if __name__ == '__main__':
    if file_or_glob := os.environ.get("FILE_OR_GLOB", ""):
        render(file_or_glob)
    else:
        streamlit_ui()  # Run Streamlit UI if no arguments


