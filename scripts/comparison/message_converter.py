from typing import List, Union, Optional, Tuple
from models import ModelOutput, ComparisonSet, ComparisonPair
from llmvm.common.objects import Message, Assistant, User, Content, TextContent


def message_to_model_output(
        message: Message,
        model_id: str = "a",
        output_id: Optional[str] = None
    ) -> ModelOutput:
    if output_id is None:
        output_id = f"Output {model_id}1"

    if isinstance(message, Assistant):
        content = message.get_str().replace('$', '\\$').replace("<helpers_result>\n\n", "<helpers_result>\n").replace("\n\n</helpers_result>", "\n</helpers_result>")
        role = "assistant"
    elif isinstance(message, User):
        content = message.get_str().replace('$', '\\$').replace("<helpers_result>\n\n", "<helpers_result>\n").replace("\n\n</helpers_result>", "\n</helpers_result>")
        role = "user"
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return ModelOutput(content=content, model_id=model_id, output_id=output_id, role=role)


def create_comparison_from_message_lists(
    model_a_messages: List[Message],
    model_b_messages: List[Message],
    commentary_list: list[str],
    model_a_id: str = "A",
    model_b_id: str = "B"
) -> ComparisonSet:
    """
    Create a ComparisonSet by grouping one or more User messages with the
    following Assistant message. This ensures that Assistant outputs line up
    across models even if they have differing numbers of User messages.
    """

    comparison_set = ComparisonSet()
    for i in range(len(model_a_messages)):
        a_output = message_to_model_output(model_a_messages[i], model_id=model_a_id, output_id=f"Output {model_a_id}{i+1}")
        b_output = message_to_model_output(model_b_messages[i], model_id=model_b_id, output_id=f"Output {model_b_id}{i+1}")
        comparison_set.add_pair(a_output, b_output, commentary=commentary_list[i])

    return comparison_set


