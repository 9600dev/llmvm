import time
from typing import Any, Dict, Optional, List, Literal
import ast
import json
import uuid

from fastapi.utils import generate_unique_id

from llmvm.common.logging_helpers import setup_logging


logging = setup_logging()


def _ast_to_python(obj):
    """
    Recursively convert an AST node for a Python literal or structure
    into an equivalent Python object (list, dict, string, int, etc.).

    This handles:
      - Constants (str, int, float, bool, None)
      - Lists
      - Tuples
      - Dicts
      - Names (treated as strings: e.g. 'True' -> 'True' if used as a name)
      - Nested Calls -> represented as a dictionary with special keys
                       so we can embed them in arguments if needed.
    """
    if isinstance(obj, ast.Constant):
        # Python 3.8+ uses ast.Constant for literals
        return obj.value
    elif isinstance(obj, ast.Str):    # (older Python versions)
        return obj.s
    elif isinstance(obj, ast.Num):    # (older Python versions)
        return obj.n
    elif isinstance(obj, ast.Name):
        # Could interpret names specially; here we just store the identifier as string
        return obj.id
    elif isinstance(obj, ast.List):
        return [_ast_to_python(elt) for elt in obj.elts]
    elif isinstance(obj, ast.Tuple):
        return tuple(_ast_to_python(elt) for elt in obj.elts)
    elif isinstance(obj, ast.Dict):
        return {
            _ast_to_python(k): _ast_to_python(v)
            for k, v in zip(obj.keys, obj.values)
        }
    elif isinstance(obj, ast.Call):
        # Nested call: Represent it as a dict describing the call.
        return _call_node_to_dict(obj)
    else:
        # You could raise an error here or handle more AST node types (e.g. BinOp).
        raise ValueError(f"Unsupported AST node type {type(obj).__name__}")


def _call_node_to_dict(call_node):
    """
    Given an AST Call node, return a dict that has:
      {
        "func_name": str,
        "args": [... converted via _ast_to_python ...],
        "kwargs": {key: value, ...}
      }
    For top-level calls, we'll transform this into the final
    OpenAI tool_call JSON. For nested calls, it appears as a normal dict.
    """
    # Function name can be something like `foo` or `mod.foo`
    # We'll handle the simplest cases: Name or Attribute
    func_name = None
    if isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    elif isinstance(call_node.func, ast.Attribute):
        # e.g. module.attribute(...) -> "module.attribute"
        # (We do a simple approach: walk until we can't. Real logic might require more care.)
        # Collect parts up the attribute chain:
        parts = []
        cur = call_node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        # Reverse, because we collected attributes inside-out
        func_name = ".".join(reversed(parts))
    else:
        raise ValueError("Unsupported function reference (not Name or simple Attribute)")

    args = [_ast_to_python(a) for a in call_node.args]

    # Keyword args
    kwargs = {kw.arg: _ast_to_python(kw.value) for kw in call_node.keywords}

    return {
        "func_name": func_name,
        "args": args,
        "kwargs": kwargs
    }


class OpenAIFunctionTranslator:
    PYTHON_TYPE_MAP = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict",
        "array": "list",
    }

    @staticmethod
    def parse_python_tool_call_result_to_oai_choices(call_string) -> list[dict[str, str]]:
        """
        Parse a Python expression string representing a list (or tuple)
        of function calls, and return a JSON-like dict structure
        mimicking an OpenAI Chat Completion response with multiple
        function calls (one per 'choice').

        Example:
        call_string = "[foo(1, 2), bar(2, 3), hello_world('testing')]"

        Returns list shaped like:
        [
            {
                "index": 0,
                "id": "call_f3MUxVV8XjnoDteR2iAxwQrC",
                "type": "function",
                "function": {
                "name": "get_stock_ask",
                "arguments": "{\"ticker\": \"NVDA\"}"
                }
            },
            {
                "index": 1,
                "id": "call_YQquXgPgcfC8Lc3Y0sUiMKHO",
                "type": "function",
                "function": {
                "name": "get_stock_bid",
                "arguments": "{\"ticker\": \"NVDA\"}"
                }
            }
        ]
        """

        # the stupid client i'm using 5ire, seems to think sending Class.function() as Class--function()
        # is a good idea, so patch that up.
        replace_call_string = False
        if "--" in call_string:
            call_string = call_string.replace("--", "__")
            replace_call_string = True

        # Parse as a Python expression
        try:
            parsed = ast.parse(call_string, mode='eval')
        except SyntaxError as e:
            # not a tool query, so return nothing
            return []

        # The top-level AST node in 'eval' mode is in parsed.body
        top_expr = parsed.body
        if isinstance(top_expr, (ast.List, ast.Tuple)):
            elements = top_expr.elts
        else:
            # If there's only a single call (no brackets) or something else
            # you can decide how strictly to handle it. We'll require a list/tuple.
            logging.error("OpenAIFunctionTranslator: Top-level expression must be a list or tuple of calls.")
            return []

        def generate_id():
            # YQquXgPgcfC8Lc3Y0sUiMKHO"
            guid = str(uuid.uuid4()).replace('-', '')
            return_guid = ''
            for i in range(24):
                return_guid += guid[i]
            return return_guid

        tool_calls = []
        for idx, elt in enumerate(elements):
            if not isinstance(elt, ast.Call):
                logging.error("OpenAIFunctionTranslator: Every top-level element must be a function call.")
                return []

            call_info = _call_node_to_dict(elt)
            name = call_info["func_name"]

            json_args = json.dumps(call_info["kwargs"], ensure_ascii=False)

            tool_calls.append({
                "id": f"call_{generate_id()}",
                "index": idx,
                "type": "function",
                "function": {
                    "name": name.replace("__", "--") if replace_call_string else name,
                    "arguments": json_args
                }
            })
        return tool_calls

    @staticmethod
    def _translate_type(param_spec: dict) -> str:
        param_type = param_spec.get("type", "string")

        if "enum" in param_spec:
            literals = ', '.join(f'"{val}"' for val in param_spec["enum"])
            return f'Literal[{literals}]'

        if param_type == "array":
            items = param_spec.get("items", {})
            item_type = OpenAIFunctionTranslator._translate_type(items)
            return f'List[{item_type}]'

        if param_type == "object":
            return "Dict[str, Any]"

        return OpenAIFunctionTranslator.PYTHON_TYPE_MAP.get(param_type, "Any")


    @staticmethod
    def _translate_oai_type(param_spec: Dict[str, Any]) -> str:
        """
        Translate a JSON Schema param spec into a Python type annotation.
        Extend or modify this as needed for your specs.
        """
        # Handle enums
        if "enum" in param_spec:
            # If there's an enum, often you'd use a Literal type. For simplicity, we'll just map to `str`.
            # You could do:  from typing import Literal
            #               return f"Literal[{', '.join(repr(v) for v in param_spec['enum'])}]"
            return "str"

        schema_type = param_spec.get("type", "any").lower()
        if schema_type == "string":
            return "str"
        elif schema_type == "number":
            return "float"
        elif schema_type == "integer":
            return "int"
        elif schema_type == "boolean":
            return "bool"
        elif schema_type == "object":
            return "dict"
        elif schema_type == "array":
            return "list"
        else:
            return "Any"

    @staticmethod
    def generate_python_function_signature_from_oai_description(
        func_desc: Dict[str, Any],
        *,
        async_method: bool = False,
        static_method: bool = False,
        return_type: str = "Any"
    ) -> str:
        """
        Generate a Python function signature (plus docstring) from an OpenAI tool (function) description.
        Expects `func_desc` to have the structure:
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                },
                ...
            }
        }
        """

        # Top-level "function" dict inside the description
        function_info = func_desc.get("function", {})

        # Extract basic metadata
        name = function_info.get("name", "unknown_function")
        description = function_info.get("description", "")
        parameters = function_info.get("parameters", {})
        properties = parameters.get("properties", {})
        required_params = parameters.get("required", [])

        # Prepare the function signature parameters
        param_strings = []
        doc_params = []

        for param_name, param_spec in properties.items():
            python_type = OpenAIFunctionTranslator._translate_oai_type(param_spec)
            is_required = param_name in required_params

            if not is_required:
                python_type = f"Optional[{python_type}]"
                default_val = " = None"
            else:
                default_val = ""

            # Build the signature piece
            param_strings.append(f"{param_name}: {python_type}{default_val}")

            # Build docstring lines
            param_description = param_spec.get("description", "")
            doc_params.append(
                f":param {param_name}: {param_description}\n"
                f":type {param_name}: {python_type}"
            )

        # Handle decorators
        decorators = []
        if static_method:
            decorators.append("@staticmethod")

        decorators_str = "\n".join(decorators)

        # Handle async / sync
        async_str = "async " if async_method else ""

        # Build the final signature line
        params_str = ", ".join(param_strings)
        signature = f"{decorators_str}\n{async_str}def {name}({params_str}) -> {return_type}:"

        # Build the docstring
        docstring_parts = []
        if description:
            docstring_parts.append(description)
        docstring_parts.extend(doc_params)
        docstring_parts.append(f":return: {return_type}\n:rtype: {return_type}")

        docstring = '"""\n' + "\n".join(docstring_parts) + '\n"""'

        # Put it all together
        full_function = f"{signature}\n{docstring}"

        return full_function
