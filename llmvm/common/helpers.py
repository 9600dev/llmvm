import ast
import asyncio
import base64
import contextlib
from dataclasses import is_dataclass, asdict
import datetime as dt
import dis
import fcntl
import glob
import gzip
import importlib
import inspect
import io
import itertools
import json
import marshal
import math
import os
import pprint
import pty
import pydoc
import re
import selectors
import signal
import struct
import subprocess
import platform
import sys
import tempfile
import termios
import threading
import dill
import traceback
import types
from bs4 import BeautifulSoup
import dateparser
import typing
import textwrap
from collections import Counter
from enum import Enum, IntEnum
from functools import reduce
from importlib import resources
from itertools import cycle, islice
from logging import Logger
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union, Tuple, cast, get_args, get_origin, Union

from urllib.parse import urljoin, urlparse
from markdownify import markdownify as md
import zlib
from zoneinfo import ZoneInfo

import httpx
import nest_asyncio
from dateutil.relativedelta import relativedelta
from docstring_parser import parse
from PIL import Image
import pyte

from llmvm.common.objects import (AstNode, Content, FunctionCall, ImageContent, MarkdownContent, HTMLContent,
                                  Message, PandasMeta, Statement, StreamNode, SupportedMessageContent, System, TextContent, User)
from llmvm.common.container import Container


def write_client_stream(obj):
    if isinstance(obj, bytes):
        obj = StreamNode(obj, type='bytes')

    frame = inspect.currentframe()
    while frame:
        # Check if 'self' exists in the frame's local namespace
        if 'stream_handler' in frame.f_locals:
            asyncio.run(frame.f_locals['stream_handler'](obj))
            return

        instance = frame.f_locals.get('self', None)
        if hasattr(instance, 'stream_handler'):
            asyncio.run(instance.stream_handler(obj))  # type: ignore
            return
        frame = frame.f_back


def get_stream_handler() -> Optional[Callable[[AstNode], Awaitable[None]]]:
    frame = inspect.currentframe()
    while frame:
        # Check if 'self' exists in the frame's local namespace
        if 'stream_handler' in frame.f_locals:
            return frame.f_locals['stream_handler']

        instance = frame.f_locals.get('self', None)
        if hasattr(instance, 'stream_handler'):
            return instance.stream_handler  # type: ignore
        frame = frame.f_back
    return None


CSI_RE = re.compile(rb'\x1b\[[0-?]*[ -/]*[@-~]')


def _winsize():
    rows, cols = os.get_terminal_size(sys.stdout.fileno())
    return rows, cols


def _set_winsize(fd, rows, cols):
    fcntl.ioctl(fd, termios.TIOCSWINSZ,
                struct.pack("HHHH", rows, cols, 0, 0))


_SENTINEL = ast.Constant(value=None)

class LateBindDefaults(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        new_body = list(node.body)  # copy so we can prepend

        for idx, (arg, default) in enumerate(
                zip(node.args.args[-len(node.args.defaults):], node.args.defaults)):

            if (isinstance(default, ast.Call)
                and isinstance(default.func, ast.Name)
                and default.func.id == "llm_var_bind"):

                # 1. replace the eager default with None
                node.args.defaults[idx] = _SENTINEL

                # 2. inject:   if <param> is None: <param> = llm_var_bind(...)
                assign = ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=arg.arg, ctx=ast.Load()),
                        ops=[ast.Is()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    body=[ast.Assign(
                        targets=[ast.Name(id=arg.arg, ctx=ast.Store())],
                        value=default
                    )],
                    orelse=[]
                )
                new_body.insert(0, assign)

        node.body = new_body
        return node


class Helpers():
    @staticmethod
    def dump_assertion(assertion: Callable[[], bool]) -> str:
        result = ''
        try:
            src = textwrap.dedent(inspect.getsource(assertion)).strip()
            header = "assertion source"
            body = src
        except (OSError, TypeError):
            header = "assertion disassembly (source unavailable)"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dis.dis(assertion)
            body = buf.getvalue().rstrip()

        result += f"── {header} ──\n"
        result += body + "\n"

        result += "\n── captured variables ──"
        cvars = inspect.getclosurevars(assertion)
        captured = {
            "globals"   : cvars.globals,
            "nonlocals" : cvars.nonlocals,
            "builtins"  : {k: v for k, v in cvars.builtins.items() if k in cvars.unbound},
        }
        result += pprint.pformat(captured, compact=True, sort_dicts=False)
        return result

    @staticmethod
    def rewrite_late_binding(code: str) -> types.CodeType:
        src = textwrap.dedent(code)
        tree = ast.parse(src, mode="exec")
        tree = LateBindDefaults().visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, "<exec>", "exec")

    @staticmethod
    def extract_program_code_block(string: str) -> str:
        blocks: list[str] = []
        lines = string.splitlines(keepends=True)
        inside_block = False
        block_text = ""

        for line in lines:
            stripped_line = line.strip()

            if not inside_block and stripped_line == "<program>":
                inside_block = True
                continue

            if inside_block and stripped_line == "</program>":
                inside_block = False
                blocks.append(block_text)
                block_text = ""

            if inside_block:
                block_text += line
        return blocks[-1]

    @staticmethod
    def to_dict(obj) -> dict:
        if isinstance(obj, (int, float, str, bool, type(None))):
            raise ValueError(f"Cannot convert {obj} to dict")

        if is_dataclass(obj):
            return Helpers.to_dict(asdict(obj))

        if isinstance(obj, dict):
            return {k: Helpers.to_dict(v) for k, v in obj.items()}

        if hasattr(obj, "__dict__"):
            return {k: Helpers.to_dict(v) for k, v in vars(obj).items()}
        return obj

    @staticmethod
    def dill_to_b64(obj, *, compress=False, protocol=None) -> str:
        raw = dill.dumps(obj, protocol=protocol)           # bytes
        if compress:
            raw = gzip.compress(raw)
        return base64.b64encode(raw).decode("ascii")       # ASCII

    @staticmethod
    def b64_to_dill(b64_string: str, *, compressed=False):
        raw = base64.b64decode(b64_string.encode("ascii"))
        if compressed:
            raw = gzip.decompress(raw)
        return dill.loads(raw)

    @staticmethod
    def str_to_type(type_name: str) -> type:
        return cast(type, pydoc.locate(type_name))

    @staticmethod
    def all(iterable, func):
        for x in iterable:
            if not func(x):
                return False
        return True

    @staticmethod
    def run_streaming(cmd: str, tui_threshold: float = 0.30) -> str:
        master, slave = pty.openpty()

        _set_winsize(slave, *_winsize())
        signal.signal(signal.SIGWINCH,
                    lambda *_: _set_winsize(slave, *_winsize()))

        proc = subprocess.Popen(cmd, shell=True,
                                stdin=slave, stdout=slave, stderr=slave,
                                close_fds=True, preexec_fn=os.setsid, text=False)
        os.close(slave)

        sel = selectors.DefaultSelector()
        sel.register(master, selectors.EVENT_READ)
        captured  = bytearray()

        try:
            while True:
                for key, _ in sel.select():
                    chunk = os.read(key.fd, 4096)
                    if not chunk:
                        raise StopIteration

                    os.write(sys.stdout.fileno(), chunk)
                    captured.extend(chunk)
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGINT)
        except StopIteration:
            pass
        finally:
            sel.unregister(master)
            os.close(master)
            proc.wait()

        tui = captured.count(b'\x1b') / max(1, len(captured)) >= tui_threshold

        if tui:
            scr, stream = pyte.Screen(*_winsize()), pyte.Stream(pyte.Screen(0, 0))
            stream.screen = scr  # type: ignore
            stream.feed(captured.decode('utf‑8', 'ignore'))
            return "\n".join(scr.display)

        try:
            os.system('reset')
        except Exception:
            pass

        return CSI_RE.sub(b'', captured).replace(b'\r', b'').decode('utf‑8', errors='replace')

    @staticmethod
    def serialize_locals_dict(locals_dict: dict[str, Any]) -> dict[str, Any]:
        temp_dict = {}
        for key, value in locals_dict.items():
            if isinstance(key, str) and key.startswith('__'):
                continue
            elif isinstance(value, types.FunctionType) and value.__module__ == 'builtins':
                continue
            elif key == 'AutoGlobalDict':
                continue
            elif isinstance(value, types.FunctionType) and value.__code__.co_filename == '<ast>':
                temp_dict[key] = Helpers.serialize_function(value)
            elif isinstance(value, dict):
                temp_dict[key] = Helpers.serialize_locals_dict(value)
            elif isinstance(value, list):
                temp_dict[key] = [Helpers.serialize_item(v) for v in value]
            elif isinstance(value, (str, int, float, bool)):
                temp_dict[key] = value
            elif isinstance(value, (Content, AstNode, Message, Statement)):
                temp_dict[key] = value
            else:
                try:
                    json.dumps(value)
                    temp_dict[key] = value
                except:
                    # keep instances of tools alive until the server winds down
                    # if not isinstance(value, types.MethodType) and not isinstance(value, types.FunctionType):
                        # self.locals_instance_state.append(InstanceState(thread_id=thread_id, locals_dict=value))
                    # actual functions can't be json serialized so we pass here
                    pass
        return temp_dict

    @staticmethod
    def serialize_item(item):
        if isinstance(item, types.FunctionType) and item.__code__.co_filename == '<ast>':
            return Helpers.serialize_function(item)
        elif isinstance(item, (str, int, float, bool)):
            return item
        elif isinstance(item, (Content, AstNode, Message, Statement)):
            return item
        elif isinstance(item, dict):
            return Helpers.serialize_locals_dict(item)
        elif isinstance(item, list):
            return [Helpers.serialize_item(v) for v in item]
        else:
            try:
                json.dumps(item)
                return item
            except:
                # keep instances of tools alive until the server winds down
                # if not isinstance(item, types.MethodType) and not isinstance(item, types.FunctionType):
                    # self.locals_instance_state.append(InstanceState(thread_id=thread_id, locals_dict=item))
                # actual functions can't be json serialized so we pass here
                pass

    @staticmethod
    def serialize_function(func):
        is_static, cls = Helpers.is_static_method(func)

        # Serialize the function's code object
        code_bytes = marshal.dumps(func.__code__)
        return {
            'type': 'function',
            'name': func.__name__,
            'code': base64.b64encode(code_bytes).decode('ascii'),
            'defaults': func.__defaults__,
            'closure': func.__closure__,
            'doc': func.__doc__,
            'annotations': func.__annotations__,
            'is_method': not is_static or cls is not None,
            'class_name': cls.__name__ if cls else None,
            'is_static_method': is_static and cls is not None,
            'qualname': func.__qualname__,
            'module': func.__module__,
            'from_ast': func.__code__.co_filename == '<ast>',
        }

    @staticmethod
    def deserialize_locals_dict(serialized_dict: dict[str, Any]) -> dict[str, Any]:
        result = {}
        # First pass: Create all basic items to establish the namespace
        for key, value in serialized_dict.items():
            if isinstance(value, dict) and value.get('type') == 'function':
                # Skip functions on first pass
                continue
            elif isinstance(value, dict):
                result[key] = Helpers.deserialize_locals_dict(value)
            elif isinstance(value, list):
                result[key] = [Helpers.deserialize_item(v, result) for v in value]
            else:
                result[key] = value

        # Second pass: Now handle functions with the established namespace
        for key, value in serialized_dict.items():
            if isinstance(value, dict) and value.get('type') == 'function':
                result[key] = Helpers.deserialize_function(value, result)

        return result

    @staticmethod
    def deserialize_item(item, context=None):
        if context is None:
            context = globals()

        if isinstance(item, dict) and item.get('type') == 'function':
            return Helpers.deserialize_function(item, context)
        elif isinstance(item, list):
            return [Helpers.deserialize_item(v, context) for v in item]
        elif isinstance(item, dict) and 'type' not in item:
            return Helpers.deserialize_locals_dict(item)
        else:
            return item

    @staticmethod
    def deserialize_function(func_dict, context):
        # Deserialize the function's code object
        code_bytes = base64.b64decode(func_dict['code'])
        code = marshal.loads(code_bytes)

        # Recreate the function with proper context
        func = types.FunctionType(
            code,
            context,
            func_dict['name'],
            func_dict['defaults'],
            func_dict['closure']
        )

        # Restore all additional attributes that were serialized
        if 'doc' in func_dict:
            func.__doc__ = func_dict['doc']
        if 'annotations' in func_dict:
            func.__annotations__ = func_dict['annotations']
        if 'qualname' in func_dict:
            func.__qualname__ = func_dict['qualname']
        if 'module' in func_dict:
            func.__module__ = func_dict['module']
        if 'from_ast' in func_dict and func_dict['from_ast']:
            func._from_ast = True  # type: ignore

        # Handle class methods properly
        if func_dict.get('is_method') and func_dict.get('class_name'):
            class_name = func_dict['class_name']
            if class_name in context:
                cls = context[class_name]
                # Handle static methods
                if func_dict.get('is_static_method'):
                    func = staticmethod(func)
        return func

    @staticmethod
    def get_class_name_of_method(func) -> Optional[str]:
        # Case 1: Bound instance method => __self__ is the instance
        if hasattr(func, '__self__') and func.__self__ is not None:
            # For a regular bound method, __self__ is the instance
            # For a bound classmethod, __self__ is the class
            cls = func.__self__ if inspect.isclass(func.__self__) else func.__self__.__class__
            return cls.__name__

        # Case 2: If it's a function or staticmethod, check __qualname__
        # __qualname__ often looks like "MyClass.my_method" or "MyClass.NestedClass.my_method"
        # If there's only one dot or none, it might just be "my_function" or something else.
        qualname = getattr(func, '__qualname__', None)
        if qualname and '.' in qualname:
            parts = qualname.split('.')
            # The last part is the function name; the rest are nested scopes/classes.
            # e.g. "MyClass.my_method" => ["MyClass", "my_method"]
            # or   "MyClass.InnerClass.my_method" => ["MyClass", "InnerClass", "my_method"]
            # Usually, the second-to-last is the immediate class name.
            if len(parts) > 1:
                return parts[-2]  # e.g. 'MyClass' or 'InnerClass'
        return None

    @staticmethod
    def annotation_to_string(annotation: Any) -> str:
        """
        Convert a possibly complex type annotation into a readable string.
        E.g., Union[str, int] -> 'str | int', List[int] -> 'list[int]', etc.
        """
        if annotation is inspect.Signature.empty:
            return "Any"
        if annotation is None or annotation is type(None):
            return "None"
        if isinstance(annotation, type):
            # e.g., <class 'int'> -> 'int'
            return annotation.__name__

        origin = get_origin(annotation)
        if origin is Union:
            # Handle union types (Python 3.10+ can appear as int | str)
            args = get_args(annotation)
            return " | ".join(Helpers.annotation_to_string(a) for a in args)
        elif origin in (list, tuple, dict, set, frozenset):
            # e.g., list[int], dict[str, int]
            args = get_args(annotation)
            if args:
                return f"{origin.__name__}[{', '.join(Helpers.annotation_to_string(a) for a in args)}]"
            else:
                return origin.__name__
        elif origin:
            # Generic type from typing (e.g. typing.Iterable, typing.Callable)
            args = get_args(annotation)
            return f"{origin.__name__}[{', '.join(Helpers.annotation_to_string(a) for a in args)}]"

        # Fallback
        return str(annotation)

    @staticmethod
    def get_function_description_new(function: Callable, openai_format: bool = False) -> Dict[str, Any]:
        """
        Inspect a callable and return a dictionary describing:
         - The parameter names (`parameters`)
         - The parameter types (`types`)
         - The return type (`return_type`)
         - The function name itself (`invoked_by`)
         - The class name (`class_name`)
         - The docstring (`description`)
        """
        # Special handling for MCPToolWrapper
        if hasattr(function, 'get_function_description'):
            return function.get_function_description()
            
        docstring = inspect.getdoc(function) or ""
        signature = inspect.signature(function)

        # Retrieve type hints (handles forward refs and generics if possible)
        type_hints = typing.get_type_hints(function, None, None)

        parameters: List[str] = []
        types: List[str] = []
        for param_name, param in signature.parameters.items():
            # Use the type hint if available, otherwise fallback to "Any"
            hint = type_hints.get(param_name, param.annotation)
            param_type_str = Helpers.annotation_to_string(hint)
            parameters.append(param_name)
            types.append(param_type_str)

        return_annotation = type_hints.get('return', signature.return_annotation)
        return_type = return_annotation if return_annotation != inspect.Signature.empty else None

        # Check if the function is async
        is_async = inspect.iscoroutinefunction(function)
        
        return {
            "parameters": parameters,
            "types": types,
            "return_type": return_type,
            "invoked_by": function.__name__,
            "class_name": Helpers.get_class_name_of_method(function),
            "description": docstring,
            "is_async": is_async,
        }

    @staticmethod
    def is_static_method_new(func: Callable) -> Tuple[bool, Optional[type]]:
        """
        Heuristic: if the first parameter name is 'self', we assume it is an instance method.
        If the first parameter is 'cls', we assume it is a class method.
        Otherwise, we assume it is a static method.

        Returns a tuple: (is_static_method: bool, owning_class_or_None)
        """
        # Special handling for MCPToolWrapper
        if hasattr(func, 'get_function_description'):
            # MCP tools are standalone functions, not methods
            return (True, None)
            
        # Attempt to discover the class (if bound method)
        cls_candidate = getattr(func, '__self__', None)
        if cls_candidate is not None:
            # if it's an instance, get its class
            if not inspect.isclass(cls_candidate):
                cls_candidate = type(cls_candidate)

        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if params and params[0] == "self":
                # Probably an instance method
                return (False, cls_candidate)
        except (ValueError, TypeError):
            # If we can't get a signature, treat as static
            pass
            
        # In more thorough code, you might also check `params[0] == "cls"` for a class method.
        # For simplicity, we treat everything else as static
        return (True, cls_candidate)

    @staticmethod
    def get_methods_and_statics(obj: object) -> List[Callable]:
        cls = obj.__class__
        result: List[Callable] = []

        for name, member in inspect.getmembers(obj):
            # skip private/dunder attributes
            if name.startswith("__"):
                continue

            # must be callable at runtime
            if not callable(member):
                continue

            # attribute must come from the class namespace
            if name not in cls.__dict__:
                continue

            class_attr = cls.__dict__[name]
            # include normal functions (instance methods) and staticmethod objects
            if inspect.isfunction(class_attr) or isinstance(class_attr, staticmethod):
                result.append(member)
        return result

    @staticmethod
    def get_function_description_flat(function: Callable) -> str:
        """
        Build a string describing the Python function signature (without body),
        including docstring (inline) and whether it's static or can be instantiated.
        """
        description = Helpers.get_function_description_new(function, openai_format=False)
        parameter_type_list = [
            f"{param}: {typ}"
            for param, typ in zip(description['parameters'], description['types'])
        ]

        # Handle return type string
        return_annotation = description['return_type']
        if return_annotation is None:
            return_type_str = "Any"
        else:
            return_type_str = Helpers.annotation_to_string(return_annotation)

        # Determine if static or instance-based
        is_static, cls = Helpers.is_static_method_new(function)

        doc = description["description"] or "No docstring"
        if not is_static and cls:
            # Instance method
            # e.g.: def my_method(a: int, b: str) -> int  # Instantiate with MyClass(). Doc here
            async_prefix = "async " if description.get("is_async", False) else ""
            return (
                f'{async_prefix}def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type_str}  # Instantiate with {cls.__name__}().\n'
                f'    """\n'
                f'{textwrap.indent(doc, " " * 4)}\n'
                f'    """\n'
            )
        elif description.get("class_name") is None:
            # Standalone function (like MCP tools)
            async_prefix = "async " if description.get("is_async", False) else ""
            return (
                f'{async_prefix}def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type_str}\n'
                f'    """\n'
                f'{textwrap.indent(doc, " " * 4)}\n'
                f'    """\n'
            )
        else:
            # Static method or function with a class
            async_prefix = "async " if description.get("is_async", False) else ""
            return (
                f'class {description["class_name"]}:\n'
                f'    @staticmethod\n'
                f'    {async_prefix}def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type_str}\n'
                f'        """\n'
                f'{textwrap.indent(doc, " " * 8)}\n'
                f'        """\n'
            )

    @staticmethod
    def find_and_run_chrome(filename: str):
        # Get file URL
        file_url = f"file://{os.path.abspath(filename)}"

        system = platform.system()  # type: ignore

        if system == "Darwin":  # macOS
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                os.path.expanduser("~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
            ]

            for path in chrome_paths:
                if os.path.exists(path):
                    subprocess.Popen([path, file_url],
                                    start_new_session=True,  # Detaches the process
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
                    return True, filename

        elif system == "Linux":
            chrome_commands = ["google-chrome", "chrome", "chromium", "chromium-browser"]

            for cmd in chrome_commands:
                try:
                    # Check if the command exists
                    which_result = subprocess.run(["which", cmd],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE,
                                                text=True)

                    if which_result.returncode == 0:
                        chrome_path = which_result.stdout.strip()
                        # Use subprocess.Popen to avoid waiting for the browser to close
                        subprocess.Popen([chrome_path, file_url],
                                        start_new_session=True,  # Detaches the process
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                        return True, filename
                except Exception as e:
                    continue

    @staticmethod
    def find_and_run_chrome_with_html(html_content, filename: Optional[str] = None):
        temp_file = None
        if not filename:
            try:
                fd, temp_path = tempfile.mkstemp(suffix='.html')
                with os.fdopen(fd, 'w') as f:
                    f.write(html_content)
                temp_file = temp_path
            except Exception as e:
                return False, None
        else:
            temp_file = filename

        Helpers.find_and_run_chrome(temp_file)

    @staticmethod
    def is_function(obj):
        if isinstance(obj, (types.FunctionType, types.LambdaType)):
            return True

        if isinstance(obj, types.MethodType):
            return True

        if isinstance(obj, types.BuiltinFunctionType):
            return True

        if callable(obj) and not inspect.isclass(obj):
            return True
        return False

    @staticmethod
    def compressed_user_messages(messages: list[Message]) -> list[str]:
        user_messages = []
        for message in messages:
            if (
                isinstance(message, User)
                and all([isinstance(c, TextContent) for c in message.message])
                and not '<helpers' in message.get_str()
            ):
                user_messages.append(message.get_str()[0:500])
        return user_messages

    @staticmethod
    def matplotlib_figure_to_image_content(fig, dpi=130):
        import matplotlib.pyplot as plt
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        image_bytes = buffer.getvalue()
        buffer.close()
        return ImageContent(image_bytes)

    @staticmethod
    def get_class_from_static_callable(callable_obj) -> Optional[type]:
        if not callable(callable_obj):
            raise ValueError("Provided object must be callable")

        qualname_parts = callable_obj.__qualname__.split('.')
        if len(qualname_parts) < 2:
            return None

        class_name = qualname_parts[-2]
        module = inspect.getmodule(callable_obj)
        if module is None:
            raise ValueError("Unable to retrieve module from callable")

        cls = getattr(module, class_name, None)
        if not isinstance(cls, type):
            raise ValueError(f"Unable to find class {class_name} in module {module.__name__}")

        return cls

    @staticmethod
    def get_value_from_parent_frame(variable_name, frame_depth=6):
        current_frame = inspect.currentframe()

        target_frame = current_frame
        for _ in range(frame_depth):
            if target_frame.f_back is None:
                return None

            if target_frame.f_locals.get(variable_name):
                return target_frame.f_locals.get(variable_name)

            target_frame = target_frame.f_back

        return None

    @staticmethod
    def str_get_str(obj):
        if hasattr(obj, 'get_str'):
            # sometimes a class gets passed in, which doesn't have an self instance
            try:
                return obj.get_str()
            except Exception as ex:
                return str(obj)
        else:
            return str(obj)

    @staticmethod
    def is_callee(func_name: str):
        import inspect
        for frame_info in inspect.stack():
            if frame_info.function == func_name:
                return True
        return False

    @staticmethod
    def deserialize_messages(json_filename: str) -> list[Message]:
        with open(os.path.expanduser(json_filename), 'r') as f:
            # find the place where there's an append
            full_str = f.read()
            pattern = r'''(?m)^ {2}\}\r?\n^\]\s*\r?\n^\s*\r?\n^\[\s*\r?\n^ {2}\{'''
            replacement = '  },\n  {'
            clean_content = re.sub(pattern, replacement, full_str)
            messages = json.loads(clean_content)
        return [Message.from_json(m) for m in messages]

    @staticmethod
    def get_google_sheet(url: str) -> PandasMeta:
            import gspread
            from gspread_dataframe import get_as_dataframe
            gp = gspread.oauth()  # type: ignore
            spreadsheet = gp.open_by_url(url)
            ws = spreadsheet.get_worksheet(0)
            df = get_as_dataframe(ws, drop_empty_rows=True, drop_empty_columns=True)
            return PandasMeta(expr_str=url, pandas_df=df)

    @staticmethod
    def parse_lists_from_string(list_str: str) -> list:
        list_str = list_str.strip()
        pattern = r'\[[^\[\]]*\]'
        list_strings = re.findall(pattern, list_str)

        lists = [eval(lst_str) for lst_str in list_strings]
        return lists

    @staticmethod
    def parse_list_string(list_string: str, default: list = []) -> list:
        try:
            list_string = list_string.strip()

            # Check if the string starts with '[' and ends with ']'
            if not (list_string.startswith('[') and list_string.endswith(']')):
                raise ValueError("Input string must start with '[' and end with ']'")

            result = eval(list_string)

            if not isinstance(result, list):
                if default: return default
                raise ValueError("Input string did not evaluate to a list")

            return result

        except SyntaxError as e:
            if default: return default
            raise SyntaxError(f"Invalid list syntax: {e}")
        except Exception as e:
            if default: return default
            raise ValueError(f"Error parsing list string: {e}")

    @staticmethod
    def is_async_iterator(obj):
        # Method 1: Check for __aiter__ and __anext__ methods
        has_aiter = hasattr(obj, '__aiter__')
        has_anext = hasattr(obj, '__anext__')

        # Method 2: Using isinstance with AsyncIterator
        is_async_iter = isinstance(obj, AsyncIterator)

        # Method 3: Using inspect (most thorough)
        is_async_gen = inspect.isasyncgen(obj)

        return has_aiter and has_anext or is_async_iter or is_async_gen

    @staticmethod
    def is_sync_iterator(obj):
        # Method 1: Check for __iter__ and __next__ methods
        has_iter = hasattr(obj, '__iter__')
        has_next = hasattr(obj, '__next__')

        # Method 2: Using isinstance with Iterator
        is_iter = isinstance(obj, Iterator)

        # Method 3: Using inspect
        is_generator = inspect.isgenerator(obj)

        return has_iter and has_next or is_iter or is_generator

    @staticmethod
    def markdown_to_minimal_text(markdown_content):
        soup = BeautifulSoup(markdown_content, 'html.parser')

        text = md(str(soup), heading_style="ATX")

        # expressive markdown
        if 'Clickable Elements' in text:
            text = text[0:text.find('Clickable Elements:')]

        text = re.sub(r'\n\s*\n', ' ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'[*_~`]', '', text)  # Remove emphasis markers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Convert links to just text
        text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)  # Convert images to just alt text
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Remove list markers
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered list markers
        text = re.sub(r'\|', ' ', text)  # Remove table separators
        text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)  # Remove blockquotes
        text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'([a-zA-Z0-9])([A-Z][a-z])', r'\1 \2', text)  # Better camelCase handling
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        text = re.sub(r'\s*([.,!?:;])', r'\1 ', text)  # Add space after punctuation
        text = re.sub(r'\s+([.,!?:;])\s+', r'\1 ', text)  # Clean up extra spaces around punctuation
        text = text.strip()
        return text

    @staticmethod
    def keep_last_browser_content(text: str, erase: bool = False) -> str:
        # Pattern to match <helpers_result> blocks
        pattern = r'(<helpers_result>(.*?)</helpers_result>)'

        # Find all matches
        matches = list(re.finditer(pattern, text, re.DOTALL))

        if len(matches) <= 1:
            return text

        # Collect replacements for all matches except the last one
        replacements = []
        for match in matches[:-1]:
            full_match = match.group(1)  # Entire <helpers_result>...</helpers_result> block
            content = match.group(2)     # Content inside the block

            # Normalize whitespace for accurate line counting
            content_lines = content.strip().split('\n')

            # Check if the block has already been processed
            # Process only if there is more than one line or if the content differs from expected
            if (
                len(content_lines) > 1
                or not content_lines[0].startswith('BrowserContent(')
                and not content_lines[0].startswith('BrowserContent(processed=true,')
            ):
                # Extract the BrowserContent(...) line
                browser_content_match = re.search(r'(BrowserContent\([^)]+\))', content)
                if browser_content_match:
                    browser_line = browser_content_match.group(1).replace('BrowserContent(', 'BrowserContent(processed=true, ')
                else:
                    browser_line = ''  # If not found, default to empty

                if not erase:
                    browser_line = browser_line + ' ' + Helpers.markdown_to_minimal_text(content)

                # Prepare the replacement
                replacement = f"<helpers_result>{browser_line}\n</helpers_result>"
                replacements.append((match.start(), match.end(), replacement))

        # Apply replacements in reverse order to avoid position shifting
        result = text
        for start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]

        return result

    @staticmethod
    def clean_tracking(url: str) -> str:
        patterns = [
            r'utm_[^&]*&?',
            r'fbclid=[^&]*&?',
            r'gclid=[^&]*&?',
            r'_ga=[^&]*&?',
        ]

        cleaned_url = url
        for pattern in patterns:
            cleaned_url = re.sub(pattern, '', cleaned_url)

        cleaned_url = re.sub(r'[?&]$', '', cleaned_url)
        return cleaned_url

    @staticmethod
    def clean_url_params(url: str, limit: int = 50) -> str:
        parsed_url = urlparse(url)

        query_params = parsed_url.query.split('&')
        cleaned_query_params = []

        for param in query_params:
            if len(param) <= limit:
                cleaned_query_params.append(param)

        cleaned_query = '&'.join(cleaned_query_params)
        cleaned_url = parsed_url._replace(query=cleaned_query).geturl()
        return cleaned_url

    @staticmethod
    def split_on_newline(text):
        return re.split(r'(?<!\\)\n', text)

    @staticmethod
    def escape_newlines_in_strings(code):
        result = []
        i = 0
        n = len(code)
        in_string = False
        in_comment = False
        string_quote = ''
        while i < n:
            c = code[i]
            if not in_string and not in_comment:
                if c == '#':
                    # Start of a comment
                    in_comment = True
                    result.append(c)
                    i += 1
                elif c in ('"', "'"):
                    # Start of a string
                    # Check for triple quotes
                    if i + 2 < n and code[i:i+3] == c * 3:
                        string_quote = c * 3
                        in_string = True
                        result.append(string_quote)
                        i += 3
                    else:
                        string_quote = c
                        in_string = True
                        result.append(c)
                        i += 1
                else:
                    # Regular character
                    result.append(c)
                    i += 1
            elif in_string:
                if c == '\\':
                    # Escape character, include next character as is
                    result.append(code[i:i+2])
                    i += 2
                elif code[i:i+len(string_quote)] == string_quote:
                    # End of string
                    result.append(string_quote)
                    i += len(string_quote)
                    in_string = False
                    string_quote = ''
                else:
                    if c == '\n':
                        # Escape newline
                        result.append('\\n')
                        i += 1
                    else:
                        result.append(c)
                        i += 1
            elif in_comment:
                if c == '\n':
                    # End of comment
                    in_comment = False
                result.append(c)
                i += 1
        return ''.join(result)

    @staticmethod
    def apply_unified_diff(original_content, diff_content):
        def in_between_including(line, start_marker, end_marker):
            start_idx = line.find(start_marker)
            end_idx = line.find(end_marker, start_idx + len(start_marker))
            if start_idx == -1 or end_idx == -1:
                return line  # fallback: return entire line
            return line[start_idx:end_idx + len(end_marker)]

        def after_end(line, start_marker, end_marker):
            start_idx = line.find(start_marker)
            if start_idx == -1:
                return line
            end_idx = line.find(end_marker, start_idx + len(start_marker))
            if end_idx == -1:
                return line
            return line[end_idx + len(end_marker):]

        original_lines = original_content.splitlines()
        diff_lines = diff_content.splitlines()
        modified_lines = original_lines.copy()

        # If first line is in the “botched” format, fix it.
        if diff_lines and diff_lines[0].startswith("@@") and not diff_lines[0].endswith("@@") and "@@" in diff_lines[0][2:]:
            # The line might look like: "@@ -6,6 +6,7 @@ extra stuff"
            new_start_line = in_between_including(diff_lines[0], '@@', '@@')
            next_line = after_end(diff_lines[0], '@@', '@@').strip()

            diff_lines[0] = new_start_line
            # Insert the leftover part as a new line if it’s non-empty
            if next_line:
                diff_lines.insert(1, next_line)

        current_line = 0

        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]

            # Detect hunk headers: @@ -oldStart,oldLen +newStart,newLen @@
            if line.startswith("@@"):
                parts = line.split()

                # We expect something like:
                # parts[0] = "@@"
                # parts[1] = "-6,6"
                # parts[2] = "+6,7"
                # parts[-1] = "@@"  (Sometimes parts[3], or maybe 4 if there's weird spacing)

                # Validate minimum length to avoid index errors
                if len(parts) >= 3 and parts[0] == "@@":
                    # old side is in parts[1], new side in parts[2]
                    old_region = parts[1]  # e.g. "-6,6"
                    new_region = parts[2]  # e.g. "+6,7"

                    # Extract the start line from old_region
                    # old_region looks like "-6,6" => start=6 length=6
                    try:
                        old_start_str = old_region.split(',')[0]  # => "-6"
                        old_start = int(old_start_str[1:])        # => 6
                    except ValueError:
                        old_start = 1  # fallback

                    current_line = old_start - 1  # adjust for 0-based indexing
                i += 1
                continue

            # Deletions ("-"), Additions ("+"), or context
            if line.startswith("-"):
                # We have a removal. If it matches the current line, pop it.
                # If your LLM sometimes omits or changes spaces, you may want a fuzzy match.
                to_remove = line[1:]
                if (0 <= current_line < len(modified_lines)
                        and modified_lines[current_line] == to_remove):
                    modified_lines.pop(current_line)
                else:
                    # If you suspect minor whitespace diffs, you could do:
                    #   if modified_lines[current_line].strip() == to_remove.strip():
                    #       ...
                    pass
            elif line.startswith("+"):
                # This is an addition
                to_add = line[1:]
                if 0 <= current_line <= len(modified_lines):
                    modified_lines.insert(current_line, to_add)
                current_line += 1
            else:
                # Context line => just move the pointer
                current_line += 1

            i += 1

        return "\n".join(modified_lines)

    @staticmethod
    def apply_context_free_diff(original_content, diff_content):
        original_lines = original_content.splitlines()
        diff_lines = diff_content.splitlines()

        # Parse diff into hunks (blocks of changes with context)
        hunks = []
        current_hunk = []

        for line in diff_lines:
            if line.startswith("@@ "):
                continue

            if not line.strip() and current_hunk:  # Empty line and we have content
                if any(l.startswith('+') or l.startswith('-') for l in current_hunk):
                    hunks.append(current_hunk)
                current_hunk = []
            else:
                current_hunk.append(line)

        # Add the last hunk if it contains changes
        if current_hunk and any(l.startswith('+') or l.startswith('-') for l in current_hunk):
            hunks.append(current_hunk)

        # Process each hunk
        result = original_lines.copy()

        for hunk in hunks:
            # Find the context before changes
            context_before = []
            for line in hunk:
                if line.startswith('+') or line.startswith('-'):
                    break
                context_before.append(line)

            # Skip if no context before changes
            if not context_before:
                continue

            # Try to find this context in the result
            start_idx = -1
            for i in range(len(result) - len(context_before) + 1):
                if all(result[i + j] == context_before[j] for j in range(len(context_before))):
                    start_idx = i
                    break

            if start_idx == -1:
                continue  # Context not found

            # Apply the hunk at the found position
            new_result = result[:start_idx + len(context_before)]  # Keep everything up to and including context

            # Process the hunk lines after context
            changes_start = len(context_before)
            result_idx = start_idx + len(context_before)

            for i in range(changes_start, len(hunk)):
                line = hunk[i]

                if line.startswith('+'):
                    # Add new line
                    new_result.append(line[1:])
                elif line.startswith('-'):
                    # Remove the next line in result if it matches
                    if result_idx < len(result) and result[result_idx] == line[1:]:
                        result_idx += 1
                else:
                    # Context line - copy from result and advance
                    if result_idx < len(result):
                        new_result.append(result[result_idx])
                        result_idx += 1

            # Add the rest of the result
            new_result.extend(result[result_idx:])
            result = new_result

        return '\n'.join(result) + '\n'

    @staticmethod
    def write_markdown(markdown: MarkdownContent, dest: io.TextIOBase):
        for content in markdown.sequence:
            if isinstance(content, TextContent):
                dest.write(content.get_str())
            elif isinstance(content, ImageContent):
                if len(content.sequence) > 0:
                    image_type = Helpers.classify_image(content.get_bytes())
                    dest.write(f"![image](data:{image_type};base64,{base64.b64encode(content.sequence).decode('utf-8')})")
                elif content.url:
                    dest.write(f"![image]({content.url})")
        dest.flush()
        return dest

    @staticmethod
    def is_markdown_simple(text):
        if '```markdown' in text:
            return True

        # Define regex patterns for common markdown elements
        patterns = [
            r'^\s{0,3}#{1,6}\s',  # Headers
            r'^\s{0,3}>\s',       # Blockquotes
            r'^\s{0,3}[-*+]\s',   # Unordered lists
            r'^\s{0,3}\d+\.\s',   # Ordered lists
            r'\[.*?\]\(.*?\)',    # Links
            r'!\[.*?\]\(.*?\)',   # Images
            r'^\s{0,3}```',       # Code blocks
            r'\$\$[\s\S]*?\$\$',  # LaTeX block equations
            r'\$.*?\$'            # LaTeX inline equations
        ]
        # Check if any pattern matches the text
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    @staticmethod
    def get_full_url(base_url: str, href: str) -> str:
        # Parse the base URL to extract scheme and domain
        parsed_base = urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

        # If href is already a full URL, return it
        if href.startswith(('http://', 'https://')):
            return href

        # If href starts with '/', join it with the base domain
        if href.startswith('/'):
            return urljoin(base_domain, href)

        # For relative URLs, join with the full base URL
        return str(urljoin(base_url, href))

    @staticmethod
    def command_substitution(input_string):
        def execute_command(match):
            command = match.group(1)

            cmd = Helpers.command_substitution(command)

            try:
                return Helpers.run_streaming(cmd).strip()
            except Exception as e:
                return f"Error: {e}"

            # try:
            #     # Recursively handle nested substitutions
            #     command = Helpers.command_substitution(command)
            #     result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            #     return result.stdout.strip()
            # except subprocess.CalledProcessError as e:
            #     return f"Error: {e}"


        pattern = r'\$\(((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*)\)'
        return re.sub(pattern, execute_command, input_string)

    @staticmethod
    def get_callsite(call_str: str, tools: list[Callable]) -> Optional[FunctionCall]:
        def __get_callsite_helper(
            call: str,
            functions: List[Callable]
        ) -> Optional[Tuple[Callable, Dict[str, Any]]]:
            function_description: Dict[str, Any] = {}

            if call.startswith('def '):
                call = call[4:]

            function_name = Helpers.in_between(call, '', '(')
            if ' ' in function_name or ',' in function_name:
                return None

            function_arg_str = Helpers.in_between(call, '(', ')')
            function_args = []

            is_str = False
            token = ''
            for i in range(0, len(function_arg_str)):
                c = function_arg_str[i]
                if c == '"' and not is_str:
                    is_str = True
                    token += c
                elif c == '"' and is_str:
                    is_str = False
                    token += c
                elif not is_str and c == ',':
                    function_args.append(token.strip())
                    token = ''
                elif not is_str and c == ' ':  # ignore spaces
                    continue
                else:
                    token += c

            if token:
                function_args.append(token.strip())

            # function_args = [p.strip() for p in Helpers.in_between(call, '(', ')').split(',')]
            func = functions[0]

            for f in functions:
                if f.__name__.lower() in function_name.lower():
                    function_description = Helpers.get_function_description(
                        f,
                        openai_format=True
                    )
                    func = f
                    break

            if not function_description:
                return None

            argument_count = 0

            for _, parameter in function_description['parameters']['properties'].items():
                if argument_count < len(function_args):
                    parameter.update({'argument': function_args[argument_count]})
                argument_count += 1

            return func, function_description

        callsite = __get_callsite_helper(call_str, tools)
        if callsite:
            func, function_description = callsite
            name = function_description['name']
            arguments = []
            types = []
            for arg_name, metadata in function_description['parameters']['properties'].items():
                # todo if we don't have an argument here, we should ensure that
                # the function has a default value for the parameter
                if 'argument' in metadata:
                    arguments.append({arg_name: metadata['argument']})
                    types.append({arg_name: metadata['type']})

            return FunctionCall(
                name=name,
                args=arguments,
                types=types,
                func=func,
            )
        return None

    @staticmethod
    def compare_ast(node1, node2):
        if type(node1) is not type(node2):
            return False
        if isinstance(node1, ast.AST):
            for k, v in vars(node1).items():
                if k in ('lineno', 'col_offset', 'ctx'):
                    continue
                if not Helpers.compare_ast(v, getattr(node2, k)):
                    return False
            return True
        elif isinstance(node1, list):
            return len(node1) == len(node2) and all(Helpers.compare_ast(n1, n2) for n1, n2 in zip(node1, node2))
        else:
            return node1 == node2

    @staticmethod
    def compare_code_blocks(code1: str, code2: str):
        try:
            tree1 = ast.parse(Helpers.escape_newlines_in_strings(code1))
        except SyntaxError:
            return False

        try:
            tree2 = ast.parse(Helpers.escape_newlines_in_strings(code2))
        except SyntaxError:
            return False

        return Helpers.compare_ast(tree1, tree2)

    @staticmethod
    def extract_stacktrace_until(stacktrace: str, cls: type):
        filename = inspect.getfile(cls)
        lines = stacktrace.split('\n')

        # Find the last occurrence of the filename in the stacktrace
        file_lines = [i for i, line in enumerate(lines) if filename in line]
        if not file_lines:
            return stacktrace

        start_index = file_lines[-1] + 1  # Start from the line after the last occurrence
        return '\n'.join(lines[start_index:])

    @staticmethod
    def remove_duplicates(lst, key_func=lambda x: x):
        seen = set()
        result = list(filter(lambda x: key_func(x) not in seen and not seen.add(key_func(x)), lst))
        try:
            # check to see if any of the strings in the list are substrings of another list item, and if so, remove that
            sub_dups = [a for a in result if not any(key_func(a) in key_func(b) for b in result if a != b)]
            return sub_dups
        except Exception as ex:
            return result

    @staticmethod
    def openai_image_tok_count(base64_encoded: str):
        def __calculate_image_tokens(width: int, height: int):
                from math import ceil

                h = ceil(height / 512)
                w = ceil(width / 512)
                n = w * h
                total = 85 + 170 * n
                return total
        # go from base64 encoded to bytes
        image = base64.b64decode(base64_encoded)
        # open the image
        img = Image.open(io.BytesIO(image))
        return __calculate_image_tokens(img.width, img.height)

    @staticmethod
    def anthropic_image_tok_count(base64_encoded: str):
        # go from base64 encoded to bytes
        image = base64.b64decode(base64_encoded)
        # open the image
        img = Image.open(io.BytesIO(image))
        return (img.width * img.height) // 750

    @staticmethod
    def anthropic_resize(image_bytes: bytes) -> bytes:
        image_type = Helpers.classify_image(image_bytes)
        pil_extension = 'JPEG'
        if image_type == 'image/png': pil_extension = 'PNG'
        if image_type == 'image/webp': pil_extension = 'WEBP'

        image = Image.open(io.BytesIO(image_bytes))
        original_width, original_height = image.size

        # Determine the aspect ratio and corresponding max dimensions
        aspect_ratio = original_width / original_height
        max_dimensions = {
            (1, 1): (1092, 1092),
            (3, 4): (951, 1268),
            (2, 3): (896, 1344),
            (9, 16): (819, 1456),
            (1, 2): (784, 1568)
        }

        # Find the closest aspect ratio and its max dimensions
        closest_ratio = min(max_dimensions.keys(), key=lambda x: abs((x[0]/x[1]) - aspect_ratio))
        max_width, max_height = max_dimensions[closest_ratio]

        # Check if the image exceeds the maximum dimensions
        if original_width > max_width or original_height > max_height:
            # Resize the image
            image.thumbnail((max_width, max_height), Image.LANCZOS)  # type: ignore

        # Save or return the image
        resized = io.BytesIO()
        image.save(resized, format=pil_extension)
        return resized.getvalue()

    @staticmethod
    async def download_bytes(url_or_file: str, throw: bool = True) -> bytes:
        url_result = urlparse(url_or_file)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        if url_result.scheme in ('http', 'https'):
            async with httpx.AsyncClient() as client:
                response = await client.get(url_or_file, headers=headers, follow_redirects=True, timeout=5)
                if response.status_code != 200:
                    if throw: raise ValueError(f'Helpers.download_bytes() Failed to download: {url_or_file}. Status code is: {response.status_code}')
                    return b''
                return response.content
        else:
            try:
                with open(url_or_file, 'rb') as file:
                    return file.read()
            except FileNotFoundError:
                if throw: raise ValueError(f'Helpers.download_bytes() The supplied argument url_or_file: {url_or_file} is not a correct filename or url.')
                return b''

    @staticmethod
    async def download(url_or_file: str) -> str:
        stream = await Helpers.download_bytes(url_or_file)
        if stream:
            return stream.decode('utf-8')
        return ''


    @staticmethod
    async def get_image_fuzzy_url(logging, url: str, image_url: str, min_width: int, min_height: int) -> bytes:
        # Return early if there's no image URL
        if not image_url:
            return b''

        # If the URL starts with "www", prepend https:// to it
        if image_url.startswith('www'):
            image_url = 'https://' + image_url

        try:
            # Handle protocol-relative URLs (e.g., //example.com/img.jpg)
            if image_url.startswith('//'):
                parsed_base = urlparse(url)
                scheme = parsed_base.scheme if parsed_base.scheme else 'https'
                full_url = f"{scheme}:{image_url}"
                result = await Helpers.download_bytes(full_url)

            # Handle full http(s) URLs
            elif image_url.startswith('http://') or image_url.startswith('https://'):
                result = await Helpers.download_bytes(image_url)

            # Handle local file URLs and paths
            elif image_url.startswith('file://'):
                local_path = image_url[len('file://'):]
                with open(local_path, 'rb') as f:
                    result = f.read()
            # Also if the file exists on disk using the given image_url directly
            elif os.path.exists(image_url):
                with open(image_url, 'rb') as f:
                    result = f.read()

            # Otherwise, assume it's a relative URL and build the full URL accordingly.
            else:
                full_url = urljoin(url, image_url)
                result = await Helpers.download_bytes(full_url)

            # Validate the image dimensions.
            width, height = Helpers.image_size(result)
            if width >= min_width and height >= min_height:
                if Helpers.is_webp(result):
                    result = Helpers.convert_image_to_png(result)

                return result

        except Exception as e:
            logging.debug(f"Helpers.get_image_fuzzy_url({image_url}) exception caught: {e}")

        return b''

    @staticmethod
    def remove_embedded_images(markdown_string):
        pattern = r'(\[.*?\]):\s*<data:image/[^;]+;base64,\s*[^>]+>'

        def replace_with_tag(match):
            return match.group(1)

        cleaned_markdown = re.sub(pattern, replace_with_tag, markdown_string)
        return cleaned_markdown

    @staticmethod
    async def markdown_content_to_supported_content(
        logging, markdown_content: MarkdownContent, min_width: int, min_height: int
    ) -> list[SupportedMessageContent]:
        # Patterns for inline images and text
        pattern = r'(.*?)!\[(.*?)\]\((.*?)\)|(.+?)$'

        # Pattern for embedded inline images
        embedded_image_pattern = r'!\[([^\]]*?)\]\(data:image/([^;]+);base64,([^)]+)\)'

        # New patterns for reference-style images
        reference_pattern = r'!\[\]\[([^\]]+)\]'
        definition_pattern = r'\[([^\]]+)\]:\s*<(data:image/([^;]+);base64,([^>]+))>'

        content = markdown_content.get_str()
        content_list: List[SupportedMessageContent] = []
        embedded_images: Dict[str, ImageContent] = {}

        # First, extract embedded inline images
        for match in re.finditer(embedded_image_pattern, content, re.MULTILINE):
            image_id, image_type, base64_data = match.groups()
            image_bytes = base64.b64decode(base64_data)
            embedded_images[image_id] = ImageContent(image_bytes, f"{image_id}.{image_type}")

        # Extract reference-style embedded images and remove the definitions
        for match in re.finditer(definition_pattern, content, re.MULTILINE):
            ref_id, full_uri, image_type, base64_data = match.groups()
            image_bytes = base64.b64decode(base64_data)
            embedded_images[ref_id] = ImageContent(image_bytes, f"{ref_id}.{image_type}")

        # Remove the image definitions from the content
        content = re.sub(definition_pattern, '', content)

        # Replace inline embedded images with placeholders
        content = re.sub(
            r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[^)]+\)',
            r'![\1](None)',  # \1 refers to the captured alt text
            content
        )

        # Replace reference-style embedded images with placeholders
        for ref_id in embedded_images.keys():
            content = re.sub(
                r'!\[\]\[' + re.escape(ref_id) + r'\]',
                f'![{ref_id}](None)',
                content
            )

        last_end = 0
        idx = 0
        tasks = []

        for match in re.finditer(pattern, content, re.DOTALL):
            before, alt_text, image_id, after = match.groups()

            if before and before != content[last_end:match.start()] and last_end != match.start():
                content_list.append(TextContent(content[last_end:match.start()]))
                idx += 1

            if before:
                content_list.append(TextContent(before))
                idx += 1

            if alt_text:
                if alt_text in embedded_images:
                    content_list.append(embedded_images[alt_text])
                else:
                    logging.debug(f"Downloading image {image_id} obtained from {markdown_content.url[:25]}")
                    task = asyncio.create_task(
                        Helpers.get_image_fuzzy_url(logging, markdown_content.url, image_id, min_width, min_height)
                    )
                    tasks.append((idx, task, image_id))
                    content_list.append(TextContent(''))
                idx += 1

            if after:
                content_list.append(TextContent(after))
                idx += 1

            last_end = match.end()

        if last_end < len(content):
            remaining_content = content[last_end:].strip()
            if remaining_content:  # Only add if there's non-whitespace content remaining
                content_list.append(TextContent(remaining_content))
                idx += 1

        for idx, task, image_url in tasks:
            image_bytes = await task
            if image_bytes:
                content_list[idx] = ImageContent(image_bytes, image_url)

        # Collapse the content list
        collapsed_content_list = [c for c in content_list if c.sequence]
        combined_content_list = []
        current_text = ""

        for c in collapsed_content_list:
            if isinstance(c, ImageContent):
                if current_text:
                    combined_content_list.append(TextContent(current_text))
                    current_text = ""
                combined_content_list.append(c)
            else:
                current_text += c.get_str()

        if current_text:
            current_text = current_text.strip()  # Remove leading/trailing whitespace
            if current_text:  # Only add if there's non-whitespace content
                combined_content_list.append(TextContent(current_text))

        return combined_content_list

    @staticmethod
    def last_day_of_quarter(year, quarter):
        start_month = 3 * quarter - 2
        end_month = start_month + 2

        if end_month > 12:
            end_month = 12

        last_day = (dt.datetime(year, end_month, 1) + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1)
        return last_day

    @staticmethod
    def parse_relative_datetime(relative_expression: str, timezone: Optional[str] = None) -> dt.datetime:
        if relative_expression.startswith('Q'):
            quarter = int(relative_expression[1:])
            return Helpers.last_day_of_quarter(dt.datetime.now().year, quarter)

        tz = dt.datetime.now().astimezone().tzinfo

        if timezone:
            tz = ZoneInfo(timezone)

        if 'now' in relative_expression:
            return dt.datetime.now(tz)

        parts = relative_expression.split()

        if len(parts) != 2:
            return dateparser.parse(relative_expression)  # type: ignore

        value = int(parts[0])
        unit = parts[1].lower()

        if unit == "days":
            return dt.datetime.now(tz) + dt.timedelta(days=value)
        elif unit == "months":
            return dt.datetime.now(tz) + relativedelta(months=value)
        elif unit == "years":
            return dt.datetime.now(tz) + relativedelta(years=value)
        elif unit == "hours":
            return dt.datetime.now(tz) + dt.timedelta(hours=value)
        else:
            return dateparser.parse(relative_expression)  # type: ignore

    @staticmethod
    def load_resize_save(raw_data: bytes, output_format='PNG', max_size=5 * 1024 * 1024) -> bytes:
        if output_format not in ['PNG', 'JPEG', 'WEBP']:
            raise ValueError('Invalid output format')

        temp_output = io.BytesIO()
        result: bytes
        with Image.open(io.BytesIO(raw_data)) as im:
            # convert to the required format
            im.save(temp_output, format=output_format)
            temp_output.seek(0)
            result = temp_output.getvalue()

            # check to see if larger than 5MB
            if len(raw_data) >= max_size:
                # Reduce the image size
                for quality in range(95, 10, -5):
                    temp_output.seek(0)
                    temp_output.truncate(0)
                    im.save(temp_output, format=output_format, quality=quality)
                    reduced_data = temp_output.getvalue()
                    if len(reduced_data) <= max_size:
                        result = reduced_data
                        raw_data = result
                        break
                else:
                    # If the image is still too large, resize the image
                    while len(raw_data) > max_size:
                        im = im.resize((int(im.width * 0.9), int(im.height * 0.9)))
                        temp_output.seek(0)
                        temp_output.truncate(0)
                        im.save(temp_output, format=output_format)
                        result = temp_output.getvalue()
                        raw_data = result
        return result

    @staticmethod
    def classify_image(raw_data):
        if raw_data:
            if raw_data[:8] == b'\x89PNG\r\n\x1a\n': return 'image/png'
            elif raw_data[:2] == b'\xff\xd8': return 'image/jpeg'
            elif raw_data[:4] == b'RIFF' and raw_data[-4:] == b'WEBP': return 'image/webp'
        return 'image/unknown'

    @staticmethod
    def log_exception(logger, e, message=None):
        exc_traceback = e.__traceback__

        while exc_traceback.tb_next:
            exc_traceback = exc_traceback.tb_next
        frame = exc_traceback.tb_frame
        lineno = exc_traceback.tb_lineno
        filename = frame.f_code.co_filename

        log_message = traceback.format_exception(type(e), e, e.__traceback__)
        if message:
            log_message += f": {message}"

        logger.error(log_message)

    @staticmethod
    def glob_exclusions(pattern):
        if not pattern.startswith('!'):
            return []

        pattern = pattern.replace('!', '')
        # Find files matching exclusion patterns
        excluded_files = set()
        excluded_files.update(glob.glob(pattern, recursive=True))
        return excluded_files

    @staticmethod
    def is_glob_pattern(s):
        return any(char in s for char in "*?[]{}!")

    @staticmethod
    def is_glob_recursive(s):
        return '**' in s

    @staticmethod
    def glob_brace(pattern):
        parts = pattern.split('{')
        if len(parts) == 1:
            # No brace found, use glob directly
            return glob.glob(pattern)

        pre = parts[0]
        post = parts[1].split('}', 1)[1]
        options = parts[1].split('}', 1)[0].split(',')

        # Create individual patterns
        patterns = [pre + option + post for option in options]

        # Apply glob to each pattern and combine results
        files = set(itertools.chain.from_iterable(glob.glob(pat) for pat in patterns))
        return list(files)

    @staticmethod
    def late_bind(module_name, class_name, method_name, *args, **kwargs):
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            method = getattr(cls, method_name)

            if isinstance(method, staticmethod):
                return method.__func__(*args, **kwargs)
            else:
                instance = cls()
                return getattr(instance, method_name)(*args, **kwargs)
        except Exception as e:
            pass

    @staticmethod
    def find_wezterm():
        import shutil
        # Try the standard way first
        wezterm_path = shutil.which("wezterm")
        if wezterm_path:
            return wezterm_path

        # Common fallback locations on macOS
        possible_paths = [
            "/Applications/WezTerm.app/Contents/MacOS/wezterm",
            os.path.expanduser("~/Applications/WezTerm.app/Contents/MacOS/wezterm"),
            "/usr/local/bin/wezterm",
            "/opt/homebrew/bin/wezterm",  # for Homebrew on Apple Silicon
        ]

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    @staticmethod
    def find_kitty():
        import shutil
        kitty_path = shutil.which("kitty")
        if kitty_path:
            return kitty_path

        possible_paths = [
            "/Applications/kitty.app/Contents/MacOS/kitty",
            os.path.expanduser("~/Applications/kitty.app/Contents/MacOS/kitty"),
            "/usr/local/bin/kitty",
            "/opt/homebrew/bin/kitty",  # Homebrew install on Apple Silicon
        ]

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    @staticmethod
    def is_running(process_name):
        try:
            # Use 'ps' to list processes and grep for process_name
            output = subprocess.check_output(['ps', 'aux'], text=True)
            return process_name.lower() in output.lower()
        except Exception:
            return False

    @staticmethod
    def __find_terminal_emulator(process):
        try:
            if process.parent():
                # Check if the process name matches known terminal emulators
                name = process.parent().name()
                if 'Terminal' in name:
                    return 'Terminal'
                elif 'iTerm' in name:
                    return 'iTerm2'
                elif 'alacritty' in name:
                    return 'alacritty'
                elif 'kitty' in name:
                    return 'kitty'
                elif 'tmux' in name:
                    return 'tmux'
                elif 'wezterm-gui' in name:
                    return 'wezterm'
                elif 'wezterm' in name:
                    return 'wezterm'
                # If no match, check the next parent
                return Helpers.__find_terminal_emulator(process.parent())
            else:
                # No more parents, terminal emulator not found
                return 'Unknown'
        except Exception as e:
            return str(e)

    @staticmethod
    def is_emulator(emulator: str):
        try:
            pid = os.getpid()
            # Get the parent process (emulator) of the current process
            output = subprocess.check_output(['ps', '-o', 'comm=', '-p', str(pid)], text=True)
            current_process_name = output.strip()
            return emulator.lower() == current_process_name.lower()
        except Exception:
            return False

    @staticmethod
    def is_pdf(byte_stream):
        # PDF files start with "%PDF-" (hex: 25 50 44 46 2D)
        pdf_signature = b'%PDF-'
        # Read the first 5 bytes to check the signature
        first_bytes = byte_stream.read(5)
        # Reset the stream position to the beginning if possible
        if hasattr(byte_stream, 'seek'):
            byte_stream.seek(0)
        # Return True if the signature matches
        return first_bytes == pdf_signature

    @staticmethod
    def convert_image_to_png(byte_stream: bytes) -> bytes:
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            buffer = io.BytesIO()
            with Image.open(io.BytesIO(byte_stream)) as im:
                im.save(buffer, format='PNG')
                buffer.seek(0)
                return buffer.getvalue()
        except Exception:
            return byte_stream

    @staticmethod
    def is_webp(byte_stream):
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return im.format == 'WEBP'
        except Exception:
            return False

    @staticmethod
    def is_image(byte_stream):
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return True
        except Exception:
            return False

    @staticmethod
    def is_markdown(byte_stream: Union[bytes, str, io.BytesIO]) -> bool:
        content = ''
        if isinstance(byte_stream, io.BytesIO):
            byte_stream = byte_stream.getvalue()

        # Convert the byte stream to a string
        if isinstance(byte_stream, bytes):
            content = byte_stream.decode('utf-8', errors='ignore')
        else:
            content = byte_stream

        # Define regex patterns for common Markdown elements
        patterns = {
            'headers': r'^#{1,6}\s',
            'lists': r'^\s*[-*+]\s',
            'numbered_lists': r'^\s*\d+\.\s',
            'code_blocks': r'```[\s\S]*?```',
            'links': r'\[([^\]]+)\]\(([^\)]+)\)',
            'images': r'!\[([^\]]+)\]\(([^\)]+)\)',
            'emphasis': r'\*\*[\s\S]*?\*\*|\*[\s\S]*?\*|__[\s\S]*?__|_[\s\S]*?_',
            'blockquotes': r'^>\s',
            'horizontal_rules': r'^(-{3,}|\*{3,}|_{3,})$',
            'tables': r'\|[^|\r\n]*\|',
            'latex_blocks': r'\$\$[\s\S]*?\$\$',  # LaTeX block equations
            'latex_inline': r'\$[^\$\n]+?\$'      # LaTeX inline equations
        }

        # Count the number of matches for each pattern
        matches = {key: len(re.findall(pattern, content, re.MULTILINE)) for key, pattern in patterns.items()}

        # Calculate the total number of matches
        total_matches = sum(matches.values())

        # Calculate the number of lines in the content
        num_lines = len(content.splitlines())

        # Check if there are multiple types of Markdown elements
        diverse_elements = sum(1 for count in matches.values() if count > 0)

        # Define thresholds
        min_matches = 3
        min_types = 2
        # todo: removed this, might have to bring it back
        # max_ratio = 0.8  # Maximum ratio of matches to lines

        # Make the decision
        if (total_matches >= min_matches and
            diverse_elements >= min_types):
            return True
        return False

    @staticmethod
    def decompress_if_compressed(byte_stream):
        try:
            # Attempt to decompress to see if it's zlib compressed
            decompressed_data = zlib.decompress(byte_stream)
            return True, decompressed_data
        except zlib.error as e:
            # If decompression fails, it's likely not zlib compressed or the data is corrupted/incomplete
            return False, byte_stream

    @staticmethod
    def image_size(byte_stream: bytes) -> tuple[int, int]:
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return im.size
        except Exception:
            return (0, 0)

    @staticmethod
    def is_base64_encoded(s):
        import binascii

        try:
            if len(s) % 4 == 0:
                base64.b64decode(s, validate=True)
                return True
        except (ValueError, binascii.Error):
            return False
        return False

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def read_netscape_cookies(cookies_file_content: str) -> List[Dict[str, Any]]:
        cookies = []
        for line in cookies_file_content.splitlines():
            if not line.startswith('#') and line.strip():  # Ignore comments and empty lines
                try:
                    domain, _, path, secure, expires_value, name, value = line.strip().split('\t')

                    if not expires_value.isnumeric():
                        if expires_value == 'Session':
                            expires_value = 1999999999
                        else:
                            import time
                            expiration_datetime = dt.datetime.strptime(expires_value, '%Y-%m-%dT%H:%M:%S.%fZ')
                            expires_value = int(time.mktime(expiration_datetime.timetuple()))

                    if int(expires_value) != -1 and int(expires_value) < 0:
                        continue  # Skip invalid cookies

                    dt_object = dt.datetime.fromtimestamp(int(expires_value))
                    if dt_object.date() < dt.datetime.now().date():
                        continue

                    cookies.append({
                        "name": name,
                        "value": value,
                        "domain": domain,
                        "path": path,
                        "expires": int(expires_value),
                        "httpOnly": False,
                        "secure": secure == "TRUE"
                    })
                except Exception as ex:
                    pass
        return cookies


    @staticmethod
    def get_callables(logging: Logger, input_str: str) -> Optional[Union[List[Callable], Callable]]:
        parts = input_str.split(".")

        if input_str.startswith('search'):
            logging.debug('hello')

        if len(parts) < 2:
            logging.error(f"Invalid input string: {input_str}")
            return None

        parts_counter = len(parts) - 1
        module_name = ''
        while parts_counter > 0:
            module_name = ".".join(parts[:parts_counter])
            try:
                module = importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                pass
            parts_counter -= 1

        if not module:
            logging.error(f"Module '{module_name}' not found")
            return None

        # module_name is the module part of the input string
        # lets check to see if the rest is a class.static_method or just a class
        counter = len(parts) - 1

        while counter >= 0:
            if getattr(module, parts[counter], None) is not None:
                func_or_class = getattr(module, parts[counter], None)
                if inspect.isfunction(func_or_class) or inspect.ismethod(func_or_class):
                    return func_or_class

                elif inspect.isclass(func_or_class):
                    # could be a static class
                    if counter < len(parts) - 1:
                        # it's a static class
                        return getattr(func_or_class, parts[counter + 1], None)
                    else:
                        # it's a class
                        return Helpers.__get_class_callables(func_or_class)
            counter -= 1

        logging.error(f"Couldn't resolve '{input_str}'")
        return None

    @staticmethod
    def __get_class_callables(class_obj) -> List[Callable]:
        return [
            member for name, member in inspect.getmembers(class_obj)
            if (inspect.isfunction(member) or inspect.ismethod(member))
            and member.__doc__
            and not name.startswith('_')
        ]

    @staticmethod
    def tfidf_similarity(query: str, text_list: list[str]) -> str:
        def tokenize(text: str) -> List[str]:
            # Simple tokenizer, can be enhanced
            return text.lower().split()

        def compute_tf(text_tokens: List[str]) -> dict:
            # Count the occurrences of each word in the text
            tf = Counter(text_tokens)
            # Divide by the total number of words for Term Frequency
            tf = {word: count / len(text_tokens) for word, count in tf.items()}
            return tf

        def compute_idf(documents: List[List[str]]) -> dict:
            # Number of documents
            N = len(documents)
            # Count the number of documents that contain each word
            idf = {}
            for document in documents:
                for word in set(document):
                    idf[word] = idf.get(word, 0) + 1
            # Calculate IDF
            idf = {word: math.log(N / df) for word, df in idf.items()}
            return idf

        def compute_tfidf(tf: dict, idf: dict) -> dict:
            # Multiply TF by IDF
            tfidf = {word: tf_value * idf.get(word, 0) for word, tf_value in tf.items()}
            return tfidf

        def cosine_similarity(vec1: dict, vec2: dict) -> float:
            # Compute the dot product
            dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in set(vec1.keys()) | set(vec2.keys()))
            # Compute the magnitudes
            mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
            mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
            # Compute cosine similarity
            if mag1 * mag2 == 0:
                return 0
            else:
                return dot_product / (mag1 * mag2)

        # Tokenize and prepare documents
        documents = [tokenize(text) for text in text_list]
        query_tokens = tokenize(query)
        documents.append(query_tokens)

        # Compute IDF for the whole corpus
        idf = compute_idf(documents)

        # Compute TF-IDF for query and documents
        query_tfidf = compute_tfidf(compute_tf(query_tokens), idf)
        documents_tfidf = [compute_tfidf(compute_tf(doc), idf) for doc in documents[:-1]]  # Exclude query

        # Compute similarity and find the most similar document
        similarities = [cosine_similarity(query_tfidf, doc_tfidf) for doc_tfidf in documents_tfidf]
        max_index = similarities.index(max(similarities))

        return text_list[max_index]

    @staticmethod
    def flatten(items: List[Any]) -> List[Any]:
        flattened = []
        for item in items:
            if isinstance(item, list):
                flattened.extend(Helpers.flatten(item))
            else:
                flattened.append(item)
        return flattened

    @staticmethod
    def extract_token(s, ident):
        if s.startswith(ident) and ' ' in s:
            return s[0:s.index(' ')]

        if ident in s:
            parts = s.split(ident)
            start = parts[0][parts[0].rfind(' ') + 1:] if ' ' in parts[0] else parts[0]
            end = parts[1][:parts[1].find(' ')] if ' ' in parts[1] else parts[1]
            return start + ident + end
        return ''

    @staticmethod
    def after_end(s: str, start: str, end: str) -> str:
        # Find the start position
        start_pos = s.find(start)
        if start_pos == -1:
            return s

        # Find the end position, starting from after the start token
        end_pos = s.find(end, start_pos + len(start))
        if end_pos == -1:
            return s

        # Extract the content after the end token
        result = s[end_pos + len(end):]
        return result

    @staticmethod
    def in_between(s, start, end):
        if end == '\n' and '\n' not in s:
            return s[s.find(start) + len(start):]

        after_start = s[s.find(start) + len(start):]
        part = after_start[:after_start.find(end)]
        return part

    @staticmethod
    def in_between_including(s, start, end):
        start_index = s.find(start)
        if start_index == -1:
            return ""  # Return empty string if start is not found

        after_start = s[start_index:]
        end_index = after_start.find(end, len(start))  # Start searching for `end` after `start`

        if end_index == -1:
            return after_start  # Return everything after start if end is not found

        return s[start_index:start_index + end_index + len(end)]

    @staticmethod
    def outside_of(s, start, end):
        if end == '\n' and '\n' not in s:
            return s[:s.find(start)]

        before_start = s[:s.find(start)]
        after_end = s[s.find(end) + len(end):]
        return before_start + after_end

    @staticmethod
    def in_between_ends(s, start, end_strs: List[str]):
        # get the text from s between start and any of the end_strs strings.
        possibilities = []
        for end in end_strs:
            if end == '\n' and '\n' not in s:
                result = s[s.find(start) + len(start):]
                possibilities.append(result)
            elif end in s:
                after_start = s[s.find(start) + len(start):]
                part = after_start[:after_start.find(end)]
                if part:
                    possibilities.append(part)

        # return the shortest one
        return min(possibilities, key=len)

    @staticmethod
    def extract_blocks(markdown_text: str, block_type: str):
        # Pattern to match code blocks with or without specified language
        pattern = fr'```{block_type}(\w+\n)?(.*?)```'

        # Using re.DOTALL to make the '.' match also newlines
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        # Extracting just the code part (ignoring the optional language part)
        code_blocks = [match[1].strip() for match in matches]
        return code_blocks

    @staticmethod
    def extract_code_blocks(markdown_text) -> list:
        # Pattern to match code blocks with or without specified language
        pattern = r'```(\w+\n)?(.*?)```'

        # Using re.DOTALL to make the '.' match also newlines
        matches = re.findall(pattern, markdown_text, re.DOTALL)

        # Extracting just the code part (ignoring the optional language part)
        code_blocks = [match[1].strip() for match in matches]
        return code_blocks

    @staticmethod
    def extract_context(s, start, end, stop_tokens=['\n', '.', '?', '!']):
        def capture(s, stop_tokens, backwards=False):
            if backwards:
                for i in range(len(s) - 1, -1, -1):
                    if s[i] in stop_tokens:
                        return s[i + 1:]
                return s
            else:
                for i in range(0, len(s)):
                    if s[i] in stop_tokens:
                        return s[:i]
                return s

        if end == '\n' and '\n' not in s:
            s += '\n'

        left_of_start = s.split(start)[0]
        right_of_end = s.split(end)[-1]
        return str(capture(left_of_start, stop_tokens, backwards=True)) + str(capture(right_of_end, stop_tokens))

    @staticmethod
    def strip_between(s: str, start: str, end: str):
        first = s[:s.find(start)]
        rest = s[s.find(start) + len(start):]
        return first + rest[rest.find(end) + len(end):]

    @staticmethod
    def split_between(s: str, start: str, end: str):
        first = s[:s.find(start)]
        rest = s[s.find(start) + len(start):]
        return (first, rest[rest.find(end) + len(end):])

    @staticmethod
    def first(predicate, iterable, default=None):
        try:
            result = next(x for x in iterable if predicate(x))
            return result
        except StopIteration as ex:
            return default

    @staticmethod
    def filter(predicate, iterable):
        return [x for x in iterable if predicate(x)]

    @staticmethod
    def remove(predicate, iterable):
        return [x for x in iterable if not predicate(x)]

    @staticmethod
    def last(predicate, iterable, default=None):
        result = [x for x in iterable if predicate(x)]
        if result:
            return result[-1]
        return default

    @staticmethod
    def resize_image(screenshot_data, base_width=500):
        # Load the image from the in-memory data
        image = Image.open(io.BytesIO(screenshot_data))

        # Calculate the height maintaining the aspect ratio
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * float(w_percent))

        # Resize the image
        image = image.resize((base_width, h_size), Image.NEAREST)  # type: ignore

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def find_string_between_tokens(text, start_token, end_token):
        start_index = text.rfind(start_token)
        if start_index == -1:
            return ''

        end_index = text.rfind(end_token, start_index)
        if end_index == -1:
            return ''

        result = text[start_index + len(start_token):end_index]
        return result.strip()

    @staticmethod
    def roundrobin(*iterables):
        num_active = len(iterables)
        nexts = cycle(iter(it).__next__ for it in iterables)
        while num_active:
            try:
                for next in nexts:
                    yield next()
            except StopIteration:
                # Remove the iterator we just exhausted from the cycle.
                num_active -= 1
                nexts = cycle(islice(nexts, num_active))

    @staticmethod
    def iter_over_async(ait, loop):
        ait = ait.__aiter__()

        async def get_next():
            try:
                obj = await ait.__anext__()
                return False, obj
            except StopAsyncIteration:
                return True, None
        while True:
            done, obj = loop.run_until_complete(get_next())
            if done:
                break
            yield obj

    @staticmethod
    def split_text_into_chunks_eol_boundary_aligned(text: str, max_chunk_length: int = 400) -> List[str]:
        lines = text.splitlines()
        sentences: List[str] = []

        for line in lines:
            parts = line.split('.')
            parts_with_period = [bel + '.' for bel in parts if bel]
            sentences.extend(parts_with_period)

        combined: List[str] = []

        for sentence in sentences:
            if not combined:
                combined.append(sentence)
                continue

            prev = combined[-1]
            if len(prev) + len(sentence) < max_chunk_length:
                combined[-1] = f'{prev.strip()} {sentence.strip()}'
            else:
                combined.append(sentence)
        return combined

    @staticmethod
    def split_text_into_chunks(text: str, max_chunk_length: int = 400) -> List[str]:
        words = text.split()
        result = []
        for i in range(0, len(words), max_chunk_length):
            result.append(' '.join(words[i:i + max_chunk_length]))
        return result

    @staticmethod
    def find_closest_sections(query: str, sections: list[str]):
        raise NotImplementedError('This is not implemented yet')
        # from sentence_transformers import SentenceTransformer, util
        # from torch import Tensor

        # model = SentenceTransformer('all-mpnet-base-v2')
        # corpus_embeddings: List[Tensor] | Any = model.encode(sections, convert_to_tensor=True)
        # query_embedding: List[Tensor] | Any = model.encode([query], convert_to_tensor=True)

        # cosine_scores = util.cos_sim(corpus_embeddings, query_embedding)  # type: ignore

        # scored_sections = list(zip(sections, cosine_scores))
        # scored_sections = sorted(scored_sections, key=lambda x: x[1], reverse=True)  # type: ignore

        # scores = [{'text': text, 'score': score.cpu().item()} for text, score in scored_sections]
        # return scores

    @staticmethod
    def chunk_and_rank(query: str, data: str, max_chunk_length=400) -> List[str]:
        """Chunks the data into sections and ranks them based on the query"""
        sections = Helpers.split_text_into_chunks(data, max_chunk_length=max_chunk_length)
        if sections:
            results = Helpers.find_closest_sections(query, sections)
        else:
            return []
        return [a['text'] for a in results]

    @staticmethod
    def prompt_data_iterable(
        prompt: str,
        data: str,
        max_tokens=4000,
        prompt_at_end: bool = False,
    ) -> Generator[str, None, None]:
        """Ensures that prompt and data are under the max token length, repeats prompt and data if necessary"""
        prompt_words = prompt.split()
        sections = Helpers.split_text_into_chunks(data, max_chunk_length=max_tokens - len(prompt_words))
        for section in sections:
            if prompt_at_end:
                yield f'{section} {prompt}'
            else:
                yield f'{prompt} {section}'

    @staticmethod
    def calculate_prompt_cost(content: str, max_chunk_length=4000):
        words = content.split()

        # confirm with user
        est_tokens = len(words) / 0.75
        cost_per_token = 0.0002 / 1000
        est_cost = est_tokens * cost_per_token
        num_chunks = round(len(words) / max_chunk_length)
        est_time = est_tokens / 4000 * 1.5  # around 1.5 mins per 4000 tokens
        return {
            'est_tokens': est_tokens,
            'cost_per_token': cost_per_token,
            'est_cost': est_cost,
            'num_chunks': num_chunks,
            'est_time': est_time,
        }

    @staticmethod
    def messages_to_str(messages: List[Dict[str, str]]) -> str:
        words = []
        for m in messages:
            words.append([w.split() for w in m.values()])
        return ' '.join(Helpers.flatten(words))

    @staticmethod
    async def generator_for_new_tokens(program, *args, **kwargs):
        future = program(*args, **kwargs, silent=True, async_mode=True)
        starting_text = future.text
        while not future._execute_complete.is_set():
            await asyncio.sleep(0.2)
            snapshot = future.text
            yield snapshot[len(starting_text):]
            starting_text = snapshot
        yield future.text[len(starting_text):]

    @staticmethod
    def run_and_stream(program, *args, **kwargs):
        try:
            other_loop = asyncio.get_event_loop()
            nest_asyncio.apply(other_loop)
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()

        full_text = ""
        for new_text in Helpers.iter_over_async(Helpers.generator_for_new_tokens(program, *args, **kwargs), loop):
            if new_text:
                full_text += new_text
                yield new_text

    @staticmethod
    def strip_roles(text: str) -> str:
        text = text.replace('{{llm.default_system_prompt}}', '')
        result = text.replace('{{#system~}}', '') \
            .replace('{{~/system}}', '') \
            .replace('{{#user~}}', '') \
            .replace('{{~/user}}', '') \
            .replace('{{#assistant~}}', '') \
            .replace('{{~/assistant}}', '')
        return result

    @staticmethod
    def __get_class_of_func(func):
        try:
            # First attempt: check if it's a bound method
            if inspect.ismethod(func):
                for cls in inspect.getmro(func.__self__.__class__):
                    if cls.__dict__.get(func.__name__) is func:
                        return cls
                func = func.__func__  # fallback to **qualname** parsing

            # Second attempt: use qualname for regular functions
            if inspect.isfunction(func):
                # Check if __qualname__ exists and has proper format
                if hasattr(func, '__qualname__') and '.' in func.__qualname__:
                    try:
                        # Get the module
                        module = inspect.getmodule(func)
                        if module:
                            # Extract class name from qualname
                            class_name = func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
                            cls = getattr(module, class_name, None)
                            if isinstance(cls, type):
                                return cls
                    except (AttributeError, ValueError):
                        pass

            # Third attempt: check for __objclass__ attribute (descriptors)
            cls = getattr(func, '__objclass__', None)
            if isinstance(cls, type):
                return cls

            # Fourth attempt: check globals dictionary directly
            if hasattr(func, '__globals__') and func.__globals__:
                # Try to find a class in globals that contains this function
                for name, obj in func.__globals__.items():
                    if isinstance(obj, type):
                        # Check if function exists in the class's dict
                        if func.__name__ in obj.__dict__ and (
                                obj.__dict__[func.__name__] is func or
                                (hasattr(obj.__dict__[func.__name__], '__func__') and
                                obj.__dict__[func.__name__].__func__ is func)):
                            return obj
        except Exception:
            # Catch any other exceptions during introspection
            pass

        # We couldn't determine the class
        return None

    @staticmethod
    def __get_class_of_func2(func):
        if inspect.ismethod(func):
            for cls in inspect.getmro(func.__self__.__class__):
                if cls.__dict__.get(func.__name__) is func:
                    return cls
            func = func.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(func):
            cls = getattr(
                inspect.getmodule(func),
                func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            )
            if isinstance(cls, type):
                return cls
        return getattr(func, '__objclass__', None)  # handle special descriptor objects

    @staticmethod
    def is_static_method(func) -> Tuple[bool, Optional[type]]:
        try:
            class_name = func.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            cls = func.__globals__.get(class_name)

            if cls is None or not isinstance(cls, type):
                return True, None  # Treat as standalone function

            for base_cls in inspect.getmro(cls):
                if func.__name__ in base_cls.__dict__:
                    # It's a method, check if it's static
                    if isinstance(base_cls.__dict__[func.__name__], staticmethod):
                        return True, None
                    return False, base_cls
        except (AttributeError, TypeError):
            pass
        return True, None

    @staticmethod
    def get_function_description(func, openai_format: bool) -> Dict[str, Any]:
        def parse_type(t):
            if t is str:
                return 'string'
            elif t is int:
                return 'integer'
            elif t is IntEnum:
                return 'integer'
            elif t is Enum:
                return 'string'
            elif t is float:
                return 'number'
            else:
                return 'object'

        import inspect

        description = ''
        if func.__doc__ and parse(func.__doc__).short_description:
            description = parse(func.__doc__).short_description
        if func.__doc__ and parse(func.__doc__).long_description:
            description += ' ' + str(parse(func.__doc__).long_description).replace('\n', ' ')  # type: ignore

        func_name = func.__name__
        func_class = Helpers.__get_class_of_func(func)
        invoked_by = f'{func_class.__name__}.{func_name}' if func_class else func_name

        params = {}

        for p in inspect.signature(func).parameters:
            param = inspect.signature(func).parameters[p]
            parameter = {
                param.name: {
                    'type': parse_type(param.annotation) if param.annotation is not inspect._empty else 'object',
                    'description': '',
                }
            }

            if param.annotation and isinstance(param.annotation, type) and issubclass(param.annotation, Enum):
                values = [v.value for v in param.annotation.__members__.values()]
                parameter[param.name]['enum'] = values  # type: ignore

            params.update(parameter)

            # if it's got doc comments, use those instead
            for p in parse(func.__doc__).params:  # type: ignore
                params.update({
                    p.arg_name: {  # type: ignore
                        'type': parse_type(p.type_name) if p.type_name is not None else 'string',  # type: ignore
                        'description': p.description,  # type: ignore
                    }  # type: ignore
                })

        def required_params(func):
            parameters = inspect.signature(func).parameters
            return [
                name for name, param in parameters.items()
                if param.default == inspect.Parameter.empty and param.kind != param.VAR_KEYWORD
            ]

        function = {
            'name': invoked_by,
            'description': description,
            'parameters': {
                'type': 'object',
                'properties': params
            },
            'required': required_params(func),
        }

        if openai_format:
            return function
        else:
            return {
                'invoked_by': invoked_by,
                'description': description,
                'parameters': list(params.keys()),
                'types': [p['type'] for p in params.values()],
                'return_type': typing.get_type_hints(func).get('return')
            }

    @staticmethod
    def get_function_description_simple(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        return (f'{description["invoked_by"]}({", ".join(description["parameters"])})  # {description["description"] or "No docstring"}')

    @staticmethod
    def get_function_description_flat_old(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        parameter_type_list = [f"{param}: {typ}" for param, typ in zip(description['parameters'], description['types'])]
        return_type = description['return_type'].__name__ if description['return_type'] else 'Any'

        is_static, cls = Helpers.is_static_method(function)
        if not is_static and cls:
            result = (f'def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type}  # Instantiate with {cls.__name__}(). {description["description"] or "No docstring"}')  # noqa: E501
        else:
            result = (f'def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type}  # @staticmethod {description["description"] or "No docstring"}')  # noqa: E501
        return result

    @staticmethod
    def load_resources_prompt(prompt_name: str, module: str = 'llmvm.server.prompts.python') -> Dict[str, Any]:
        prompt_file = resources.files(module) / prompt_name

        if not os.path.exists(str(prompt_file)):
            raise ValueError(f'Prompt file {prompt_file} does not exist')

        with open(prompt_file, 'r') as f:  # type: ignore
            prompt = f.read()

            if '[system_message]' not in prompt:
                raise ValueError('Prompt file must contain [system_message]')

            if '[user_message]' not in prompt:
                raise ValueError('Prompt file must contain [user_message]')

            system_message = Helpers.in_between(prompt, '[system_message]', '[user_message]').strip()
            user_message = prompt[prompt.find('[user_message]') + len('[user_message]'):].strip()
            templates = []

            temp_prompt = prompt
            while '{{' and '}}' in temp_prompt:
                templates.append(Helpers.in_between(temp_prompt, '{{', '}}'))
                temp_prompt = temp_prompt.split('}}', 1)[-1]

            return {
                'system_message': system_message,
                'user_message': user_message,
                'templates': templates
            }

    @staticmethod
    def get_prompts(
        prompt_text: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        scratchpad_token: str = 'scratchpad',
        append_token: str = '',
    ) -> Tuple[System, User]:
        if '[system_message]' not in prompt_text:
            raise ValueError('Prompt file must contain [system_message]')

        if '[user_message]' not in prompt_text:
            raise ValueError('Prompt file must contain [user_message]')

        system_message = Helpers.in_between(prompt_text, '[system_message]', '[user_message]').strip()
        user_message = prompt_text[prompt_text.find('[user_message]') + len('[user_message]'):].strip()
        templates = []

        temp_prompt = prompt_text
        while '{{' and '}}' in temp_prompt:
            templates.append(Helpers.in_between(temp_prompt, '{{', '}}'))
            temp_prompt = temp_prompt.split('}}', 1)[-1]

        prompt = {
            'system_message': system_message,
            'user_message': user_message,
            'templates': templates
        }

        if not template.get('user_token'):
            template['user_token'] = user_token
            template['user_colon_token'] = user_token + ':'
        if not template.get('assistant_token'):
            template['assistant_token'] = assistant_token
            template['assistant_colon_token'] = assistant_token + ':'
        if not template.get('scratchpad_token'):
            template['scratchpad_token'] = scratchpad_token

        for key, value in template.items():
            prompt['system_message'] = prompt['system_message'].replace('{{' + key + '}}', value)
            prompt['user_message'] = prompt['user_message'].replace('{{' + key + '}}', value)

        # deal with exec() statements to inject things like datetime
        import datetime
        for message_key in ['system_message', 'user_message']:
            message = prompt[message_key]
            while '{{' in message and '}}' in message:
                start = message.find('{{')
                end = message.find('}}', start)
                if end == -1:  # No closing '}}' found
                    break

                key = message[start+2:end]
                replacement = ''

                if key.startswith('exec('):
                    try:
                        replacement = str(eval(key[5:-1]))
                    except Exception as e:
                        pass
                else:
                    replacement = key

                message = message[:start] + replacement + message[end+2:]

            prompt[message_key] = message
        return (System(prompt['system_message']), User(TextContent(prompt['user_message'] + append_token)))

    @staticmethod
    def load_and_populate_prompt(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        scratchpad_token: str = 'scratchpad',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.python'
    ) -> Dict[str, Any]:
        prompt: Dict[str, Any] = Helpers.load_resources_prompt(prompt_name, module)

        try:
            if not template.get('user_token'):
                template['user_token'] = user_token
                template['user_colon_token'] = user_token + ':'
            if not template.get('assistant_token'):
                template['assistant_token'] = assistant_token
                template['assistant_colon_token'] = assistant_token + ':'
            if not template.get('scratchpad_token'):
                template['scratchpad_token'] = scratchpad_token

            for key, value in template.items():
                prompt['system_message'] = prompt['system_message'].replace('{{' + key + '}}', value)
                prompt['user_message'] = prompt['user_message'].replace('{{' + key + '}}', value)

            # todo hack!
            result = Helpers.get_value_from_parent_frame('thread')
            thread_id = 0
            if result:
                thread_id = result.id

            # deal with exec() statements to inject things like datetime
            import datetime
            import tzlocal

            for message_key in ['system_message', 'user_message']:
                message = prompt[message_key]
                while '{{' in message and '}}' in message:
                    start = message.find('{{')
                    end = message.find('}}', start)
                    if end == -1:  # No closing '}}' found
                        break

                    key = message[start+2:end]
                    replacement = ''

                    if key.startswith('exec('):
                        try:
                            replacement = str(eval(key[5:-1]))
                        except Exception as e:
                            pass
                    else:
                        replacement = key

                    message = message[:start] + replacement + message[end+2:]

                prompt[message_key] = message

            prompt['user_message'] += f'{append_token}'
            prompt['prompt_name'] = prompt_name
            return prompt
        except Exception as e:
            result = {
                'system_message': f'Error loading prompt: {str(e)}',
                'user_message': f'Error loading prompt: {str(e)}',
                'templates': []
            }
            return result

    @staticmethod
    def prompt_user(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        scratchpad_token: str = 'scratchpad',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.python'
    ) -> Message:
        prompt = Helpers.load_and_populate_prompt(prompt_name, template, user_token, assistant_token, scratchpad_token, append_token, module)
        return User([TextContent(prompt['user_message'])])

    @staticmethod
    def prompts(
        prompt_name: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        scratchpad_token: str = 'scratchpad',
        append_token: str = '',
        module: str = 'llmvm.server.prompts.python'
    ) -> Tuple[System, User]:
        prompt = Helpers.load_and_populate_prompt(prompt_name, template, user_token, assistant_token, scratchpad_token, append_token, module)
        return (System(prompt['system_message']), User([TextContent(prompt['user_message'])]))
