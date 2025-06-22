import asyncio
import base64
import html
import json
import marshal
import os
from pathlib import Path
import re
import shutil
import sys
import time
import types
import datetime as dt
from importlib import resources
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, Optional, cast

from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

import jsonpickle
import nest_asyncio
from pydantic import BaseModel
import rich
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, HTTPException, Query, Request, Response,
                     UploadFile)
from fastapi.param_functions import File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llmvm.client.client import LLMVMClient, get_client
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.bedrock_executor import BedrockExecutor
from llmvm.common.container import Container
from llmvm.common.deepseek_executor import DeepSeekExecutor
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content,
                                  DownloadItemModel, Executor, LLMCall, Message,
                                  MessageModel, QueueBreakNode,
                                  SessionThreadModel, Statement,
                                  StreamingStopNode, System, TextContent, TokenCompressionMethod,
                                  TokenPriceCalculator, User)
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.common.openai_tool_translator import OpenAIFunctionTranslator
from llmvm.server.persistent_cache import MemoryCache, PersistentCache
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.python_runtime_host import PythonRuntimeHost
from llmvm.server.runtime import Runtime
from llmvm.server.mcp_monitor import MCPMonitor
# needed for pydantic late binding serialization
from anthropic.types import Completion as AnthropicCompletion, Message as AnthropicMessage, ContentBlock
from anthropic import BaseModel

nest_asyncio.apply()

logging = setup_logging()

# check to see if the config file exists, if not, create it
try:
    Container().get_config_variable('executor', 'LLMVM_EXECUTOR', default='')
except ValueError:
    rich.print('[cyan]Configuration file not found. Adding default config in ~/.config/llmvm/config.yaml[/cyan]')
    os.makedirs(os.path.expanduser('~/.config/llmvm'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/cache'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/download'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/logs'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/memory'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/memory/programs'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/traces'), exist_ok=True)

    config_file = resources.files('llmvm') / 'config.yaml'
    shutil.copy(str(config_file), os.path.expanduser('~/.config/llmvm/config.yaml'))


app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global helpers, mcp_helpers

# Initialize helpers as a list that can be updated
base_helpers = Helpers.flatten(list(
    filter(
        lambda x: x is not None, [Helpers.get_callables(logging, helper) for helper in Container().get('helper_functions')]
    )
))


helpers = list(base_helpers)  # Make it mutable
mcp_helpers: Dict[str, Callable] = {}  # Track MCP-specific helpers


# MCP helpers will be discovered dynamically on each request

# Initialize MCP monitor with configurable patterns
mcp_server_patterns = Container().get_config_variable('mcp_server_patterns', default=None)
mcp_exclude_patterns = Container().get_config_variable('mcp_exclude_patterns', default=None)

# Create a global MCP monitor instance (no callback needed for sync approach)
mcp_monitor = MCPMonitor(
    server_patterns=mcp_server_patterns,
    exclude_patterns=mcp_exclude_patterns
)


os.makedirs(Container().get('cache_directory'), exist_ok=True)
os.makedirs(Container().get('log_directory'), exist_ok=True)
os.makedirs(Container().get('memory_directory'), exist_ok=True)

cache_session = PersistentCache(cache_directory=Container().get('cache_directory'))
runtime_dict_cache: MemoryCache[int, dict[str, Any]] = MemoryCache()


if (
    not os.environ.get('OPENAI_API_KEY')
    and not os.environ.get('ANTHROPIC_API_KEY')
    and not os.environ.get('GEMINI_API_KEY')
    and not os.environ.get('DEEPSEEK_API_KEY')
    and not os.environ.get('BEDROCK_API_KEY')
):
    rich.print('Neither OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, BEDROCK_API_KEY, or DEEPSEEK_API_KEY are set.')
    rich.print('One of these API keys needs to be set in your terminal environment[/red]')
    sys.exit(1)


def __get_unserializable_locals(locals_dict: dict[str, Any]) -> dict[str, Any]:
    unserializable_locals = {}
    for key, value in locals_dict.items():
        if (
            not isinstance(value, types.FunctionType) and
            not isinstance(value, types.MethodType) and
            not isinstance(value, types.ModuleType) and
            not isinstance(value, types.BuiltinFunctionType) and
            not isinstance(value, types.BuiltinMethodType) and
            isinstance(value, object)
            and not isinstance(value, type)
            and type(value).__module__ != 'builtins'
            and type(value).__module__ != PythonRuntimeHost.__module__
            and type(value).__module__ != Runtime.__module__
        ):
            try:
                json.dumps(value)
            except:
                unserializable_locals[key] = value
    return unserializable_locals


class PrettyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(", ", ": "),
        ).encode("utf-8") + b"\n"

def get_executor(
    executor: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
 ) -> Executor:
    if not executor:
        executor = Container().get_config_variable('executor', 'LLMVM_EXECUTOR', default='')

    if not executor:
        raise EnvironmentError('No executor specified in environment or config file')

    default_model_config = Container().get_config_variable(f'default_{executor}_model', 'LLMVM_MODEL')
    override_max_input_len = Container().get_config_variable(f'override_max_input_tokens', 'LLMVM_OVERRIDE_MAX_INPUT_TOKENS', default=None)
    override_max_output_len = Container().get_config_variable(f'override_max_output_tokens', 'LLMVM_OVERRIDE_MAX_OUTPUT_TOKENS', default=None)

    executor_instance: Executor

    if executor == 'anthropic':
        executor_instance = AnthropicExecutor(
            api_key=os.environ.get('ANTHROPIC_API_KEY', ''),
            default_model=default_model_config,
            api_endpoint=Container().get_config_variable('anthropic_api_base', 'ANTHROPIC_API_BASE'),
            default_max_input_len=max_input_tokens or override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='anthropic', default=200000),
            default_max_output_len=max_output_tokens or override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='anthropic', default=8192),
        )
    elif executor == 'gemini':
        executor_instance = GeminiExecutor(
            api_key=os.environ.get('GEMINI_API_KEY', ''),
            default_model=default_model_config,
            default_max_input_len=max_input_tokens or override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='gemini', default=2000000),
            default_max_output_len=max_output_tokens or override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='gemini', default=8192),
        )
    elif executor == 'deepseek':
        executor_instance = DeepSeekExecutor(
            api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
            default_model=default_model_config,
            default_max_input_len=max_input_tokens or override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='deepseek', default=64000),
            default_max_output_len=max_output_tokens or override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='deepseek', default=4096),
        )
    elif executor == 'bedrock':
        executor_instance = BedrockExecutor(
            api_key='',
            default_model=default_model_config,
            default_max_input_len=max_input_tokens or override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='bedrock', default=300000),
            default_max_output_len=max_output_tokens or override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='bedrock', default=4096),
            region_name=Container().get_config_variable('bedrock_api_base', 'BEDROCK_API_BASE'),
        )
    else:
        executor_instance = OpenAIExecutor(
            api_key=os.environ.get('OPENAI_API_KEY', ''),
            default_model=default_model_config,
            api_endpoint=Container().get_config_variable('openai_api_base', 'OPENAI_API_BASE'),
            default_max_input_len=max_input_tokens or override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='openai', default=128000),
            default_max_output_len=max_output_tokens or override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='openai', default=4096),
        )
    return executor_instance


async def __get_helpers_async() -> list[Callable]:
    """Get all helpers including dynamically discovered MCP tools (async version)"""
    global helpers, mcp_helpers, mcp_monitor

    # Get current MCP tools
    try:
        current_mcp_tools = await mcp_monitor.get_current_tools_async()

        # Update our tracking of MCP helpers
        mcp_helpers = current_mcp_tools

        # Combine base helpers with current MCP tools
        all_helpers = list(helpers)  # Start with base helpers

        # Add MCP tools
        for name, tool in current_mcp_tools.items():
            # Check if already in list by name
            if not any(h.__name__ == name for h in all_helpers):
                all_helpers.append(tool)

        logging.debug(f"Returning {len(all_helpers)} helpers ({len(current_mcp_tools)} MCP tools)")
        return all_helpers

    except Exception as e:
        logging.error(f"Error getting MCP tools: {e}")
        # Return just base helpers if MCP detection fails
        return list(helpers)

def __get_helpers() -> list[Callable]:
    """Get all helpers - sync wrapper that returns base helpers only"""
    # This is used in places where we can't easily make things async
    # It returns only the base helpers without MCP tools
    global helpers
    return list(helpers)


def get_controller(
        thread_id: int = 0,
        executor: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None
    ) -> ExecutionController:
    executor_instance = get_executor(executor, max_input_tokens, max_output_tokens)
    return ExecutionController(
        executor=executor_instance,
        helpers=__get_helpers(),  # Pass the list, not the function
        thread_id=thread_id
    )


def __get_threads() -> list[SessionThreadModel]:
    threads = []

    for id in cache_session.keys():
        raw = cache_session.get(id)

        if isinstance(raw, dict):
            raw = SessionThreadModel(**raw)

        threads.append(cast(SessionThreadModel, raw))
    return threads


def __get_thread(id: int) -> SessionThreadModel:
    if not cache_session.has_key(id) or id <= 0:
        id = cache_session.gen_key()
        thread = SessionThreadModel(id=id)
        cache_session.set(thread.id, thread)
    return cast(SessionThreadModel, cache_session.get(id))


async def stream_response(response):
    # this was changed mar 6, to support idle timeout so there's streaming issues, this will be the root cause
    content = ''
    response_iterator = response.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(response_iterator.__anext__(), timeout=120)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Stream timed out")
        except StopAsyncIteration:
            yield "data: [DONE]"
            break
        content += str(chunk)
        yield f"data: {jsonpickle.encode(chunk)}\n\n"


@app.get("/files/{id}/", response_class=HTMLResponse)
async def list_memory_files(id: int, request: Request):
    folder = (Path(Container().get('memory_directory')) / str(id)).resolve()

    if folder.is_dir() is False or Path(Container().get('memory_directory')) not in folder.parents:
        raise HTTPException(status_code=404)

    rows = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file():
            size = entry.stat().st_size
            mtime = dt.datetime.fromtimestamp(entry.stat().st_mtime)
            rows.append(
                f"<tr>"
                f"<td><a href='{html.escape(request.url.path + entry.name)}'>{html.escape(entry.name)}</a></td>"
                f"<td>{size:,} bytes</td>"
                f"<td>{mtime:%Y-%m-%d %H:%M:%S}</td>"
                f"</tr>"
            )

    body = (
        "<html><head><title>Memory files</title>"
        "<style>table{border-collapse:collapse}td,th{padding:4px 8px;border:1px solid #ccc}</style></head><body>"
        f"<h1>Files for ID {id}</h1>"
        "<table><tr><th>Name</th><th>Size</th><th>Last modified</th></tr>"
        + "\n".join(rows)
        + "</table></body></html>"
    )
    return HTMLResponse(body)


# must be after the list_memory_files endpoint
app.mount(
    "/files",
    StaticFiles(directory=Container().get('memory_directory')),
    name="memory-files",
)


@app.post('/download')
async def download(
    download_item: DownloadItemModel,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    thread = __get_thread(download_item.id)

    if not cache_session.has_key(thread.id) or thread.id <= 0:
        temp = __get_thread(0)
        thread.id = temp.id

    queue = asyncio.Queue()
    controller = get_controller()

    async def callback(token: AstNode):
        queue.put_nowait(token)

    async def stream():
        async def execute_and_signal():
            stream_handler = callback
            from llmvm.server.base_library.content_downloader import \
                WebAndContentDriver

            # todo thread cookies through here
            downloader = WebAndContentDriver()
            content: Content = downloader.download(download={
                'url': download_item.url,
                'goal': '',
                'search_term': ''
            })
            queue.put_nowait(QueueBreakNode())

            # Removed vector_search.ingest_text functionality
            return content

        task = asyncio.create_task(execute_and_signal())

        while True:
            data = await queue.get()
            if isinstance(data, QueueBreakNode):
                break
            yield data

        await task

        content = task.result()
        thread.messages.append(MessageModel.from_message(User(content)))
        cache_session.set(thread.id, thread)
        yield thread.model_dump()

    return StreamingResponse(stream_response(stream()), media_type='text/event-stream')  # media_type="application/json")

@app.get('/v1/chat/get_thread')
async def get_thread(id: int) -> SessionThreadModel:
    logging.debug(f'/v1/chat/get_thread?id={id}')
    thread = __get_thread(id)
    return thread

@app.post('/v1/chat/set_thread_title')
async def set_thread_title(request: Request):
    request = await request.json()
    id = request['id']
    title = request['title']

    thread = __get_thread(id)
    thread.title = title
    cache_session.set(thread.id, thread)
    return thread

@app.post('/v1/chat/set_thread')
async def set_thread(request: SessionThreadModel) -> SessionThreadModel:
    thread = request

    if not cache_session.has_key(thread.id) or thread.id <= 0:
        temp = __get_thread(0)
        thread.id = temp.id
    else:
        # set the locals_dict from the cached thread
        l_cached_thread = cache_session.get(thread.id)
        thread.locals_dict = l_cached_thread.locals_dict

    cache_session.set(thread.id, thread)
    return cast(SessionThreadModel, cache_session.get(thread.id))

@app.get('/v1/chat/get_threads', response_model=list[SessionThreadModel])
async def get_threads() -> list[SessionThreadModel]:
    threads = []

    for id in cache_session.keys():
        raw = cache_session.get(id)

        if isinstance(raw, dict):
            raw = SessionThreadModel(**raw)

        if isinstance(raw, SessionThreadModel):
            thread_dict = raw.model_dump(exclude={'locals_dict'})

            # Add properly serialized locals_dict if client needs it
            if raw.locals_dict:
                thread_dict['locals_dict'] = Helpers.serialize_locals_dict(raw.locals_dict)
            else:
                thread_dict['locals_dict'] = {}

            threads.append(thread_dict)
    return threads

@app.get('/v1/chat/get_programs', response_model=list[SessionThreadModel])
async def get_programs() -> list[SessionThreadModel]:
    threads = __get_threads()
    programs = []
    for thread in threads:
        if thread.current_mode == 'program':
            programs.append(thread)
    return programs

@app.get("/v1/chat/get_program", response_model=SessionThreadModel)
async def get_program(
    id: Optional[int] = None,
    program_name: Optional[str] = None,
) -> SessionThreadModel:
    if id:
        thread = __get_thread(id)
        if thread.current_mode != 'program':
            raise HTTPException(status_code=400, detail='Thread not found or is not in program mode')
        return thread

    if program_name:
        for thread in __get_threads():
            if thread.title.lower() == program_name.lower():
                return thread
    raise HTTPException(status_code=400, detail='Program not found')

@app.get('v1/chat/clear_threads')
async def clear_threads() -> None:
    for id in cache_session.keys():
        cache_session.delete(id)

@app.get('/health')
async def health():
    logging.debug('/health')
    return {'status': 'ok'}

@app.get('/helpers')
async def list_helpers():
    """Debug endpoint to list current helpers"""
    helpers = await __get_helpers_async()
    return {
        'total_helpers': len(helpers),
        'helper_names': [h.__name__ for h in helpers],
        'mcp_helpers': list(mcp_helpers.keys()) if mcp_helpers else []
    }

async def _tools_completions_generator(thread: SessionThreadModel) -> AsyncIterator[Any]:
    # Get helpers with MCP tools asynchronously
    helpers = await __get_helpers_async()

    """
    Core logic of tools_completions, returning an async generator instead of StreamingResponse.
    This allows other endpoints to use this logic and intercept the stream.
    """
    if not cache_session.has_key(thread.id) or thread.id == 0:
        temp = __get_thread(0)
        thread.id = temp.id

    # locals_dict needs to be set, grab it from the cache
    elif cache_session.has_key(thread.id) and not thread.locals_dict:
        l_cached_thread = cache_session.get(thread.id)
        thread.locals_dict = l_cached_thread.locals_dict

    messages = [MessageModel.to_message(m) for m in thread.messages]  # type: ignore
    mode = thread.current_mode
    compression = TokenCompressionMethod.from_str(thread.compression)
    queue = asyncio.Queue()
    cookies = thread.cookies if thread.cookies else []

    # set the defaults, or use what the SessionThread thread asks
    if thread.executor and thread.model:
        controller = get_controller(thread_id=thread.id, executor=thread.executor)
        model = thread.model if thread.model else controller.get_executor().default_model
    # either the executor or the model is not set, so use the defaults
    # and update the thread
    else:
        logging.debug('Either the executor or the model is not set. Updating thread.')
        controller = get_controller(thread_id=thread.id)
        model = controller.get_executor().default_model
        thread.executor = controller.get_executor().name()
        thread.model = model

    logging.debug(f'/v1/tools/completions?id={thread.id}&mode={mode}&thinking={thread.thinking}&model={model}&executor={thread.executor}&compression={thread.compression}&cookies={thread.cookies}&temperature={thread.temperature}')  # NOQA: E501

    if len(messages) == 0:
        raise HTTPException(status_code=400, detail='No messages provided')

    async def callback(token: AstNode):
        queue.put_nowait(token)

    async def stream():
        def handle_exception(task):
            if not task.cancelled() and task.exception() is not None:
                Helpers.log_exception(logging, task.exception())
                queue.put_nowait(QueueBreakNode())

        async def execute_and_signal() -> list[Message]:
            # don't touch, required for walking up the stack
            stream_handler = callback

            if thread.current_mode == 'direct':
                result = await controller.aexecute(
                    messages=messages,
                    temperature=thread.temperature,
                    model=model,
                    compression=compression,
                    thinking=thread.thinking,
                    stream_handler=callback,
                )
                queue.put_nowait(QueueBreakNode())
                return result
            else:
                # deserialize the locals_dict, then merge it with the in-memory locals_dict we have in MemoryCache
                runtime_dict = Helpers.deserialize_locals_dict(thread.locals_dict)
                runtime_dict.update(runtime_dict_cache.get(thread.id) or {})

                # Create a local copy of helpers for this request
                # Note: We use the already-fetched helpers from above which includes MCP tools
                local_helpers = list(helpers)

                # add the runtime defined tools to the list of tools
                for key, value in runtime_dict.items():
                    if isinstance(value, types.FunctionType) and value.__code__.co_filename == '<ast>':
                        local_helpers.append(value)

                # todo: this is a hack
                result, runtime_state = await controller.aexecute_continuation(
                    messages=messages,
                    temperature=thread.temperature,
                    stream_handler=callback,
                    model=model,
                    max_output_tokens=thread.output_token_len,
                    compression=compression,
                    cookies=cookies,
                    helpers=cast(list[Callable], local_helpers),
                    runtime_state=AutoGlobalDict(runtime_dict),
                    thinking=thread.thinking,
                )
                queue.put_nowait(QueueBreakNode())

                # update the in-memory locals_dict with unserializable locals
                runtime_dict_cache.set(thread.id, __get_unserializable_locals(runtime_state))
                thread.locals_dict = Helpers.serialize_locals_dict(runtime_state)
                return result

        task = asyncio.create_task(execute_and_signal())
        task.add_done_callback(handle_exception)

        while True:
            data = await queue.get()

            # this controls the end of the stream
            if isinstance(data, QueueBreakNode):
                # this tells the client to deal with carriage returns etc for pretty printing
                yield StreamingStopNode('\n\n' if thread.executor == 'openai' else '\n')
                break

            yield data

        try:
            await task
        except Exception as e:
            pass

        # error handling
        if task.exception() is not None:
            thread.messages.append(MessageModel.from_message(Assistant(TextContent(f'Error: {str(task.exception())}'))))
            yield thread.model_dump()
            return

        messages_result: list[Message] = task.result()
        thread.messages = [MessageModel.from_message(m) for m in messages_result]
        cache_session.set(thread.id, thread)
        yield thread.model_dump()

    async for item in stream():
        yield item

@app.post('/v1/tools/completions', response_model=None)
async def tools_completions(request: SessionThreadModel):
    """
    Main endpoint for tools completions. Uses the extracted generator function.
    """
    return StreamingResponse(
        stream_response(_tools_completions_generator(request)),
        media_type='text/event-stream'
    )

@app.post('/v1/tools/compile')
async def compile(request: SessionThreadModel) -> StreamingResponse:
    thread = request
    compile_instructions = request.compile_prompt

    if not cache_session.has_key(thread.id) or thread.id == 0:
        raise HTTPException(status_code=400, detail='Thread id must be in cache and greater than 0.')

    thread_model_base = cache_session.get(thread.id)
    thread = thread_model_base.copy()

    thread.id = -1
    thread.locals_dict = thread_model_base.locals_dict  # type: ignore
    controller = get_controller(thread_id=thread.id, executor=thread.executor)

    compile_prompt=Helpers.prompt_user(
        prompt_name='thread_to_program.prompt',
        template={
            'compile_instructions': compile_instructions,
            'program_title': thread.title
        },
        user_token=controller.get_executor().user_token(),
        assistant_token=controller.get_executor().assistant_token(),
        scratchpad_token=controller.get_executor().scratchpad_token(),
        append_token=controller.get_executor().append_token(),
    )
    thread.messages.append(MessageModel.from_message(compile_prompt))

    # Create a variable to capture the final thread
    captured_thread = None

    async def intercepting_stream():
        """Stream that intercepts the final SessionThreadModel while passing everything through"""
        nonlocal captured_thread

        async for item in _tools_completions_generator(thread):
            # Check if this is the final SessionThreadModel dump
            if isinstance(item, dict) and 'id' in item and 'messages' in item:
                try:
                    captured_thread = SessionThreadModel(**item)
                except Exception:
                    pass  # Not a valid SessionThreadModel

            yield item

        if captured_thread:
            # add the compiled code to the "program" thread so the user can interact with it
            logging.debug(f'compile() compiled thread: {captured_thread.id}')
            captured_thread.current_mode = 'program'
            await set_thread(captured_thread)

            from llmvm.server.python_runtime_host import PythonRuntimeHost

            code_block: str = Helpers.extract_program_code_block(MessageModel.to_message(captured_thread.messages[-1]).get_str())

            # deal with the program title
            _TITLE_RE = re.compile(
                r'(?is)<program_title\s*>(.*?)</program_title\s*>'
            )
            matches = _TITLE_RE.findall(MessageModel.to_message(captured_thread.messages[-1]).get_str())
            title = matches[-1].strip() if matches else None

            if title and not captured_thread.title:
                captured_thread.title = title
            elif not captured_thread.title:
                captured_thread.title = str(captured_thread.id)

            # execute the code block in the thread
            python_runtime_host = PythonRuntimeHost(
                controller=get_controller(),
                answer_error_correcting=False,
                thread_id=thread.id
            )
            runtime_state = AutoGlobalDict(captured_thread.locals_dict)
            try:
                python_runtime_host.compile_and_execute_code_block(
                    python_code=code_block,
                    messages_list=[MessageModel.to_message(m) for m in captured_thread.messages],
                    helpers=__get_helpers(),
                    runtime_state=runtime_state,
                )
            except Exception as ex:
                logging.error(f'PythonRuntime.compile() threw an exception while executing:\n{code_block}\n')

            # update the caches
            runtime_dict_cache.set(captured_thread.id, runtime_state)
            await set_thread(captured_thread)
            yield captured_thread.model_dump()

    return StreamingResponse(stream_response(intercepting_stream()), media_type='text/event-stream')

@app.get('/python')
async def execute_python_in_thread(thread_id: int, python_str: str):
    logging.debug('/python')

    if thread_id <= 0:
        raise HTTPException(status_code=400, detail='Thread id must be greater than 0')

    thread = __get_thread(thread_id)

    # locals_dict needs to be set, grab it from the cache
    if cache_session.has_key(thread.id) and not thread.locals_dict:
        thread.locals_dict = cache_session.get(thread.id).locals_dict  # type: ignore

    messages = [MessageModel.to_message(m) for m in thread.messages]  # type: ignore

    from llmvm.server.python_runtime_host import PythonRuntimeHost
    python_runtime_host = PythonRuntimeHost(
        controller=get_controller(),
        answer_error_correcting=False,
        thread_id=thread.id
    )
    runtime_state = AutoGlobalDict(thread.locals_dict)
    try:
        list_answers = python_runtime_host.compile_and_execute_code_block(
            python_code=python_str,
            messages_list=messages,
            helpers=__get_helpers(),
            runtime_state=runtime_state,
        )
        state_result = python_runtime_host.get_last_statement(python_str, runtime_state=runtime_state)

    except Exception as ex:
        logging.error(f'PythonRuntime.compile_and_execute() threw an exception while executing:\n{python_str}\n')
        return Response(jsonpickle.encode({'var_name': '', 'var_value': '', 'results': [], 'error': str(ex)}, unpicklable=False), media_type='application/json')

    if state_result or list_answers:
        var_name, var_value = state_result or ('', '')
        runtime_dict_cache.set(thread.id, __get_unserializable_locals(runtime_state))
        thread.locals_dict = Helpers.serialize_locals_dict(runtime_state)
        cache_session.set(thread.id, thread)

        return_result = {
            'var_name': var_name,
            'var_value': var_value,
            'results': list_answers,
            'error': ''
        }
        return Response(jsonpickle.encode(return_result, unpicklable=False), media_type='application/json')
    else:
        return Response(jsonpickle.encode({'var_name': '', 'var_value': '', 'results': [], 'error': ''}, unpicklable=False), media_type='application/json')

@app.post('/v1/chat/cookies', response_model=None)
async def set_cookies(requests: Request):
    request_body = await requests.json()
    id = request_body['id']
    cookies = request_body['cookies']

    logging.debug(f'/v1/chat/cookies?id={id}')
    thread = __get_thread(id)
    thread.cookies = cookies
    cache_session.set(thread.id, thread)
    return thread

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request = await request.json()

    if request.get('messages') and request.get('messages')[-1].get('content'):  # type: ignore
        logging.debug(f'/v1/chat/completions: request {str(request["messages"][-1]["role"])}: {str(request["messages"][-1]["content"])[0:50]}')

    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def streamify_tool_call(full_response_json):
        """
        Given a JSON dict that represents the final completion with a tool call,
        yield multiple SSE-like chunks that represent partial tokens of the function call.
        """
        base_id = full_response_json["id"]
        base_created = full_response_json["created"]
        base_model = full_response_json["model"]

        # The typical structure:
        for choice in full_response_json["choices"]:
            tool_call = choice["message"]["tool_calls"][0]

            function_name = tool_call["function"]["name"]
            arguments_full = str(tool_call["function"]["arguments"]).replace('\'', '"')  # e.g. {"ticker": "NVDA"}

            # 1) First chunk: declare function name with empty args
            first_chunk = {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": base_created,
                "model": base_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": tool_call["index"],
                                    "id": tool_call["id"],
                                    "type": tool_call["type"],
                                    "function": {
                                        "name": function_name,
                                        "arguments": ""
                                    }
                                }
                            ]
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            # 2) Then chunk up the arguments string in small pieces
            #    (In a real scenario, the chunks might correspond to tokens.)
            chunk_size = 20  # Just an example
            for start_idx in range(0, len(arguments_full), chunk_size):
                partial_args = arguments_full[start_idx : start_idx + chunk_size]

                chunk = {
                    "id": base_id,
                    "object": "chat.completion.chunk",
                    "created": base_created,
                    "model": base_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": tool_call["index"],
                                        # Notice in streaming, we just keep adding partial
                                        # arguments in each chunk.
                                        "function": {
                                            "arguments": partial_args
                                        }
                                    }
                                ]
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # 3) Final chunk that includes the finish_reason
            final_chunk = {
                "id": base_id,
                "object": "chat.completion.chunk",
                "created": base_created,
                "model": base_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        # At the end, the streaming API typically sets finish_reason to
                        # "function_call" or "stop" or something else.
                        "finish_reason": "tool_calls"
                    }
                ]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        yield f"data: [DONE]\n\n"

    async def stream_text_response(text_resp: str, model: str):
        tokens = text_resp.split(" ")

        for i, token in enumerate(tokens):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}-{i}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": token + " "}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Indicate end of stream
        yield "data: [DONE]\n\n"

    async def stream_handler(ast_node: AstNode):
        token_text = str(ast_node)
        await queue.put(token_text)

    async def event_stream():
        chunk_idx = 0

        while True:
            token = await queue.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            chunk = {
                "id": f"chatcmpl-{int(time.time())}-{chunk_idx}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request['model'],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            chunk_idx += 1
            yield f"data: {json.dumps(chunk)}\n\n"

    # get default controller
    controller = get_controller()
    llmvm_client: LLMVMClient = get_client(executor_name=controller.get_executor().name(),
                                           model_name=controller.get_executor().default_model)

    def role_to_message(m: dict) -> Message:
        if not isinstance(m, dict):
            raise ValueError('Message must be a dict')

        if not m.get('role'):
            raise ValueError('Message must have a role')

        content_str = ''

        if m.get('content') and isinstance(m.get('content'), str):
            content_str = str(m['content'])
        elif m.get('content') and isinstance(m['content'], list):
            for item in m['content']:
                if isinstance(item, dict) and item.get('type') == 'text':
                    content_str += str(item['text'])
                elif isinstance(item, dict) and item.get('type') == 'image':
                    content_str += f"[ImageContent({item['image_url']})]"
        elif not m.get('content') and m.get('tool_calls'):
            tool_calls: list[dict[str, Any]] = m.get('tool_calls')  # type: ignore
            content_str = '\n'.join([OpenAIFunctionTranslator.generate_python_function_signature_from_oai_description(cast(dict[str, Any], tool)) for tool in tool_calls])

        if m.get('role') == 'user':
            return User(TextContent(content_str))
        elif m.get('role') == 'assistant':
            return Assistant(TextContent(content_str))
        elif m.get('role') == 'tool':
            return User(TextContent(content_str))
        elif m.get('role') == 'system':
            return System(content_str)
        else:
            return User(TextContent(content_str))

    llmvm_messages: list[Message] = [role_to_message(m) for m in request['messages']]

    # tool calling
    if request.get('tools') and not request.get('messages')[-1].get('role') == 'tool':  # type: ignore
        result = await llmvm_client.openai_tool_call(
            messages=llmvm_messages,
            tools=request['tools'],
            executor=controller.get_executor(),
            model=controller.get_executor().default_model,
            temperature=float(request['temperature']) if request.get('temperature') else 1.0,
            output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 8192,
        )

        if request.get('stream') and request['stream'] == True and result.get('choices')[0].get('message').get('tool_calls'):  # type: ignore
            return StreamingResponse(streamify_tool_call(result))
        elif request.get('stream') and request['stream'] == True:
            return StreamingResponse(stream_text_response(result['choices'][0]['message']['content'], request['model']))
        else:
            return PrettyJSONResponse(result)

    # streaming
    if request.get('stream'):
        async def llm_call():
            try:
                await llmvm_client.call_direct(
                    messages=llmvm_messages,
                    executor=controller.get_executor(),
                    model=controller.get_executor().default_model,
                    temperature=float(request['temperature']) if request.get('temperature') else 1.0,
                    output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 8192,
                    stream_handler=stream_handler,
                )
            finally:
                await queue.put(None)

        asyncio.create_task(llm_call())
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream"
        )

    # not streaming
    else:
        assistant = await llmvm_client.call_direct(
            messages=llmvm_messages,
            executor=controller.get_executor(),
            model=controller.get_executor().default_model,
            temperature=float(request['temperature']) if request.get('temperature') else 1.0,
            output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 8192,
        )

        assistant_content = assistant.get_str().strip()

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request['model'],
            "choices": [
                {
                    "index": 0,
                    "message":
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "refusal": None,
                            "annotations": []
                        },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(' '.join(str(msg['content']) for msg in request['messages']).split()),
                "completion_tokens": len(str(assistant_content).split()),
                "total_tokens": len(' '.join(str(msg['content']) for msg in request['messages']).split()) + len(str(assistant_content).split())
            }
        }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.exception(exc)
    return JSONResponse(content={"error": f"Unhandled exception: {exc}"})


if __name__ == '__main__':
    default_controller = Container().get_config_variable('executor', 'LLMVM_EXECUTOR', default='')
    default_model_str = f'{default_controller}_model'
    default_model = Container().get_config_variable(default_model_str, 'LLMVM_MODEL', default='')
    role_color = Container().get_config_variable('client_info_bold_color', default='cyan')
    helper_color = Container().get_config_variable('client_info_color', default='bold green')
    port=int(Container().get_config_variable('server_port', 'LLMVM_SERVER_PORT'))

    rich.print(f'[{role_color}]Default executor is: {default_controller}[/{role_color}]')
    rich.print(f'[{role_color}]Default model is: {default_model}[/{role_color}]')
    rich.print(f'[{role_color}]Default port is: {port} $LLMVM_SERVER_PORT or config to change.[/{role_color}]')
    rich.print()
    rich.print(f'[{role_color}]Make sure to `playwright install`.[/{role_color}]')
    rich.print(f'[{role_color}]If you have pip upgraded, delete ~/.config/llmvm/config.yaml to get latest config and helpers.[/{role_color}]')

    for helper in base_helpers:
        rich.print(f'[{helper_color}]Loaded helper: {helper.__name__}[/{helper_color}]')  # type: ignore

    rich.print(f'\n[{role_color}]MCP support enabled - servers will be detected on request[/{role_color}]')

    # you can run this using uvicorn to get better asynchronousy, but don't count on it yet.
    # uvicorn server:app --loop asyncio --workers 4 --log-level debug --host 0.0.0.0 --port 8011
    config = uvicorn.Config(
        app='llmvm.server.server:app',
        host=Container().get_config_variable('server_host', 'LLMVM_SERVER_HOST'),
        port=int(Container().get_config_variable('server_port', 'LLMVM_SERVER_PORT')),
        reload=False,
        loop='asyncio',
        log_level='debug',
    )

    server = uvicorn.Server(config)
    server.run()