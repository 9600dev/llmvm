import asyncio
import base64
import json
import marshal
import os
import shutil
import sys
import time
import types
from importlib import resources
from typing import Any, Callable, Optional, cast

import jsonpickle
import nest_asyncio
import rich
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, HTTPException, Request,
                     UploadFile)
from fastapi.param_functions import File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from llmvm.client.client import LLMVMClient, get_client
from llmvm.common.anthropic_executor import AnthropicExecutor
from llmvm.common.bedrock_executor import BedrockExecutor
from llmvm.common.container import Container
from llmvm.common.deepseek_executor import DeepSeekExecutor
from llmvm.common.gemini_executor import GeminiExecutor
from llmvm.common.helpers import Helpers
from llmvm.common.logging_helpers import setup_logging
from llmvm.common.objects import (Assistant, AstNode, Content,
                                  DownloadItemModel, Executor, Message,
                                  MessageModel, QueueBreakNode,
                                  SessionThreadModel, Statement,
                                  StreamingStopNode, System, TextContent,
                                  TokenPriceCalculator, User, compression_enum)
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.server.auto_global_dict import AutoGlobalDict
from llmvm.common.openai_tool_translator import OpenAIFunctionTranslator
from llmvm.server.persistent_cache import MemoryCache, PersistentCache
from llmvm.server.python_execution_controller import ExecutionController
from llmvm.server.python_runtime_host import PythonRuntimeHost
from llmvm.server.runtime import Runtime
from llmvm.server.vector_search import VectorSearch
from llmvm.server.vector_store import VectorStore

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
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/cdn'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/logs'), exist_ok=True)
    os.makedirs(os.path.expanduser('~/.local/share/llmvm/faiss'), exist_ok=True)

    config_file = resources.files('llmvm') / 'config.yaml'
    shutil.copy(str(config_file), os.path.expanduser('~/.config/llmvm/config.yaml'))


app = FastAPI()

helpers = Helpers.flatten(list(
    filter(
        lambda x: x is not None, [Helpers.get_callables(logging, helper) for helper in Container().get('helper_functions')]
    )
))


os.makedirs(Container().get('cache_directory'), exist_ok=True)
os.makedirs(Container().get('cdn_directory'), exist_ok=True)
os.makedirs(Container().get('log_directory'), exist_ok=True)
os.makedirs(Container().get('vector_store_index_directory'), exist_ok=True)

cache_session = PersistentCache(cache_directory=Container().get('cache_directory'))
runtime_dict_cache: MemoryCache[int, dict[str, Any]] = MemoryCache()
cdn_directory = Container().get('cdn_directory')


if (
    not os.environ.get('OPENAI_API_KEY')
    and not os.environ.get('ANTHROPIC_API_KEY')
    and not os.environ.get('GEMINI_API_KEY')
    and not os.environ.get('DEEPSEEK_API_KEY')
):
    rich.print('[red]Neither OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY are set.[/red]')
    rich.print('One of these API keys needs to be set in your terminal environment[/red]')
    sys.exit(1)


vector_store = VectorStore(
    store_directory=Container().get('vector_store_index_directory'),
    index_name='index',
    embedding_model=Container().get('vector_store_embedding_model'),
    chunk_size=int(Container().get('vector_store_chunk_size')),
    chunk_overlap=10
)
vector_search = VectorSearch(vector_store=vector_store)


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


def get_controller(thread_id: int = 0, controller: Optional[str] = None) -> ExecutionController:
    if not controller:
        controller = Container().get_config_variable('executor', 'LLMVM_EXECUTOR', default='')

    if not controller:
        raise EnvironmentError('No executor specified in environment or config file')

    default_model_config = Container().get_config_variable(f'default_{controller}_model', 'LLMVM_MODEL')
    override_max_input_len = Container().get_config_variable(f'override_max_input_tokens', 'LLMVM_OVERRIDE_MAX_INPUT_TOKENS', default=None)
    override_max_output_len = Container().get_config_variable(f'override_max_output_tokens', 'LLMVM_OVERRIDE_MAX_OUTPUT_TOKENS', default=None)

    executor: Executor

    if controller == 'anthropic':
        executor = AnthropicExecutor(
            api_key=os.environ.get('ANTHROPIC_API_KEY', ''),
            default_model=default_model_config,
            api_endpoint=Container().get_config_variable('anthropic_api_base', 'ANTHROPIC_API_BASE'),
            default_max_input_len=override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='anthropic', default=200000),
            # default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='anthropic', default=4096),
            default_max_output_len=64000,
        )
    elif controller == 'gemini':
        executor = GeminiExecutor(
            api_key=os.environ.get('GEMINI_API_KEY', ''),
            default_model=default_model_config,
            default_max_input_len=override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='gemini', default=2000000),
            default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='gemini', default=4096),
        )
    elif controller == 'deepseek':
        executor = DeepSeekExecutor(
            api_key=os.environ.get('DEEPSEEK_API_KEY', ''),
            default_model=default_model_config,
            default_max_input_len=override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='deepseek', default=64000),
            default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='deepseek', default=4096),
        )
    elif controller == 'bedrock':
        executor = BedrockExecutor(
            api_key='',
            default_model=default_model_config,
            default_max_input_len=override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='bedrock', default=300000),
            default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='bedrock', default=4096),
            region_name=Container().get_config_variable('bedrock_api_base', 'BEDROCK_API_BASE'),
        )
    else:
        executor = OpenAIExecutor(
            api_key=os.environ.get('OPENAI_API_KEY', ''),
            default_model=default_model_config,
            api_endpoint=Container().get_config_variable('openai_api_base', 'OPENAI_API_BASE'),
            default_max_input_len=override_max_input_len or TokenPriceCalculator().max_input_tokens(default_model_config, executor='openai', default=128000),
            default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='openai', default=4096),
        )

    return ExecutionController(
        executor=executor,
        tools=helpers,
        vector_search=vector_search,
        thread_id=thread_id
    )


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


@app.get('/cdn/{filename}')
def get_file(filename: str):
    file_path = os.path.join(cdn_directory, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(path=file_path)

@app.get('/search/{query}')
def search(query: str):
    results = vector_search.search(query, max_results=10, min_score=0.7)
    return results

@app.post('/ingest')
async def ingest(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        name = os.path.basename(str(file.filename))

        with open(f"{cdn_directory}/{name}", "wb") as buffer:
            buffer.write(file.file.read())
            background_tasks.add_task(
                vector_search.ingest_file,
                f"{cdn_directory}/{name}",
                '',
                str(file.filename),
                {}
            )
        return {"filename": file.filename, "detail": "Ingestion started."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exception: {e}")

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

            if content:
                background_tasks.add_task(
                    vector_search.ingest_text,
                    controller.statement_to_str(content),
                    controller.statement_to_str(content)[:25],
                    download_item.url,
                    {}
                )
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

@app.post('/v1/chat/set_thread')
async def set_thread(request: SessionThreadModel) -> SessionThreadModel:
    thread = request

    if not cache_session.has_key(thread.id) or thread.id <= 0:
        temp = __get_thread(0)
        thread.id = temp.id

    cache_session.set(thread.id, thread)
    return cast(SessionThreadModel, cache_session.get(thread.id))

@app.get('/v1/chat/get_threads')
async def get_threads():
    def safe_json(obj):
        try:
            json.dumps(obj)  # Fast check
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Fallback to string

    threads = []

    for id in cache_session.keys():
        raw = cache_session.get(id)

        if isinstance(raw, dict):
            raw = SessionThreadModel(**raw)

        # Sanitize locals_dict
        if isinstance(raw, SessionThreadModel):
            raw.locals_dict = {
                k: safe_json(v)
                for k, v in raw.locals_dict.items()
            }
            threads.append(raw)

    return threads
    threads = []
    for id in cache_session.keys():
        raw = cache_session.get(id)
        model = SessionThreadModel(**raw) if isinstance(raw, dict) else raw
        threads.append(model.model_dump())
    return threads


    model = SessionThreadModel(**raw) if isinstance(raw, dict) else raw
    result = [cast(SessionThreadModel, cache_session.get(id)).model_dump() for id in cache_session.keys()]
    return result

@app.get('v1/chat/clear_threads')
async def clear_threads() -> None:
    for id in cache_session.keys():
        cache_session.delete(id)

@app.get('/health')
async def health():
    logging.debug('/health')
    return {'status': 'ok'}

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
            temperature=float(request['temperature']) if request.get('temperature') else 0.0,
            output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 4096,
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
                    temperature=float(request['temperature']) if request.get('temperature') else 0.0,
                    output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 4096,
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
            temperature=float(request['temperature']) if request.get('temperature') else 0.0,
            output_token_len=int(request['max_tokens']) if request.get('max_tokens') else 4096,
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

@app.post('/v1/tools/completions', response_model=None)
async def tools_completions(request: SessionThreadModel):
    thread = request

    if not cache_session.has_key(thread.id) or thread.id == 0:
        temp = __get_thread(0)
        thread.id = temp.id

    # locals_dict needs to be set, grab it from the cache
    elif cache_session.has_key(thread.id) and not thread.locals_dict:
        thread.locals_dict = cache_session.get(thread.id).locals_dict  # type: ignore

    messages = [MessageModel.to_message(m) for m in thread.messages]  # type: ignore
    mode = thread.current_mode
    compression = compression_enum(thread.compression)
    queue = asyncio.Queue()
    cookies = thread.cookies if thread.cookies else []

    # set the defaults, or use what the SessionThread thread asks
    if thread.executor and thread.model:
        controller = get_controller(thread_id=thread.id, controller=thread.executor)
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

    # todo perform some merge logic of the messages here
    # I don't think we want to have the client always grab the full thread
    # before posting

    async def callback(token: AstNode):
        queue.put_nowait(token)

    async def stream():
        def handle_exception(task):
            if not task.cancelled() and task.exception() is not None:
                Helpers.log_exception(logging, task.exception())
                queue.put_nowait(QueueBreakNode())

        async def execute_and_signal() -> list[Message]:
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

                # add the runtime defined tools to the list of tools
                for key, value in runtime_dict.items():
                    if isinstance(value, types.FunctionType) and value.__code__.co_filename == '<ast>':
                        helpers.append(value)

                # todo: this is a hack
                result, runtime_state = await controller.aexecute_continuation(
                    messages=messages,
                    temperature=thread.temperature,
                    stream_handler=callback,
                    model=model,
                    compression=compression,
                    cookies=cookies,
                    helpers=cast(list[Callable], helpers),
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

    return StreamingResponse(stream_response(stream()), media_type='text/event-stream')  # media_type="application/json")


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

    for helper in helpers:
        rich.print(f'[{helper_color}]Loaded helper: {helper.__name__}[/{helper_color}]')  # type: ignore

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