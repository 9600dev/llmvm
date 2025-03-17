import asyncio
import base64
import json
import marshal
import os
import shutil
import sys
import types
from importlib import resources
from typing import Any, Callable, Optional, cast

import async_timeout
import jsonpickle
import nest_asyncio
import rich
import uvicorn
from fastapi import (BackgroundTasks, FastAPI, HTTPException, Request,
                     UploadFile)
from fastapi.param_functions import File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from openai import AsyncOpenAI, OpenAI

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
                                  StreamingStopNode, TextContent,
                                  TokenPriceCalculator, User, compression_enum)
from llmvm.common.openai_executor import OpenAIExecutor
from llmvm.server.auto_global_dict import AutoGlobalDict
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


def __serialize_locals_dict(locals_dict: dict[str, Any]) -> dict[str, Any]:
    temp_dict = {}
    for key, value in locals_dict.items():
        if key == 'hello_world':
            print('asdfasdfasdf')
        if isinstance(key, str) and key.startswith('__'):
            continue
        elif isinstance(value, types.FunctionType) and value.__module__ == 'builtins':
            continue
        elif key == 'AutoGlobalDict':
            continue
        elif isinstance(value, types.FunctionType) and value.__code__.co_filename == '<ast>':
            is_static, cls = Helpers.is_static_method(value)

            # Serialize the function's code object
            code_bytes = marshal.dumps(value.__code__)
            temp_dict[key] = {
                'type': 'function',
                'name': value.__name__,
                'code': base64.b64encode(code_bytes).decode('ascii'),
                'defaults': value.__defaults__,
                'closure': value.__closure__,
                'doc': value.__doc__,
                'annotations': value.__annotations__,
                'is_method': not is_static or cls is not None,
                'class_name': cls.__name__ if cls else None,
                'is_static_method': is_static and cls is not None,
                'qualname': value.__qualname__,
                'module': value.__module__
            }
        elif isinstance(value, dict):
            temp_dict[key] = __serialize_locals_dict(value)
        elif isinstance(value, list):
            temp_dict[key] = [__serialize_item(v) for v in value]
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

def __serialize_item(item):
    if isinstance(item, types.FunctionType) and item.__code__.co_filename == '<ast>':
        code_bytes = marshal.dumps(item.__code__)
        return {
            'type': 'function',
            'name': item.__name__,
            'code': base64.b64encode(code_bytes).decode('ascii'),
            'defaults': item.__defaults__,
            'closure': item.__closure__
        }
    elif isinstance(item, (str, int, float, bool)):
        return item
    elif isinstance(item, (Content, AstNode, Message, Statement)):
        return item
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

def __deserialize_locals_dict(serialized_dict: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in serialized_dict.items():
        if isinstance(value, dict) and value.get('type') == 'function':
            # Deserialize the function's code object
            code_bytes = base64.b64decode(value['code'])
            code = marshal.loads(code_bytes)
            # Recreate the function
            func = types.FunctionType(code, result, value['name'], value['defaults'], value['closure'])

            # add the docstrings etc.
            if 'doc' in value:
                    func.__doc__ = value['doc']

            if 'annotations' in value:
                    func.__annotations__ = value['annotations']

            if 'qualname' in value:
                func.__qualname__ = value['qualname']

            if 'module' in value:
                func.__module__ = value['module']

            if value.get('is_method') and value.get('class_name') and value['class_name'] in result:
                cls = result[value['class_name']]
                if value.get('is_static_method'):
                    func = staticmethod(func)

            result[key] = func
        elif isinstance(value, dict):
            result[key] = __deserialize_locals_dict(value)
        elif isinstance(value, list):
            result[key] = [__deserialize_item(v) for v in value]
        else:
            result[key] = value
    return result

def __deserialize_item(item):
    if isinstance(item, dict) and item.get('type') == 'function':
        code_bytes = base64.b64decode(item['code'])
        code = marshal.loads(code_bytes)
        return types.FunctionType(code, globals(), item['name'], item['defaults'], item['closure'])
    return item


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
            default_max_output_len=override_max_output_len or TokenPriceCalculator().max_output_tokens(default_model_config, executor='anthropic', default=4096),
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
    # this was changed mar 6, to support idle timoue so there's streaming issues, this will be the root cause
    content = ''
    response_iterator = response.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(response_iterator.__anext__(), timeout=60)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Stream timed out")
        except StopAsyncIteration:
            yield "data: [DONE]"
            break
        content += str(chunk)
        yield f"data: {jsonpickle.encode(chunk)}\n\n"


@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
    api_key = os.environ.get('OPENAI_API_KEY', default=''),
    aclient = AsyncOpenAI(api_key=cast(str, api_key))
    client = OpenAI(api_key=cast(str, api_key))

    try:
        # Construct the prompt from the messages
        data = await request.json()
        messages = data.get('messages', [])
        prompt = ""
        for msg in messages:
            prompt += f'{msg["role"]}: {msg["content"]}\n'

        # Get the JSON body of the request
        if 'stream' in data and data['stream']:
            response = await aclient.chat.completions.create(
                model=data['model'],
                temperature=0.0,
                max_tokens=150,
                messages=messages,
                stream=True
            )
            return StreamingResponse(stream_response(response), media_type='text/event-stream')  # media_type="application/json")
        else:
            response = client.chat.completions.create(
                model=data['model'],
                temperature=0.0,
                max_tokens=150,
                messages=messages,
                stream=False
            )
            return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
async def get_threads() -> list[SessionThreadModel]:
    result = [cast(SessionThreadModel, cache_session.get(id)) for id in cache_session.keys()]
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
                runtime_dict = __deserialize_locals_dict(thread.locals_dict)
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
                thread.locals_dict = __serialize_locals_dict(runtime_state)
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