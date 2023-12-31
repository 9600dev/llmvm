import asyncio
import base64
import datetime as dt
import importlib
import inspect
import io
import os
import re
import typing
from enum import Enum, IntEnum
from itertools import cycle, islice
from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import nest_asyncio
import psutil
from docstring_parser import parse
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from objects import Content, Message, StreamNode, User


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
            asyncio.run(instance.stream_handler(obj))
            return
        frame = frame.f_back


class Helpers():
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
    def is_running(process_name):
        for proc in psutil.process_iter():
            try:
                if process_name.lower() in proc.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
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
                # If no match, check the next parent
                return Helpers.__find_terminal_emulator(process.parent())
            else:
                # No more parents, terminal emulator not found
                return 'Unknown'
        except Exception as e:
            return str(e)

    @staticmethod
    def is_emulator(emulator: str):
        current_process = psutil.Process(os.getpid())
        result = Helpers.__find_terminal_emulator(current_process)
        return emulator == result

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
    def is_image(byte_stream):
        try:
            if isinstance(byte_stream, io.BytesIO):
                byte_stream = byte_stream.getvalue()

            with Image.open(io.BytesIO(byte_stream)) as im:
                return True
        except Exception:
            return False

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
    def get_callable(logging: Logger, method_str) -> Optional[Callable]:
        parts = method_str.split(".")
        class_name = ''

        if len(parts) == 1:
            # If it's a function in the current module
            module = importlib.import_module("__main__")
            method_name = parts[0]
        else:
            # If it's a method in some other module or class
            module_name = ".".join(parts[:-2]) if len(parts) > 2 else ".".join(parts[:-1])
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                logging.error(f"Module '{module_name}' not found")
                return None

            if len(parts) > 2:
                class_name = parts[-2]
                class_obj = getattr(module, class_name, None)
                if class_obj is None:
                    raise ValueError(f"Class '{class_name}' not found in module '{module_name}'")
                method_name = parts[-1]
                method = getattr(class_obj, method_name, None)
            else:
                method_name = parts[-1]
                method = getattr(module, method_name, None)

            if method is None:
                logging.error(f"Method '{method_name}' not found in {'class ' + class_name + ' in ' if len(parts) > 2 else ''}module '{module_name}'")
            return method

    @staticmethod
    def tfidf_similarity(query: str, text_list: list[str]):
        lowered_list = []
        for item in text_list:
            lowered_list.append(re.sub('[^a-zA-Z0-9\s]', '', item.lower()))  # noqa: W605 # type: ignore

        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(lowered_list)

        user_query_tfidf = tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(user_query_tfidf, tfidf_matrix)
        most_similar_index = similarities.argmax()
        # Get the most likely prompt match
        most_likely = text_list[most_similar_index]
        return most_likely

    @staticmethod
    def flatten(lst):
        def __has_list(lst):
            for item in lst:
                if isinstance(item, list):
                    return True
            return False

        def __inner_flatten(lst):
            return [item for sublist in lst for item in sublist]

        while __has_list(lst):
            lst = __inner_flatten(lst)
        return lst

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
    def in_between(s, start, end):
        if end == '\n' and '\n' not in s:
            return s[s.find(start) + len(start):]

        after_start = s[s.find(start) + len(start):]
        part = after_start[:after_start.find(end)]
        return part

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
    def first(predicate, iterable):
        try:
            result = next(x for x in iterable if predicate(x))
            return result
        except StopIteration as ex:
            return None

    @staticmethod
    def filter(predicate, iterable):
        return [x for x in iterable if predicate(x)]

    @staticmethod
    def last(predicate, iterable):
        result = [x for x in iterable if predicate(x)]
        if result:
            return result[-1]
        return None

    @staticmethod
    def resize_image(screenshot_data, base_width=500):
        # Load the image from the in-memory data
        image = Image.open(io.BytesIO(screenshot_data))

        # Calculate the height maintaining the aspect ratio
        w_percent = base_width / float(image.size[0])
        h_size = int(float(image.size[1]) * float(w_percent))

        # Resize the image
        image = image.resize((base_width, h_size), Image.NEAREST)

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
            print('No sections found')
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
                return 'string'

        import inspect

        description = ''
        if func.__doc__ and parse(func.__doc__).short_description:
            description = parse(func.__doc__).short_description
        if func.__doc__ and parse(func.__doc__).long_description:
            description += ' ' + str(parse(func.__doc__).long_description).replace('\n', ' ')  # type: ignore

        func_name = func.__name__
        func_class = Helpers.__get_class_of_func(func)
        invoked_by = f'{func_class.__name__}.{func_name}' if func_class else func_name

        if openai_format:
            params = {}

            for p in inspect.signature(func).parameters:
                param = inspect.signature(func).parameters[p]
                parameter = {
                    param.name: {
                        'type': parse_type(param.annotation) if param.annotation is not inspect._empty else 'string',
                        'description': '',
                    }
                }

                if param.annotation and issubclass(param.annotation, Enum):
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
            return function
        else:
            # check to see if parameters are specified in the __doc__, often they're not
            if not parse(func.__doc__).params:
                parameters = list(inspect.signature(func).parameters.keys())
            else:
                parameters = [p.arg_name for p in parse(func.__doc__).params]

            types = [p.__name__ for p in typing.get_type_hints(func).values()]
            return_type = typing.get_type_hints(func).get('return')

            return {
                # todo: refactor this to be name
                'invoked_by': invoked_by,
                'description': description,
                'parameters': parameters,
                'types': types,
                'return_type': return_type,
            }

    @staticmethod
    def get_function_description_flat(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        return (f'{description["invoked_by"]}({", ".join(description["parameters"])})  # {description["description"]}')

    @staticmethod
    def get_function_description_flat_extra(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        parameter_type_list = [f"{param}: {typ}" for param, typ in zip(description['parameters'], description['types'])]
        return_type = description['return_type'].__name__ if description['return_type'] else ''
        return (f'def {description["invoked_by"]}({", ".join(parameter_type_list)}) -> {return_type}  # {description["description"]}')  # noqa: E501

    @staticmethod
    def load_prompt(prompt_filename: str) -> Dict[str, Any]:
        with open(prompt_filename, 'r') as f:
            prompt = f.read()

            if '[system_message]' not in prompt:
                raise ValueError('Prompt file must contain [system_message]')

            if '[user_message]' not in prompt:
                raise ValueError('Prompt file must contain [user_message]')

            system_message = Helpers.in_between(prompt, '[system_message]', '[user_message]')
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
    def load_and_populate_prompt(
        prompt_filename: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
    ) -> Dict[str, Any]:
        prompt: Dict[str, Any] = Helpers.load_prompt(prompt_filename)

        if not template.get('user_token'):
            template['user_token'] = user_token
            template['user_colon_token'] = user_token + ':'
        if not template.get('assistant_token'):
            template['assistant_token'] = assistant_token
            template['assistant_colon_token'] = assistant_token + ':'

        for key, value in template.items():
            prompt['system_message'] = prompt['system_message'].replace('{{' + key + '}}', value)
            prompt['user_message'] = prompt['user_message'].replace('{{' + key + '}}', value)

        prompt['user_message'] += f'{append_token}'
        prompt['prompt_filename'] = prompt_filename
        return prompt

    @staticmethod
    def load_and_populate_message(
        prompt_filename: str,
        template: Dict[str, str],
        user_token: str = 'User',
        assistant_token: str = 'Assistant',
        append_token: str = '',
    ) -> Message:
        prompt = Helpers.load_and_populate_prompt(prompt_filename, template, user_token, assistant_token, append_token)
        return User(Content(prompt['user_message']))
