import asyncio
import inspect
import re
from enum import Enum, IntEnum
from itertools import cycle, islice
from typing import Any, Callable, Dict, Generator, List, Tuple

import guidance
import nest_asyncio
from docstring_parser import parse
from guidance.llms import LLM, OpenAI
from guidance.llms.transformers import LLaMA, Vicuna
from sentence_transformers import SentenceTransformer, util
from torch import Tensor


class Helpers():
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
        model = SentenceTransformer('all-mpnet-base-v2')
        corpus_embeddings: List[Tensor] | Any = model.encode(sections, convert_to_tensor=True)
        query_embedding: List[Tensor] | Any = model.encode([query], convert_to_tensor=True)

        cosine_scores = util.cos_sim(corpus_embeddings, query_embedding)  # type: ignore

        scored_sections = list(zip(sections, cosine_scores))
        scored_sections = sorted(scored_sections, key=lambda x: x[1], reverse=True)  # type: ignore

        scores = [{'text': text, 'score': score.cpu().item()} for text, score in scored_sections]
        return scores

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
    def calculate_tokens(content: str | List[Dict[str, str]]):
        if isinstance(content, list):
            return len(Helpers.messages_to_str(content).split()) / 0.75
        else:
            return len(content.split()) / 0.75

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
    async def execute_llm_template_async(template: str, llm: LLM, **kwargs) -> Dict[str, str]:
        prompt = guidance(template, llm, stream=True)  # type: ignore

        token_results = []
        for text in Helpers.run_and_stream(prompt, **kwargs):
            print(text, end='')
            token_results.append(text)

        prompt_result = ''.join(token_results)
        assistant_result = Helpers.find_string_between_tokens(prompt_result, 'ASSISTANT: ', '</s>')

        return {'prompt_result': str(prompt_result), 'answer': assistant_result}

    @staticmethod
    def execute_llm_template(template: str, llm: LLM, **kwargs) -> Dict[str, str]:
        prompt = guidance(template, llm)  # type: ignore   # , stream=True, async_mode=True)  # type: ignore
        prompt_result = prompt(**kwargs)

        assistant_result = ''
        pattern = r'ASSISTANT: (.*?)</s>'
        matches = re.findall(pattern, str(prompt_result), re.MULTILINE | re.DOTALL)

        if matches:
            assistant_result = matches[-1]

        return {'prompt_result': str(prompt_result), 'answer': assistant_result}

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
    def get_function_description(func, openai_format: bool):
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
        if func.__doc__ and parse(func.__doc__).long_description:
            description = parse(func.__doc__).long_description
        elif func.__doc__ and parse(func.__doc__).short_description:
            description = parse(func.__doc__).short_description

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

            function = {
                'name': invoked_by,
                'description': description,
                'parameters': {
                    'type': 'object',
                    'properties': params
                },
                'required': list(params.keys()),
            }
            return function
        else:
            if not func.__doc__:
                return {
                    'invoked_by': invoked_by,
                    'description': description,
                    'parameters': list(inspect.signature(func).parameters.keys())
                }
            return {
                # todo: refactor this to be name
                'invoked_by': invoked_by,
                'description': description,
                'parameters': [p.arg_name for p in parse(func.__doc__).params],
            }

    @staticmethod
    def get_function_description_flat(function: Callable) -> str:
        description = Helpers.get_function_description(function, openai_format=False)
        return (f'{description["invoked_by"]}({", ".join(description["parameters"])})  # {description["description"]}')

    @staticmethod
    def parse_function_call(call: str, functions: List[Callable]):
        function_description: Dict[str, Any] = {}

        function_name = Helpers.in_between(call, '', '(')
        function_args = [p.strip() for p in Helpers.in_between(call, '(', ')').split(',')]

        for f in functions:
            if f.__name__.lower() in function_name.lower():
                function_description = Helpers.get_function_description(
                    f,
                    openai_format=True
                )
                continue

        if not function_description:
            return None

        argument_count = 0

        for name, parameter in function_description['parameters']['properties'].items():
            if len(function_args) < argument_count:
                parameter.update({'argument': function_args[argument_count]})
            argument_count += 1

        return function_description

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
    ):
        prompt = Helpers.load_prompt(prompt_filename)

        for key, value in template.items():
            prompt['system_message'] = prompt['system_message'].replace(f'{{{key}}}', value)
            prompt['user_message'] = prompt['user_message'].replace(f'{{{key}}}', value)

        return prompt
