import copy
import datetime as dt
import inspect
import math
import os
import random
import time
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

from openai import InvalidRequestError

from eightbitvicuna import VicunaEightBit
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import console_debug, setup_logging
from helpers.vector_store import VectorStore
from objects import (Agent, Answer, Assistant, AstNode, Content, DataFrame,
                     ExecutionFlow, Executor, ForEach, FunctionCall, Get,
                     LLMCall, Message, Order, Program, Set, StackNode,
                     Statement, System, UncertainOrError, User, tree_map)

logging = setup_logging()
def response_writer(callee, message):
    with (open('logs/ast.log', 'a')) as f:
        f.write(f'{str(dt.datetime.now())} {callee}: {message}\n')


# https://glean.com/product/ai-search
# https://dust.tt/
# https://support.apple.com/guide/automator/welcome/mac


def vector_store():
    from helpers.vector_store import VectorStore

    return VectorStore(openai_key=os.environ.get('OPENAI_API_KEY'), store_filename='faiss_index')  # type: ignore

def load_vicuna():
    return VicunaEightBit('models/applied-vicuna-7b', device_map='auto')


def invokeable(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.debug(f"{func.__name__} ran in {end_time - start_time} seconds")
        return result
    return wrapper


# <program> ::= { <statement> }
# <statement> ::= <llm_call> | <foreach> | <function_call> | <answer> | <set> | <get> | <uncertain_or_error>
# <llm_call> ::= 'llm_call' '(' [ <stack> ',' <text> | <stack_pop> ',' <text> | <text> ] ')'
# <foreach> ::= 'foreach' '(' [ <stack> | <stack_pop> | <dataframe> ] ',' <statement> ')'
# <function_call> ::= 'function_call' '(' <helper_function> ')'
# <answer> ::= 'answer' '(' [ <stack> | <stack_pop> | <text> ] ')'
# <get> ::= 'get' '(' <text> ')'
# <set> ::= 'set' '(' <text> ')'
# <text> ::= '"' { <any_character> } '"'
# <dataframe> ::= 'dataframe' '(' [ <list> | <stack> | <stack_pop> ] ')'
# <list> ::= '[' { <text> ',' } <text> ']'
# <stack> ::= 'stack' '(' ')'
# <stack_pop> ::= 'stack_pop' '(' <digit> ')'
# <digit> ::= '0'..'9'
# <any_character> ::= <all_printable_characters_except_double_quotes>
# <helper_function> ::= <function_call_from_available_helper_functions>


class Parser():
    def __init__(
        self,
        message_type=User,
    ):
        self.message: str = ''
        self.remainder: str = ''
        self.index = 0
        self.agents: List[Callable] = []
        self.message_type: type = message_type
        self.errors: List[UncertainOrError] = []

    def consume(self, token: str):
        if token in self.remainder:
            self.remainder = self.remainder[self.remainder.index(token) + len(token):]
            self.remainder = self.remainder.strip()

    def to_function_call(self, call_str: str) -> Optional[FunctionCall]:
        parsed_function = Helpers.parse_function_call(call_str, self.agents)
        if parsed_function:
            func, function_description = parsed_function
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

    def parse_function_call(
        self,
    ) -> Optional[FunctionCall]:
        text = self.remainder
        if (
                ('(' not in text and '))' not in text)
        ):
            return None

        while (
            ('(' in text and '))' in text)
        ):
            start_token = ''
            end_token = ''

            match text:
                case _ if '(' and ')' in text:
                    start_token = '('
                    end_token = '))'

            function_call_str = Helpers.in_between(text, start_token, end_token) + ')'
            function_call: Optional[FunctionCall] = self.to_function_call(function_call_str)
            if function_call:
                function_call.context = Content(text)
                function_call._ast_text = f'function_call({start_token}{function_call_str})'

            # remainder is the stuff after the end_token
            self.remainder = text[text.index(end_token) + len(end_token):]
            return function_call
        return None

    def parse_string(
        self,
    ) -> Optional[str]:
        if self.remainder.startswith('"') and '"' in self.remainder:
            string_result = Helpers.in_between(self.remainder, '"', '"')
            self.consume('"')
            self.consume('"')
            return string_result
        elif self.remainder.startswith("'") and "'" in self.remainder:
            string_result = Helpers.in_between(self.remainder, "'", "'")
            self.consume("'")
            self.consume("'")
            return string_result
        return None

    def __strip_string(self, s):
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
        if s.startswith('\'') and s.endswith('\''):
            s = s[1:-1]
        return s

    def parse_dataframe(self) -> Optional[DataFrame]:
        if self.remainder.startswith('dataframe('):
            self.consume('dataframe(')
            elements = []
            while self.remainder.strip() != '' and not self.remainder.startswith(')'):
                # todo this is probably broken
                element = self.parse_string() or self.parse_stack()
                if element:
                    elements.append(element)
                self.consume(',')
            self.consume(')')
            return DataFrame(elements)
        else:
            return None

    def parse_stack(self) -> Optional[StackNode]:
        re = self.remainder.strip()

        if re.startswith('stack_pop(') and ')' in re:
            num = Helpers.in_between(re, 'stack_pop(', ')')
            self.consume(')')
            return StackNode(int(num))
        elif re.startswith('stack(') and ')' in re:
            num = 0
            self.consume(')')
            return StackNode(0)
        else:
            return None

    def parse_get(self) -> Optional[Get]:
        re = self.remainder.strip()

        if re.startswith('get(') and ')' in re:
            self.consume('get(')
            variable = cast(str, self.parse_string())
            self.consume(')')
            return Get(variable)
        else:
            return None

    def parse_set(self) -> Optional[Set]:
        re = self.remainder.strip()

        if re.startswith('set(') and ')' in re:
            self.consume('set(')
            variable = cast(str, self.parse_string())
            name = ''
            if self.remainder.startswith(','):
                self.consume(',')
                name = cast(str, self.parse_string())
            self.consume(')')
            return Set(variable, name)
        else:
            return None

    def parse_statement(
        self,
        stack: List[Statement],
    ) -> Statement:

        re = self.remainder.strip()

        while re != '':
            # get rid of any prepend stuff that the LLM might throw up
            if re.startswith('Assistant:'):
                self.consume('Assistant:')
                re = self.remainder.strip()

            if re.startswith('answer(') and ')' in re:
                # todo: sort out these '\n' things.
                self.consume('answer(')
                result = ''
                if self.remainder.startswith('stack'):
                    result = self.parse_stack()
                else:
                    result = self.parse_string()

                self.consume(')')

                return Answer(
                    conversation=[Content(result)],
                    ast_text=f'answer({result}))',
                    result=result,
                )

            if re.startswith('set(') and ')' in re:
                set_node = self.parse_set()
                if set_node: return set_node

            if re.startswith('get(') and ')' in re:
                get_node = self.parse_get()
                if get_node: return get_node

            if re.startswith('function_call(') and '))' in re:
                function_call = self.parse_function_call()
                if not function_call:
                    # push past the function call and keep parsing
                    function_call_text = Helpers.in_between(re, 'function_call(', '))')
                    self.consume('))')
                    re = self.remainder

                    error = UncertainOrError(
                        error_message=Content(f'The helper function {function_call_text} could not be found.')
                    )
                    self.errors.append(error)
                else:
                    # end token was consumed in parse_function_call()
                    return function_call

            if re.startswith('llm_call(') and ')' in re:
                self.consume('llm_call(')
                llm_call = LLMCall()
                llm_call.context = self.parse_stack()
                if self.remainder.startswith(','):
                    self.consume(',')
                string_message = Content(self.parse_string())
                self.consume(')')
                llm_call.message = User(string_message)
                llm_call._ast_text = f'llm_call({str(llm_call.context)}, "{str(llm_call.message)}")'
                return llm_call

            if re.startswith('uncertain_or_error(') and ')' in re:
                language = Helpers.in_between(re, 'uncertain_or_error(', ')')
                self.consume(')')
                error = UncertainOrError(
                    error_message=Content(language),
                )
                self.errors.append(error)
                return error

            if re.startswith('foreach(') and ')' in re:
                # <foreach> ::= 'foreach' '(' [ <stack> | <stack_pop> ] ',' <statement> ')'
                self.consume('foreach(')
                context = self.parse_stack() or self.parse_dataframe()
                if not context:
                    raise ValueError('no foreach context')
                else:
                    fe = ForEach(
                        lhs=context,
                        rhs=Statement()
                    )
                    self.consume(',')

                    fe.rhs = self.parse_statement(stack)
                    fe._ast_text = f'foreach({str(fe.lhs)}, {str(fe.rhs)})'.format()
                    self.consume(')')
                    return fe

            if re.startswith('dataframe(') and ')' in re:
                result = self.parse_dataframe()
                if not result: raise ValueError()
                return result

            # might be a string
            if re.startswith('"'):
                # message = self.message_type(self.parse_string())
                # result = LLMCall(messages=[message])
                # result._ast_text = f'"{message}"'
                # return result
                raise ValueError('we should probably figure out how to deal with random strings')

            # we have no idea, so start packaging tokens into a Natural Language call until we see something we know
            known_tokens = [
                'function_call',
                'foreach',
                'uncertain_or_error',
                'natural_language',
                'answer',
                'continuation',
                'get',
                'set',
            ]
            tokens = re.split(' ')
            consumed_tokens = []
            while len(tokens) > 0:
                if tokens[0] in known_tokens:
                    self.remainder = ' '.join(tokens)
                    return LLMCall(message=self.message_type(Content(' '.join(consumed_tokens))))

                consumed_tokens.append(tokens[0])
                self.remainder = self.remainder[len(tokens[0]) + 1:]
                tokens = tokens[1:]

            self.remainder = ''
            return LLMCall(
                message=self.message_type(Content(' '.join(consumed_tokens))),
                ast_text=' '.join(consumed_tokens)
            )
        return Statement()

    def parse_program(
        self,
        message: str,
        agents: List[Callable],
        executor: Executor,
    ) -> Program:
        self.original_message = message
        self.message = message
        self.agents = agents
        self.remainder = message

        program = Program(executor)
        stack: List[Statement] = []

        while self.remainder.strip() != '':
            statement = self.parse_statement(stack)
            program.statements.append(statement)
            stack.append(statement)

        self.message = ''
        self.agents = []
        program.errors = copy.deepcopy(self.errors)

        return program


class ExecutionController():
    def __init__(
        self,
        execution_contexts: List[Executor],
        agents: List[Callable] = [],
        vector_store: VectorStore = VectorStore(),
        cache: PersistentCache = PersistentCache(),
    ):
        self.execution_contexts: List[Executor] = execution_contexts
        self.agents = agents
        self.parser = Parser()
        self.vector_store: VectorStore = vector_store
        self.cache = cache

    def __statement_to_message_no_chunk(
        self,
        statement: Statement,
    ) -> User:
        statement_result_prompts = {
            'answer': 'prompts/answer_result.prompt',
            'function_call': 'prompts/function_call_result.prompt',
            'llm_call': 'prompts/llm_call_result.prompt',
            'uncertain_or_error': 'prompts/uncertain_or_error_result.prompt',
            'foreach': 'prompts/foreach_result.prompt',
        }

        if isinstance(statement, FunctionCall):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[statement.token()],
                template={
                    'function_call': statement.to_code_call(),
                    'function_signature': statement.to_definition(),
                    'function_result': str(statement.result()),
                }
            )
            return (User(Content(result_prompt['user_message'])))

        elif isinstance(statement, LLMCall) or isinstance(statement, ForEach):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[statement.token()],
                template={
                    f'{statement.token()}_result': str(statement.result()),
                }
            )
            return (User(Content(result_prompt['user_message'])))

        elif isinstance(statement, UncertainOrError):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[statement.token()],
                template={
                    'error_message': str(statement.error_message),
                    'supporting_context': str(statement.supporting_error) if statement.supporting_error else '',
                }
            )

        return User(Content(''))

    def __statement_to_message(
        self,
        context: Statement,
        query: User,
        max_tokens: int,
    ) -> User:
        statement_result_prompts = {
            'answer': 'prompts/answer_result.prompt',
            'function_call': 'prompts/function_call_result.prompt',
            'llm_call': 'prompts/llm_call_result.prompt',
            'uncertain_or_error': 'prompts/uncertain_or_error_result.prompt',
            'foreach': 'prompts/foreach_result.prompt',
        }

        if isinstance(context, FunctionCall):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[context.token()],
                template={
                    'function_call': context.to_code_call(),
                    'function_signature': context.to_definition(),
                    'function_result': self.__chunk_content(
                        query=query.message,
                        current_str=context.result(),
                        max_token_count=max_tokens,
                    ),
                }
            )
            return (User(Content(result_prompt['user_message'])))

        if isinstance(context, LLMCall) or isinstance(context, ForEach):
            result_prompt = Helpers.load_and_populate_prompt(
                prompt_filename=statement_result_prompts[context.token()],
                template={
                    f'{context.token()}_result': self.__chunk_content(
                        query=query.message,
                        current_str=context.result(),
                        max_token_count=max_tokens,
                    ),
                }
            )
            return (User(Content(result_prompt['user_message'])))

        return User(Content(''))

    def __stacknode_to_statement(
        self,
        program: Program,
        stack_node: StackNode,
    ) -> List[Statement]:
        return program.runtime_stack.peek(stack_node.value)

    def __stacknode_to_runtime_pop(
        self,
        program: Program,
        stack_node: StackNode,
    ):
        if stack_node.value == 0:
            while len(program.runtime_stack.stack) > 0:
                program.runtime_stack.pop()
        else:
            for i in range(0, stack_node.value):
                program.runtime_stack.pop()

    def __package_context(
        self,
        node: StackNode | DataFrame,
        program: Program,
    ) -> List[User]:
        """
        Either returns a Dataframe, or peeks at the stack to the depth specified by the StackNode.
        It does not pop the stack.
        """
        if isinstance(node, StackNode):
            return [User(Content(str(n.result()))) for n in program.runtime_stack.peek(node.value)]
        elif isinstance(node, DataFrame):
            logging.debug('todo')
            return [User(Content('\n'.join([str(s.result()) for s in node.elements])))]
        return []

    def __marshal(self, value: object, type: str) -> Any:
        def strip_quotes(value: str) -> str:
            if value.startswith('\'') or value.startswith('"'):
                value = value[1:]
            if value.endswith('\'') or value.endswith('"'):
                value = value[:-1]
            return value

        if type == 'str':
            result = str(value)
            return strip_quotes(result)
        elif type == 'int':
            return int(strip_quotes(str(value)))
        elif type == 'float':
            return float(strip_quotes(str(value)))
        else:
            return value

    def __chunk_messages(
        self,
        query: Any,
        messages: List[Message],
        max_token_count: int,
    ) -> List[Message]:
        # force conversion to str
        query = str(query)
        total_message = '\n'.join([str(m.message) for m in messages])

        if Helpers.calculate_tokens(total_message) + Helpers.calculate_tokens(query) < max_token_count:
            return messages

        total_tokens_per_message = math.floor(max_token_count / len(messages))

        chunked_messages: List[Message] = []

        for message in messages:
            ranked: List[Tuple[str, float]] = self.vector_store.chunk_and_rank(
                query,
                str(message.message),
                chunk_token_count=math.floor(total_tokens_per_message / 5),
                max_tokens=total_tokens_per_message,
            )
            ranked_str = '\n\n'.join([r[0] for r in ranked])

            while Helpers.calculate_tokens(ranked_str) + Helpers.calculate_tokens(query) > max_token_count:
                ranked_str = ranked_str[0:len(ranked_str) - 16]

            chunked_messages.append(message.__class__(Content(ranked_str)))
        return chunked_messages

    def __chunk_content(
        self,
        query: Any,
        current_str: Any,
        max_token_count: int,
    ) -> str:
        # force conversion to str
        query = str(query)
        current_str = str(current_str)

        if Helpers.calculate_tokens(current_str) + Helpers.calculate_tokens(query) < max_token_count:
            return str(current_str)

        ranked: List[Tuple[str, float]] = self.vector_store.chunk_and_rank(
            query,
            current_str,
            chunk_token_count=math.floor((max_token_count / 10)),
        )

        ranked_str = '\n\n'.join([r[0] for r in ranked])
        while Helpers.calculate_tokens(ranked_str) + Helpers.calculate_tokens(query) > max_token_count:
            ranked_str = ranked_str[0:len(ranked_str) - 16]

        return ranked_str

    def __lhs_to_str(self, lhs: List[Statement]) -> str:
        results = [r for r in lhs if r.result()]
        # unique
        results = list(set(results))
        return '\n'.join([str(r) for r in results])

    def __summarize_to_messages(self, node: AstNode) -> List[Message]:
        def summarize_conversation(ast_node: AstNode) -> Optional[Message]:
            if isinstance(ast_node, FunctionCall) and ast_node.result():
                return User(Content(str(ast_node.result())))
            elif isinstance(ast_node, LLMCall):
                return ast_node.message
            else:
                return None

        # we probably need to do stuff to fit inside the context window etc
        messages: List[Message] = []
        context_messages = cast(List[Message], tree_map(node, summarize_conversation))
        context_messages = [m for m in context_messages if m is not None]
        messages.extend(context_messages)

        return messages

    def classify_tool_or_direct(
        self,
        prompt: str,
    ) -> Dict[str, float]:
        def parse_result(result: str) -> Dict[str, float]:
            if ',' in result:
                if result.startswith('Assistant: '):
                    result = result[len('Assistant: '):].strip()

                first = result.split(',')[0].strip()
                second = result.split(',')[1].strip()
                try:
                    if first.startswith('tool') or first.startswith('"tool"'):
                        return {'tool': float(second)}
                    elif first.startswith('direct') or first.startswith('"direct"'):
                        return {'direct': float(second)}
                    else:
                        return {'tool': 1.0}
                except ValueError as ex:
                    return {'tool': 1.0}
            return {'tool': 1.0}

        executor = self.execution_contexts[0]

        # assess the type of task
        function_list = [Helpers.get_function_description_flat(f) for f in self.agents]
        query_understanding = Helpers.load_and_populate_prompt(
            prompt_filename='prompts/query_understanding.prompt',
            template={
                'functions': '\n'.join(function_list),
                'user_input': prompt,
            }
        )

        assistant: Assistant = executor.execute(
            messages=[
                System(Content(query_understanding['system_message'])),
                User(Content(query_understanding['user_message']))
            ],
            temperature=0.0,
        )

        if assistant.error or not parse_result(str(assistant.message)):
            return {'tool': 1.0}
        return parse_result(str(assistant.message))

    def __llm_call(
        self,
        user_message: Message,
        context_messages: List[Message],
        executor: Executor,
        prompt_filename: Optional[str] = None,
    ) -> Assistant:
        if not prompt_filename:
            prompt_filename = ''
        # execute the call to check to see if the Answer satisfies the original query
        messages: List[Message] = copy.deepcopy(context_messages)
        messages.append(user_message)
        try:
            assistant: Assistant = executor.execute(messages)
            console_debug(prompt_filename, 'User', str(user_message.message))
            console_debug(prompt_filename, 'Assistant', str(assistant.message))
        except InvalidRequestError as ex:
            console_debug(prompt_filename, 'User', str(user_message.message))
            raise ex
        response_writer(prompt_filename, assistant)
        return assistant

    def __llm_call_prompt(
        self,
        prompt_filename: str,
        context_messages: List[Message],
        executor: Executor,
        template: Dict[str, Any],
    ) -> Assistant:
        prompt = Helpers.load_and_populate_prompt(
            prompt_filename=prompt_filename,
            template=template,
        )
        return self.__llm_call(
            User(Content(prompt['user_message'])),
            context_messages,
            executor,
            prompt_filename=prompt_filename
        )

    def execute_llm_call(
        self,
        query_or_task: str,
        load_execute_message: Message,
        executor: Executor,
        program: Program,
        messages: List[Message],
        vector_search_query: Optional[str] = None,
        prompt_filename: Optional[str] = None,
    ) -> Assistant:
        """
        Executes an LLM call on a prompt_message with a context of messages.
        Performs either a chunk_and_rank, or a map/reduce depending on the
        context relavence to the prompt_message.
        """
        assistant_result: Assistant
        if not vector_search_query:
            logging.debug('todo this almost certainly doesnt work properly')
            vector_search_query = str(query_or_task)

        # I have either a message, or a list of messages. They might need to be map/reduced.
        if executor.calculate_tokens(messages + [load_execute_message]) > executor.max_prompt_tokens():
            message = User(Content('\n\n'.join([str(m.message) for m in messages])))

            # see if we can do a similarity search or not.
            similarity_chunks = self.vector_store.chunk_and_rank(
                query=vector_search_query,
                content=str(message.message),
                chunk_token_count=1024,
                chunk_overlap=10,
                max_tokens=executor.max_prompt_tokens() - executor.calculate_tokens([load_execute_message])
            )

            # randomize and sample from the similarity_chunks
            twenty_percent = math.floor(len(similarity_chunks) * 0.2)
            similarity_chunks = random.sample(similarity_chunks, min(len(similarity_chunks), twenty_percent))

            decision_criteria: List[str] = []
            for chunk, rank in similarity_chunks:
                assistant_similarity = self.__llm_call_prompt(
                    prompt_filename='prompts/document_chunk.prompt',
                    context_messages=[],
                    executor=executor,
                    template={
                        'query': str(query_or_task),
                        'document_chunk': chunk,
                    })
                decision_criteria.append(str(assistant_similarity.message))
                logging.debug('map_reduce_required, query_or_task: {}, response: {}'.format(
                    query_or_task,
                    assistant_similarity.message,
                ))
                if 'No' in str(assistant_similarity.message):
                    # we can break early, as the 'map_reduced_required' flag will not be set below
                    break

            map_reduce_required = all(['Yes' in d for d in decision_criteria])

            # either similarity search, or map reduce required
            # here, we're not doing a map_reduce, we're simply populating the context window
            # with the highest ranked chunks from each message.
            if not map_reduce_required:
                tokens_per_message = math.floor((executor.max_prompt_tokens() - executor.calculate_tokens([load_execute_message])) / len(messages))

                # for all messages, do a similarity search
                similarity_messages = []
                for i in range(len(messages)):
                    message = messages[i]

                    similarity_chunks = self.vector_store.chunk_and_rank(
                        query=vector_search_query,
                        content=str(message),
                        chunk_token_count=512,
                        chunk_overlap=0,
                        max_tokens=tokens_per_message - 32,
                    )
                    similarity_message = '\n\n'.join([content for content, rank in similarity_chunks])
                    similarity_messages.append(User(Content(similarity_message)))

                assistant_result = self.__llm_call(
                    user_message=load_execute_message,
                    context_messages=similarity_messages,
                    executor=executor,
                    prompt_filename=prompt_filename
                )

            # do the map reduce instead of similarity
            else:
                # collapse the message
                context_message = User(Content('\n\n'.join([str(m.message) for m in messages])))
                chunk_results = []

                # iterate over the data.
                map_reduce_prompt_tokens = executor.calculate_tokens(
                    [User(Content(open('prompts/map_reduce_map.prompt', 'r').read()))]
                )
                chunk_size = (executor.max_prompt_tokens() - map_reduce_prompt_tokens) - (
                    executor.calculate_tokens([load_execute_message]) - 32
                )

                chunks = self.vector_store.chunk(
                    content=str(context_message.message),
                    chunk_size=chunk_size,
                    overlap=0
                )

                for chunk in chunks:
                    chunk_assistant = self.__llm_call_prompt(
                        prompt_filename='prompts/map_reduce_map.prompt',
                        context_messages=[],
                        executor=executor,
                        template={
                            'original_query': str(program.original_query),
                            'query': query_or_task,
                            'data': chunk,
                        })
                    chunk_results.append(str(chunk_assistant.message))

                # perform the reduce
                map_results = '\n\n====\n\n' + '\n\n====\n\n'.join(chunk_results)

                assistant_result = self.__llm_call_prompt(
                    prompt_filename='prompts/map_reduce_reduce.prompt',
                    context_messages=[],
                    executor=executor,
                    template={
                        'original_query': str(program.original_query),
                        'query': query_or_task,
                        'map_results': map_results
                    })
        else:
            assistant_result = self.__llm_call(
                user_message=cast(User, load_execute_message),
                context_messages=messages,
                executor=executor,
                prompt_filename=prompt_filename,
            )
        return assistant_result

    def execute_statement(
        self,
        statement: Statement,
        executor: Executor,
        program: Program,
    ) -> Statement:

        # we have an answer to the query, return it
        if isinstance(statement, Answer):
            messages: List[Message] = []

            original_query = program.original_query
            stack_node_query = program.original_query
            answer = str(statement.result())

            context_messages: List[Message] = [
                self.__statement_to_message(
                    context=s,
                    query=User(Content(program.original_query)),
                    max_tokens=executor.max_prompt_tokens() // 2,
                ) for s in program.executed_stack.stack
            ]

            if statement.result() and isinstance(statement.result(), StackNode):
                answer = '\n\n====\n\n'.join(
                    [
                        str(s.result())
                        for s in self.__stacknode_to_statement(program, cast(StackNode, statement.result()))
                    ])

                stack_node_query = '\n or '.join(
                    [
                        str(s.message)
                        for s in self.__stacknode_to_statement(program, cast(StackNode, statement.result()))
                        if isinstance(s, LLMCall)
                    ])

            # if it's a stack() based answer, we treat this case differently
            # as it's an aggregation of possibly many different calls prior.
            # we ask the LLM to summarize and keep consistent formatting.
            if (
                statement.result()
                and isinstance(statement.result(), StackNode)
                and cast(StackNode, statement.result()).value == 0
                and program.runtime_stack.count() > 1
            ):
                # sometimes we have a massive stack of function_calls with no
                # llm map reduce. In this case, answer(stack()) should probably kick this up

                answer_assistant: Assistant = self.__llm_call_prompt(
                    prompt_filename='prompts/answer_result_stack.prompt',
                    context_messages=context_messages,
                    executor=executor,
                    template={
                        'original_query': original_query,
                        'answer': '====\n\n' + answer,
                    })
            else:
                answer_assistant: Assistant = self.__llm_call_prompt(
                    prompt_filename='prompts/answer_result.prompt',
                    context_messages=context_messages,
                    executor=executor,
                    template={
                        'original_query': original_query,
                        'stack_node_query': stack_node_query,
                        'answer': answer,
                    })

                # pop off any consumed elements on the stack
            if answer_assistant and isinstance(statement.result(), StackNode):
                self.__stacknode_to_runtime_pop(program, cast(StackNode, statement.result()))

            # check for comments
            if "[##]" in str(answer_assistant.message):
                answer_assistant.message = Content(str(answer_assistant.message).split("[##]")[0].strip())

            statement._result = answer_assistant
            program.answers.append(statement)
            return statement

        # get and set registers
        elif (
            isinstance(statement, Get)
            and not statement.result()
        ):
            name, temp_statement = program.runtime_registers.get(statement.variable)  # type: ignore
            if temp_statement:
                statement._result = temp_statement
                if name and temp_statement._result and isinstance(temp_statement._result, str):
                    temp_statement._result = f'The data below is named: {name}' + '\n\n' + str(temp_statement._result)
                elif name and temp_statement._result and isinstance(temp_statement._result, Content):
                    temp_statement._result = Content(f'The data below is named: {name}' + '\n\n' + str(temp_statement._result))
                elif name and temp_statement._result and isinstance(temp_statement._result, Assistant):
                    temp_statement._result.message = Content(f'The data below is named: {name}' + '\n\n' + str(temp_statement._result.message))

                # outer loop pushes this on to the runtime stack
                return temp_statement

        elif (
            isinstance(statement, Set)
            and not statement.result()
        ):
            temp_statement = program.runtime_stack.pop()
            if temp_statement:
                program.runtime_registers[statement.variable] = (statement.name, temp_statement)
            # outer loop ignores Set and Answer
            return statement

        # execute function call
        elif (
            isinstance(statement, FunctionCall)
            and not statement.result()
        ):
            # todo:
            # if there is something on the stack (say an llm_call) then it's likely
            # that we need to extract args for the function call.

            # unpack the args, call the function
            function_call = statement
            function_args_desc = statement.args
            function_args = {}

            # Note: the JSON response from the model may not be valid JSON
            func: Callable | None = Helpers.first(lambda f: f.__name__ in function_call.name, self.agents)

            if not func:
                logging.error('Could not find function named: {}'.format(function_call.name))
                return UncertainOrError(
                    error_message=Content('I could not find the function named: {}'.format(function_call.name)),
                )

            # check for enum types and marshal from string to enum
            counter = 0
            for p in inspect.signature(func).parameters.values():
                if p.annotation != inspect.Parameter.empty and p.annotation.__class__.__name__ == 'EnumMeta':
                    function_args[p.name] = p.annotation(self.__marshal(function_args_desc[counter][p.name], 'str'))
                elif counter < len(function_args_desc):
                    function_args[p.name] = self.__marshal(function_args_desc[counter][p.name], p.annotation.__name__)
                else:
                    function_args[p.name] = p.default
                counter += 1

            try:
                # let's check the cache first
                if self.cache.has_key(function_args):
                    function_response = self.cache.get(function_args)
                else:
                    function_response = func(**function_args)
                    self.cache.set(function_args, function_response)

            except Exception as e:
                logging.error(e)
                return UncertainOrError(
                    error_message=Content('The function could not execute. It raised an exception: {}'.format(e)),
                    supporting_error=e,
                )

            function_call._result = function_response
            return function_call

        elif (
            isinstance(statement, LLMCall)
            and not statement.result()
        ):
            context_messages: List[Message] = []
            if statement.supporting_system: context_messages.append(statement.supporting_system)
            if statement.supporting_messages: context_messages.extend(statement.supporting_messages)
            if statement.context and isinstance(statement.context, StackNode):
                # dataframe or stack context peek
                stack_statements = self.__stacknode_to_statement(program, statement.context)
                context = [
                    self.__statement_to_message_no_chunk(statement=s) for s in stack_statements
                ]
                context_messages.extend(context)

            # updates
            statement._result = self.execute_llm_call(
                query_or_task=str(statement.message),
                load_execute_message=Helpers.load_and_populate_message(
                    prompt_filename='prompts/llm_call.prompt',
                    template={
                        'llm_call_message': str(statement.message),
                    }),
                # prompt_message=statement.message,
                executor=executor,
                program=program,
                messages=context_messages,
                vector_search_query=str(statement.message),
                prompt_filename='prompts/llm_call.prompt'
            )
            # todo: do a check to see if looks valid

            # pop off any consumed elements on the stack
            if statement.context and isinstance(statement.context, StackNode):
                self.__stacknode_to_runtime_pop(program, statement.context)

            # return the result
            return statement

        # foreach
        elif (
            isinstance(statement, ForEach)
            and not statement.result()
        ):
            # deal with the first argument (lhs), which is the list context for the foreach loop
            # we will need to shape that into a dataframe that the foreach can execute
            stack_statements: List[Statement] = []
            assistant: Assistant

            # deal with the left hand side
            if isinstance(statement.lhs, StackNode):
                stack_statements = self.__stacknode_to_statement(program, statement.lhs)
            else:
                stack_statements = [statement.lhs]

            # now deal with the right hand side
            # it can either be a something that has an unmarshalled list,
            # or it can be an actual list of things

            # do the actual foreach
            for stack_statement in stack_statements:
                # foreach FunctionCall
                if isinstance(statement.rhs, FunctionCall):
                    context_messages: List[Message] = [
                        self.__statement_to_message(
                            stack_statement,
                            User(Content(program.original_query)),
                            executor.max_prompt_tokens()
                        ),
                    ]

                    assistant = self.execute_llm_call(
                        query_or_task=statement.rhs.to_definition(),
                        load_execute_message=Helpers.load_and_populate_message(
                            prompt_filename='prompts/foreach_function_call.prompt',
                            template={
                                'function_definition': statement.rhs.to_definition(),
                                'function_call': statement.rhs.to_code_call(),
                                'goal': program.original_query,
                            }),
                        executor=executor,
                        program=program,
                        messages=[
                            self.__statement_to_message_no_chunk(stack_statement)
                        ],
                        # todo: I think this is wrong, but this is what the previous code did
                        vector_search_query=program.original_query + str(statement.rhs.to_definition()),
                        prompt_filename='prompts/foreach_function_call.prompt',
                    )

                    foreach_function_response = str(assistant.message)

                    # check to see if the function call had any "missing" parameters
                    # try pushing the LLM to re-write the correct function calls
                    # todo: I think this is wrong, context should be enough.
                    if '"missing"' in str(assistant.message):
                        previous_messages = [str(s.result()) for s in program.runtime_stack.peek(0)]

                        assistant_function_call_rewrite = self.execute_llm_call(
                            query_or_task=statement.rhs.to_definition(),
                            load_execute_message=Helpers.load_and_populate_message(
                                prompt_filename='prompts/function_call_error_correction.prompt',
                                template={
                                    'function_calls_missing': str(assistant.message),
                                    'function_call_signatures': '\n'.join(
                                        [Helpers.get_function_description_flat_extra(f) for f in self.agents]
                                    ),
                                    'previous_messages': '\n'.join(previous_messages)
                                }),
                            executor=executor,
                            program=program,
                            messages=[],
                            vector_search_query='',
                            prompt_filename='prompts/function_call_error_correction.prompt',
                        )
                        foreach_function_response = str(assistant_function_call_rewrite.message)

                    # if the assistant result is a program, we need to compile and interpret
                    foreach_program = Parser().parse_program(
                        message=foreach_function_response,
                        agents=self.agents,
                        executor=program.executor
                    )

                    # execute it
                    if foreach_program:
                        statement._result.extend([  # type: ignore
                            self.execute_statement(s, executor, foreach_program)
                            for s in foreach_program.statements
                        ])
                    else:
                        # todo: why is this string, and not assistant?
                        statement._result.append(foreach_function_response)  # type: ignore

                # foreach llm call
                elif isinstance(statement.rhs, LLMCall):
                    # if I have a stack of elements, then I need to be specific about
                    # how the llm should extract the task list.
                    if (
                        isinstance(statement.lhs, StackNode)
                        and cast(StackNode, statement.lhs).value == 0
                        and program.runtime_stack.count() > 1
                    ):
                        assistant_llm_call = self.execute_llm_call(
                            query_or_task=str(statement.rhs.message),
                            load_execute_message=Helpers.load_and_populate_message(
                                prompt_filename='prompts/foreach_llm_call_stack.prompt',
                                template={
                                    'message': str(statement.rhs.message),
                                }),
                            executor=executor,
                            program=program,
                            messages=[self.__statement_to_message_no_chunk(stack_statement)],
                            vector_search_query=program.original_query + ' ' + str(statement.rhs.message),
                            prompt_filename='prompts/foreach_llm_call_stack.prompt',
                        )
                        statement.rhs._result = assistant_llm_call
                        statement._result.append(copy.deepcopy(statement.rhs))  # type: ignore
                    else:
                        assistant_llm_call = self.execute_llm_call(
                            query_or_task=str(statement.rhs.message),
                            load_execute_message=Helpers.load_and_populate_message(
                                prompt_filename='prompts/foreach_llm_call.prompt',
                                template={
                                    'message': str(statement.rhs.message),
                                }),
                            executor=executor,
                            program=program,
                            messages=[self.__statement_to_message_no_chunk(stack_statement)],
                            vector_search_query=program.original_query + ' ' + str(statement.rhs.message),
                            prompt_filename='prompts/foreach_llm_call.prompt',
                        )
                        statement.rhs._result = assistant_llm_call
                        statement._result.append(copy.deepcopy(statement.rhs))  # type: ignore

                # foreach Answer
                elif isinstance(statement.rhs, Answer):
                    logging.error('todo: this should never happen')
                    assistant_llm_call = self.execute_llm_call(
                        query_or_task=str(statement.rhs.result()),
                        load_execute_message=Helpers.load_and_populate_message(
                            prompt_filename='prompts/foreach_answer.prompt',
                            template={
                                'message': str(statement.rhs.result()),
                            }),
                        executor=executor,
                        program=program,
                        messages=[self.__statement_to_message_no_chunk(stack_statement)],
                        vector_search_query='',
                        prompt_filename='prompts/foreach_answer.prompt',
                    )
                    statement._result.append(copy.deepcopy(assistant_llm_call))  # type: ignore

            # end of foreach tests
            # pop off any consumed elements on the stack
            if statement.lhs and isinstance(statement.lhs, StackNode):
                self.__stacknode_to_runtime_pop(program, statement.lhs)

            # not sure if I should return here
            return statement

        else:
            return Statement()

    def execute_program(
        self,
        program: Program,
    ) -> Program:

        flow = ExecutionFlow(Order.QUEUE)
        for s in reversed(program.statements):
            flow.push(s)

        while statement := flow.pop():
            result = self.execute_statement(
                statement,
                program.executor,
                program,
            )
            # these are nodes that are control flow or do not have a result
            if (
                not isinstance(result, Answer)
                and not isinstance(result, Set)
                and not isinstance(result, ForEach)
            ):
                program.runtime_stack.push(result)
            elif (
                isinstance(result, ForEach)
            ):
                if (
                    isinstance(result.result(), list)
                    and all([isinstance(r, Statement) for r in cast(list, result.result())])
                ):
                    # foreach generated a list of statements that we should push on the
                    # runtime stack
                    for r in cast(list, result.result()):
                        program.runtime_stack.push(r)

            # track all executed nodes so that Answer can do a final check
            if (
                not isinstance(result, Answer)
                and not isinstance(result, Set)
                and not isinstance(result, Get)
            ):
                program.executed_stack.push(copy.deepcopy(result))

        return program

    def execute_chat(
        self,
        messages: List[Message],
    ) -> Assistant:
        executor = self.execution_contexts[0]
        return executor.execute(messages)

    def execute(
        self,
        call: LLMCall
    ) -> List[Statement]:
        # pick the right execution context that will get the task done
        # for now, we just grab the first
        results: List[Statement] = []
        executor = self.execution_contexts[0]

        # assess the type of task
        classification = self.classify_tool_or_direct(str(call.message))

        # if it requires tooling, hand it off to the AST execution engine
        if 'tool' in classification:
            response = executor.execute_with_agents(
                call=call,
                agents=self.agents,
                temperature=0.0,
            )
            assistant_response = str(response.message)
            logging.debug('Abstract Syntax Tree:')
            # debug out AST
            lines = str(assistant_response).split('\n')
            for line in lines:
                logging.debug(f'  {str(line)}')

            # debug output
            response_writer('llm_call', assistant_response)

            # parse the response
            program = Parser().parse_program(assistant_response, self.agents, executor)
            program.conversation.append(response)
            program.original_query = str(call.message)

            # deal with errors
            if program.errors:
                logging.debug('Abstract Syntax Tree had errors: {}'.format(program.errors))
                error_reply = self.__llm_call_prompt(
                    prompt_filename='prompts/tool_execution_error.prompt',
                    context_messages=response._messages_context + [Assistant(response.message)],
                    executor=executor,
                    template={
                        'program': assistant_response,
                        'errors': '\n\n'.join([str(e.error_message) for e in program.errors])
                    }
                )
                program = Parser().parse_program(str(error_reply.message), self.agents, executor)
                program.conversation.append(response)
                program.original_query = str(call.message)

            # execute the program
            program = self.execute_program(program)
            results.extend(program.answers)
        else:
            assistant_reply: Assistant = self.execute_chat(call.supporting_messages + [call.message])
            results.append(Answer(
                conversation=[Content(str(assistant_reply.message))],
                result=assistant_reply.message
            ))

        return results

    def execute_completion(
        self,
        user_message: str,
        system_message: str = 'You are a helpful assistant.'
    ) -> List[Statement]:
        call = LLMCall(
            context=None,
            message=User(Content(user_message)),
            supporting_system=System(Content(system_message)),
        )

        return self.execute(call)
