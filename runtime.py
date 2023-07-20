import datetime as dt
import inspect
import math
import os
import time
from typing import (Any, Callable, Dict, Generator, Generic, List, Optional,
                    Sequence, Tuple, TypeVar, Union, cast)

from eightbitvicuna import VicunaEightBit
from helpers.helpers import Helpers, PersistentCache
from helpers.logging_helpers import setup_logging
from helpers.vector_store import VectorStore
from objects import (Agent, Answer, Assistant, AstNode, Content, DataFrame,
                     ExecutionFlow, Executor, ForEach, FunctionCall, LLMCall,
                     Message, Order, Program, StackNode, Statement, System,
                     UncertainOrError, User, tree_map)

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
# <statement> ::= <llm_call> | <foreach> | <function_call> | <answer> | <uncertain_or_error>
# <llm_call> ::= 'llm_call' '(' [ <stack> | <stack_pop> | <text> ] ')'
# <foreach> ::= 'foreach' '(' [ <stack> | <stack_pop> ] ',' <statement> ')'
# <function_call> ::= 'function_call' '(' <helper_function> ')'
# <answer> ::= 'answer' '(' [ <stack> | <stack_pop> | <text> ] ')'
# <uncertain_or_error> ::= 'uncertain_or_error' '(' <text> ')'
# <text> ::= '"' { <any_character> } '"'
# <dataframe> ::= '[' { <element> ',' } <element> ']'
# <element> ::= <text> | <stack_pop> | <stack>
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
    ) -> Optional[Content]:
        if self.remainder.startswith('"') and '"' in self.remainder:
            string_result = Helpers.in_between(self.remainder, '"', '"')
            self.consume('"')
            self.consume('"')
            return Content(string_result)
        elif self.remainder.startswith("'") and "'" in self.remainder:
            string_result = Helpers.in_between(self.remainder, "'", "'")
            self.consume("'")
            self.consume("'")
            return Content(string_result)
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

            if re.startswith('function_call(') and '))' in re:
                function_call = self.parse_function_call()
                if not function_call:
                    # push past the function call and keep parsing
                    self.consume(')')
                    re = self.remainder
                else:
                    # end token was consumed in parse_function_call()
                    return function_call

            if re.startswith('llm_call(') and ')' in re:
                self.consume('llm_call(')
                llm_call = LLMCall()
                llm_call.context = self.parse_stack()
                if self.remainder.startswith(','):
                    self.consume(',')
                string_message = self.parse_string() or Content('')
                self.consume(')')
                llm_call.message = User(string_message)
                llm_call._ast_text = f'llm_call({str(llm_call.context)}, "{str(llm_call.message)}")'
                return llm_call

            if re.startswith('uncertain_or_error(') and ')' in re:
                language = Helpers.in_between(re, 'uncertain_or_error(', ')')
                self.consume(')')
                return UncertainOrError(
                    error_message=Content(language),
                )

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
            known_tokens = ['function_call', 'foreach', 'uncertain_or_error', 'natural_language', 'answer', 'continuation']
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

    def __context_to_message(
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
    ):
        return [node for node in program.runtime_stack.peek(stack_node.value)]

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

    def execute_statement(
        self,
        statement: Statement,
        executor: Executor,
        program: Program,
    ) -> Statement:

        # we have an answer to the query, return it
        if isinstance(statement, Answer):
            if isinstance(statement.result(), StackNode):
                statement._result = program.runtime_stack.pop()

            # check to see if the answer is partially or fully satisfactory of the
            # original query.
            # dataframe or stack context peek
            messages: List[Message] = []
            context = [
                self.__context_to_message(
                    context=s,
                    query=User(Content(program.original_query)),
                    max_tokens=executor.max_tokens()
                ) for s in program.executed_stack.peek(0)
            ]

            messages.extend(context)

            prompt = Helpers.load_and_populate_prompt(
                prompt_filename=f'prompts/{statement.token()}_result.prompt',
                template={
                    'original_query': str(program.original_query),
                    'answer': str(statement.result()),
                }
            )

            # execute the llm_call
            messages.append(User(Content(prompt['user_message'])))
            assistant: Assistant = executor.execute(messages)

            response_writer(prompt['prompt_filename'], assistant)
            statement._result = assistant

            program.answers.append(statement)
            return statement

        # execute function call
        elif (
            isinstance(statement, FunctionCall)
            and not statement.result()
        ):
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
            messages: List[Message] = []
            if statement.supporting_system: messages.append(statement.supporting_system)
            if statement.supporting_messages: messages.extend(statement.supporting_messages)

            if statement.context and isinstance(statement.context, StackNode):
                # dataframe or stack context peek
                stack_statements = self.__stacknode_to_statement(program, statement.context)
                context = [
                    self.__context_to_message(
                        context=s,
                        query=User(Content(program.original_query)),
                        max_tokens=executor.max_tokens()
                    ) for s in stack_statements
                ]
                messages.extend(context)

            prompt = Helpers.load_and_populate_prompt(
                prompt_filename=f'prompts/{statement.token()}.prompt',
                template={
                    'llm_call_message': str(statement.message),
                }
            )

            # execute the llm_call
            messages.append(User(Content(prompt['user_message'])))
            assistant: Assistant = executor.execute(messages)

            response_writer(prompt['prompt_filename'], assistant)
            statement._result = assistant
            return statement

        # foreach
        elif (
            isinstance(statement, ForEach)
            and not statement.result()
        ):
            # deal with the first argument (lhs), which is the list context for the foreach loop
            # we will need to shape that into a dataframe that the foreach can execute
            context: List[Message] = []
            assistant: Assistant

            if isinstance(statement.lhs, StackNode):
                # dataframe or stack context peek
                stack_statements = self.__stacknode_to_statement(program, statement.lhs)
                context_messages = [
                    self.__context_to_message(
                        context=s,
                        query=User(Content(program.original_query)),
                        max_tokens=executor.max_tokens()
                    ) for s in stack_statements
                ]
                context.extend(context_messages)

            # now deal with the right hand side
            if isinstance(statement.rhs, FunctionCall):
                prompt = Helpers.load_and_populate_prompt(
                    prompt_filename=f'prompts/{statement.token()}_{statement.rhs.token()}.prompt',
                    template={
                        'function_call': statement.rhs.to_definition(),
                    }
                )

                foreach_function_response = ''
                messages = context + [User(Content(prompt['user_message']))]

                assistant = executor.execute(messages=messages)
                response_writer(prompt['prompt_filename'], assistant)
                foreach_function_response = str(assistant.message)

                if '"missing"' in str(assistant.message):
                    # try pushing the LLM to re-write the correct function call
                    # todo: I think this is wrong, context should be enough.
                    previous_messages = [str(s.result()) for s in program.runtime_stack.peek(0)]

                    function_call_rewrite = Helpers.load_and_populate_prompt(
                        f'prompts/{statement.token()}_error_correction.prompt',
                        template={
                            'function_calls_missing': str(assistant.message),
                            'function_call_signatures': '\n'.join(
                                [Helpers.get_function_description_flat_extra(f) for f in self.agents]
                            ),

                            'previous_messages': '\n'.join(previous_messages)
                        }
                    )
                    assistant = executor.execute(messages=[User(Content(function_call_rewrite['user_message']))])
                    response_writer(function_call_rewrite['prompt_filename'], assistant)
                    # statement._result = assistant.message
                    foreach_function_response = str(assistant.message)

                # if this is a program, we need to compile and interpret
                foreach_program = Parser().parse_program(
                    message=foreach_function_response,
                    agents=self.agents,
                    executor=program.executor
                )

                # execute it
                if foreach_program:
                    statement._result = [
                        self.execute_statement(s, executor, foreach_program).result()
                        for s in foreach_program.statements
                    ]
                    return statement
                else:
                    statement._result = foreach_function_response

            elif isinstance(statement.rhs, LLMCall):
                prompt = Helpers.load_and_populate_prompt(
                    prompt_filename=f'prompts/{statement.token()}_{statement.rhs.token()}.prompt',
                    template={
                        'message': str(statement.rhs.message),
                    }
                )

                messages = context + [User(Content(prompt['user_message']))]
                assistant = executor.execute(messages=messages)

                response_writer(prompt['prompt_filename'], assistant)
                statement._result = assistant.message
                return statement

            elif isinstance(statement.rhs, Answer):
                prompt = Helpers.load_and_populate_prompt(
                    prompt_filename=f'prompts/{statement.token()}_{statement.rhs.token()}.prompt',
                    template={
                        'message': str(statement.rhs._result),
                    }
                )

                messages = context + [User(Content(prompt['user_message']))]
                assistant = executor.execute(messages=messages)

                response_writer(prompt['prompt_filename'], assistant)
                statement._result = assistant.message
                return statement

            else:
                raise ValueError('shouldnt be here')

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
            if not isinstance(result, Answer):
                program.runtime_stack.push(result)
                # track all executed nodes so that Answer
                # can do a double check.
                program.executed_stack.push(result)

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

            # debug output
            response_writer('llm_call', assistant_response)

            # parse the response
            program = Parser().parse_program(assistant_response, self.agents, executor)
            program.conversation.append(response)
            program.original_query = str(call.message)

            # execute the program
            program = self.execute_program(program)
            results.extend(program.answers)
        else:
            assistant_reply: Assistant = self.execute_chat(call.supporting_messages + [call.message])
            results.append(Answer(conversation=[Content(str(assistant_reply.message))]))

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
