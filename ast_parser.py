import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from helpers.helpers import Helpers
from objects import (Answer, Content, DataFrame, Executor, ForEach,
                     FunctionCall, Get, LLMCall, Program, Set, StackNode,
                     Statement, UncertainOrError, User)

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
    ):
        self.message: str = ''
        self.remainder: str = ''
        self.index = 0
        self.agents: List[Callable] = []
        self.errors: List[UncertainOrError] = []

    def __parse_string(
        self,
    ) -> Optional[str]:
        # todo: should deal with \" char in string
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

    def __get_callsite_helper(
        self,
        call: str,
        functions: List[Callable]
    ) -> Optional[Tuple[Callable, Dict[str, Any]]]:
        function_description: Dict[str, Any] = {}

        function_name = Helpers.in_between(call, '', '(')
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

    def get_callsite(self, call_str: str) -> Optional[FunctionCall]:
        callsite = self.__get_callsite_helper(call_str, self.agents)
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

    def consume(self, token: str):
        if token in self.remainder:
            self.remainder = self.remainder[self.remainder.index(token) + len(token):]
            self.remainder = self.remainder.strip()

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
            function_call: Optional[FunctionCall] = self.get_callsite(function_call_str)
            if function_call:
                function_call.context = Content(text)
                function_call._ast_text = f'function_call({start_token}{function_call_str})'

            # remainder is the stuff after the end_token
            self.remainder = text[text.index(end_token) + len(end_token):]
            return function_call
        return None

    def parse_dataframe(self) -> Optional[DataFrame]:
        if self.remainder.startswith('dataframe('):
            self.consume('dataframe(')
            elements = []
            while self.remainder.strip() != '' and not self.remainder.startswith(')'):
                # todo this is probably broken
                element = self.__parse_string() or self.parse_stack()
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
            variable = cast(str, self.__parse_string())
            self.consume(')')
            return Get(variable)
        else:
            return None

    def parse_set(self) -> Optional[Set]:
        re = self.remainder.strip()

        if re.startswith('set(') and ')' in re:
            self.consume('set(')
            variable = cast(str, self.__parse_string())
            name = ''
            if self.remainder.startswith(','):
                self.consume(',')
                name = cast(str, self.__parse_string())
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
                    result = self.__parse_string()

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
                string_message = Content(self.__parse_string())
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
                    return LLMCall(message=User(Content(' '.join(consumed_tokens))))

                consumed_tokens.append(tokens[0])
                self.remainder = self.remainder[len(tokens[0]) + 1:]
                tokens = tokens[1:]

            self.remainder = ''
            return LLMCall(
                message=User(Content(' '.join(consumed_tokens))),
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
