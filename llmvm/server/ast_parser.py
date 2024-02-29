from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from llmvm.common.helpers import Helpers
from llmvm.common.objects import Content, FunctionCall

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
        self.errors = []

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
